/**
 * Gateway HTTP server: Slack Events API ingress + A2A cross-node delivery.
 *
 * Flow (DESIGN.md §6): persist → dedup → gate → enqueue. Everything after the
 * gate is async (an enqueued agent run) so the HTTP response always lands
 * well inside Slack's 3-second ack window. The lane debounce coalesces a
 * same-thread burst into ONE batched turn — one runAgent call per batch, not
 * per message (§6: "debounce ~500ms same-author burst → one batched turn").
 *
 * Behavioral config (model, thresholds, gate policy) lives in the `settings`
 * table, not here. The gateway receives only bootstrap EnvConfig + an
 * explicit tenantId; the CLI-owned runAgent callback re-loads settings per
 * wake, which keeps the hot-reload boundary outside the gateway.
 *
 * The event-store surface is the narrow `EventSink` interface below: either
 * the real core EventStore or a test fake. It exposes exactly one method,
 * ingest(), because dedup and append have to be ONE transaction here — see
 * the note on EventSink.
 */
import type { EnvConfig, Principal, ThreadEventData, ThreadRef } from "@pinky/core";
import { gateEvent } from "./slack/gate";
import { normalizeSlackEvent, type NormalizedSlackMessage } from "./slack/normalize";
import { verifySlackRequest } from "./slack/verify";
import { handleA2ADeliver } from "./a2a-relay";
import { LaneQueue } from "./lanes";
import type { Messenger } from "@pinky/runtime";

/**
 * Minimal event-store surface the gateway depends on.
 *
 * Deliberately just ingest(): claiming the dedup id and writing the events it
 * unlocks is one atomic step. Split into `dedup()` then `append()`, a failure
 * between them leaves the id claimed with nothing in the log, and Slack's
 * retry of that event_id is then discarded as a duplicate — the message is
 * lost permanently. ingest() returns null when the id was already seen (retry:
 * ack it and do nothing) and the appended events otherwise.
 */
export interface EventSink {
  ingest(
    ref: ThreadRef,
    externalId: string,
    data: ThreadEventData[],
  ): Promise<unknown[] | null>;
}

/** One gated ingress message, already persisted to the log. */
export interface RawIngress {
  text: string;
  author: Principal;
  externalId: string;
}

export interface GatewayOpts {
  /** Bootstrap config only (db url, secrets, port). Behavior lives in settings. */
  env: EnvConfig;
  /** Thread identity + dedup scope — from the loaded settings snapshot at CLI startup. */
  tenantId: string;
  messenger: Messenger;
  events: EventSink;
  /** The bot's own Slack user id (mention/reply detection). Empty string disables detection. */
  botUserId?: string;
  /**
   * Resolves one debounced ingress batch into exactly ONE agent run. The batch
   * is already in the event log, so the run's projection sees every message in
   * it; running per item would re-answer the same burst N times.
   * Errors are logged, never rethrown into the lane.
   */
  runAgent: (thread: ThreadRef, batch: RawIngress[]) => Promise<void>;
}

function respond(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Build the fetch handler. Routing: GET /healthz, POST /slack/events,
 * POST /a2a/deliver. Everything else is 404.
 */
export function createGateway(opts: GatewayOpts): (req: Request) => Promise<Response> {
  const slack = new LaneQueue<RawIngress>(async (key, batch) => {
    if (batch.length === 0) return;
    // The lane key IS the conversation identity (`${channelId}:${threadId}`),
    // so a batch never spans conversations: derive the thread ref once, from
    // the key, and make exactly one run for the whole debounced burst.
    const [channelId, threadId] = key.split(":", 2);
    const thread: ThreadRef = {
      tenantId: opts.tenantId,
      channelId: `slack:${channelId}`,
      threadId: threadId ?? "",
    };
    try {
      await opts.runAgent(thread, batch);
    } catch (error) {
      console.error(`runAgent failed for ${thread.channelId}/${thread.threadId}:`, error);
    }
  });

  return async (req: Request): Promise<Response> => {
    const url = new URL(req.url);

    if (req.method === "GET" && url.pathname === "/healthz") {
      return new Response("ok", { status: 200 });
    }

    if (req.method === "POST" && url.pathname === "/slack/events") {
      return handleSlackEvent(req, opts, slack);
    }

    if (url.pathname === "/a2a/deliver") {
      // An empty HMAC key is not "no auth", it is "auth anyone can forge":
      // signA2ABody("", ...) is computable by any caller. Refuse the route
      // outright rather than accept everything that reaches it.
      if (opts.env.a2aSecret.trim() === "") {
        return respond(503, { ok: false, error: "a2a disabled: no A2A_SECRET" });
      }
      return handleA2ADeliver(req, { secret: opts.env.a2aSecret, messenger: opts.messenger });
    }

    return respond(404, { ok: false, error: "not found" });
  };
}

async function handleSlackEvent(
  req: Request,
  opts: GatewayOpts,
  slack: LaneQueue<RawIngress>,
): Promise<Response> {
  const rawBody = await req.text();
  const signature = req.headers.get("x-slack-signature") ?? "";
  const timestamp = req.headers.get("x-slack-request-timestamp") ?? "";

  if (!verifySlackRequest(opts.env.slack.signingSecret, timestamp, rawBody, signature)) {
    return respond(401, { ok: false, error: "invalid signature" });
  }

  let body: unknown;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return respond(400, { ok: false, error: "invalid JSON" });
  }

  // Slack endpoint verification handshake.
  if (
    typeof body === "object" &&
    body !== null &&
    (body as Record<string, unknown>).type === "url_verification"
  ) {
    const challenge = (body as Record<string, unknown>).challenge;
    return respond(200, { challenge: typeof challenge === "string" ? challenge : "" });
  }

  const normalized = normalizeSlackEvent(body, opts.botUserId ?? "");
  if (!normalized) {
    // Non-message event (reaction, join, …): ack Slack and drop it.
    return respond(200, { ok: true });
  }

  const gate = gateEvent(normalized);
  const thread: ThreadRef = {
    tenantId: opts.tenantId,
    channelId: `slack:${normalized.channelId}`,
    threadId: normalized.threadId,
  };

  // Dedup claim + both events in ONE transaction: either Slack's event_id is
  // recorded WITH its ingress and decision, or neither is and the retry gets
  // another chance. Never a claimed id with an empty log.
  let written: unknown[] | null;
  try {
    written = await opts.events.ingest(thread, normalized.externalId, [
      ingressData(normalized),
      {
        type: "decision",
        action: gate.action === "engage" ? "reply" : "silent",
        reason: gate.reason,
      },
    ]);
  } catch (error) {
    // The transaction rolled back, so the event_id is still unclaimed. Answer
    // non-2xx on purpose: Slack's retry of this id will be handled as fresh.
    console.error(`ingest failed for ${normalized.externalId}:`, error);
    return respond(500, { ok: false, error: "ingest failed" });
  }
  if (written === null) {
    // Retried delivery — already recorded. Ack immediately.
    return respond(200, { ok: true });
  }

  if (gate.action === "engage") {
    slack.enqueue(`${normalized.channelId}:${normalized.threadId}`, {
      text: normalized.text,
      author: normalized.author,
      externalId: normalized.externalId,
    });
  }

  return respond(200, { ok: true });
}

function ingressData(msg: NormalizedSlackMessage): ThreadEventData {
  return {
    type: "ingress",
    platform: "slack",
    author: msg.author,
    text: msg.text,
    refs: [],
    externalId: msg.externalId,
  };
}

/** Bun.serve entry point. */
export function startGateway(opts: GatewayOpts): ReturnType<typeof Bun.serve> {
  return Bun.serve({
    port: opts.env.port,
    fetch: createGateway(opts),
  });
}
