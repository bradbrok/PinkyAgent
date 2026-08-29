/**
 * Gateway HTTP server: A2A cross-node delivery (DESIGN.md §7).
 *
 * This process used to also front a Slack Events API ingress. It no longer
 * does: the primary interface is the JSONL headless mode (headless.ts), which
 * needs no socket at all. What is left here is the one thing A2A genuinely
 * requires a listening port for — a peer node POSTing a signed envelope at
 * /a2a/deliver so this node's mailbox can claim delivery and wake the agent.
 *
 * Routing: GET /healthz, POST /a2a/deliver, everything else 404.
 *
 * Behavioral config (model, thresholds, gate policy) lives in the `settings`
 * table, not here. The server receives only bootstrap EnvConfig, which keeps
 * the hot-reload boundary outside the gateway.
 */
import type { EnvConfig, Principal, ThreadEventData, ThreadRef } from "@pinky/core";
import { handleA2ADeliver } from "./a2a-relay";
import type { Messenger } from "@pinky/runtime";

/**
 * Minimal event-store surface an ingress front-end depends on (headless.ts is
 * the consumer today; it lives here because it is the package's ingress
 * vocabulary, not the JSONL protocol's).
 *
 * Deliberately just ingest(): claiming the dedup id and writing the events it
 * unlocks is one atomic step. Split into `dedup()` then `append()`, a failure
 * between them leaves the id claimed with nothing in the log, and a client's
 * retry of that id is then discarded as a duplicate — the message is lost
 * permanently. ingest() returns null when the id was already seen (retry: ack
 * it and do nothing) and the appended events otherwise.
 */
export interface EventSink {
  ingest(
    ref: ThreadRef,
    externalId: string,
    data: ThreadEventData[],
  ): Promise<unknown[] | null>;
}

/** One ingress message, already persisted to the log. */
export interface RawIngress {
  text: string;
  author: Principal;
  externalId: string;
}

export interface GatewayOpts {
  /** Bootstrap config only (db url, secrets, port). Behavior lives in settings. */
  env: EnvConfig;
  messenger: Messenger;
}

function respond(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Build the fetch handler. */
export function createGateway(opts: GatewayOpts): (req: Request) => Promise<Response> {
  return async (req: Request): Promise<Response> => {
    const url = new URL(req.url);

    if (req.method === "GET" && url.pathname === "/healthz") {
      return new Response("ok", { status: 200 });
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

/** Bun.serve entry point. */
export function startGateway(opts: GatewayOpts): ReturnType<typeof Bun.serve> {
  return Bun.serve({
    port: opts.env.port,
    fetch: createGateway(opts),
  });
}
