import { describe, expect, test, vi } from "bun:test";
import type { EnvConfig, ThreadEventData, ThreadRef } from "@pinky/core";
import type { A2AEnvelope, Messenger } from "@pinky/runtime";
import { createGateway, type EventSink, type RawIngress } from "../src/server";
import { signSlackRequest } from "../src/slack/verify";
import { signA2ABody } from "../src/a2a-relay";

/** Microtask drain: lets a settled handler promise's .finally chain run. */
async function flushAsync(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

const TENANT = "t1";
const SLACK_SECRET = "slack-signing";
const A2A_SECRET = "a2a-secret";
const BOT = "U0BOT";

/**
 * Atomic sink double: the dedup claim and the appends land together, or the
 * whole call is a no-op. `failOn` makes an append blow up so the test can
 * assert the id is NOT left claimed (the lost-message defect).
 */
class FakeEvents implements EventSink {
  readonly seen = new Set<string>();
  readonly appended: { ref: ThreadRef; data: ThreadEventData }[] = [];
  /** One entry per ingest() call — the batch is what makes it atomic. */
  readonly calls: { externalId: string; types: string[] }[] = [];
  /** Throw from ingest when the external id matches. */
  failOn: string | null = null;

  ingest(ref: ThreadRef, externalId: string, data: ThreadEventData[]): Promise<unknown[] | null> {
    this.calls.push({ externalId, types: data.map((d) => d.type) });
    const key = `${ref.tenantId}:${externalId}`;
    if (this.seen.has(key)) return Promise.resolve(null);
    if (this.failOn === externalId) {
      // Transaction rolls back: no dedup row, no events.
      return Promise.reject(new Error("append exploded"));
    }
    this.seen.add(key);
    for (const d of data) this.appended.push({ ref, data: d });
    return Promise.resolve(data.map(() => ({})));
  }
}

class FakeMessenger implements Messenger {
  readonly nodeId = "local";
  readonly sent: Omit<A2AEnvelope, "id" | "sentAt">[] = [];
  readonly received: A2AEnvelope[] = [];
  private readonly seen = new Set<string>();

  send(env: Omit<A2AEnvelope, "id" | "sentAt">): Promise<string> {
    this.sent.push(env);
    return Promise.resolve("m1");
  }

  inbox(_agentId: string): Promise<A2AEnvelope[]> {
    return Promise.resolve([]);
  }

  onMessage(_agentId: string, _handler: (env: A2AEnvelope) => void): () => void {
    return () => {};
  }

  receive(env: A2AEnvelope): Promise<boolean> {
    this.received.push(env);
    const fresh = !this.seen.has(env.id);
    this.seen.add(env.id);
    return Promise.resolve(fresh);
  }
}

function makeEnv(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    databaseUrl: "postgres://unused",
    databaseAdminUrl: "postgres://unused",
    nodeId: "local",
    peers: {},
    a2aSecret: A2A_SECRET,
    slack: { botToken: "xoxb-test", signingSecret: SLACK_SECRET },
    port: 0,
    ...overrides,
  };
}

function slackBody(overrides: Record<string, unknown> = {}, eventId = "Ev1"): string {
  return JSON.stringify({
    type: "event_callback",
    event_id: eventId,
    event: {
      type: "message",
      channel: "C1",
      ts: "1700000000.000001",
      user: "U0HUMAN",
      text: "hi <@U0BOT>",
      ...overrides,
    },
  });
}

function slackRequest(body: string, sign = true): Request {
  const ts = String(Math.floor(Date.now() / 1000));
  const headers = new Headers({
    "x-slack-request-timestamp": ts,
    "x-slack-signature": sign ? signSlackRequest(SLACK_SECRET, ts, body) : "v0=bad",
  });
  return new Request("http://gw/slack/events", { method: "POST", headers, body });
}

function a2aRequest(envelope: A2AEnvelope, sign = true): Request {
  const body = JSON.stringify(envelope);
  const headers = new Headers({
    "x-pinky-signature": sign ? signA2ABody(A2A_SECRET, envelope.id, envelope.sentAt, body) : "bad",
  });
  return new Request("http://gw/a2a/deliver", { method: "POST", headers, body });
}

function makeGateway(overrides: {
  events?: FakeEvents;
  env?: Partial<EnvConfig>;
  runAgent?: (thread: ThreadRef, batch: RawIngress[]) => Promise<void>;
} = {}) {
  const events = overrides.events ?? new FakeEvents();
  const messenger = new FakeMessenger();
  // One entry per runAgent invocation — the batch is what the defect was about.
  const runAgentCalls: { thread: ThreadRef; texts: string[] }[] = [];
  const runAgent =
    overrides.runAgent ??
    ((thread: ThreadRef, batch: RawIngress[]): Promise<void> => {
      runAgentCalls.push({ thread, texts: batch.map((i) => i.text) });
      return Promise.resolve();
    });
  const handle = createGateway({
    env: makeEnv(overrides.env ?? {}),
    tenantId: TENANT,
    messenger,
    events,
    botUserId: BOT,
    runAgent,
  });
  return { handle, events, messenger, runAgentCalls };
}

describe("createGateway", () => {
  test("GET /healthz returns ok", async () => {
    const { handle } = makeGateway();
    const res = await handle(new Request("http://gw/healthz"));
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("ok");
  });

  test("unknown path is 404", async () => {
    const { handle } = makeGateway();
    const res = await handle(new Request("http://gw/nope", { method: "POST" }));
    expect(res.status).toBe(404);
  });

  test("url_verification echoes the challenge", async () => {
    const { handle } = makeGateway();
    const body = JSON.stringify({ type: "url_verification", challenge: "CHAL" });
    const res = await handle(slackRequest(body));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ challenge: "CHAL" });
  });

  test("bad slack signature is 401", async () => {
    const { handle } = makeGateway();
    const res = await handle(slackRequest(slackBody(), false));
    expect(res.status).toBe(401);
  });

  test("duplicate event short-circuits without appending", async () => {
    const { handle, events } = makeGateway();
    const first = await handle(slackRequest(slackBody()));
    expect(first.status).toBe(200);
    const appendedAfterFirst = events.appended.length;

    const second = await handle(slackRequest(slackBody())); // same event_id
    expect(second.status).toBe(200);
    expect(events.appended.length).toBe(appendedAfterFirst);
  });

  test("ingress + decision are written in ONE ingest call, not two appends", async () => {
    vi.useFakeTimers();
    const { handle, events } = makeGateway();
    expect((await handle(slackRequest(slackBody()))).status).toBe(200);
    expect(events.calls).toEqual([{ externalId: "Ev1", types: ["ingress", "decision"] }]);
    vi.useRealTimers();
  });

  test("a failed ingest leaves the event id unclaimed so Slack's retry is processed", async () => {
    vi.useFakeTimers();
    const events = new FakeEvents();
    events.failOn = "EvBoom";
    const { handle } = makeGateway({ events });
    // The handler logs the failure on purpose; keep it out of the test output.
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});

    // The transaction rolls back: nothing recorded, nothing claimed, and
    // Slack is told to retry rather than being acked.
    const boom = await handle(slackRequest(slackBody({}, "EvBoom")));
    expect(boom.status).toBe(500);
    expect(events.appended).toHaveLength(0);
    expect(events.seen.size).toBe(0);

    // Slack retries the same event_id — it must land, not be dropped as a dup.
    events.failOn = null;
    const retry = await handle(slackRequest(slackBody({}, "EvBoom")));
    expect(retry.status).toBe(200);
    expect(events.appended.map((a) => a.data.type)).toEqual(["ingress", "decision"]);
    expect(logged).toHaveBeenCalledTimes(1);
    logged.mockRestore();
    vi.useRealTimers();
  });

  test("non-message events ack without appending", async () => {
    const { handle, events } = makeGateway();
    const body = JSON.stringify({
      type: "event_callback",
      event_id: "Ev2",
      event: { type: "reaction_added" },
    });
    const res = await handle(slackRequest(body));
    expect(res.status).toBe(200);
    expect(events.appended.length).toBe(0);
  });

  test("silent path writes ingress + silent decision, no agent run", async () => {
    vi.useFakeTimers();
    const { handle, events } = makeGateway();
    const body = slackBody({ text: "ambient chatter" });
    const res = await handle(slackRequest(body));
    expect(res.status).toBe(200);

    expect(events.appended.map((a) => a.data.type)).toEqual(["ingress", "decision"]);
    const decision = events.appended[1]?.data;
    expect(decision).toMatchObject({ type: "decision", action: "silent", reason: "ambient" });
    vi.useRealTimers();
  });

  test("engage path writes reply decision and eventually runs the agent", async () => {
    vi.useFakeTimers();
    const { handle, events, runAgentCalls } = makeGateway();
    const res = await handle(slackRequest(slackBody()));
    expect(res.status).toBe(200);

    expect(events.appended.map((a) => a.data.type)).toEqual(["ingress", "decision"]);
    expect(events.appended[1]?.data).toMatchObject({
      type: "decision",
      action: "reply",
      reason: "mention",
    });

    // Debounced: nothing yet.
    expect(runAgentCalls.length).toBe(0);
    vi.advanceTimersByTime(500);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(runAgentCalls).toEqual([
      {
        thread: { tenantId: TENANT, channelId: "slack:C1", threadId: "1700000000.000001" },
        texts: ["hi"], // mention stripped
      },
    ]);
    vi.useRealTimers();
  });

  test("a 3-message burst in one thread produces exactly ONE run with a 3-item batch", async () => {
    vi.useFakeTimers();
    const { handle, events, runAgentCalls } = makeGateway();

    // Same channel + same thread_ts → same lane key, inside the debounce window.
    for (const [i, word] of ["one", "two", "three"].entries()) {
      const body = slackBody(
        {
          text: `<@${BOT}> ${word}`,
          ts: `1700000000.00000${i + 2}`,
          thread_ts: "1700000000.000001",
        },
        `EvBurst${i}`,
      );
      const res = await handle(slackRequest(body));
      expect(res.status).toBe(200);
      vi.advanceTimersByTime(100); // still inside the 500ms window
    }

    // All three are already in the log; the run must not be repeated per message.
    expect(events.appended.filter((a) => a.data.type === "ingress").length).toBe(3);
    expect(runAgentCalls.length).toBe(0);

    vi.advanceTimersByTime(500);
    await flushAsync();

    expect(runAgentCalls.length).toBe(1);
    expect(runAgentCalls[0]).toEqual({
      thread: { tenantId: TENANT, channelId: "slack:C1", threadId: "1700000000.000001" },
      texts: ["one", "two", "three"],
    });
    vi.useRealTimers();
  });

  test("two threads run independently and no batch spans conversations", async () => {
    vi.useFakeTimers();
    const { handle, runAgentCalls } = makeGateway();

    const messages: [string, string, string, string][] = [
      // [channel, thread_ts, text, event_id]
      ["C1", "1700000000.000001", "a1", "EvA1"],
      ["C2", "1700000000.000009", "b1", "EvB1"],
      ["C1", "1700000000.000001", "a2", "EvA2"],
      ["C2", "1700000000.000009", "b2", "EvB2"],
    ];
    for (const [channel, threadTs, text, eventId] of messages) {
      const body = slackBody(
        { channel, thread_ts: threadTs, ts: `${threadTs}${eventId}`, text: `<@${BOT}> ${text}` },
        eventId,
      );
      expect((await handle(slackRequest(body))).status).toBe(200);
    }

    vi.advanceTimersByTime(500);
    await flushAsync();

    expect(runAgentCalls.length).toBe(2);
    const byChannel = new Map(runAgentCalls.map((c) => [c.thread.channelId, c]));
    expect(byChannel.get("slack:C1")).toEqual({
      thread: { tenantId: TENANT, channelId: "slack:C1", threadId: "1700000000.000001" },
      texts: ["a1", "a2"],
    });
    expect(byChannel.get("slack:C2")).toEqual({
      thread: { tenantId: TENANT, channelId: "slack:C2", threadId: "1700000000.000009" },
      texts: ["b1", "b2"],
    });
    vi.useRealTimers();
  });

  test("POST /a2a/deliver accepts a valid signed envelope", async () => {
    const { handle, messenger } = makeGateway();
    const env: A2AEnvelope = {
      id: "e1",
      from: "agent-a@node1",
      to: "agent-b@local",
      kind: "message",
      text: "hello",
      sentAt: new Date().toISOString(),
    };
    const res = await handle(a2aRequest(env));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, duplicate: false });
    // receive() gets the peer's envelope verbatim — same id, same sentAt.
    expect(messenger.received).toEqual([env]);
    expect(messenger.sent).toHaveLength(0); // never re-minted through send()
  });

  test("POST /a2a/deliver is 503 when A2A_SECRET is empty", async () => {
    const { handle, messenger } = makeGateway({ env: { a2aSecret: "" } });
    const env: A2AEnvelope = {
      id: "e3",
      from: "agent-a@node1",
      to: "agent-b@local",
      kind: "message",
      text: "hello",
      sentAt: new Date().toISOString(),
    };
    // Signed with the empty key — exactly what an attacker could compute.
    const body = JSON.stringify(env);
    const req = new Request("http://gw/a2a/deliver", {
      method: "POST",
      headers: { "x-pinky-signature": signA2ABody("", env.id, env.sentAt, body) },
      body,
    });
    const res = await handle(req);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ ok: false, error: "a2a disabled: no A2A_SECRET" });
    expect(messenger.received).toHaveLength(0);
  });

  test("POST /a2a/deliver rejects a bad signature with 401", async () => {
    const { handle, messenger } = makeGateway();
    const env: A2AEnvelope = {
      id: "e2",
      from: "agent-a@node1",
      to: "agent-b@local",
      kind: "request",
      text: "ping",
      sentAt: new Date().toISOString(),
    };
    const res = await handle(a2aRequest(env, false));
    expect(res.status).toBe(401);
    expect(messenger.received.length).toBe(0);
    expect(messenger.sent.length).toBe(0);
  });

  test("POST /a2a/deliver rejects malformed bodies with 400", async () => {
    const { handle } = makeGateway();
    const req = new Request("http://gw/a2a/deliver", {
      method: "POST",
      headers: { "x-pinky-signature": "whatever" },
      body: JSON.stringify({ kind: "message" }),
    });
    const res = await handle(req);
    expect(res.status).toBe(400);
  });

  test("GET on /a2a/deliver is 405", async () => {
    const { handle } = makeGateway();
    const res = await handle(new Request("http://gw/a2a/deliver"));
    expect(res.status).toBe(405);
  });
});
