/**
 * A2A relay tests (DESIGN.md §7). The handler is the receiving half of
 * at-least-once: verify the HMAC, then hand the envelope to
 * messenger.receive(), which CLAIMS delivery of the envelope id. A sender
 * retry loses the claim and must be a 200 with no second row and no second
 * wakeup — that is what `duplicate: true` reports.
 */
import { describe, expect, test } from "bun:test";
import type { A2AEnvelope, Messenger } from "@pinky/runtime";
import { handleA2ADeliver, signA2ABody } from "../src/a2a-relay";

const SECRET = "a2a-secret";

/** Messenger double with the durable receive() contract: the first caller to
 *  claim an id wins and wakes subscribers; everyone after it gets false. */
class FakeMessenger implements Messenger {
  readonly nodeId = "node2";
  readonly received: A2AEnvelope[] = [];
  readonly sent: Omit<A2AEnvelope, "id" | "sentAt">[] = [];
  readonly wakeups: A2AEnvelope[] = [];
  private readonly seen = new Set<string>();

  send(env: Omit<A2AEnvelope, "id" | "sentAt">): Promise<string> {
    this.sent.push(env);
    return Promise.resolve("minted-id");
  }
  inbox(): Promise<A2AEnvelope[]> {
    return Promise.resolve([]);
  }
  onMessage(): () => void {
    return () => {};
  }
  receive(env: A2AEnvelope): Promise<boolean> {
    this.received.push(env);
    if (this.seen.has(env.id)) return Promise.resolve(false); // claim lost
    this.seen.add(env.id);
    this.wakeups.push(env); // subscribers fire only for the winning claim
    return Promise.resolve(true);
  }
  // Consumption edge (issue #4). The relay never touches it — delivery is not
  // consumption — so the double only has to satisfy the interface.
  redeliverUnconsumed(): Promise<number> {
    return Promise.resolve(0);
  }
  claimConsumption(): Promise<boolean> {
    return Promise.resolve(false);
  }
}

function envelope(overrides: Partial<A2AEnvelope> = {}): A2AEnvelope {
  return {
    id: "peer-msg-1",
    from: "alpha@local",
    to: "beta@node2",
    kind: "request",
    text: "what is 2+2?",
    sentAt: new Date().toISOString(),
    ...overrides,
  };
}

function request(env: A2AEnvelope, opts: { sign?: boolean } = {}): Request {
  const body = JSON.stringify(env);
  const signature = opts.sign === false ? "bad" : signA2ABody(SECRET, env.id, env.sentAt, body);
  return new Request("http://gw/a2a/deliver", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-pinky-signature": signature,
      "x-pinky-ts": env.sentAt,
    },
    body,
  });
}

describe("handleA2ADeliver", () => {
  test("accepts a valid signed envelope and passes it to receive() intact", async () => {
    const messenger = new FakeMessenger();
    const env = envelope({ threadHint: "alpha" });
    const res = await handleA2ADeliver(request(env), { secret: SECRET, messenger });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, duplicate: false });
    expect(messenger.received).toEqual([env]); // ORIGINAL id + sentAt preserved
    expect(messenger.sent).toHaveLength(0); // never re-mints via send()
    expect(messenger.wakeups).toHaveLength(1);
  });

  test("a duplicate delivery is 200 with no second wakeup", async () => {
    const messenger = new FakeMessenger();
    const env = envelope();

    const first = await handleA2ADeliver(request(env), { secret: SECRET, messenger });
    const second = await handleA2ADeliver(request(env), { secret: SECRET, messenger });

    expect(first.status).toBe(200);
    expect(await first.json()).toEqual({ ok: true, duplicate: false });
    expect(second.status).toBe(200);
    expect(await second.json()).toEqual({ ok: true, duplicate: true });
    expect(messenger.received).toHaveLength(2); // both reached the messenger
    expect(messenger.wakeups).toHaveLength(1); // ...but only one woke anyone
  });

  test("rejects a bad signature with 401 and never touches the messenger", async () => {
    const messenger = new FakeMessenger();
    const res = await handleA2ADeliver(request(envelope(), { sign: false }), {
      secret: SECRET,
      messenger,
    });
    expect(res.status).toBe(401);
    expect(messenger.received).toHaveLength(0);
  });

  test("rejects a stale envelope outside the 300s freshness window", async () => {
    const messenger = new FakeMessenger();
    const stale = envelope({ sentAt: new Date(Date.now() - 301_000).toISOString() });
    const res = await handleA2ADeliver(request(stale), { secret: SECRET, messenger });
    expect(res.status).toBe(401);
    expect(messenger.received).toHaveLength(0);
  });

  test("rejects a missing signature header", async () => {
    const messenger = new FakeMessenger();
    const env = envelope();
    const req = new Request("http://gw/a2a/deliver", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(env),
    });
    expect((await handleA2ADeliver(req, { secret: SECRET, messenger })).status).toBe(401);
    expect(messenger.received).toHaveLength(0);
  });

  test("rejects malformed JSON with 400 and a non-envelope body with 400", async () => {
    const messenger = new FakeMessenger();
    const bad = new Request("http://gw/a2a/deliver", { method: "POST", body: "{nope" });
    expect((await handleA2ADeliver(bad, { secret: SECRET, messenger })).status).toBe(400);

    const notEnvelope = new Request("http://gw/a2a/deliver", {
      method: "POST",
      body: JSON.stringify({ id: "x", from: "a@b" }),
    });
    expect((await handleA2ADeliver(notEnvelope, { secret: SECRET, messenger })).status).toBe(400);
    expect(messenger.received).toHaveLength(0);
  });

  test("GET is 405", async () => {
    const messenger = new FakeMessenger();
    const res = await handleA2ADeliver(new Request("http://gw/a2a/deliver"), {
      secret: SECRET,
      messenger,
    });
    expect(res.status).toBe(405);
  });
});
