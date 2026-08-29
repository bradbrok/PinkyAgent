import { describe, expect, test } from "bun:test";
import type { EnvConfig } from "@pinky/core";
import type { A2AEnvelope, Messenger } from "@pinky/runtime";
import { createGateway } from "../src/server";
import { signA2ABody } from "../src/a2a-relay";

const A2A_SECRET = "a2a-secret";

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

  // The consumption edge (issue #4) is the agent's, not the relay's.
  redeliverUnconsumed(_agentId: string): Promise<number> {
    return Promise.resolve(0);
  }

  claimConsumption(_id: string): Promise<boolean> {
    return Promise.resolve(false);
  }
}

function makeEnv(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    databaseUrl: "postgres://unused",
    databaseAdminUrl: "postgres://unused",
    nodeId: "local",
    peers: {},
    a2aSecret: A2A_SECRET,
    port: 0,
    ...overrides,
  };
}

function a2aRequest(envelope: A2AEnvelope, sign = true): Request {
  const body = JSON.stringify(envelope);
  const headers = new Headers({
    "x-pinky-signature": sign ? signA2ABody(A2A_SECRET, envelope.id, envelope.sentAt, body) : "bad",
  });
  return new Request("http://gw/a2a/deliver", { method: "POST", headers, body });
}

function makeGateway(overrides: { env?: Partial<EnvConfig> } = {}) {
  const messenger = new FakeMessenger();
  const handle = createGateway({ env: makeEnv(overrides.env ?? {}), messenger });
  return { handle, messenger };
}

describe("createGateway (A2A relay)", () => {
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

  test("the retired Slack ingress route is just another 404", async () => {
    const { handle } = makeGateway();
    const res = await handle(
      new Request("http://gw/slack/events", { method: "POST", body: "{}" }),
    );
    expect(res.status).toBe(404);
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
