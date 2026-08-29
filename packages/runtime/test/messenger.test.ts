/**
 * Messenger tests: LocalMessenger over a FakeDb implementing core Mailbox's
 * exact SQL. Row shape mirrors the a2a_messages table; `node_to` is the node
 * partition (compared against the *parameterized* node id, never a literal),
 * delivered_at/read_at are the lifecycle markers.
 */
import { describe, expect, test } from "bun:test";
import type { Db } from "@pinky/core";
import { a2aSignature, LocalMessenger } from "../src/messenger";
import type { A2AEnvelope } from "../src/types";

interface MailRow {
  id: string;
  from_agent: string;
  to_agent: string;
  node_from: string;
  node_to: string;
  kind: string;
  text: string;
  thread_hint: string | null;
  delivered_at: string | null;
  read_at: string | null;
  created_at: string;
}

const norm = (sql: string): string => sql.replace(/\s+/g, " ");

class FakeDb implements Db {
  readonly rows: MailRow[] = [];
  private clock = 0;

  private toRecord(r: MailRow): Record<string, unknown> {
    return { ...r };
  }

  private forAgent(agent: string, node: string): MailRow[] {
    return this.rows.filter(
      (r) => (r.to_agent === agent || r.to_agent === "broadcast") && r.node_to === node,
    );
  }

  private unread(agent: string, node: string): MailRow[] {
    return this.forAgent(agent, node).filter(
      (r) => r.delivered_at !== null && r.read_at === null,
    );
  }

  private undelivered(agent: string, node: string): MailRow[] {
    return this.forAgent(agent, node).filter((r) => r.delivered_at === null);
  }

  query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    const s = norm(sql);
    const p = params ?? [];

    if (/insert into a2a_messages/.test(s)) {
      const [id, from, to, nodeFrom, nodeTo, kind, text, threadHint] = p as [
        string, string, string, string, string, string, string, string | null,
      ];
      if (/on conflict \(id\) do nothing/.test(s) && this.rows.some((r) => r.id === id)) {
        return Promise.resolve([] as T[]); // duplicate: no row, no `returning`
      }
      this.rows.push({
        id,
        from_agent: from,
        to_agent: to,
        node_from: nodeFrom,
        node_to: nodeTo,
        kind,
        text,
        thread_hint: threadHint,
        delivered_at: null,
        read_at: null,
        created_at: `c${++this.clock}`,
      });
      return Promise.resolve((/returning id/.test(s) ? [{ id }] : []) as T[]);
    }

    // claimDelivery(id): receiver-side atomic claim, node-scoped, returns the
    // row only for the caller that flipped delivered_at. Must be matched
    // BEFORE the markDelivered branch — the two statements share a prefix.
    if (/update a2a_messages set delivered_at = now\(\) where id = \$1 and node_to = \$2/.test(s)) {
      const row = this.rows.find(
        (r) => r.id === p[0] && r.node_to === p[1] && r.delivered_at === null,
      );
      if (!row) return Promise.resolve([] as T[]);
      row.delivered_at = `d${++this.clock}`;
      return Promise.resolve([{ id: row.id }] as T[]);
    }
    // markDelivered(id)
    if (/update a2a_messages set delivered_at = now\(\) where id = \$1/.test(s)) {
      const row = this.rows.find((r) => r.id === p[0] && r.delivered_at === null);
      if (row) row.delivered_at = `d${++this.clock}`;
      return Promise.resolve([] as T[]);
    }
    // deliverLocal sweep
    if (/update a2a_messages set delivered_at = now\(\)/.test(s)) {
      for (const r of this.undelivered(p[0] as string, p[1] as string)) {
        r.delivered_at = `d${++this.clock}`;
      }
      return Promise.resolve([] as T[]);
    }
    if (/update a2a_messages set read_at = now\(\)/.test(s)) {
      for (const r of this.unread(p[0] as string, p[1] as string)) r.read_at = `r${++this.clock}`;
      return Promise.resolve([] as T[]);
    }
    if (/delivered_at is not null and read_at is null/.test(s)) {
      const limit = (p[2] as number) ?? 50;
      return Promise.resolve(
        this.unread(p[0] as string, p[1] as string).slice(0, limit).map((r) => this.toRecord(r)) as T[],
      );
    }
    if (/node_to = \$1 and node_to != \$2 and delivered_at is null/.test(s)) {
      const [target, self] = p as [string, string];
      return Promise.resolve(
        this.rows
          .filter((r) => r.node_to === target && r.node_to !== self && r.delivered_at === null)
          .map((r) => this.toRecord(r)) as T[],
      );
    }
    if (/delivered_at is null/.test(s)) {
      return Promise.resolve(
        this.undelivered(p[0] as string, p[1] as string).map((r) => this.toRecord(r)) as T[],
      );
    }
    return Promise.resolve([] as T[]);
  }

  queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    return this.query<T>(sql, params).then((rows) => rows[0] ?? null);
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }

  async close(): Promise<void> {}
}

function envelope(to: string, text = "hello"): Omit<A2AEnvelope, "id" | "sentAt"> {
  return { from: "pinky@local", to, kind: "message", text };
}
/** Flush the microtask queue so fire-and-forget delivery settles. */
function flushMicrotasks(): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  queueMicrotask(() => queueMicrotask(() => queueMicrotask(resolve)));
  return promise;
}

/** Capture console.warn for the duration of fn. */
async function withCapturedWarnings(fn: () => Promise<void>): Promise<unknown[][]> {
  const warnings: unknown[][] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };
  try {
    await fn();
  } finally {
    console.warn = originalWarn;
  }
  return warnings;
}

describe("a2aSignature", () => {
  test("is deterministic hex HMAC over id.ts.body", () => {
    const sig = a2aSignature("s3cret", "m1", "2026-08-28T00:00:00.000Z", '{"x":1}');
    expect(sig).toMatch(/^[0-9a-f]{64}$/);
    expect(a2aSignature("s3cret", "m1", "2026-08-28T00:00:00.000Z", '{"x":1}')).toBe(sig);
    expect(a2aSignature("other", "m1", "2026-08-28T00:00:00.000Z", '{"x":1}')).not.toBe(sig);
  });
});

describe("LocalMessenger", () => {
  test("local send persists, marks delivered, and fires subscribers", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "local" });
    const seen: A2AEnvelope[] = [];
    m.onMessage("worker", (env) => seen.push(env));

    const id = await m.send(envelope("worker@local", "ping"));
    expect(typeof id).toBe("string");
    expect(db.rows).toHaveLength(1);
    expect(db.rows[0]).toMatchObject({ to_agent: "worker", node_to: "local", text: "ping" });
    await flushMicrotasks();
    expect(db.rows[0]!.delivered_at).not.toBeNull();

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ id, from: "pinky@local", to: "worker@local", text: "ping" });
    expect(seen[0]!.sentAt).toBeTruthy();
  });

  test("unqualified recipient is treated as local", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "local" });
    const seen: A2AEnvelope[] = [];
    m.onMessage("worker", (env) => seen.push(env));
    await m.send(envelope("worker"));
    expect(seen).toHaveLength(1);
    expect(db.rows[0]!.node_to).toBe("local");
  });

  test("on node2, agent@node2 is local: stored, delivered, and readable from inbox", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "node2" });
    const seen: A2AEnvelope[] = [];
    m.onMessage("agent", (env) => seen.push(env));

    await m.send({ from: "alpha@node2", to: "agent@node2", kind: "request", text: "2+2?" });
    await flushMicrotasks();

    expect(db.rows[0]).toMatchObject({ to_agent: "agent", node_to: "node2", node_from: "node2" });
    expect(db.rows[0]!.delivered_at).not.toBeNull();
    expect(seen).toHaveLength(1);

    const inbox = await m.inbox("agent");
    expect(inbox.map((e) => e.text)).toEqual(["2+2?"]);
    expect(inbox[0]!.to).toBe("agent@node2");
  });

  test("on node2, rows for other nodes never surface in the local inbox", async () => {
    const db = new FakeDb();
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, { nodeId: "node2", peers: {} });
      await m.send(envelope("agent@local", "for the other node"));
      await m.send(envelope("agent@node2", "for me"));
      await flushMicrotasks();
      expect((await m.inbox("agent")).map((e) => e.text)).toEqual(["for me"]);
    });
    expect(warnings.length).toBeGreaterThan(0); // no route to node "local"
    expect(db.rows.map((r) => r.node_to)).toEqual(["local", "node2"]);
  });

  test("broadcast stays local-only and also fires the '*' subscribers", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "node2", peers: { other: "https://other.test" } });
    const star: A2AEnvelope[] = [];
    m.onMessage("*", (env) => star.push(env));
    await m.send(envelope("broadcast", "hear ye"));
    await flushMicrotasks();
    expect(db.rows[0]).toMatchObject({ to_agent: "broadcast", node_to: "node2" });
    expect(star).toHaveLength(1);
  });

  test("unsubscribe stops delivery to that handler", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "local" });
    const seen: A2AEnvelope[] = [];
    const off = m.onMessage("worker", (env) => seen.push(env));
    await m.send(envelope("worker@local", "one"));
    off();
    await m.send(envelope("worker@local", "two"));
    expect(seen.map((e) => e.text)).toEqual(["one"]);
  });

  test("inbox returns unread envelopes and marks them read", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "local" });
    await m.send(envelope("worker@local", "a"));
    await m.send(envelope("worker@local", "b"));
    await flushMicrotasks();

    const first = await m.inbox("worker");
    expect(first.map((e) => e.text)).toEqual(["a", "b"]);
    expect(first[0]).toMatchObject({ from: "pinky@local", kind: "message" });
    expect(db.rows.every((r) => r.read_at !== null)).toBe(true);

    expect(await m.inbox("worker")).toEqual([]);
  });

  test("remote send persists durable-first and posts signed envelope to the peer", async () => {
    const db = new FakeDb();
    const posts: { url: string; headers: Headers; body: string }[] = [];
    // Hold the peer's response open so "durable first, delivered after" is
    // observable rather than racing the fire-and-forget POST.
    const gate = Promise.withResolvers<void>();
    const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
      posts.push({
        url: String(url),
        headers: new Headers(init?.headers),
        body: String(init?.body),
      });
      await gate.promise;
      return new Response("ok", { status: 200 });
    }) as unknown as typeof fetch;
    const m = new LocalMessenger(db, {
      nodeId: "local",
      peers: { node2: "https://node2.test" },
      a2aSecret: "shh",
      fetchFn,
    });

    const id = await m.send(envelope("pinky@node2", "cross-machine"));
    // Durable first: row exists, not yet delivered.
    expect(db.rows).toHaveLength(1);
    expect(db.rows[0]).toMatchObject({ id, to_agent: "pinky", node_to: "node2" });
    expect(db.rows[0]!.delivered_at).toBeNull();

    // Fire-and-forget: release the peer and flush microtasks so it settles.
    gate.resolve();
    await flushMicrotasks();
    expect(posts).toHaveLength(1);
    expect(db.rows[0]!.delivered_at).not.toBeNull();
    expect(posts[0]!.url).toBe("https://node2.test/a2a/deliver");
    const ts = posts[0]!.headers.get("X-Pinky-Ts")!;
    const sig = posts[0]!.headers.get("X-Pinky-Signature")!;
    expect(sig).toBe(a2aSignature("shh", id, ts, posts[0]!.body));
    const payload = JSON.parse(posts[0]!.body) as A2AEnvelope;
    expect(payload).toMatchObject({ id, from: "pinky@local", to: "pinky@node2", text: "cross-machine" });
  });

  test("a 2xx from the peer completes delivery (row marked delivered)", async () => {
    const db = new FakeDb();
    const fetchFn = (async () => new Response("ok", { status: 202 })) as unknown as typeof fetch;
    const m = new LocalMessenger(db, {
      nodeId: "local",
      peers: { node2: "https://node2.test" },
      a2aSecret: "shh",
      fetchFn,
    });
    await m.send(envelope("pinky@node2", "delivered please"));
    await flushMicrotasks();
    expect(db.rows[0]!.delivered_at).not.toBeNull();
  });

  test("a non-2xx from the peer leaves the row pending", async () => {
    const db = new FakeDb();
    const fetchFn = (async () => new Response("nope", { status: 500 })) as unknown as typeof fetch;
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, {
        nodeId: "local",
        peers: { node2: "https://node2.test" },
        a2aSecret: "shh",
        fetchFn,
      });
      await m.send(envelope("pinky@node2", "server error"));
      await flushMicrotasks();
    });
    expect(db.rows[0]!.delivered_at).toBeNull();
    expect(warnings.length).toBeGreaterThan(0);
  });

  test("remote delivery failure leaves the row undelivered for retry", async () => {
    const db = new FakeDb();
    const fetchFn = (async () => {
      throw new Error("connection refused");
    }) as unknown as typeof fetch;
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, {
        nodeId: "local",
        peers: { node2: "https://node2.test" },
        a2aSecret: "shh",
        fetchFn,
      });
      await m.send(envelope("pinky@node2", "retry me"));
      await flushMicrotasks();
    });
    expect(db.rows[0]!.delivered_at).toBeNull();
    expect(warnings.length).toBeGreaterThan(0);
  });

  test("unknown peer leaves the row undelivered without a fetch attempt", async () => {
    const db = new FakeDb();
    let fetched = 0;
    const fetchFn = (async () => {
      fetched += 1;
      return new Response("ok");
    }) as unknown as typeof fetch;
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, { nodeId: "local", peers: {}, fetchFn });
      await m.send(envelope("pinky@nowhere"));
      await flushMicrotasks();
    });
    expect(fetched).toBe(0);
    expect(db.rows[0]!.delivered_at).toBeNull();
    expect(warnings.length).toBeGreaterThan(0);
  });
});

describe("LocalMessenger.receive", () => {
  const relayed: A2AEnvelope = {
    id: "peer-msg-1",
    from: "alpha@local",
    to: "beta@node2",
    kind: "request",
    text: "hello from afar",
    sentAt: "2026-08-28T10:00:00.000Z",
  };

  test("persists under the original id, marks delivered, and wakes subscribers", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "node2" });
    const seen: A2AEnvelope[] = [];
    m.onMessage("beta", (env) => seen.push(env));

    expect(await m.receive(relayed)).toBe(true);
    expect(db.rows).toHaveLength(1);
    expect(db.rows[0]).toMatchObject({
      id: "peer-msg-1", // ORIGINAL id, not a freshly minted one
      to_agent: "beta",
      node_to: "node2",
      node_from: "local",
    });
    expect(db.rows[0]!.delivered_at).not.toBeNull();
    expect(seen).toHaveLength(1);
    expect((await m.inbox("beta")).map((e) => e.text)).toEqual(["hello from afar"]);
  });

  test("a retry loses the claim: no second row, no second wakeup", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "node2" });
    const seen: A2AEnvelope[] = [];
    m.onMessage("beta", (env) => seen.push(env));

    expect(await m.receive(relayed)).toBe(true);
    expect(await m.receive(relayed)).toBe(false);
    expect(await m.receive({ ...relayed, text: "tampered but same id" })).toBe(false);

    expect(db.rows).toHaveLength(1);
    expect(seen).toHaveLength(1);
    expect(seen[0]!.text).toBe("hello from afar");
  });

  test("an envelope for a third node is stored pending, not delivered here", async () => {
    const db = new FakeDb();
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, { nodeId: "node2" });
      const seen: A2AEnvelope[] = [];
      m.onMessage("gamma", (env) => seen.push(env));
      expect(await m.receive({ ...relayed, id: "mis-1", to: "gamma@node9" })).toBe(true);
      expect(seen).toHaveLength(0);
      expect(await m.inbox("gamma")).toEqual([]);
    });
    expect(db.rows[0]).toMatchObject({ id: "mis-1", node_to: "node9" });
    expect(db.rows[0]!.delivered_at).toBeNull(); // still pending for a forward
    expect(warnings.length).toBeGreaterThan(0);
  });

  test("shared database: the sender's own row does not swallow the delivery", async () => {
    // Both nodes on ONE database (`PINKY_NODE_ID=node2 pinky smoke`). A's
    // send() has already written the row, so the receiver's insert conflicts —
    // which used to be read as "duplicate, already handled" and silently
    // skipped the wakeup. The CLAIM is what makes this a real delivery.
    const db = new FakeDb();
    const a = new LocalMessenger(db, { nodeId: "local", peers: {} });
    const b = new LocalMessenger(db, { nodeId: "node2" });
    const seen: A2AEnvelope[] = [];
    b.onMessage("beta", (env) => seen.push(env));

    let id = "";
    await withCapturedWarnings(async () => {
      id = await a.send({
        from: "alpha@local",
        to: "beta@node2",
        kind: "message",
        text: "shared db",
      });
      await flushMicrotasks();
    });
    expect(db.rows).toHaveLength(1);
    expect(db.rows[0]!.delivered_at).toBeNull(); // no route: still pending

    const env: A2AEnvelope = {
      id,
      from: "alpha@local",
      to: "beta@node2",
      kind: "message",
      text: "shared db",
      sentAt: new Date().toISOString(),
    };
    expect(await b.receive(env)).toBe(true);
    expect(db.rows).toHaveLength(1); // the sender's row, claimed — not a copy
    expect(db.rows[0]!.delivered_at).not.toBeNull();
    expect(seen.map((e) => e.text)).toEqual(["shared db"]);

    // Replay of the same envelope: claim fails, nobody is woken twice.
    expect(await b.receive(env)).toBe(false);
    expect(seen).toHaveLength(1);
  });

  test("a broadcast envelope from a peer also fires '*'", async () => {
    const db = new FakeDb();
    const m = new LocalMessenger(db, { nodeId: "node2" });
    const star: A2AEnvelope[] = [];
    m.onMessage("*", (env) => star.push(env));
    await m.receive({ ...relayed, id: "b-1", to: "broadcast" });
    expect(star).toHaveLength(1);
    expect(db.rows[0]).toMatchObject({ to_agent: "broadcast", node_to: "node2" });
  });
});

describe("LocalMessenger.flushPending", () => {
  test("re-attempts pending rows for every peer and marks them delivered", async () => {
    const db = new FakeDb();
    let up = false;
    const posts: string[] = [];
    const fetchFn = (async (_url: string | URL | Request, init?: RequestInit) => {
      posts.push(String(init?.body));
      return up ? new Response("ok", { status: 200 }) : new Response("down", { status: 503 });
    }) as unknown as typeof fetch;

    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, {
        nodeId: "local",
        peers: { node2: "https://node2.test" },
        a2aSecret: "shh",
        fetchFn,
      });
      const id = await m.send(envelope("pinky@node2", "queued"));
      await flushMicrotasks();
      expect(db.rows[0]!.delivered_at).toBeNull();

      // Peer still down: attempted, not delivered, still pending.
      expect(await m.flushPending()).toEqual({ attempted: 1, delivered: 0 });
      expect(db.rows[0]!.delivered_at).toBeNull();

      // Peer back up: the sweep completes delivery.
      up = true;
      expect(await m.flushPending()).toEqual({ attempted: 1, delivered: 1 });
      expect(db.rows[0]!.delivered_at).not.toBeNull();

      // Nothing left to do.
      expect(await m.flushPending()).toEqual({ attempted: 0, delivered: 0 });

      // Retries keep the original id (the receiver dedups on it)...
      const bodies = posts.map((b) => JSON.parse(b) as A2AEnvelope);
      expect(bodies.every((b) => b.id === id)).toBe(true);
      // ...but re-stamp sentAt so the peer's freshness window accepts them.
      const age = Date.now() - new Date(bodies.at(-1)!.sentAt).getTime();
      expect(age).toBeLessThan(300_000);
    });
    expect(warnings.length).toBeGreaterThan(0); // the 503 attempt warned
  });

  test("ignores local rows and unrouted nodes", async () => {
    const db = new FakeDb();
    let fetched = 0;
    const fetchFn = (async () => {
      fetched += 1;
      return new Response("ok", { status: 200 });
    }) as unknown as typeof fetch;
    const warnings = await withCapturedWarnings(async () => {
      const m = new LocalMessenger(db, {
        nodeId: "node2",
        peers: { node3: "https://node3.test" },
        a2aSecret: "shh",
        fetchFn,
      });
      await m.send(envelope("beta@node2", "local, already delivered"));
      await m.send(envelope("beta@nowhere", "no route"));
      await flushMicrotasks();
      expect(await m.flushPending()).toEqual({ attempted: 0, delivered: 0 });
    });
    expect(fetched).toBe(0);
    expect(warnings.length).toBeGreaterThan(0); // no route to "nowhere"
  });
});
