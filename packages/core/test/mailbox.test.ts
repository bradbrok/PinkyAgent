import { describe, expect, it } from "bun:test";
import { Mailbox, parseA2AAddress, type A2AEnvelope } from "../src/mailbox";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
}

class FakeDb implements Db {
  calls: Call[] = [];
  private script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>;

  constructor(script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>) {
    this.script = script;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params });
    for (const s of this.script) {
      if (s.pattern.test(sql)) return s.respond(params) as T[];
    }
    throw new Error(`FakeDb: no script for SQL: ${sql}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }
  tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }
  async close(): Promise<void> {}
}

const baseEnv: A2AEnvelope = {
  id: "msg-1",
  from: "weather@local",
  to: "planner@local",
  kind: "message",
  text: "hi",
  sentAt: "2026-08-28T10:00:00Z",
};

function captureInsert(): { db: FakeDb; captured: () => unknown[] | undefined } {
  let captured: unknown[] | undefined;
  const db = new FakeDb([
    {
      pattern: /insert into a2a_messages/,
      respond: (params?: unknown[]) => {
        captured = params;
        return [{ id: (params ?? [])[0] }];
      },
    },
  ]);
  return { db, captured: () => captured };
}

describe("parseA2AAddress", () => {
  it("resolves unqualified addresses against the caller's node", () => {
    expect(parseA2AAddress("planner", "node2")).toEqual({ agentId: "planner", nodeId: "node2" });
    expect(parseA2AAddress("broadcast", "node2")).toEqual({ agentId: "broadcast", nodeId: "node2" });
  });

  it("splits on the LAST '@' and ignores a leading one", () => {
    expect(parseA2AAddress("planner@node2", "local")).toEqual({
      agentId: "planner",
      nodeId: "node2",
    });
    expect(parseA2AAddress("a@b@node2", "local")).toEqual({ agentId: "a@b", nodeId: "node2" });
    expect(parseA2AAddress("@weird", "local")).toEqual({ agentId: "@weird", nodeId: "local" });
  });
});

describe("Mailbox.put", () => {
  it("parses 'agent@node' and inserts the row", async () => {
    const { db, captured } = captureInsert();
    const mbox = new Mailbox(db);
    await mbox.put(baseEnv);
    expect(captured()).toEqual([
      "msg-1",
      "weather",
      "planner",
      "local",
      "local",
      "message",
      "hi",
      null,
    ]);
  });

  it("treats bare 'broadcast' as to_agent='broadcast' on this node", async () => {
    const { db, captured } = captureInsert();
    const mbox = new Mailbox(db);
    await mbox.put({ ...baseEnv, to: "broadcast" });
    expect(captured()?.[2]).toBe("broadcast");
    expect(captured()?.[4]).toBe("local"); // broadcast is local-only
  });

  it("normalizes '@<thisNode>' and unqualified addresses to this node", async () => {
    const { db, captured } = captureInsert();
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.put({ ...baseEnv, from: "weather", to: "planner@node2" });
    expect(captured()?.[3]).toBe("node2"); // node_from (unqualified sender)
    expect(captured()?.[4]).toBe("node2"); // node_to

    const second = captureInsert();
    const mbox2 = new Mailbox(second.db, { nodeId: "node2" });
    await mbox2.put({ ...baseEnv, to: "planner" });
    expect(second.captured()?.[4]).toBe("node2");
  });

  it("keeps a foreign node id for outbound rows", async () => {
    const { db, captured } = captureInsert();
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.put({ ...baseEnv, to: "planner@node7" });
    expect(captured()?.[4]).toBe("node7");
  });
});

describe("Mailbox.putIfAbsent", () => {
  it("returns true when the row is new", async () => {
    const db = new FakeDb([
      { pattern: /on conflict \(id\) do nothing/, respond: () => [{ id: "msg-1" }] },
    ]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.putIfAbsent(baseEnv)).toBe(true);
    expect(db.calls[0]!.sql).toContain("on conflict (id) do nothing");
    expect(db.calls[0]!.params?.[0]).toBe("msg-1");
  });

  it("returns false when the id already exists", async () => {
    const db = new FakeDb([{ pattern: /on conflict \(id\) do nothing/, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.putIfAbsent(baseEnv)).toBe(false);
  });
});

describe("Mailbox.markDelivered", () => {
  it("updates only the given id, and only while undelivered", async () => {
    const db = new FakeDb([
      { pattern: /update a2a_messages set delivered_at = now\(\) where id = \$1/, respond: () => [] },
    ]);
    const mbox = new Mailbox(db);
    await mbox.markDelivered("msg-1");
    expect(db.calls[0]!.sql).toContain("where id = $1 and delivered_at is null");
    expect(db.calls[0]!.params).toEqual(["msg-1"]);
  });
});

describe("Mailbox.claimDelivery", () => {
  const CLAIM = /update a2a_messages set delivered_at = now\(\)[\s\S]*returning id/;

  it("claims an undelivered row in THIS node's partition", async () => {
    const db = new FakeDb([{ pattern: CLAIM, respond: () => [{ id: "msg-1" }] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.claimDelivery("msg-1")).toBe(true);
    // The claim is the idempotency point of receive(): node-scoped, and only
    // while the row is still undelivered.
    expect(db.calls[0]!.sql).toContain("node_to = $2");
    expect(db.calls[0]!.sql).toContain("delivered_at is null");
    expect(db.calls[0]!.sql).toContain("returning id");
    expect(db.calls[0]!.params).toEqual(["msg-1", "node2"]);
  });

  it("returns false when no row was claimed (already delivered, or another node's)", async () => {
    const db = new FakeDb([{ pattern: CLAIM, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.claimDelivery("msg-1")).toBe(false);
  });
});

describe("Mailbox.claimRead", () => {
  const CLAIM = /update a2a_messages set read_at = now\(\)[\s\S]*returning id/;

  it("claims an unread row in THIS node's partition", async () => {
    const db = new FakeDb([{ pattern: CLAIM, respond: () => [{ id: "msg-1" }] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.claimRead("msg-1")).toBe(true);
    // The receipt is node-scoped and only claimable while unread — the same
    // shape as claimDelivery, on the column that means "an agent consumed it".
    expect(db.calls[0]!.sql).toContain("node_to = $2");
    expect(db.calls[0]!.sql).toContain("read_at is null");
    expect(db.calls[0]!.sql).toContain("returning id");
    expect(db.calls[0]!.params).toEqual(["msg-1", "node2"]);
  });

  it("returns false when nothing was claimed (already consumed, or another node's)", async () => {
    const db = new FakeDb([{ pattern: CLAIM, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    expect(await mbox.claimRead("msg-1")).toBe(false);
  });

  it("runs on the caller's handle when given one, so the receipt joins their tx", async () => {
    const own = new FakeDb([{ pattern: CLAIM, respond: () => [] }]);
    const tx = new FakeDb([{ pattern: CLAIM, respond: () => [{ id: "msg-1" }] }]);
    const mbox = new Mailbox(own, { nodeId: "node2" });
    expect(await mbox.claimRead("msg-1", tx)).toBe(true);
    // The whole point: the statement went to the consumer's transaction, so it
    // commits (or rolls back) with the work it is a receipt for.
    expect(tx.calls).toHaveLength(1);
    expect(own.calls).toHaveLength(0);
  });
});

describe("Mailbox.unconsumedFor", () => {
  const row = {
    id: "msg-1",
    from_agent: "weather",
    to_agent: "planner",
    node_from: "node2",
    node_to: "node2",
    kind: "message" as const,
    text: "hi",
    thread_hint: null,
    // Delivered — and still unconsumed. This is exactly the row a crash
    // between the delivery claim and the agent's turn leaves behind.
    delivered_at: "2026-08-28T10:01:00Z",
    read_at: null,
    created_at: "2026-08-28T10:00:00Z",
  };

  it("selects unread rows regardless of delivered_at, oldest first", async () => {
    const db = new FakeDb([{ pattern: /select .* from a2a_messages/i, respond: () => [row] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    const out = await mbox.unconsumedFor("planner");
    expect(out).toHaveLength(1);
    expect(out[0]!.from).toBe("weather@node2");
    const sql = db.calls[0]!.sql;
    expect(sql).toContain("read_at is null");
    // The bug this exists for: filtering on delivered_at here would skip the
    // orphans, because they ARE delivered. (It is still SELECTed, just never
    // a predicate.)
    expect(sql).not.toMatch(/delivered_at is (not )?null/);
    expect(sql).toContain("order by created_at asc");
    expect(sql).toContain("to_agent in ($1, 'broadcast')");
    expect(sql).toContain("node_to = $2");
    expect(db.calls[0]!.params).toEqual(["planner", "node2", 100]);
  });

  it("takes an explicit limit and never marks anything", async () => {
    const db = new FakeDb([{ pattern: /select .* from a2a_messages/i, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.unconsumedFor("planner", { limit: 5 });
    expect(db.calls[0]!.params).toEqual(["planner", "node2", 5]);
    // Reading the backlog is not consuming it: only the consumer's own
    // transaction may stamp a receipt.
    expect(db.calls.some((c) => /update/i.test(c.sql))).toBe(false);
  });
});

describe("Mailbox.deliverLocal", () => {
  it("returns undelivered rows for the agent and marks delivered_at", async () => {
    const rows = [
      {
        id: "msg-1",
        from_agent: "weather",
        to_agent: "planner",
        node_from: "local",
        node_to: "local",
        kind: "message" as const,
        text: "hi",
        thread_hint: null,
        delivered_at: null,
        read_at: null,
        created_at: "2026-08-28T10:00:00Z",
      },
    ];
    const db = new FakeDb([
      { pattern: /select .* from a2a_messages/i, respond: () => rows },
      { pattern: /update a2a_messages set delivered_at = now\(\)/i, respond: () => [] },
    ]);
    const mbox = new Mailbox(db);
    const out = await mbox.deliverLocal("planner");
    expect(out).toHaveLength(1);
    expect(out[0]!.from).toBe("weather@local");
    expect(out[0]!.to).toBe("planner@local");
    expect(out[0]!.sentAt).toBe("2026-08-28T10:00:00Z");
    expect(db.calls.some((c) => /update a2a_messages set delivered_at = now\(\)/i.test(c.sql))).toBe(true);
  });

  it("filters on this mailbox's node id, passed as a parameter", async () => {
    const db = new FakeDb([{ pattern: /select .* from a2a_messages/i, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.deliverLocal("planner");
    expect(db.calls[0]!.sql).toContain("node_to = $2");
    expect(db.calls[0]!.params).toEqual(["planner", "node2"]);
    expect(db.calls[0]!.sql).not.toContain("'local'");
  });

  it("converts a Date created_at to an ISO sentAt", async () => {
    const rows = [
      {
        id: "msg-1",
        from_agent: "weather",
        to_agent: "planner",
        node_from: "node2",
        node_to: "node2",
        kind: "message" as const,
        text: "hi",
        thread_hint: null,
        delivered_at: null,
        read_at: null,
        created_at: new Date("2026-08-28T10:00:00.000Z"),
      },
    ];
    const db = new FakeDb([
      { pattern: /select .* from a2a_messages/i, respond: () => rows },
      { pattern: /update a2a_messages set delivered_at = now\(\)/i, respond: () => [] },
    ]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    const out = await mbox.deliverLocal("planner");
    expect(out[0]!.sentAt).toBe("2026-08-28T10:00:00.000Z");
  });
});

describe("Mailbox.inbox", () => {
  it("returns delivered-but-unread rows and marks read_at", async () => {
    const rows = [
      {
        id: "msg-1",
        from_agent: "weather",
        to_agent: "planner",
        node_from: "local",
        node_to: "local",
        kind: "message" as const,
        text: "hi",
        thread_hint: null,
        delivered_at: "2026-08-28T10:01:00Z",
        read_at: null,
        created_at: "2026-08-28T10:00:00Z",
      },
    ];
    const db = new FakeDb([
      { pattern: /select .* from a2a_messages/i, respond: () => rows },
      { pattern: /update a2a_messages set read_at = now\(\)/i, respond: () => [] },
    ]);
    const mbox = new Mailbox(db);
    const out = await mbox.inbox("planner", { limit: 10 });
    expect(out).toHaveLength(1);
    expect(db.calls.some((c) => /update a2a_messages set read_at = now\(\)/i.test(c.sql))).toBe(true);
  });

  it("scopes to this node id and keeps limit as $3", async () => {
    const db = new FakeDb([{ pattern: /select .* from a2a_messages/i, respond: () => [] }]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.inbox("planner", { limit: 10 });
    expect(db.calls[0]!.sql).toContain("node_to = $2");
    expect(db.calls[0]!.params).toEqual(["planner", "node2", 10]);
  });
});

describe("Mailbox.pendingForNode", () => {
  it("returns outbound rows for the remote node, not yet delivered", async () => {
    const rows = [
      {
        id: "msg-1",
        from_agent: "weather",
        to_agent: "planner",
        node_from: "local",
        node_to: "remote",
        kind: "message" as const,
        text: "hi",
        thread_hint: null,
        delivered_at: null,
        read_at: null,
        created_at: "2026-08-28T10:00:00Z",
      },
    ];
    const db = new FakeDb([
      { pattern: /node_to = \$1 and node_to != \$2 and delivered_at is null/i, respond: () => rows },
    ]);
    const mbox = new Mailbox(db);
    const out = await mbox.pendingForNode("remote");
    expect(out).toHaveLength(1);
    expect(out[0]!.to).toBe("planner@remote");
    expect(db.calls[0]!.params).toEqual(["remote", "local"]);
  });

  it("excludes this node's own partition by parameter, not by the literal 'local'", async () => {
    const db = new FakeDb([
      { pattern: /node_to = \$1 and node_to != \$2 and delivered_at is null/i, respond: () => [] },
    ]);
    const mbox = new Mailbox(db, { nodeId: "node2" });
    await mbox.pendingForNode("local");
    expect(db.calls[0]!.params).toEqual(["local", "node2"]);
  });
});
