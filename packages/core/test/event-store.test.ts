import { describe, expect, it } from "bun:test";
import { DEFAULT_CONTEXT_EVENT_CAP, DEFAULT_HISTORY_PAGE, EventStore } from "../src/event-store";
import type { Db } from "../src/db";
import type { ThreadEvent, ThreadEventData, ThreadRef } from "../src/events";

/**
 * Minimal in-memory Db: precise enough for the queries EventStore issues,
 * callable without a network. Scripted SQL via route entries.
 */
interface FakeDbOptions {
  route: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>;
  /** Fired when the OUTERMOST transaction settles — the durability boundary. */
  onCommit?: () => void;
  onRollback?: () => void;
}

class FakeDb implements Db {
  /** `txDepth` is 0 for a statement issued outside any transaction. */
  calls: Array<{ sql: string; params: unknown[] | undefined; txDepth: number }> = [];
  /** "begin"/"commit"/"rollback" for the OUTERMOST tx only, in order. */
  txLog: string[] = [];
  private route: FakeDbOptions["route"];
  private opts: FakeDbOptions;
  private txDepth = 0;

  constructor(opts: FakeDbOptions) {
    this.route = opts.route;
    this.opts = opts;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params, txDepth: this.txDepth });
    for (const r of this.route) {
      if (r.pattern.test(sql)) return r.respond(params) as T[];
    }
    throw new Error(`FakeDb: no route for SQL: ${sql}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    // Same object; nested tx re-entrant. Only the outermost one commits.
    const outermost = this.txDepth === 0;
    if (outermost) this.txLog.push("begin");
    this.txDepth += 1;
    try {
      const out = await fn(this);
      if (outermost) {
        this.txLog.push("commit");
        this.opts.onCommit?.();
      }
      return out;
    } catch (err) {
      if (outermost) {
        this.txLog.push("rollback");
        this.opts.onRollback?.();
      }
      throw err;
    } finally {
      this.txDepth -= 1;
    }
  }

  async close(): Promise<void> {}
}

const ref: ThreadRef = { tenantId: "tenant1", channelId: "chan1", threadId: "t1" };

const eventData: ThreadEventData = {
  type: "ingress",
  platform: "slack",
  author: { platform: "slack", userId: "u1" },
  text: "hello",
  refs: [],
};

describe("EventStore.append", () => {
  it("locks the thread row, inserts, and returns the event with next seq", async () => {
    const route = [
      {
        pattern: /insert into threads/i,
        respond: () => [],
      },
      {
        pattern: /from threads[\s\S]*for update/i,
        respond: () => [{ seq: 5 }],
      },
      {
        pattern: /select coalesce\(max\(seq\), 0\) \+ 1 as next/i,
        respond: () => [{ next: 6 }],
      },
      {
        pattern: /insert into events/i,
        respond: (params?: unknown[]) => [
          {
            id: (params as unknown[])[0],
            tenant_id: ref.tenantId,
            channel_id: ref.channelId,
            thread_id: ref.threadId,
            seq: (params as unknown[])[4],
            ts: "2026-08-28T10:00:00Z",
            data: eventData,
          },
        ],
      },
    ];
    let counters: Record<string, number> = { insertThreads: 0, lockSelect: 0, maxSeq: 0, insertEvents: 0 };
    const routed = route.map((r, i) => ({
      pattern: r.pattern,
      respond: (p?: unknown[]) => {
        const k = Object.keys(counters)[i];
        if (k !== undefined) counters[k] = (counters[k] ?? 0) + 1;
        return r.respond(p);
      },
    }));
    const db = new FakeDb({ route: routed });
    const store = new EventStore(db);
    const ev = await store.append(ref, eventData);
    expect(ev.id).toMatch(/[0-9a-f-]{36}/);
    expect(ev.seq).toBe(6);
    expect(ev.data).toEqual(eventData);
    expect(db.calls[0]!.sql).toMatch(/insert into threads/i);
    expect(db.calls[1]!.sql).toMatch(/for update/i);
    expect(db.calls[2]!.sql).toMatch(/coalesce\(max\(seq\)/i);
    expect(db.calls[3]!.sql).toMatch(/insert into events/i);
    // `data` is bound as the PLAIN object (pg.ts's JSONB CONTRACT). Binding
    // JSON.stringify(d) here is what stored every event as a jsonb *string*,
    // where `data->>'type'` is NULL and the log cannot be queried as JSON.
    expect(db.calls[3]!.params![6]).toEqual(eventData);
    expect(typeof db.calls[3]!.params![6]).toBe("object");
    expect(counters["insertThreads"]).toBe(1);
  });

  it("appendBatch assigns ascending seqs in order", async () => {
    let seq = 10;
    const route = [
      { pattern: /insert into threads/i, respond: () => [] },
      { pattern: /from threads[\s\S]*for update/i, respond: () => [{ seq }] },
      { pattern: /select coalesce\(max\(seq\), 0\) \+ 1 as next/i, respond: () => [{ next: seq + 1 }] },
      {
        pattern: /insert into events/i,
        respond: (params?: unknown[]) => {
          const p = params as unknown[];
          return [{ id: p[0], tenant_id: ref.tenantId, channel_id: ref.channelId, thread_id: ref.threadId, seq: p[4], ts: "t", data: p[6] }];
        },
      },
    ];
    const db = new FakeDb({ route });
    const store = new EventStore(db);
    seq = 10;
    const batch = await store.appendBatch(ref, [eventData, eventData]);
    expect(batch[0]!.seq).toBe(11);
    expect(batch[1]!.seq).toBe(12); // appendBatch re-locks on the same thread and increments client-side per item
  });
});

describe("EventStore.history", () => {
  it("returns asc-ordered events mapped to camelCase", async () => {
    const route = [
      {
        pattern: /select id, tenant_id, channel_id, thread_id, seq, ts, data from events/i,
        respond: () => [
          { id: "a", tenant_id: ref.tenantId, channel_id: ref.channelId, thread_id: ref.threadId, seq: 1, ts: "t1", data: eventData },
          { id: "b", tenant_id: ref.tenantId, channel_id: ref.channelId, thread_id: ref.threadId, seq: 2, ts: "t2", data: eventData },
        ],
      },
    ];
    const db = new FakeDb({ route });
    const store = new EventStore(db);
    const out = await store.history(ref, { afterSeq: 0, limit: 100 });
    expect(out.map((e) => e.seq)).toEqual([1, 2]);
    expect(out[0]!.tenantId).toBe("tenant1");
    expect(db.calls[0]!.params).toEqual(["tenant1", "chan1", "t1", 0, 100]);
  });
});

describe("EventStore.dedup", () => {
  it("returns true on first sight (a row inserted), false on duplicate", async () => {
    const firstResp = [{ ok: 1 }];
    const secondResp: unknown[] = [];
    let first = true;
    const route = [
      { pattern: /insert into ingress_dedup/i, respond: () => (first ? firstResp : secondResp) },
    ];
    const db = new FakeDb({ route });
    const store = new EventStore(db);
    expect(await store.dedup("t1", "ext1")).toBe(true);
    first = false;
    expect(await store.dedup("t1", "ext1")).toBe(false);
  });
});

describe("EventStore.ingest", () => {
  const decisionData: ThreadEventData = { type: "decision", action: "reply", reason: "mention" };

  /** Routes for a healthy thread whose next seq is 1; events insert normally. */
  const happyRoutes = (
    claim: (params?: unknown[]) => unknown[],
    insertEvent?: (params?: unknown[]) => unknown[],
  ) => [
    { pattern: /insert into ingress_dedup/i, respond: claim },
    { pattern: /insert into threads/i, respond: () => [] },
    { pattern: /from threads[\s\S]*for update/i, respond: () => [] },
    { pattern: /select coalesce\(max\(seq\), 0\) \+ 1 as next/i, respond: () => [{ next: 1 }] },
    {
      pattern: /insert into events/i,
      respond:
        insertEvent ??
        ((params?: unknown[]) => [
          {
            id: (params as unknown[])[0],
            tenant_id: ref.tenantId,
            channel_id: ref.channelId,
            thread_id: ref.threadId,
            seq: (params as unknown[])[4],
            ts: "2026-08-28T10:00:00Z",
            data: (params as unknown[])[6],
          },
        ]),
    },
  ];

  it("wraps the dedup claim and every insert in ONE transaction", async () => {
    const db = new FakeDb({ route: happyRoutes(() => [{ ok: 1 }]) });
    const written = await new EventStore(db).ingest(ref, "ext1", [eventData, decisionData]);

    expect(written?.map((e) => e.seq)).toEqual([1, 2]);
    expect(written?.map((e) => e.data.type)).toEqual(["ingress", "decision"]);

    // Exactly one transaction, committed once — not three.
    expect(db.txLog).toEqual(["begin", "commit"]);
    // ...and every statement, dedup claim included, ran inside it.
    expect(db.calls.every((c) => c.txDepth > 0)).toBe(true);
    expect(db.calls[0]!.sql).toMatch(/insert into ingress_dedup/);
    expect(db.calls[0]!.sql).toMatch(/on conflict do nothing returning 1/);
    expect(db.calls.filter((c) => /insert into events/i.test(c.sql))).toHaveLength(2);
  });

  it("returns null and writes nothing when the external id was already claimed", async () => {
    const db = new FakeDb({ route: happyRoutes(() => []) }); // conflict: no row back
    expect(await new EventStore(db).ingest(ref, "ext1", [eventData, decisionData])).toBeNull();
    expect(db.calls).toHaveLength(1); // the claim, and nothing after it
    expect(db.txLog).toEqual(["begin", "commit"]);
  });

  it("a failure mid-transaction leaves no dedup row, so the retry is processed", async () => {
    // Durable state the transaction protects: ids only become visible on commit.
    const committed = new Set<string>();
    let pending: string | null = null;
    let explode = true;

    const db = new FakeDb({
      route: happyRoutes(
        (params) => {
          const id = (params as string[])[1]!;
          if (committed.has(id)) return []; // on conflict do nothing
          pending = id;
          return [{ ok: 1 }];
        },
        (params) => {
          if (explode) throw new Error("insert into events failed");
          return [
            {
              id: (params as unknown[])[0],
              tenant_id: ref.tenantId,
              channel_id: ref.channelId,
              thread_id: ref.threadId,
              seq: (params as unknown[])[4],
              ts: "2026-08-28T10:00:00Z",
              data: (params as unknown[])[6],
            },
          ];
        },
      ),
      onCommit: () => {
        if (pending !== null) committed.add(pending);
        pending = null;
      },
      onRollback: () => {
        pending = null;
      },
    });
    const store = new EventStore(db);

    await expect(store.ingest(ref, "Ev1", [eventData, decisionData])).rejects.toThrow(
      "insert into events failed",
    );
    expect(db.txLog).toEqual(["begin", "rollback"]);
    expect(committed.size).toBe(0); // the claim rolled back with the inserts

    // Slack retries the same event_id: it must be treated as first sight.
    explode = false;
    const retry = await store.ingest(ref, "Ev1", [eventData, decisionData]);
    expect(retry?.map((e) => e.data.type)).toEqual(["ingress", "decision"]);
    expect(committed.has("Ev1")).toBe(true);
  });
});

describe("EventStore.latestContinuitySeq", () => {
  it("returns the newest continuity seq, or 0 when none", async () => {
    const store1 = new EventStore(new FakeDb({ route: [{ pattern: /type = 'continuity'/i, respond: () => [{ seq: 42 }] }] }));
    expect(await store1.latestContinuitySeq(ref)).toBe(42);
    const store2 = new EventStore(new FakeDb({ route: [{ pattern: /continuity/, respond: () => [] }] }));
    expect(await store2.latestContinuitySeq(ref)).toBe(0);
  });
});

describe("EventStore.history limit semantics", () => {
  it("defaults to a 500-event forward page starting after afterSeq", async () => {
    const db = new FakeDb({
      route: [{ pattern: /select id, tenant_id, channel_id, thread_id, seq, ts, data from events/i, respond: () => [] }],
    });
    await new EventStore(db).history(ref);
    expect(db.calls[0]!.params).toEqual(["tenant1", "chan1", "t1", 0, DEFAULT_HISTORY_PAGE]);
    expect(DEFAULT_HISTORY_PAGE).toBe(500);
    expect(db.calls[0]!.sql).toMatch(/seq > \$4/);
    expect(db.calls[0]!.sql).toMatch(/order by seq asc/);
  });
});

describe("EventStore.contextEvents", () => {
  // NOTE the string `data`: that is a LEGACY doubly-encoded row, kept here on
  // purpose to pin mapRow()'s parse-on-read tolerance. New rows are written as
  // plain objects and old ones are repaired by
  // schema/0004_jsonb_repair.rerun.sql, but a database that has not migrated
  // yet must still read.
  const row = (seq: number) => ({
    id: `e${seq}`,
    tenant_id: ref.tenantId,
    channel_id: ref.channelId,
    thread_id: ref.threadId,
    seq,
    ts: `t${seq}`,
    data: JSON.stringify(eventData),
  });

  const contextDb = (continuitySeq: number | null, rowsDesc: ReturnType<typeof row>[]): FakeDb =>
    new FakeDb({
      route: [
        { pattern: /type = 'continuity'/i, respond: () => (continuitySeq === null ? [] : [{ seq: continuitySeq }]) },
        { pattern: /seq >= \$4/i, respond: () => rowsDesc },
      ],
    });

  it("loads the continuity event and everything after it, ascending", async () => {
    const db = contextDb(7, [row(9), row(8), row(7)]);
    const out = await new EventStore(db).contextEvents(ref);
    expect(out.events.map((e) => e.seq)).toEqual([7, 8, 9]); // boundary included
    expect(out.boundarySeq).toBe(7);
    expect(out.truncated).toBe(false);
    expect(out.events[0]!.tenantId).toBe("tenant1");
    expect(out.events[0]!.data).toEqual(eventData); // json string parsed
    // Query asks for seq >= boundary, newest-first, cap + 1.
    expect(db.calls[1]!.params).toEqual(["tenant1", "chan1", "t1", 7, DEFAULT_CONTEXT_EVENT_CAP + 1]);
    expect(db.calls[1]!.sql).toMatch(/order by seq desc/);
  });

  it("returns the whole log when the thread has no continuity event", async () => {
    const db = contextDb(null, [row(2), row(1)]);
    const out = await new EventStore(db).contextEvents(ref);
    expect(out.boundarySeq).toBe(0);
    expect(out.events.map((e) => e.seq)).toEqual([1, 2]);
    expect(db.calls[1]!.params![3]).toBe(0);
  });

  it("has no silent cap: the default is far above any plausible window", () => {
    expect(DEFAULT_CONTEXT_EVENT_CAP).toBeGreaterThanOrEqual(5000);
    expect(DEFAULT_CONTEXT_EVENT_CAP).toBeGreaterThan(DEFAULT_HISTORY_PAGE);
  });

  it("keeps the NEWEST events and reports truncation when the cap is exceeded", async () => {
    // maxEvents 2 -> the store asks for 3 and gets 3, so one was dropped.
    const db = contextDb(3, [row(5), row(4), row(3)]);
    const out = await new EventStore(db).contextEvents(ref, { maxEvents: 2 });
    expect(out.truncated).toBe(true);
    expect(out.events.map((e) => e.seq)).toEqual([4, 5]); // oldest dropped, newest kept
    expect(db.calls[1]!.params![4]).toBe(3);
  });

  it("does not report truncation for a window sitting exactly on the cap", async () => {
    const db = contextDb(3, [row(5), row(4), row(3)]);
    const out = await new EventStore(db).contextEvents(ref, { maxEvents: 3 });
    expect(out.truncated).toBe(false);
    expect(out.events.map((e) => e.seq)).toEqual([3, 4, 5]);
  });
});

/**
 * The per-thread lock as a step of its own (slice 6).
 *
 * The sleep-time worker has to HOLD the lock across work that is not an append
 * — re-read its cursor, write memory rows, then journal the receipt — so the
 * discipline appendLockedTx already used is exposed rather than copied. A
 * second copy of "insert if absent, then select for update" is exactly how the
 * two would drift.
 */
describe("EventStore.lockThreadTx", () => {
  const lockRoutes = () => [
    { pattern: /insert into threads/i, respond: () => [] },
    { pattern: /from threads[\s\S]*for update/i, respond: () => [] },
  ];

  it("inserts the thread row if absent, then locks THAT row", async () => {
    const db = new FakeDb({ route: lockRoutes() });
    await db.tx((tx) => EventStore.lockThreadTx(tx, ref));

    expect(db.calls).toHaveLength(2);
    expect(db.calls[0]!.sql).toMatch(/insert into threads/i);
    // Insert-if-absent: the row must exist for EVERY writer, or concurrent
    // first appends lock nothing and collide on seq 1.
    expect(db.calls[0]!.sql).toMatch(/on conflict do nothing/i);
    expect(db.calls[1]!.sql).toMatch(/from threads[\s\S]*for update/i);
    for (const call of db.calls) {
      expect(call.params).toEqual(["tenant1", "chan1", "t1"]);
      expect(call.txDepth).toBeGreaterThan(0); // the lock is worthless outside a tx
    }
  });

  it("is the ONE locking routine: append does not issue its own", async () => {
    const db = new FakeDb({
      route: [
        ...lockRoutes(),
        { pattern: /select coalesce\(max\(seq\), 0\) \+ 1 as next/i, respond: () => [{ next: 1 }] },
        {
          pattern: /insert into events/i,
          respond: (params?: unknown[]) => {
            const p = params as unknown[];
            return [
              {
                id: p[0],
                tenant_id: ref.tenantId,
                channel_id: ref.channelId,
                thread_id: ref.threadId,
                seq: p[4],
                ts: "t",
                data: p[6],
              },
            ];
          },
        },
      ],
    });
    await new EventStore(db).append(ref, eventData);
    const matching = (re: RegExp) => db.calls.filter((c) => re.test(c.sql));
    expect(matching(/insert into threads/i)).toHaveLength(1);
    expect(matching(/for update/i)).toHaveLength(1);
    // ...and in that order, ahead of the seq computation.
    expect(db.calls.map((c) => /insert into threads/i.test(c.sql))).toEqual([
      true,
      false,
      false,
      false,
    ]);
  });

  it("is re-entrant: taking it twice in one transaction is harmless", async () => {
    // Postgres row locks are held per TRANSACTION, so a caller that locks and
    // then calls appendTx (which locks again) pays one extra statement and
    // never blocks on itself. That is what makes the sleep pass's
    // lock-then-append shape safe without a "already locked" flag to get wrong.
    const db = new FakeDb({ route: lockRoutes() });
    await db.tx(async (tx) => {
      await EventStore.lockThreadTx(tx, ref);
      await EventStore.lockThreadTx(tx, ref);
    });
    expect(db.calls).toHaveLength(4);
    expect(db.txLog).toEqual(["begin", "commit"]);
  });

  it("lets a caller lock FIRST, do other work, and append in the same transaction", async () => {
    // The sleep worker's shape (DESIGN.md §5.3 item 3): whatever it writes
    // between the lock and the receipt commits with the receipt or not at all.
    const db = new FakeDb({
      route: [
        ...lockRoutes(),
        { pattern: /insert into memories/i, respond: () => [{ id: "m1" }] },
        { pattern: /select coalesce\(max\(seq\), 0\) \+ 1 as next/i, respond: () => [{ next: 4 }] },
        {
          pattern: /insert into events/i,
          respond: (params?: unknown[]) => {
            const p = params as unknown[];
            return [
              {
                id: p[0],
                tenant_id: ref.tenantId,
                channel_id: ref.channelId,
                thread_id: ref.threadId,
                seq: p[4],
                ts: "t",
                data: p[6],
              },
            ];
          },
        },
      ],
    });

    const receipt: ThreadEventData = {
      type: "sleep",
      phase: "extract",
      fromSeq: 1,
      toSeq: 3,
      scanned: 2,
      candidates: 1,
      added: 1,
      updated: 0,
      invalidated: 0,
      noop: 0,
      model: "fake/sleep",
      ms: 7,
    };
    const written = await db.tx(async (tx) => {
      await EventStore.lockThreadTx(tx, ref);
      await tx.query("insert into memories (id) values ($1)", ["m1"]);
      return await EventStore.appendTx(tx, ref, [receipt]);
    });

    expect(written[0]!.seq).toBe(4);
    expect(db.txLog).toEqual(["begin", "commit"]); // ONE transaction
    expect(db.calls.every((c) => c.txDepth > 0)).toBe(true);
    // The lock is taken BEFORE the memory write, which is the whole point: a
    // concurrent pass either waits here or is seen by the cursor re-read.
    const order = db.calls.map((c) => c.sql.replace(/\s+/g, " ").slice(0, 30));
    expect(order[0]).toMatch(/insert into threads/i);
    expect(order[1]).toMatch(/from threads/i);
    expect(order[2]).toMatch(/insert into memories/i);
    expect(db.calls.some((c) => /insert into events/i.test(c.sql))).toBe(true);
  });
});
