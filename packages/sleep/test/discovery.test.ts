/**
 * Thread discovery (slice 6, contract §3.5).
 *
 * Unit scope is the REQUEST: the SQL this file emits and the params bound to
 * it. Whether Postgres agrees with the plan — the laterals, the idle
 * arithmetic, the `sleep:` exclusion actually excluding — is proved against a
 * real server in packages/sleep/test/integration/sleep.test.ts. Both halves
 * matter: the text below encodes decisions (index-backed laterals, an
 * injectable clock, bigints coerced at the boundary) a rewrite would silently
 * undo while every integration assertion stayed green.
 */
import { describe, expect, it } from "bun:test";
import { discoverDueThreads } from "../src/discovery";
import { EXTRACT_EVENT_TYPES } from "../src/types";
import { FakeDb, dueRow, installDueThreads } from "./helpers";

const DUE_SQL = /from threads t cross join lateral/;
const NOW = new Date("2026-08-29T12:00:00.000Z");

function dueDb(rows: Record<string, unknown>[] = []): FakeDb {
  const db = new FakeDb();
  installDueThreads(db, rows);
  return db;
}

describe("discoverDueThreads — the query shape", () => {
  it("laterals over threads, excludes sleep: channels, and gates on an injectable clock", async () => {
    const db = dueDb();

    await discoverDueThreads(db, {
      tenantId: "t1",
      idleMs: 600_000,
      limit: 10,
      now: NOW,
    });

    const sql = (db.find(DUE_SQL)?.sql ?? "").replace(/\s+/g, " ");
    // Index-backed per thread — two `order by seq desc limit 1` probes — never
    // a group-by over the whole event log, which is the biggest table here and
    // would be rescanned every sweep forever.
    expect(sql).toContain("cross join lateral");
    expect(sql).toContain("left join lateral");
    expect(sql).toContain("order by e.seq desc limit 1");
    // The worker journals its own receipts in `sleep:<agent>`; extracting
    // memories from the record of extracting memories is a feedback loop.
    expect(sql).toContain("t.channel_id not like 'sleep:%'");
    // `now` as a param, not now(): the gate has to be positionable by a test
    // and by the integration suite.
    expect(sql).toContain("last.ts <= $2::timestamptz - ($3::bigint * interval '1 millisecond')");
    // Only threads with material after their own cursor.
    expect(sql).toContain("e.seq > coalesce(cur.to_seq, 0)");
    expect(sql).toContain("order by last.ts asc");

    expect(db.find(DUE_SQL)?.params).toEqual([
      "t1",
      NOW.toISOString(),
      600_000,
      [...EXTRACT_EVENT_TYPES],
      10,
    ]);
  });

  it("binds the extractable event types as the text[] param", async () => {
    const db = dueDb();

    await discoverDueThreads(db, { tenantId: "t1", idleMs: 0, limit: 5, now: NOW });

    const call = db.find(DUE_SQL);
    // The shared list, not a copy that drifted: a thread whose only new events
    // are audit-only (`decision`, `restart`, another `sleep` receipt) is not due.
    expect(call?.params?.[3]).toEqual([...EXTRACT_EVENT_TYPES]);
    expect((call?.sql ?? "").replace(/\s+/g, " ")).toContain("e.type = any($4::text[])");
  });

  it("defaults the clock to now and clamps a negative idle and a zero limit", async () => {
    const db = dueDb();
    const before = Date.now();

    await discoverDueThreads(db, { tenantId: "t1", idleMs: -5, limit: 0 });

    const params = db.find(DUE_SQL)?.params ?? [];
    const stamped = Date.parse(String(params[1]));
    expect(Number.isNaN(stamped)).toBe(false);
    expect(stamped).toBeGreaterThanOrEqual(before);
    expect(params[2]).toBe(0);
    expect(params[4]).toBe(1);
  });
});

describe("discoverDueThreads — filters", () => {
  it("adds no channel/thread predicate when neither is given", async () => {
    const db = dueDb();

    await discoverDueThreads(db, { tenantId: "t1", idleMs: 0, limit: 3, now: NOW });

    const sql = (db.find(DUE_SQL)?.sql ?? "").replace(/\s+/g, " ");
    expect(sql).not.toMatch(/and t\.channel_id = \$/);
    expect(sql).not.toMatch(/and t\.thread_id = \$/);
    expect((db.find(DUE_SQL)?.params ?? []).length).toBe(5);
  });

  it("appends each filter with its own param, ahead of the limit", async () => {
    const db = dueDb();

    await discoverDueThreads(db, {
      tenantId: "t1",
      idleMs: 0,
      limit: 3,
      channelId: "cli:main",
      threadId: "root",
      now: NOW,
    });

    const call = db.find(DUE_SQL);
    const sql = (call?.sql ?? "").replace(/\s+/g, " ");
    expect(sql).toContain("and t.channel_id = $5");
    expect(sql).toContain("and t.thread_id = $6");
    expect(sql).toContain("limit $7");
    expect(call?.params).toEqual([
      "t1",
      NOW.toISOString(),
      0,
      [...EXTRACT_EVENT_TYPES],
      "cli:main",
      "root",
      3,
    ]);
  });
});

describe("discoverDueThreads — results", () => {
  it("coerces the bigint and timestamptz columns and rebuilds the thread ref", async () => {
    // postgres.js hands int8 back as a STRING and timestamptz as a DATE.
    // Letting a seq escape means `cursorSeq + 1` concatenates and `>` goes
    // lexicographic ("9" > "10"); letting a Date escape means the CLI prints
    // "Fri Aug 29 2026 …" where every other surface prints ISO.
    const db = dueDb([
      dueRow("cli:a", {
        last_seq: "42",
        cursor_seq: "7",
        last_ts: new Date("2026-08-29T10:00:00.000Z"),
      }),
      dueRow("cli:b", { last_seq: 3, cursor_seq: null, last_ts: "2026-08-29T11:00:00.000Z" }),
    ]);

    const due = await discoverDueThreads(db, { tenantId: "t1", idleMs: 0, limit: 10, now: NOW });

    expect(due).toEqual([
      {
        thread: { tenantId: "t1", channelId: "cli:a", threadId: "main" },
        lastSeq: 42,
        lastTs: "2026-08-29T10:00:00.000Z",
        cursorSeq: 7,
      },
      {
        thread: { tenantId: "t1", channelId: "cli:b", threadId: "main" },
        lastSeq: 3,
        lastTs: "2026-08-29T11:00:00.000Z",
        // A thread nobody has ever swept: cursor 0, so its whole log is material.
        cursorSeq: 0,
      },
    ]);
    expect(typeof due[0]?.lastSeq).toBe("number");
    expect(typeof due[0]?.cursorSeq).toBe("number");
    // A string, ISO, whichever shape the driver used.
    expect(typeof due[0]?.lastTs).toBe("string");
    expect(typeof due[1]?.lastTs).toBe("string");
  });

  it("returns an empty list when nothing is due", async () => {
    const db = dueDb([]);
    expect(await discoverDueThreads(db, { tenantId: "t1", idleMs: 0, limit: 10 })).toEqual([]);
  });
});
