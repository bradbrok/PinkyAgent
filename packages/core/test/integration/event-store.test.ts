/**
 * EventStore against the live database (DESIGN.md §3).
 *
 * The unit suite drives EventStore through a hand-written FakeDb, which is why
 * it never noticed that Postgres hands `bigint` columns back as JS *strings*
 * (postgres.js does not coerce int8, to protect precision). Everything that
 * does arithmetic on `seq` is therefore untested until it runs here.
 *
 * Skipped unless PINKY_INTEGRATION=1 — same gate as rls.test.ts:
 *
 *   docker compose up -d postgres
 *   bun run test:integration
 *
 * The connection comes from loadEnvConfig() (DATABASE_URL), never a literal
 * port: local dev is 5544, CI is 5432.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { loadEnvConfig } from "../../src/config";
import { createDb } from "../../src/pg";
import { migrate } from "../../src/migrate";
import { EventStore } from "../../src/event-store";
import type { Db } from "../../src/db";
import type { ContinuityDoc, ThreadEventData, ThreadRef } from "../../src/events";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const DB_URL = loadEnvConfig().databaseUrl;
const SCHEMA_DIR = new URL("../../schema", import.meta.url).pathname;

/** Every row this file writes is stamped with this prefix, so cleanup is a
 *  scoped DELETE and never a TRUNCATE. The random run id keeps two runs (or a
 *  rerun after a crash) from colliding on thread identity. */
const PREFIX = "it-events-";
const RUN = crypto.randomUUID().slice(0, 8);
const TENANT = `${PREFIX}${RUN}`;

function thread(name: string): ThreadRef {
  return { tenantId: TENANT, channelId: "it:core", threadId: `${name}-${RUN}` };
}

function decision(reason: string): ThreadEventData {
  return { type: "decision", action: "silent", reason };
}

const DOC: ContinuityDoc = {
  goal: "integration boundary",
  plan: { done: [], now: "test", next: [] },
  workingSet: {},
  decisions: [],
  openLoops: [],
  lessons: [],
  memoryHints: [],
};

function continuity(tokensBefore: number): ThreadEventData {
  return { type: "continuity", document: DOC, tokensBefore };
}

/** seq is typed `number` and now IS one (toSeq coerces the int8 string at the
 *  boundary — see the DEFECT tests below). Number() is kept so a regression
 *  shows up as the seq-type assertion failing, not as every test failing. */
const seqs = (events: { seq: number }[]): number[] => events.map((e) => Number(e.seq));

suite("EventStore (live postgres)", () => {
  let db: Db;
  let store: EventStore;

  beforeAll(async () => {
    // max >= the widest concurrent fan-out below, or the 8-writer race just
    // queues in the client pool and proves nothing about the row lock.
    db = createDb(DB_URL, { max: 12 });
    await migrate(db, SCHEMA_DIR);
    await purge(db);
    store = new EventStore(db);
  });

  afterAll(async () => {
    if (!db) return;
    await purge(db);
    await db.close();
  });

  /** Drop this file's rows *and* any left by an earlier interrupted run. */
  async function purge(handle: Db): Promise<void> {
    await handle.query(`delete from events where tenant_id like $1`, [`${PREFIX}%`]);
    await handle.query(`delete from threads where tenant_id like $1`, [`${PREFIX}%`]);
    await handle.query(`delete from ingress_dedup where tenant_id like $1`, [`${PREFIX}%`]);
  }

  it("8 concurrent first appends on a fresh thread all get a distinct seq 1..8", async () => {
    const ref = thread("race");
    const results = await Promise.all(
      Array.from({ length: 8 }, (_, i) => store.append(ref, decision(`writer-${i}`))),
    );

    // The interesting part is that nobody got a duplicate and nobody crashed on
    // the unique (thread, seq) constraint: the `insert into threads ... on
    // conflict do nothing` + `select ... for update` pair serializes them.
    expect(seqs(results).sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const stored = await store.history(ref);
    expect(seqs(stored)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(new Set(stored.map((e) => e.id)).size).toBe(8);
  });

  it("appendBatch returns the rows it wrote, in order, on one thread", async () => {
    const ref = thread("batch-order");
    const written = await store.appendBatch(ref, [decision("b0"), decision("b1"), decision("b2")]);
    expect(written).toHaveLength(3);
    expect(written.map((e) => (e.data as { reason: string }).reason)).toEqual(["b0", "b1", "b2"]);

    const stored = await store.history(ref);
    expect(stored.map((e) => (e.data as { reason: string }).reason)).toEqual(["b0", "b1", "b2"]);
  });

  it(
    "DEFECT: appendBatch assigns CONTIGUOUS seqs (live db returned 1, 11, 111)",
    async () => {
      // Root cause: Postgres `bigint` arrives from postgres.js as a JS string,
      // so `coalesce(max(seq),0)+1` was "1", and appendLockedTx's `nextSeq +=
      // 1` CONCATENATED: "1" -> "11" -> "111". The unit suite's FakeDb returns
      // numbers, so this is invisible there.
      // Fixed by coercing at the boundary: toSeq() in appendLockedTx, mapRow
      // and latestContinuitySeq. This test is the regression guard.
      const ref = thread("batch-contiguous");
      const first = await store.append(ref, decision("solo"));
      expect(first.seq).toBe(1);

      const batch = await store.appendBatch(ref, [decision("b1"), decision("b2"), decision("b3")]);
      expect(batch.map((e) => e.seq)).toEqual([2, 3, 4]);
      expect((await store.history(ref)).map((e) => e.seq)).toEqual([1, 2, 3, 4]);
    },
  );

  it("DEFECT: seq round-trips as a number, not an int8 string", async () => {
    // The same root cause, stated directly. It is not cosmetic: buildContext()
    // (packages/core/src/projection.ts) filters `e.seq >= boundarySeq`, and
    // once a thread passes 10 events that string comparison is lexicographic —
    // "9" >= "10" is true, so a pre-boundary event leaks into the model
    // context that the continuity boundary was supposed to drop.
    const ref = thread("seq-type");
    const e = await store.append(ref, decision("typed"));
    expect(typeof e.seq).toBe("number");
    expect(typeof (await store.latestContinuitySeq(ref))).toBe("number");
  });

  it("contextEvents returns the window from the latest continuity boundary", async () => {
    const ref = thread("boundary");
    await store.append(ref, decision("pre-1"));
    await store.append(ref, decision("pre-2"));
    const boundary = await store.append(ref, continuity(1234));
    await store.append(ref, decision("post-1"));
    await store.append(ref, decision("post-2"));

    const window = await store.contextEvents(ref);
    expect(Number(window.boundarySeq)).toBe(Number(boundary.seq));
    expect(Number(window.boundarySeq)).toBe(3);
    expect(window.truncated).toBe(false);
    // Boundary event included, ascending, pre-boundary events dropped.
    expect(seqs(window.events)).toEqual([3, 4, 5]);
    expect(window.events[0]!.data.type).toBe("continuity");
  });

  it("a second continuity event moves the boundary forward", async () => {
    const ref = thread("boundary-moves");
    await store.append(ref, decision("old"));
    await store.append(ref, continuity(10));
    await store.append(ref, decision("middle"));
    await store.append(ref, continuity(20));
    await store.append(ref, decision("new"));

    const window = await store.contextEvents(ref);
    expect(Number(window.boundarySeq)).toBe(4);
    expect(seqs(window.events)).toEqual([4, 5]);
  });

  it("with no continuity event the whole log is the window", async () => {
    const ref = thread("no-boundary");
    await store.append(ref, decision("a"));
    await store.append(ref, decision("b"));

    const window = await store.contextEvents(ref);
    expect(Number(window.boundarySeq)).toBe(0);
    expect(seqs(window.events)).toEqual([1, 2]);
    expect(window.truncated).toBe(false);
  });

  it("over the cap it keeps the NEWEST events and says so", async () => {
    const ref = thread("cap");
    for (let i = 1; i <= 10; i++) await store.append(ref, decision(`e${i}`));

    const capped = await store.contextEvents(ref, { maxEvents: 4 });
    expect(capped.truncated).toBe(true);
    expect(seqs(capped.events)).toEqual([7, 8, 9, 10]);
    // The tail is what the model needs; the dropped prefix stays in the log.
    expect((capped.events.at(-1)!.data as { reason: string }).reason).toBe("e10");

    // Exactly at the cap is NOT truncation (the cap+1 fetch exists for this).
    const exact = await store.contextEvents(ref, { maxEvents: 10 });
    expect(exact.truncated).toBe(false);
    expect(exact.events).toHaveLength(10);
  });

  it("dedup returns true on first sight of an external id and false after", async () => {
    const externalId = `Ev-${RUN}-1`;
    expect(await store.dedup(TENANT, externalId)).toBe(true);
    expect(await store.dedup(TENANT, externalId)).toBe(false);
    expect(await store.dedup(TENANT, externalId)).toBe(false);
  });

  it("dedup is scoped per tenant — the same external id is fresh elsewhere", async () => {
    const externalId = `Ev-${RUN}-shared`;
    expect(await store.dedup(TENANT, externalId)).toBe(true);
    expect(await store.dedup(`${PREFIX}${RUN}-other`, externalId)).toBe(true);
  });

  it("DEFECT: events.data is stored as queryable jsonb, not a jsonb string", async () => {
    // Second, independent defect (same class as the settings one). Db.query
    // used to pass `JSON.stringify(data)` as a bind parameter; postgres.js
    // learns the parameter's type from the server's Describe (jsonb) and
    // applies its OWN jsonb serializer — JSON.stringify — so the document was
    // encoded TWICE and landed as a jsonb *string*. `select jsonb_typeof(data)
    // from events` was 'string' for every row in this database and
    // `data->>'type'` NULL, so the event log could not be queried or indexed
    // as JSON at all.
    //
    // It was invisible from the app because mapRow() defends against it
    // (`typeof r.data === "string" ? JSON.parse(...)`) — the read side undid
    // the damage on the way out. That branch is now legacy-row tolerance only.
    //
    // Fixed in appendLockedTx: `[..., d.type, jsonbParam(d)]`, the plain
    // object, serialized once by the driver. Rows already written are repaired
    // by schema/0004_jsonb_repair.rerun.sql.
    const ref = thread("jsonb");
    await store.append(ref, decision("queryable"));
    const row = await db.queryOne<{ jtype: string; type_via_jsonb: string | null }>(
      `select jsonb_typeof(data) as jtype, data ->> 'type' as type_via_jsonb
         from events where (tenant_id, channel_id, thread_id) = ($1, $2, $3)`,
      [ref.tenantId, ref.channelId, ref.threadId],
    );
    expect(row?.jtype).toBe("object");
    expect(row?.type_via_jsonb).toBe("decision");
  });

  it("history pages forward from afterSeq", async () => {
    const ref = thread("paging");
    for (let i = 1; i <= 6; i++) await store.append(ref, decision(`p${i}`));

    const page1 = await store.history(ref, { limit: 4 });
    expect(seqs(page1)).toEqual([1, 2, 3, 4]);
    const page2 = await store.history(ref, { afterSeq: Number(page1.at(-1)!.seq), limit: 4 });
    expect(seqs(page2)).toEqual([5, 6]);
  });
});
