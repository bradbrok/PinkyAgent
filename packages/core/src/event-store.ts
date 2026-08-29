/**
 * Append-only event log (DESIGN.md §3). seq is per-thread; the append path
 * takes a lock on the `threads` row (insert ... on conflict do nothing; then
 * select ... for update) so concurrent writers serialize on the same thread.
 *
 * append(), appendBatch() and ingest() all funnel through ONE locking routine
 * (appendLockedTx) so the lock discipline exists in a single place; ingest()
 * additionally claims the platform's dedup id inside that same transaction.
 */
import type { Db } from "./db";
import { jsonbParam } from "./pg";
import type { ThreadEvent, ThreadEventData, ThreadRef } from "./events";

/** Default forward page size for {@link EventStore.history} (audit/replay). */
export const DEFAULT_HISTORY_PAGE = 500;

/**
 * Safety cap for {@link EventStore.contextEvents}. Deliberately far above any
 * plausible post-continuity window: the continuity engine (DESIGN.md §4) is
 * what bounds context, not a row limit. Hitting this means something is wrong,
 * so it is reported rather than applied silently.
 */
export const DEFAULT_CONTEXT_EVENT_CAP = 5000;

/** Result of {@link EventStore.contextEvents}: the model-visible slice of the log. */
export interface ContextWindow {
  /** Ascending by seq, starting at the latest continuity event (inclusive). */
  events: ThreadEvent[];
  /** Seq of the continuity event at the boundary, or 0 when there is none. */
  boundarySeq: number;
  /**
   * True when the safety cap dropped the OLDEST events of the window.
   *
   * The loop treats this as hard context pressure and forces a shed
   * (runtime/loop.ts): the cap keeps the NEWEST events, so a truncated
   * window's start rolls forward with every append and a prefix whose first
   * bytes move each turn can never hit a provider cache (DESIGN.md §4.5).
   */
  truncated: boolean;
}

interface EventRow {
  id: string;
  tenant_id: string;
  channel_id: string;
  thread_id: string;
  /** `bigint` on the wire, so postgres.js hands it back as a STRING. */
  seq: number | string;
  ts: string;
  /** jsonb. `string` only for legacy doubly-encoded rows — see mapRow(). */
  data: ThreadEventData | string;
}

/**
 * Coerce a Postgres `bigint` to a JS number at the boundary.
 *
 * postgres.js returns int8 as a string (it does not fit in a double in
 * general), and every seq we produce is small enough that a number is right.
 * Letting the string escape is not cosmetic: `nextSeq += 1` would CONCATENATE
 * ("1" -> "11" -> "111"), and buildContext() compares `seq >= boundarySeq`,
 * which on strings is lexicographic — "9" >= "10" is true, so a pre-boundary
 * event would leak back into the model's context past 10 events.
 */
function toSeq(value: number | string | null | undefined, fallback: number): number {
  if (value === null || value === undefined) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Row -> ThreadEvent.
 *
 * `data` is jsonb and is WRITTEN as a plain object (pg.ts's JSONB CONTRACT),
 * so postgres.js hands it back as an object. The string branch is pure
 * tolerance for LEGACY rows written by the doubly-encoding version of this
 * file — those land as a jsonb string and would otherwise surface as an
 * unusable `data` of type string. schema/0004_jsonb_repair.rerun.sql repairs
 * them in place; this branch keeps a database that has not been migrated yet
 * readable in the meantime.
 */
function mapRow(r: EventRow): ThreadEvent {
  return {
    id: r.id,
    tenantId: r.tenant_id,
    channelId: r.channel_id,
    threadId: r.thread_id,
    seq: toSeq(r.seq, 0),
    ts: r.ts,
    data: typeof r.data === "string" ? (JSON.parse(r.data) as ThreadEventData) : r.data,
  };
}

export class EventStore {
  private db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  /**
   * The one append implementation: take the per-thread write lock, then insert
   * `data` in order under contiguous seqs. MUST be called with `tx` already
   * inside a transaction — the lock is worthless otherwise, and ingest() needs
   * the dedup row to commit or roll back together with these inserts.
   */
  private static async appendLockedTx(
    tx: Db,
    ref: ThreadRef,
    data: ThreadEventData[],
    label: string,
  ): Promise<ThreadEvent[]> {
    // Ensure the thread row exists, then lock THAT row while computing the
    // next seq. The lock target has to be a row that exists for every writer,
    // which the newest *event* is not: on a fresh thread there is none, so
    // `select ... from events ... for update` locks nothing and N concurrent
    // first appends all compute seq 1 and collide on the unique constraint.
    // The threads row is the stable per-conversation mutex (DESIGN.md §6:
    // "per-conversation lock; one in-flight run per thread").
    await tx.query(
      `insert into threads (tenant_id, channel_id, thread_id) values ($1, $2, $3)
       on conflict do nothing`,
      [ref.tenantId, ref.channelId, ref.threadId],
    );
    await tx.query(
      `select 1 from threads where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
       for update`,
      [ref.tenantId, ref.channelId, ref.threadId],
    );
    const seqRow = await tx.queryOne<{ next: number | string }>(
      `select coalesce(max(seq), 0) + 1 as next from events
       where (tenant_id, channel_id, thread_id) = ($1, $2, $3)`,
      [ref.tenantId, ref.channelId, ref.threadId],
    );
    // Coerce: `next` is bigint, i.e. a string, and `nextSeq += 1` below would
    // otherwise concatenate ("1" -> "11" -> "111").
    let nextSeq = toSeq(seqRow?.next, 1);
    const out: ThreadEvent[] = [];
    for (const d of data) {
      const id = crypto.randomUUID();
      const row = await tx.queryOne<EventRow>(
        `insert into events (id, tenant_id, channel_id, thread_id, seq, ts, type, data)
         values ($1, $2, $3, $4, $5, now(), $6, $7)
         returning id, tenant_id, channel_id, thread_id, seq, ts, data`,
        // `d` PLAIN, never JSON.stringify(d): the driver encodes a jsonb
        // param once by itself (pg.ts's JSONB CONTRACT). Pre-encoding stores
        // the event as a jsonb string, where `data->>'type'` is NULL.
        [id, ref.tenantId, ref.channelId, ref.threadId, nextSeq, d.type, jsonbParam(d)],
      );
      if (!row) throw new Error(`${label}: insert returned no row`);
      out.push(mapRow(row));
      nextSeq += 1;
    }
    return out;
  }

  /**
   * Append inside a transaction the CALLER owns, under the same per-thread
   * lock as every other append.
   *
   * For work that must commit with something else — the A2A consumption
   * receipt (issue #4) is the case this exists for: stamp `read_at` and
   * journal the `a2a` event in ONE transaction, so a message is marked
   * consumed if and only if the event that proves it exists. `ingest()` does
   * the same trick for the platform dedup id; this is the open version for
   * callers whose other write is not a dedup row.
   *
   * The tx MUST really be a transaction (`db.tx(...)`): outside one the thread
   * lock is released at once and concurrent appends can collide on seq.
   */
  static appendTx(tx: Db, ref: ThreadRef, data: ThreadEventData[]): Promise<ThreadEvent[]> {
    return EventStore.appendLockedTx(tx, ref, data, "appendTx");
  }

  async append(ref: ThreadRef, data: ThreadEventData): Promise<ThreadEvent> {
    const [event] = await this.db.tx((tx) =>
      EventStore.appendLockedTx(tx, ref, [data], "append"),
    );
    if (!event) throw new Error("append: insert returned no row");
    return event;
  }

  async appendBatch(ref: ThreadRef, data: ThreadEventData[]): Promise<ThreadEvent[]> {
    return await this.db.tx((tx) => EventStore.appendLockedTx(tx, ref, data, "appendBatch"));
  }

  /**
   * Atomic "first sight of this external id" + append (DESIGN.md §6:
   * "persist raw event → dedup → gate → enqueue").
   *
   * Dedup and append must commit together or not at all. Doing them as two
   * statements loses messages: if the dedup row lands and an append then
   * fails, Slack's retry of the very same event_id is rejected as a duplicate
   * and the message is gone for good, with nothing in the log to replay.
   *
   * So: ONE transaction. Insert the dedup row with `on conflict do nothing
   * returning 1` — no row back means someone already claimed this id, and we
   * return null having written nothing. Otherwise the same transaction takes
   * the thread lock and appends every event in `data`. Any failure inside
   * rolls the dedup row back with the inserts, leaving the id unclaimed so the
   * platform's retry is processed as fresh.
   */
  async ingest(
    ref: ThreadRef,
    externalId: string,
    data: ThreadEventData[],
  ): Promise<ThreadEvent[] | null> {
    return await this.db.tx(async (tx) => {
      const claimed = await tx.query<{ ok: number }>(
        `insert into ingress_dedup (tenant_id, external_id) values ($1, $2)
         on conflict do nothing returning 1 as ok`,
        [ref.tenantId, externalId],
      );
      if (claimed.length === 0) return null; // already seen — nothing written
      return await EventStore.appendLockedTx(tx, ref, data, "ingest");
    });
  }

  /**
   * Raw log page for audit, replay, and memory extraction — NOT for building
   * the model prompt (use {@link EventStore.contextEvents} for that).
   *
   * Limit semantics are explicit and deliberate: this returns at most `limit`
   * events (default {@link DEFAULT_HISTORY_PAGE}) in ASCENDING seq order,
   * starting at the OLDEST event with `seq > afterSeq`. It is a forward page,
   * so a thread longer than `limit` is truncated at the TAIL: callers that
   * want the whole log must page by feeding the last returned seq back in as
   * `afterSeq` until fewer than `limit` rows come back.
   */
  async history(ref: ThreadRef, opts?: { afterSeq?: number; limit?: number }): Promise<ThreadEvent[]> {
    const after = opts?.afterSeq ?? 0;
    const limit = opts?.limit ?? DEFAULT_HISTORY_PAGE;
    const rows = await this.db.query<EventRow>(
      `select id, tenant_id, channel_id, thread_id, seq, ts, data from events
       where (tenant_id, channel_id, thread_id) = ($1, $2, $3) and seq > $4
       order by seq asc limit $5`,
      [ref.tenantId, ref.channelId, ref.threadId, after, limit],
    );
    return rows.map(mapRow);
  }

  /**
   * The events the model is allowed to see (DESIGN.md §3): the latest
   * `continuity` event and everything after it, ascending. The continuity
   * event itself is included because `buildContext` renders it as the
   * boundary payload; when the thread has no continuity event yet the whole
   * log is returned.
   *
   * Unlike {@link EventStore.history} there is no silent forward cap. A very
   * large explicit safety cap (`maxEvents`, default
   * {@link DEFAULT_CONTEXT_EVENT_CAP}) keeps a runaway thread from OOMing the
   * process; it keeps the NEWEST events and reports the loss via
   * `truncated`, so the caller is never quietly handed a stale prefix.
   */
  async contextEvents(ref: ThreadRef, opts?: { maxEvents?: number }): Promise<ContextWindow> {
    const boundarySeq = await this.latestContinuitySeq(ref);
    const cap = Math.max(1, opts?.maxEvents ?? DEFAULT_CONTEXT_EVENT_CAP);
    // Fetch newest-first with cap+1 so an exact-cap window is not misreported
    // as truncated, then flip back to ascending for the projection.
    const rows = await this.db.query<EventRow>(
      `select id, tenant_id, channel_id, thread_id, seq, ts, data from events
       where (tenant_id, channel_id, thread_id) = ($1, $2, $3) and seq >= $4
       order by seq desc limit $5`,
      [ref.tenantId, ref.channelId, ref.threadId, boundarySeq, cap + 1],
    );
    const truncated = rows.length > cap;
    const kept = truncated ? rows.slice(0, cap) : rows;
    const events = kept.map(mapRow).reverse();
    return { events, boundarySeq, truncated };
  }

  /**
   * Returns true iff this call was the first sight of the external id.
   *
   * Standalone claim, committed on its own. Prefer {@link EventStore.ingest}
   * for anything that also writes events: claiming here and appending later
   * means a failed append leaves the id claimed and the retry discarded.
   */
  async dedup(tenantId: string, externalId: string): Promise<boolean> {
    const rows = await this.db.query<{ ok: number }>(
      `insert into ingress_dedup (tenant_id, external_id) values ($1, $2)
       on conflict do nothing returning 1 as ok`,
      [tenantId, externalId],
    );
    return rows.length > 0;
  }

  /** Seq of the newest `continuity` event, or 0 when the thread has none. */
  async latestContinuitySeq(ref: ThreadRef): Promise<number> {
    const row = await this.db.queryOne<{ seq: number | string }>(
      `select seq from events
       where (tenant_id, channel_id, thread_id) = ($1, $2, $3) and type = 'continuity'
       order by seq desc limit 1`,
      [ref.tenantId, ref.channelId, ref.threadId],
    );
    // Number, not the bigint string: buildContext filters `seq >= boundarySeq`.
    return toSeq(row?.seq, 0);
  }

  close(): Promise<void> {
    return this.db.close();
  }
}
