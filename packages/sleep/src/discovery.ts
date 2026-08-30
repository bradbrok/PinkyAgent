/**
 * Which threads the sleep worker owes a pass (DESIGN.md §5.3 item 3, slice 6).
 *
 * The scheduler holds NO state (CLAUDE.md invariant #6: a timer emits, the
 * consumer journals the receipt). So "due" is a question asked of the log
 * itself, every sweep, from scratch:
 *
 *   due = the thread has events after its cursor that are worth extracting,
 *         AND its newest event is at least `idleMs` old.
 *
 * The cursor is the `toSeq` of the newest `sleep`/`extract` receipt on the
 * thread — the same value the pass itself re-reads under the lock — so a
 * discovery race can only ever cost a wasted pass that then reports
 * `lost-claim`, never a lost or duplicated write.
 *
 * The idle gate does double duty: it keeps the worker off a live conversation
 * (DESIGN.md §5.3: "runs on cron/idle"), and it is the retry backoff. A failed
 * pass journals an `error` event, which becomes the thread's newest event and
 * pushes the thread back out of the window for another `idleMs`.
 */
import type { Db, ThreadRef } from "@pinky/core";
import { EXTRACT_EVENT_TYPES } from "./types";
import { toIso, toNum } from "./util";

/** A thread with unextracted material, plus what discovery already knows about it. */
export interface DueThread {
  thread: ThreadRef;
  /** seq of the thread's newest event of ANY type. */
  lastSeq: number;
  /** ISO timestamp of that event — what the idle gate compared. */
  lastTs: string;
  /** `toSeq` of the newest extract receipt; 0 when the thread has never been swept. */
  cursorSeq: number;
}

interface DueRow {
  channel_id: string;
  thread_id: string;
  /** `bigint` on the wire => postgres.js hands these back as STRINGS. */
  last_seq: number | string;
  /** `timestamptz` => postgres.js hands this back as a Date, not a string. */
  last_ts: string | Date;
  cursor_seq: number | string;
}

/**
 * Threads owed an extraction pass, oldest-idle first.
 *
 * Shape, deliberately: a lateral over `threads`, never a `group by` over
 * `events`. The event log is the biggest table in the system and grows without
 * bound; `max(seq) group by thread` reads all of it every sweep, while two
 * `order by seq desc limit 1` laterals are two index probes per thread.
 *
 * `sleep:` channels are excluded because that is where the worker journals its
 * OWN receipts (`reflectThread`): extracting memories from the record of
 * extracting memories is a feedback loop, and every event there is audit-only
 * anyway.
 *
 * `now` is a parameter, not `now()`, so a test can put the clock wherever it
 * needs the idle gate to fall.
 */
export async function discoverDueThreads(
  db: Db,
  opts: {
    tenantId: string;
    idleMs: number;
    limit: number;
    channelId?: string;
    threadId?: string;
    now?: Date;
  },
): Promise<DueThread[]> {
  const params: unknown[] = [];
  params.push(opts.tenantId);
  const pTenant = `$${params.length}`;
  params.push((opts.now ?? new Date()).toISOString());
  const pNow = `$${params.length}`;
  params.push(Math.max(0, Math.floor(opts.idleMs)));
  const pIdle = `$${params.length}`;
  // Spread to a plain array: EXTRACT_EVENT_TYPES is `as const` (readonly), and
  // the driver binds a mutable string[] to `text[]`.
  params.push([...EXTRACT_EVENT_TYPES]);
  const pTypes = `$${params.length}`;

  let filters = "";
  if (opts.channelId !== undefined) {
    params.push(opts.channelId);
    filters += ` and t.channel_id = $${params.length}`;
  }
  if (opts.threadId !== undefined) {
    params.push(opts.threadId);
    filters += ` and t.thread_id = $${params.length}`;
  }

  params.push(Math.max(1, Math.floor(opts.limit)));
  const pLimit = `$${params.length}`;

  const rows = await db.query<DueRow>(
    `select t.channel_id, t.thread_id, last.seq as last_seq, last.ts as last_ts,
            coalesce(cur.to_seq, 0) as cursor_seq
     from threads t
     cross join lateral (
       select e.seq, e.ts from events e
       where (e.tenant_id, e.channel_id, e.thread_id) = (t.tenant_id, t.channel_id, t.thread_id)
       order by e.seq desc limit 1) last
     left join lateral (
       select (e.data->>'toSeq')::bigint as to_seq from events e
       where (e.tenant_id, e.channel_id, e.thread_id) = (t.tenant_id, t.channel_id, t.thread_id)
         and e.type = 'sleep' and e.data->>'phase' = 'extract'
       order by e.seq desc limit 1) cur on true
     where t.tenant_id = ${pTenant}
       and t.channel_id not like 'sleep:%'
       and last.ts <= ${pNow}::timestamptz - (${pIdle}::bigint * interval '1 millisecond')
       and exists (select 1 from events e
                   where (e.tenant_id, e.channel_id, e.thread_id) = (t.tenant_id, t.channel_id, t.thread_id)
                     and e.seq > coalesce(cur.to_seq, 0) and e.type = any(${pTypes}::text[]))${filters}
     order by last.ts asc limit ${pLimit}`,
    params,
  );

  return rows.map((r) => ({
    thread: { tenantId: opts.tenantId, channelId: r.channel_id, threadId: r.thread_id },
    lastSeq: toNum(r.last_seq, 0),
    lastTs: toIso(r.last_ts),
    cursorSeq: toNum(r.cursor_seq, 0),
  }));
}
