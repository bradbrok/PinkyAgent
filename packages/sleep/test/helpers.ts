/**
 * Shared doubles for the sleep-worker unit suite (slice 6).
 *
 * Everything here is hand-rolled: no database, no network, no clock. Two
 * things it does NOT fake away, because they are where the pass's defects
 * would hide:
 *
 * - **`seq` comes back as a STRING**, the way postgres.js hands back a bigint.
 *   A fake that returned numbers would make every missing `Number()` coercion
 *   invisible (event-store.ts `toSeq` explains what that costs).
 * - **`tx` is recorded**, so "one transaction" is an assertion (`txLog`) rather
 *   than a claim in a comment — that is the whole safety argument of a pass.
 *
 * SECTIONS: agent B owns everything down to the "END OF SECTION B" marker;
 * agent C appends below it. Re-read the file before editing.
 */
import { DEFAULT_SETTINGS, EventStore } from "@pinky/core";
import type {
  Db,
  MemoryHit,
  MemoryKind,
  MemoryRow,
  MemorySinceInput,
  MemoryStore,
  MemoryVisibility,
  RetainInput,
  SearchInput,
  ThreadEvent,
  ThreadEventData,
  ThreadRef,
  ToolCall,
} from "@pinky/core";
import { FakeProvider } from "@pinky/runtime";
import type { AssistantTurn, CompleteOptions, Embedder } from "@pinky/runtime";
import type { FakeScript } from "@pinky/runtime";
import type { SleepDeps, SleepScope, SleepSettings } from "../src/types";

export const THREAD: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

const norm = (sql: string): string => sql.replace(/\s+/g, " ").trim();

// ---------------------------------------------------------------------------
// FakeDb — the event log, plus the two statements a pass issues by hand
// ---------------------------------------------------------------------------

export interface Call {
  sql: string;
  params: unknown[] | undefined;
  /** 0 outside a transaction; the pass's writes must all be > 0. */
  txDepth: number;
}

export interface Route {
  pattern: RegExp;
  respond: (params: unknown[] | undefined) => unknown[];
}

/**
 * A Db that really keeps an event log, so `EventStore.appendTx`, `history()`
 * and the worker's cursor query all behave as they do against Postgres —
 * including the bits that bite: bigints arrive as strings, and `tx()` is a
 * real nesting boundary whose begin/commit/rollback is recorded.
 *
 * `routes` are consulted BEFORE the built-in handling, so a test can override
 * one statement (or make it throw) without reimplementing the log.
 */
export class FakeDb implements Db {
  readonly calls: Call[] = [];
  /** "begin"/"commit"/"rollback" for the OUTERMOST tx only, in order. */
  readonly txLog: string[] = [];
  readonly events: ThreadEvent[] = [];
  /** Extra routes, tried first. Used to inject failures. */
  readonly routes: Route[] = [];
  private seqByThread = new Map<string, number>();
  private txDepth = 0;
  private idSeq = 0;

  private key(ref: Pick<ThreadEvent, "tenantId" | "channelId" | "threadId">): string {
    return `${ref.tenantId}:${ref.channelId}:${ref.threadId}`;
  }

  private eventsFor(key: string): ThreadEvent[] {
    return this.events.filter((e) => this.key(e) === key).sort((a, b) => a.seq - b.seq);
  }

  /** Land events directly, bypassing the append path (fixtures, not writes). */
  seed(ref: ThreadRef, datas: ThreadEventData[]): ThreadEvent[] {
    const key = this.key(ref);
    let seq = this.seqByThread.get(key) ?? 0;
    const out: ThreadEvent[] = [];
    for (const data of datas) {
      seq += 1;
      this.idSeq += 1;
      const event: ThreadEvent = {
        ...ref,
        id: `seed-${this.idSeq}`,
        seq,
        ts: new Date(Date.UTC(2026, 7, 1, 0, 0, seq)).toISOString(),
        data,
      };
      this.events.push(event);
      out.push(event);
    }
    this.seqByThread.set(key, seq);
    return out;
  }

  /** Fail every statement matching `pattern` — the mid-apply rollback case. */
  failOn(pattern: RegExp, message: string): void {
    this.routes.push({
      pattern,
      respond: () => {
        throw new Error(message);
      },
    });
  }

  /** postgres.js hands a jsonb column back as a parsed value (pg.ts JSONB CONTRACT). */
  private toRow(e: ThreadEvent): Record<string, unknown> {
    return {
      id: e.id,
      tenant_id: e.tenantId,
      channel_id: e.channelId,
      thread_id: e.threadId,
      // A STRING: `seq` is bigint on the wire.
      seq: String(e.seq),
      ts: e.ts,
      data: e.data,
    };
  }

  query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params, txDepth: this.txDepth });
    const s = norm(sql);
    for (const route of this.routes) {
      if (route.pattern.test(s)) return Promise.resolve(route.respond(params) as T[]);
    }
    const p = params ?? [];
    const threadKey = `${String(p[0])}:${String(p[1])}:${String(p[2])}`;

    if (/insert into threads/.test(s)) {
      if (!this.seqByThread.has(threadKey)) this.seqByThread.set(threadKey, 0);
      return Promise.resolve([] as T[]);
    }
    if (/from threads .* for update/.test(s)) return Promise.resolve([] as T[]);

    // The worker's cursor: newest sleep/extract receipt's toSeq, as a bigint
    // string — the coercion the pass has to do.
    //
    // The pattern is ANCHORED, and that is load-bearing. Discovery's SQL also
    // contains `(e.data->>'toSeq')::bigint` and `data->>'phase' = 'extract'`,
    // so a loose /data->>'toSeq'/ swallows it and answers with this route's
    // one-row shape — which a sweep test that forgot `installDueThreads` reads
    // as a perfectly healthy "no due threads" instead of the loud
    // `FakeDb: no route for SQL` it should get. Only the cursor query starts
    // with the UNALIASED projection (discovery's starts `select t.channel_id`).
    if (/^select \(data->>'toSeq'\)::bigint .*data->>'phase' = 'extract'.* order by seq desc/.test(s)) {
      const last = [...this.eventsFor(threadKey)]
        .reverse()
        .find((e) => e.data.type === "sleep" && e.data.phase === "extract");
      const toSeq =
        last && last.data.type === "sleep" && last.data.phase === "extract"
          ? String(last.data.toSeq)
          : null;
      return Promise.resolve((toSeq === null ? [] : [{ to_seq: toSeq }]) as T[]);
    }
    if (/coalesce\(max\(seq\), 0\) \+ 1 as next/.test(s)) {
      // Also a bigint string.
      return Promise.resolve([{ next: String((this.seqByThread.get(threadKey) ?? 0) + 1) }] as T[]);
    }
    if (/insert into events/.test(s)) {
      const [id, tenantId, channelId, threadId, seq, , data] = p as [
        string,
        string,
        string,
        string,
        number,
        string,
        ThreadEventData,
      ];
      const event: ThreadEvent = {
        id,
        tenantId,
        channelId,
        threadId,
        seq,
        ts: new Date(Date.UTC(2026, 7, 2, 0, 0, seq)).toISOString(),
        data,
      };
      this.events.push(event);
      this.seqByThread.set(this.key(event), seq);
      return Promise.resolve([this.toRow(event)] as T[]);
    }
    // history(): forward page, ascending.
    if (/from events/.test(s) && /seq > \$4/.test(s)) {
      const after = Number(p[3]);
      const limit = Number(p[4]);
      const rows = this.eventsFor(threadKey)
        .filter((e) => e.seq > after)
        .slice(0, limit);
      return Promise.resolve(rows.map((r) => this.toRow(r)) as T[]);
    }
    // contextEvents(): newest-first, capped.
    if (/from events/.test(s) && /seq >= \$4/.test(s)) {
      const from = Number(p[3]);
      const limit = Number(p[4]);
      const rows = this.eventsFor(threadKey)
        .filter((e) => e.seq >= from)
        .sort((a, b) => b.seq - a.seq)
        .slice(0, limit);
      return Promise.resolve(rows.map((r) => this.toRow(r)) as T[]);
    }
    if (/type = 'continuity'/.test(s)) {
      const last = [...this.eventsFor(threadKey)]
        .reverse()
        .find((e) => e.data.type === "continuity");
      return Promise.resolve((last ? [{ seq: String(last.seq) }] : []) as T[]);
    }
    throw new Error(`FakeDb: no route for SQL: ${s}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    const outermost = this.txDepth === 0;
    if (outermost) this.txLog.push("begin");
    this.txDepth += 1;
    try {
      const out = await fn(this);
      if (outermost) this.txLog.push("commit");
      return out;
    } catch (err) {
      if (outermost) this.txLog.push("rollback");
      throw err;
    } finally {
      this.txDepth -= 1;
    }
  }

  close(): Promise<void> {
    return Promise.resolve();
  }

  find(pattern: RegExp): Call | undefined {
    return this.calls.find((c) => pattern.test(norm(c.sql)));
  }

  all(pattern: RegExp): Call[] {
    return this.calls.filter((c) => pattern.test(norm(c.sql)));
  }

  /** Every event data on a thread, in seq order. */
  dataFor(ref: ThreadRef): ThreadEventData[] {
    return this.eventsFor(this.key(ref)).map((e) => e.data);
  }
}

// ---------------------------------------------------------------------------
// Memory-plane double
// ---------------------------------------------------------------------------

let rowSeq = 0;

export function makeMemoryRow(over: Partial<MemoryRow> = {}): MemoryRow {
  rowSeq += 1;
  return {
    id: `${String(rowSeq).padStart(8, "0")}-0000-4000-8000-000000000000`,
    tenantId: "t1",
    agentId: "pinky",
    visibility: "channel" as MemoryVisibility,
    userId: null,
    channelId: "c1",
    kind: "semantic" as MemoryKind,
    text: `memory ${rowSeq}`,
    importance: 5,
    validFrom: "2026-08-01T00:00:00.000Z",
    validTo: null,
    recordedAt: "2026-08-01T00:00:00.000Z",
    embeddingModel: null,
    meta: {},
    ...over,
  };
}

export function makeMemoryHit(over: Partial<MemoryHit> = {}): MemoryHit {
  const { score, voices, ...rowOver } = over;
  return { ...makeMemoryRow(rowOver), score: score ?? 0.5, voices: voices ?? { fts: 1 } };
}

export interface FakeMemoryOptions {
  /** Neighbours every `search` returns, unless `hitsFor` overrides. */
  hits?: MemoryHit[];
  /** Per-call neighbours, by 0-based search index. */
  hitsFor?: (input: SearchInput, index: number) => MemoryHit[];
  /** Rows `since()` answers with (the reflect pass's batch). */
  since?: MemoryRow[];
  supportsVectors?: boolean;
  /** `invalidate()`'s return; false = "already retired by someone else". */
  invalidateReturns?: boolean;
  /** Make one write throw — the mid-apply rollback case. */
  failOn?: { op: "retain" | "update" | "invalidate"; message: string };
}

export interface FakeMemory {
  /** Typed as the class the deps want; only its methods are ever called. */
  store: MemoryStore;
  /** Every Db `bind()` was handed, in order. Empty until a pass binds. */
  boundTo: Db[];
  retains: RetainInput[];
  updates: { id: string; replacement: Record<string, unknown> }[];
  invalidations: { id: string; reason?: string }[];
  searches: SearchInput[];
  sinces: MemorySinceInput[];
  /** Rows written by retain/update, newest last. */
  written: MemoryRow[];
  supportsVectorsCalls: number;
}

/**
 * A MemoryStore double that records every call.
 *
 * `bind(db)` returns THE SAME double and records the handle it was given —
 * which is the only way a unit test can prove the pass's writes were issued on
 * the transaction it locked in, rather than on the outer pool.
 */
export function makeFakeMemory(opts: FakeMemoryOptions = {}): FakeMemory {
  const boundTo: Db[] = [];
  const retains: RetainInput[] = [];
  const updates: { id: string; replacement: Record<string, unknown> }[] = [];
  const invalidations: { id: string; reason?: string }[] = [];
  const searches: SearchInput[] = [];
  const sinces: MemorySinceInput[] = [];
  const written: MemoryRow[] = [];
  const state = { supportsVectorsCalls: 0 };

  const fail = (op: "retain" | "update" | "invalidate"): void => {
    if (opts.failOn?.op === op) throw new Error(opts.failOn.message);
  };

  const store = {
    bind(db: Db) {
      boundTo.push(db);
      return store;
    },
    supportsVectors(): Promise<boolean> {
      state.supportsVectorsCalls += 1;
      return Promise.resolve(opts.supportsVectors ?? true);
    },
    search(input: SearchInput): Promise<MemoryHit[]> {
      const index = searches.length;
      searches.push(input);
      return Promise.resolve(opts.hitsFor ? opts.hitsFor(input, index) : (opts.hits ?? []));
    },
    since(input: MemorySinceInput): Promise<MemoryRow[]> {
      sinces.push(input);
      return Promise.resolve(opts.since ?? []);
    },
    retain(input: RetainInput): Promise<MemoryRow> {
      fail("retain");
      retains.push(input);
      const row = makeMemoryRow({
        agentId: input.agentId,
        visibility: input.visibility,
        userId: input.userId ?? null,
        channelId: input.channelId ?? null,
        kind: input.kind,
        text: input.text,
        importance: input.importance ?? 5,
        embeddingModel: input.embeddingModel ?? null,
        meta: input.meta ?? {},
      });
      written.push(row);
      return Promise.resolve(row);
    },
    update(id: string, replacement: Record<string, unknown>): Promise<MemoryRow> {
      fail("update");
      updates.push({ id, replacement });
      const row = makeMemoryRow({
        text: String(replacement["text"] ?? ""),
        meta: { ...(replacement["meta"] as Record<string, unknown> | undefined), supersedes: id },
      });
      written.push(row);
      return Promise.resolve(row);
    },
    invalidate(id: string, o?: { reason?: string }): Promise<boolean> {
      fail("invalidate");
      invalidations.push({ id, ...(o?.reason ? { reason: o.reason } : {}) });
      return Promise.resolve(opts.invalidateReturns ?? true);
    },
  };

  return {
    store: store as unknown as MemoryStore,
    boundTo,
    retains,
    updates,
    invalidations,
    searches,
    sinces,
    written,
    get supportsVectorsCalls() {
      return state.supportsVectorsCalls;
    },
  };
}

// ---------------------------------------------------------------------------
// Provider doubles
// ---------------------------------------------------------------------------

/** A turn that calls exactly one tool. */
export function toolTurn(
  name: string,
  args: Record<string, unknown>,
  over: Partial<AssistantTurn> = {},
): AssistantTurn {
  return {
    text: "",
    toolCalls: [{ id: `${name}-1`, name, args }],
    stopReason: "tool_calls",
    ...over,
  };
}

/** A turn that calls nothing — what a model does when it ignores the forcing. */
export function textTurn(text: string, over: Partial<AssistantTurn> = {}): AssistantTurn {
  return { text, toolCalls: [], stopReason: "stop", ...over };
}

/**
 * A script keyed on the tool the caller FORCED, i.e. `opts.tools[0].name`.
 *
 * The worker's calls are not a conversation — each is a one-shot request with a
 * different system prompt and a different forced tool — so "which call is this"
 * is answered by the request, never by a shift()-ing counter that a skipped
 * decide call would silently desynchronize.
 */
export function byTool(
  handlers: Record<string, AssistantTurn | ((opts: CompleteOptions) => AssistantTurn)>,
): FakeScript {
  return (_messages, opts): AssistantTurn => {
    const name = opts.tools[0]?.name ?? "";
    const handler = handlers[name];
    if (!handler) return textTurn(`no scripted answer for tool ${JSON.stringify(name)}`);
    return typeof handler === "function" ? handler(opts) : handler;
  };
}

export function makeProvider(script: FakeScript): FakeProvider {
  return new FakeProvider(script);
}

/** Embedder double: deterministic vectors, or a rejection. */
export function makeEmbedder(
  opts: { model?: string; dimensions?: number; fail?: string } = {},
): Embedder & { calls: string[][] } {
  const calls: string[][] = [];
  const dimensions = opts.dimensions ?? 4;
  return {
    model: opts.model ?? "fake/embed",
    dimensions,
    calls,
    embed(texts: string[]): Promise<number[][]> {
      calls.push([...texts]);
      if (opts.fail) return Promise.reject(new Error(opts.fail));
      return Promise.resolve(
        texts.map((t, i) => Array.from({ length: dimensions }, (_, d) => (t.length + i + d) / 100)),
      );
    },
  };
}

// ---------------------------------------------------------------------------
// Deps
// ---------------------------------------------------------------------------

export const WIDE_SCOPE: SleepScope = { includeUser: true, includePrivate: true };
export const NARROW_SCOPE: SleepScope = { includeUser: false, includePrivate: false };

export function sleepSettings(over: Partial<SleepSettings> = {}): SleepSettings {
  return { ...DEFAULT_SETTINGS.sleep, ...over };
}

export interface DepsOverrides {
  db?: FakeDb;
  memory?: FakeMemory;
  provider?: FakeProvider;
  embedder?: Embedder;
  model?: string;
  agentId?: string;
  tenantId?: string;
  settings?: Partial<SleepSettings>;
  scope?: SleepScope;
  now?: () => Date;
  signal?: AbortSignal;
}

export interface Harness {
  deps: SleepDeps;
  db: FakeDb;
  memory: FakeMemory;
  provider: FakeProvider;
  /** Every line the worker wrote to its log (stderr in production). */
  logs: string[];
}

/**
 * Assemble `SleepDeps` out of doubles. A fixed clock by default, so a receipt's
 * `ms` is 0 and a snapshot of one is stable.
 */
export function makeDeps(over: DepsOverrides = {}): Harness {
  const db = over.db ?? new FakeDb();
  const memory = over.memory ?? makeFakeMemory();
  const provider = over.provider ?? makeProvider(byTool({}));
  const logs: string[] = [];
  const fixed = new Date("2026-08-29T00:00:00.000Z");
  const deps: SleepDeps = {
    db,
    events: new EventStore(db),
    memory: memory.store,
    provider,
    model: over.model ?? "fake/sleep",
    agentId: over.agentId ?? "pinky",
    tenantId: over.tenantId ?? "t1",
    settings: sleepSettings(over.settings ?? {}),
    scope: over.scope ?? WIDE_SCOPE,
    log: (msg) => logs.push(msg),
    now: over.now ?? ((): Date => fixed),
    ...(over.embedder ? { embedder: over.embedder } : {}),
    ...(over.signal ? { signal: over.signal } : {}),
  };
  return { deps, db, memory, provider, logs };
}

/** An `ingress` fixture; `userId` is what a `user`-visible candidate may claim. */
export function ingress(text: string, userId = "u1"): ThreadEventData {
  return {
    type: "ingress",
    platform: "cli",
    author: { platform: "cli", userId },
    text,
    refs: [],
  };
}

export function assistant(text: string, toolCalls: ToolCall[] = []): ThreadEventData {
  return { type: "message", role: "assistant", text, toolCalls, model: "fake/sleep" };
}

// ============================ END OF SECTION B ============================
// Agent C appends below this line.

// ---------------------------------------------------------------------------
// Section C — reflect, discovery, sweep
// ---------------------------------------------------------------------------

// Kept in this section rather than merged into the imports above so the two
// sections stay independently editable (an import declaration is legal at any
// top-level position).
// `ThreadEvent` is already imported by section B above and is in scope for the
// whole module; re-importing it here is a duplicate-identifier error.
import type { ReflectReceipt } from "../src/types";

/**
 * Answer `runReflectPass`'s watermark query out of the FakeDb's own event log.
 *
 * Two statements the log itself cannot serve are the pass's own by-hand
 * queries; this is the reflect one (`data->>'phase' = 'reflect'`). Answering it
 * from `db.events` rather than a scripted response is what makes the
 * lost-claim test honest: a receipt seeded DURING the provider call is invisible
 * to the read that already happened and visible to the in-transaction re-read,
 * exactly as a concurrent pass's commit would be in Postgres.
 */
export function installReflectCursor(db: FakeDb): void {
  // The idle gate's probe: the newest event on the thread, whatever its type.
  // `ts` comes back as a DATE from postgres.js, so the double hands one over
  // too — a fake that returned an ISO string would hide the coercion.
  db.routes.push({
    pattern: /select ts from events/,
    respond: (params) => {
      const [tenantId, channelId, threadId] = (params ?? []) as string[];
      const last = [...db.events]
        .reverse()
        .find(
          (e) => e.tenantId === tenantId && e.channelId === channelId && e.threadId === threadId,
        );
      return last ? [{ ts: new Date(last.ts) }] : [];
    },
  });
  db.routes.push({
    pattern: /data->>'phase' = 'reflect'/,
    respond: (params) => {
      const [tenantId, channelId, threadId] = (params ?? []) as string[];
      const last = [...db.events]
        .reverse()
        .find(
          (e) =>
            e.tenantId === tenantId &&
            e.channelId === channelId &&
            e.threadId === threadId &&
            e.data.type === "sleep" &&
            e.data.phase === "reflect",
        );
      return last ? [{ data: last.data }] : [];
    },
  });
}

/**
 * Seed events and stamp them at `ts` — the idle gate compares the newest
 * event's timestamp against the clock, and `seed()` dates everything to
 * 2026-08-01 (a month before the harness's fixed "now", i.e. always idle).
 */
export function seedAt(
  db: FakeDb,
  ref: ThreadRef,
  datas: ThreadEventData[],
  ts: Date,
): ThreadEvent[] {
  const out = db.seed(ref, datas);
  for (const event of out) event.ts = ts.toISOString();
  return out;
}

/** A `sleep`/`reflect` receipt to seed as a prior pass's watermark. */
export function reflectReceipt(over: Partial<ReflectReceipt> = {}): ThreadEventData {
  return {
    type: "sleep",
    phase: "reflect",
    after: null,
    through: { recordedAt: "2026-08-01T00:00:00.000Z", id: "m0" },
    scanned: 1,
    candidates: 0,
    added: 0,
    updated: 0,
    invalidated: 0,
    noop: 0,
    model: "fake/sleep",
    ms: 0,
    ...over,
  };
}

/**
 * Route `discoverDueThreads`'s lateral query to fixed rows. The unit suite only
 * asserts the REQUEST (text + params); that Postgres agrees with the plan is
 * the integration suite's job.
 */
export function installDueThreads(db: FakeDb, rows: Record<string, unknown>[] = []): void {
  db.routes.push({ pattern: /from threads t cross join lateral/, respond: () => rows });
}

/**
 * One discovery row, shaped like the wire: bigint columns as STRINGS and
 * `timestamptz` as a DATE, which is what postgres.js actually returns.
 */
export function dueRow(
  channelId: string,
  over: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    channel_id: channelId,
    thread_id: "main",
    last_seq: "10",
    last_ts: new Date("2026-08-29T10:00:00.000Z"),
    cursor_seq: "0",
    ...over,
  };
}
