/**
 * The memory plane (DESIGN.md §5) — storage and hybrid retrieval.
 *
 * Four rules run through everything below.
 *
 * 1. SCOPE IS A PREDICATE, NOT A FILTER APPLIED LATER (§5.1). Every read goes
 *    through {@link scopePredicate}: recall in a shared channel sees
 *    tenant + channel + global; a DM additionally sees the subject user's
 *    rows; the agent's `private` scratch is never projected into shared
 *    context. Below that, Postgres row-level security keys on the
 *    `pinky.tenant_id` GUC, so the Db handed to the constructor SHOULD be
 *    withTenant()-wrapped (packages/core/src/tenant.ts). Belt and braces: the
 *    store also writes `tenant_id` explicitly and repeats it in every WHERE,
 *    because RLS is bypassed outright by a superuser connection and a dev
 *    checkout runs as one.
 *
 * 2. INVALIDATION, NEVER DELETION (§5.2). Nothing here issues a DELETE.
 *    `invalidate()` stamps `valid_to`; `update()` stamps the old row and
 *    inserts its replacement in ONE transaction, so a reader never observes
 *    either both versions or neither. Current truth = `valid_to is null`;
 *    the history stays queryable for temporal reconstruction.
 *
 * 3. RETRIEVAL IS TWO VOICES PLUS A RESCORE (§5.4). A vector voice (pgvector
 *    cosine) and a lexical voice (Postgres FTS) each return their own ranked
 *    candidates; {@link fuseHits} merges them with reciprocal rank fusion and
 *    then applies Generative-Agents-style `α·recency + β·relevance +
 *    γ·importance`. Either voice can be absent — no pgvector, no embedder, or
 *    an empty query — and the other still answers. With BOTH absent, search
 *    degrades to a newest-first listing rather than returning nothing.
 *
 * 4. THE VECTOR PARAM IS A STRING (§5.5). pgvector's text input form is
 *    '[0.1,0.2,...]'. Handing postgres.js a JS number[] binds a float8[],
 *    which the server will not coerce to `vector`. Always
 *    {@link vectorLiteral} + an explicit `::vector` cast.
 *
 * jsonb (`meta`) obeys pg.ts's JSONB CONTRACT: plain values, never
 * JSON.stringify — see schema/0004_jsonb_repair.rerun.sql for what pre-encoding
 * costs.
 */
import type { Db } from "./db";
import { jsonbParam } from "./pg";

export type MemoryKind = "semantic" | "episodic" | "procedural";
export type MemoryVisibility = "private" | "user" | "channel" | "tenant" | "global";

const KINDS: readonly MemoryKind[] = ["semantic", "episodic", "procedural"];
const VISIBILITIES: readonly MemoryVisibility[] = [
  "private",
  "user",
  "channel",
  "tenant",
  "global",
];

/** A memory row as the application sees it: timestamps ISO, meta an object. */
export interface MemoryRow {
  id: string;
  tenantId: string;
  agentId: string;
  visibility: MemoryVisibility;
  userId: string | null;
  channelId: string | null;
  kind: MemoryKind;
  text: string;
  /** 1..10, LLM-assigned at write time (§5.4). */
  importance: number;
  validFrom: string;
  /** Non-null => invalidated; the row is history, not current truth. */
  validTo: string | null;
  recordedAt: string;
  /** "provider/model-id" that produced `embedding`; null = no embedding. */
  embeddingModel: string | null;
  meta: Record<string, unknown>;
}

/** Who is asking, and from where (DESIGN.md §5.1). */
export interface RecallScope {
  agentId: string;
  /** Channel whose `channel`-visibility rows are visible. */
  channelId?: string;
  /** Subject user whose `user`-visibility rows are visible (DMs only). */
  userId?: string;
  /**
   * SLEEP-WORKER ONLY (DESIGN.md §5.3 item 3, slice 6): read `channel`-visibility
   * rows of EVERY channel, not just `channelId`'s.
   *
   * Never set by a conversation run — a run sees exactly one channel, and that
   * is the whole point of `channel` visibility (§5.1). The reflect pass is the
   * one reader that is legitimately cross-thread: it consolidates the plane
   * itself, so it has to be able to SEE what extraction wrote, and extraction
   * writes `channel` by default. What it may then WRITE stays narrow — an
   * insight drawn from one channel stays in that channel, and one drawn from
   * two is dropped rather than widened to the tenant (see reflect.ts).
   */
  allChannels?: boolean;
  /** DM or trusted local surface: include `user` rows for userId. */
  includeUser: boolean;
  /** Trusted local surface (cli) or the agent's own DM: include the agent's `private` rows. */
  includePrivate: boolean;
}

export interface RetainInput {
  agentId: string;
  visibility: MemoryVisibility;
  userId?: string;
  channelId?: string;
  kind: MemoryKind;
  text: string;
  /** 1..10; default 5. Out of range throws rather than clamping silently. */
  importance?: number;
  /** Omitted, empty, or no pgvector on this database => stored without one. */
  embedding?: number[];
  embeddingModel?: string;
  meta?: Record<string, unknown>;
}

export interface SearchInput {
  scope: RecallScope;
  /** FTS text; blank => the lexical voice is skipped. */
  query: string;
  /** Vector voice; omitted or unsupported => skipped. */
  queryEmbedding?: number[];
  /** Final hit count after fusion; default 20. */
  limit?: number;
  kinds?: MemoryKind[];
  /** Score weights (§5.4: α·recency + β·relevance + γ·importance). */
  weights?: { recency?: number; relevance?: number; importance?: number };
  /** Injectable clock for the recency decay. */
  now?: Date;
}

/**
 * Watermark read for the sleep-time worker (DESIGN.md §5.3 item 3, slice 6):
 * "every current row in scope recorded after this point, oldest first".
 *
 * The watermark is a TUPLE, not a timestamp. `recorded_at` defaults to `now()`,
 * which in Postgres is the transaction's start time, so every row a single
 * transaction retains shares it exactly — a timestamp-only cursor set to that
 * value would skip every sibling but one, permanently. `(recorded_at, id)` is
 * unique because `id` is the primary key.
 */
export interface MemorySinceInput {
  scope: RecallScope;
  /** Exclusive: rows strictly after this tuple. `null` = from the beginning. */
  after: { recordedAt: string; id: string } | null;
  limit: number;
  kinds?: MemoryKind[];
  /**
   * Narrow to these visibilities, INTERSECTED with the scope predicate — it can
   * only ever remove rows the scope already allowed, never add one. The reflect
   * pass uses it to keep `user`/`private` rows out of a batch whose synthesized
   * insight lands at `tenant` visibility (§5.1: a shared insight must not carry
   * one user's facts into the shared scope).
   */
  visibilities?: MemoryVisibility[];
  /**
   * Drop rows whose `meta.source` is one of these (slice 6).
   *
   * The reflect pass excludes `"sleep:reflect"`, because otherwise it CONSUMES
   * ITS OWN OUTPUT: an insight lands at tenant/channel visibility under the same
   * agent, after the watermark, so the next pass reads it back as fresh
   * material. With a low `reflectMinMemories` that is a loop which sustains
   * itself forever on zero new conversation — consolidating consolidations.
   */
  excludeSources?: string[];
}

export interface MemoryHit extends MemoryRow {
  /** Fused final score; results are sorted by it, descending. */
  score: number;
  /** Which voices ranked it, with the 1-based rank each gave. */
  voices: { vector?: number; fts?: number };
}

/** Default fusion weights (§5.4). Relevance leads; recency and importance nudge. */
export const DEFAULT_WEIGHTS = { relevance: 0.6, recency: 0.2, importance: 0.2 } as const;

/** Reciprocal-rank-fusion constant. 60 is the value the RRF paper settled on. */
export const RRF_K = 60;

/** Per-hour multiplier for the recency term (§5.4: "recency decay γ≈0.995"). */
export const RECENCY_DECAY_PER_HOUR = 0.995;

/** Default final hit count for {@link MemoryStore.search}. */
export const DEFAULT_SEARCH_LIMIT = 20;

/** Default row count for {@link MemoryStore.list}. */
export const DEFAULT_LIST_LIMIT = 50;

/** Hard cap on per-voice candidates, so a big `limit` cannot scan the plane. */
export const MAX_CANDIDATES = 60;

/**
 * Width of `memories.embedding` (schema/0002_embeddings.rerun.sql declares
 * `vector(1536)`). Postgres refuses any other length outright, so a store that
 * simply forwarded whatever the embedder produced would lose the whole memory
 * to a 3072-wide model — see {@link MemoryStore.vectorDimensions}.
 */
export const MEMORY_VECTOR_DIMENSIONS = 1536;

/** Rows a prefix lookup returns unless the caller says otherwise. Two, not one,
 *  so the caller can tell "unique" from "ambiguous" in one query. */
export const DEFAULT_ID_PREFIX_LIMIT = 2;

/**
 * What an id prefix may contain. Memory ids are `crypto.randomUUID()`, so
 * lowercase hex and dashes cover every real prefix — and the character class is
 * exactly what keeps `%` and `_` (LIKE's wildcards) out of a value that is
 * concatenated into a LIKE pattern.
 */
const ID_PREFIX_RE = /^[0-9a-f-]{4,36}$/;

/**
 * `recorded_at` at the precision the application can actually see it
 * (see {@link MemoryStore.since}): postgres.js hands a timestamptz back through
 * `new Date(text)`, which truncates microseconds, so a watermark round-tripped
 * through JS is a MILLISECOND value. Both the cursor comparison and the ORDER
 * BY use this expression so the two agree exactly.
 */
const RECORDED_MS = `date_trunc('milliseconds', recorded_at)`;

/** Every column the store reads. `embedding`/`tsv` are never selected: the
 *  first may not exist (no pgvector) and neither is useful in JS. */
const COLUMNS = `id, tenant_id, agent_id, visibility, user_id, channel_id, kind, text,
         importance, valid_from, valid_to, recorded_at, embedding_model, meta`;

interface MemoryRowRaw {
  id: string;
  tenant_id: string;
  agent_id: string;
  visibility: string;
  user_id: string | null;
  channel_id: string | null;
  kind: string;
  text: string;
  importance: number | string;
  valid_from: Date | string;
  valid_to: Date | string | null;
  recorded_at: Date | string;
  embedding_model: string | null;
  /** jsonb. `string` only for legacy doubly-encoded rows — see mapRow(). */
  meta: Record<string, unknown> | string | null;
}

/**
 * timestamptz -> ISO string. postgres.js hands back a Date; a FakeDb (and a
 * driver configured with `types.date` off) hands back the text form. Anything
 * unparseable is passed through verbatim rather than becoming "Invalid Date".
 */
function toIso(value: Date | string): string {
  if (value instanceof Date) return value.toISOString();
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString();
}

function toIsoOrNull(value: Date | string | null | undefined): string | null {
  return value === null || value === undefined ? null : toIso(value);
}

/**
 * Row -> MemoryRow. The `meta` string branch is tolerance for legacy
 * doubly-encoded jsonb (repaired in place by 0004_jsonb_repair.rerun.sql),
 * exactly as EventStore.mapRow does for `events.data`.
 */
function mapRow(r: MemoryRowRaw): MemoryRow {
  let meta: Record<string, unknown> = {};
  if (typeof r.meta === "string") {
    try {
      const parsed: unknown = JSON.parse(r.meta);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        meta = parsed as Record<string, unknown>;
      }
    } catch {
      meta = {};
    }
  } else if (r.meta && typeof r.meta === "object") {
    meta = r.meta;
  }
  return {
    id: r.id,
    tenantId: r.tenant_id,
    agentId: r.agent_id,
    visibility: r.visibility as MemoryVisibility,
    userId: r.user_id ?? null,
    channelId: r.channel_id ?? null,
    kind: r.kind as MemoryKind,
    text: r.text,
    importance: Number(r.importance),
    validFrom: toIso(r.valid_from),
    validTo: toIsoOrNull(r.valid_to),
    recordedAt: toIso(r.recorded_at),
    embeddingModel: r.embedding_model ?? null,
    meta,
  };
}

/**
 * pgvector's text input form. NOT `JSON.stringify` by accident: the shape
 * coincides, but the intent is the documented '[1,2,3]' literal, and every
 * element must be a finite number or the server rejects the whole statement
 * with a parse error naming no column.
 */
function vectorLiteral(v: number[]): string {
  for (const n of v) {
    if (typeof n !== "number" || !Number.isFinite(n)) {
      throw new Error("memory: embedding must contain only finite numbers");
    }
  }
  return `[${v.join(",")}]`;
}

function assertKind(kind: unknown): MemoryKind {
  if (typeof kind !== "string" || !KINDS.includes(kind as MemoryKind)) {
    throw new Error(`memory: unknown kind ${JSON.stringify(kind)} (expected ${KINDS.join(", ")})`);
  }
  return kind as MemoryKind;
}

function assertVisibility(visibility: unknown): MemoryVisibility {
  if (typeof visibility !== "string" || !VISIBILITIES.includes(visibility as MemoryVisibility)) {
    throw new Error(
      `memory: unknown visibility ${JSON.stringify(visibility)} (expected ${VISIBILITIES.join(", ")})`,
    );
  }
  return visibility as MemoryVisibility;
}

/** Default 5; 1..10 integers only. Out of range throws — a silent clamp would
 *  quietly reshape the importance term of every later score. */
function normalizeImportance(value: number | undefined): number {
  if (value === undefined) return 5;
  if (!Number.isInteger(value) || value < 1 || value > 10) {
    throw new Error(`memory: importance must be an integer 1..10, got ${JSON.stringify(value)}`);
  }
  return value;
}

/**
 * The scope predicate (DESIGN.md §5.1). Appends its bind values to `params`
 * and returns SQL whose `$n` placeholders start at the old `params.length + 1`,
 * so it composes with whatever the caller has already bound.
 *
 *   agent_id = $a AND valid_to IS NULL AND (
 *        visibility = 'global'
 *     OR visibility = 'tenant'
 *     OR (visibility = 'channel' AND channel_id = $c)   -- only with scope.channelId
 *     OR visibility = 'channel'                         -- only with allChannels
 *     OR (visibility = 'user'    AND user_id    = $u)   -- only with includeUser + userId
 *     OR visibility = 'private'                         -- only with includePrivate
 *   )
 *
 * `global` is NOT cross-tenant in v1: every row is still fenced by tenant_id
 * (RLS + the store's own predicate). It means "every channel and user of this
 * tenant"; a genuinely cross-tenant global scope is a later slice.
 *
 * The channel/user arms are omitted rather than bound to NULL, because
 * `channel_id = NULL` is NULL, not false — a subtle way to write a clause that
 * looks like a filter and matches nothing. Omitting says what is meant.
 *
 * `allChannels` REPLACES the channelId arm rather than joining it: the bare
 * `visibility = 'channel'` strictly subsumes `visibility = 'channel' and
 * channel_id = $c`, so emitting both would bind a parameter that cannot change
 * a single row — and a redundant clause inside a privacy predicate is one a
 * later reader has to re-derive as harmless before they can touch it. One arm,
 * one meaning.
 *
 * `includeInvalid` drops the `valid_to is null` conjunct; that is the ONLY
 * caller-visible switch, and only `list()` uses it (audit/CLI history).
 */
export function scopePredicate(
  scope: RecallScope,
  params: unknown[],
  opts?: { includeInvalid?: boolean },
): string {
  if (!scope.agentId || scope.agentId.trim() === "") {
    throw new Error("memory: scope.agentId must be a non-empty string");
  }
  params.push(scope.agentId);
  const agent = `$${params.length}`;

  const arms: string[] = ["visibility = 'global'", "visibility = 'tenant'"];
  if (scope.allChannels) {
    // Cross-channel read (slice 6). Subsumes the channelId arm, so that one is
    // not also emitted — see the note above.
    arms.push("visibility = 'channel'");
  } else if (scope.channelId) {
    params.push(scope.channelId);
    arms.push(`(visibility = 'channel' and channel_id = $${params.length})`);
  }
  if (scope.includeUser && scope.userId) {
    params.push(scope.userId);
    arms.push(`(visibility = 'user' and user_id = $${params.length})`);
  }
  if (scope.includePrivate) arms.push("visibility = 'private'");

  const validity = opts?.includeInvalid ? "" : " and valid_to is null";
  return `agent_id = ${agent}${validity} and (${arms.join(" or ")})`;
}

/** Optional `and kind = any(...)` fragment; empty string when unfiltered. */
function kindsClause(kinds: MemoryKind[] | undefined, params: unknown[]): string {
  if (!kinds || kinds.length === 0) return "";
  for (const k of kinds) assertKind(k);
  params.push(kinds);
  // Explicit ::text[] cast: without it the driver's inferred array type and
  // the column's text type do not always unify on `= any($n)`.
  return ` and kind = any($${params.length}::text[])`;
}

/**
 * Optional `and visibility = any(...)` fragment (slice 6). Unlike
 * {@link kindsClause}, an EMPTY array throws rather than meaning "unfiltered":
 * this narrows a privacy boundary, and a filter that silently widens itself
 * when its list comes back empty is exactly the bug §5.1 cannot afford. Omit
 * the field to mean no restriction.
 */
function visibilitiesClause(
  visibilities: MemoryVisibility[] | undefined,
  params: unknown[],
): string {
  if (visibilities === undefined) return "";
  if (visibilities.length === 0) {
    throw new Error(
      "memory: visibilities must name at least one visibility (omit the field for no restriction)",
    );
  }
  for (const v of visibilities) assertVisibility(v);
  params.push(visibilities);
  return ` and visibility = any($${params.length}::text[])`;
}

/**
 * Optional `and coalesce(meta->>'source', '') <> all(...)` fragment (slice 6).
 *
 * The COALESCE is the whole point: `meta->>'source'` is NULL for every row that
 * never recorded one (anything a human or the agent retained), and `NULL <> all
 * (...)` is NULL, not true — so without it the filter would silently drop every
 * row it was not asked about, which is most of the plane. `''` stands in for
 * "no source" and matches no real source string.
 *
 * Empty array throws, for the same reason as {@link visibilitiesClause}: an
 * exclusion list that came back empty must not read as "exclude nothing" by
 * accident. Omit the field to mean that.
 */
function excludeSourcesClause(sources: string[] | undefined, params: unknown[]): string {
  if (sources === undefined) return "";
  if (sources.length === 0) {
    throw new Error(
      "memory: excludeSources must name at least one source (omit the field for no restriction)",
    );
  }
  params.push(sources);
  return ` and coalesce(meta->>'source', '') <> all($${params.length}::text[])`;
}

export interface FuseOptions {
  /** Final hit count; default {@link DEFAULT_SEARCH_LIMIT}. */
  limit?: number;
  weights?: { recency?: number; relevance?: number; importance?: number };
  /** Injectable clock; default `new Date()`. */
  now?: Date;
  /** RRF constant; default {@link RRF_K}. */
  k?: number;
}

/**
 * Reciprocal rank fusion + the §5.4 rescore. Pure, so the ranking is testable
 * without a database.
 *
 *   rrf      = Σ over voices of 1 / (k + rank)          (rank is 1-based)
 *   rrfNorm  = rrf / max(rrf) over the candidate set     (0..1)
 *   decay    = 0.995 ^ hoursSince(recordedAt)            (0..1)
 *   score    = relevance·rrfNorm + recency·decay + importance·(importance/10)
 *
 * Normalizing by the best rrf in THIS candidate set (rather than by a
 * theoretical maximum) keeps the relevance term comparable across queries that
 * matched one voice and queries that matched both — a row ranked by both
 * voices still beats a row ranked by one, because its raw rrf is larger.
 *
 * Ties break newer-first, then by id, so a fused list is deterministic.
 */
export function fuseHits(
  voices: { vector?: MemoryRow[]; fts?: MemoryRow[] },
  opts: FuseOptions = {},
): MemoryHit[] {
  const k = opts.k ?? RRF_K;
  const limit = Math.max(1, Math.floor(opts.limit ?? DEFAULT_SEARCH_LIMIT));
  const wRelevance = opts.weights?.relevance ?? DEFAULT_WEIGHTS.relevance;
  const wRecency = opts.weights?.recency ?? DEFAULT_WEIGHTS.recency;
  const wImportance = opts.weights?.importance ?? DEFAULT_WEIGHTS.importance;
  const nowMs = (opts.now ?? new Date()).getTime();

  interface Candidate {
    row: MemoryRow;
    voices: { vector?: number; fts?: number };
    rrf: number;
  }
  const byId = new Map<string, Candidate>();

  const absorb = (rows: MemoryRow[] | undefined, voice: "vector" | "fts"): void => {
    if (!rows) return;
    rows.forEach((row, i) => {
      const rank = i + 1;
      const existing = byId.get(row.id);
      const candidate: Candidate = existing ?? { row, voices: {}, rrf: 0 };
      // A row can only be ranked once per voice; the first (better) rank wins.
      if (candidate.voices[voice] === undefined) {
        candidate.voices[voice] = rank;
        candidate.rrf += 1 / (k + rank);
      }
      if (!existing) byId.set(row.id, candidate);
    });
  };
  absorb(voices.vector, "vector");
  absorb(voices.fts, "fts");

  const candidates = Array.from(byId.values());
  let maxRrf = 0;
  for (const c of candidates) if (c.rrf > maxRrf) maxRrf = c.rrf;

  const hits: MemoryHit[] = candidates.map((c) => {
    const rrfNorm = maxRrf > 0 ? c.rrf / maxRrf : 0;
    const score =
      wRelevance * rrfNorm +
      wRecency * recencyDecay(c.row.recordedAt, nowMs) +
      wImportance * (clampImportance(c.row.importance) / 10);
    return { ...c.row, score, voices: c.voices };
  });

  hits.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    const ta = Date.parse(a.recordedAt);
    const tb = Date.parse(b.recordedAt);
    if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb - ta;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return hits.slice(0, limit);
}

/**
 * 0.995 ^ hours. A future timestamp (clock skew) counts as 0 hours rather than
 * scoring above 1; an unparseable one gets no recency credit at all, which is
 * the conservative direction — it can never outrank a row we can date.
 */
function recencyDecay(recordedAt: string, nowMs: number): number {
  const t = Date.parse(recordedAt);
  if (!Number.isFinite(t)) return 0;
  const hours = Math.max(0, (nowMs - t) / 3_600_000);
  return Math.pow(RECENCY_DECAY_PER_HOUR, hours);
}

function clampImportance(value: number): number {
  if (!Number.isFinite(value)) return 5;
  return Math.min(10, Math.max(1, value));
}

export interface MemoryStoreOptions {
  /**
   * Where a degraded write is reported — today, an embedding whose width does
   * not match the column (see {@link MemoryStore.vectorDimensions}). Default
   * `console.warn`, i.e. STDERR: stdout is the headless JSONL protocol.
   */
  onWarning?: (message: string) => void;
}

/**
 * The `supportsVectors()` cache, held in a cell rather than a field so
 * {@link MemoryStore.bind} can hand the SAME cache to a tx-scoped store instead
 * of a copy: whichever store probes first answers for both, and a bound store
 * never issues a second information_schema query inside a caller's transaction.
 */
interface VectorProbe {
  /** Null = not probed yet, or the last probe failed (so it is retried). */
  promise: Promise<boolean> | null;
}

/**
 * Read/write access to the memory plane for one tenant.
 *
 * `db` should be withTenant()-wrapped so the RLS policy has its GUC; the
 * tenant id is passed separately because every write states its own
 * `tenant_id` rather than relying on a default the policy would have to
 * supply.
 */
export class MemoryStore {
  private db: Db;
  private tenantId: string;
  private onWarning: (message: string) => void;
  /** Cached promise, so N concurrent recalls issue ONE probe (and share it).
   *  Shared by reference with every store {@link MemoryStore.bind} produced. */
  private vectorProbe: VectorProbe = { promise: null };

  /**
   * The width `memories.embedding` accepts. A vector of any other length is
   * DROPPED (with a warning) rather than sent: `memory.embeddingModel` is a
   * setting, so "openai/text-embedding-3-large" is a legal thing for a human
   * to write, and its 3072-wide output would otherwise make every `retain`
   * throw 22000 — losing the memory the agent was trying to keep. Storing the
   * text without a vector costs the vector voice for that row; throwing costs
   * the row.
   */
  readonly vectorDimensions: number = MEMORY_VECTOR_DIMENSIONS;

  constructor(db: Db, tenantId: string, opts: MemoryStoreOptions = {}) {
    if (!tenantId || tenantId.trim() === "") {
      throw new Error("MemoryStore: tenantId must be a non-empty string");
    }
    this.db = db;
    this.tenantId = tenantId;
    this.onWarning = opts.onWarning ?? ((message: string): void => console.warn(message));
  }

  /**
   * Does this database have `memories.embedding` (i.e. did pgvector exist when
   * 0002_embeddings.rerun.sql last ran)? Probed once per instance and cached;
   * a failed probe is un-cached so a transient error does not permanently
   * disable the vector voice.
   *
   * `table_schema = current_schema()` matters: information_schema lists every
   * schema this role can see, so a `memories` table in some other schema (a
   * tenant sandbox, an old copy left in `public` while the search_path points
   * elsewhere) would otherwise answer for ours.
   */
  supportsVectors(): Promise<boolean> {
    const cell = this.vectorProbe;
    if (cell.promise) return cell.promise;
    const probe = this.db
      .query<{ ok: number }>(
        `select 1 as ok from information_schema.columns
         where table_name = 'memories' and column_name = 'embedding'
           and table_schema = current_schema()`,
      )
      .then((rows) => rows.length > 0);
    cell.promise = probe;
    probe.catch(() => {
      if (cell.promise === probe) cell.promise = null;
    });
    return probe;
  }

  /**
   * The same store over another {@link Db} — a transaction handle — sharing
   * this store's tenant id, warning sink AND vector probe (slice 6).
   *
   * Why it composes with a transaction the CALLER owns: pg.ts's `tx()` reuses a
   * tx-scoped client IN PLACE when a nested transaction is requested (no nested
   * BEGIN), so `store.bind(tx).update(...)` — which opens its own `tx()` to pair
   * the invalidate with the replacement insert — runs inside the caller's
   * transaction and commits or rolls back with it. The tenant GUC that
   * withTenant() set at the head of that transaction is still in force too, so
   * RLS keys on the right tenant.
   *
   * The probe is shared rather than re-run because it is an information_schema
   * read: issuing it from inside the caller's transaction would be a second
   * statement on the critical path for an answer the parent already has, and
   * running it as a fresh statement per pass would multiply it by every write.
   *
   * This is the ONLY way to write memories inside someone else's transaction —
   * the sleep worker's whole safety argument is that its rows and its receipt
   * commit together (DESIGN.md §5.3 item 3).
   */
  bind(db: Db): MemoryStore {
    const bound = new MemoryStore(db, this.tenantId, { onWarning: this.onWarning });
    bound.vectorProbe = this.vectorProbe; // the same cell, not a copy
    return bound;
  }

  /** Write a new memory (§5.2: rows are only ever added). */
  async retain(input: RetainInput): Promise<MemoryRow> {
    const withVector = await this.wantsVector(input.embedding);
    return await MemoryStore.insertRow(this.db, this.tenantId, input, withVector);
  }

  async get(id: string): Promise<MemoryRow | null> {
    const row = await this.db.queryOne<MemoryRowRaw>(
      `select ${COLUMNS} from memories where id = $1 and tenant_id = $2`,
      [id, this.tenantId],
    );
    return row ? mapRow(row) : null;
  }

  /**
   * Retire a memory: stamp `valid_to`, never DELETE (§5.2). Returns false when
   * the id is unknown to this tenant or the row is already invalid — the
   * `valid_to is null` guard makes the call idempotent, so a retry cannot
   * rewrite the original invalidation time.
   */
  async invalidate(id: string, opts?: { reason?: string }): Promise<boolean> {
    const reason = opts?.reason?.trim();
    const params: unknown[] = [id, this.tenantId];
    let metaSet = "";
    if (reason) {
      params.push(jsonbParam({ invalidatedReason: reason }));
      // `|| $n::jsonb` merges at the top level, preserving the rest of meta.
      metaSet = `, meta = meta || $${params.length}::jsonb`;
    }
    const rows = await this.db.query<{ id: string }>(
      `update memories set valid_to = now()${metaSet}
       where id = $1 and tenant_id = $2 and valid_to is null
       returning id`,
      params,
    );
    return rows.length > 0;
  }

  /**
   * Replace a memory: invalidate the old row and insert its successor in ONE
   * transaction (§5.2 — a contradicting fact retires the old one; both states
   * must become visible together). The new row records `meta.supersedes`, so
   * the chain is walkable backwards through the history.
   *
   * Throws when the id is unknown to this tenant or already invalid: silently
   * inserting an orphan replacement would leave two "current" versions of a
   * fact the caller believed it had superseded.
   */
  async update(
    id: string,
    replacement: Omit<RetainInput, "agentId"> & Partial<Pick<RetainInput, "agentId">>,
  ): Promise<MemoryRow> {
    // Probe before opening the transaction: it is an unrelated read, and
    // holding the row's lock across it buys nothing.
    const withVector = await this.wantsVector(replacement.embedding);
    const tenantId = this.tenantId;
    return await this.db.tx(async (tx) => {
      const existing = await tx.queryOne<MemoryRowRaw>(
        `select ${COLUMNS} from memories where id = $1 and tenant_id = $2 for update`,
        [id, tenantId],
      );
      if (!existing) throw new Error(`memory update: ${id} not found`);
      const old = mapRow(existing);
      if (old.validTo !== null) {
        throw new Error(`memory update: ${id} was already invalidated at ${old.validTo}`);
      }
      await tx.query(`update memories set valid_to = now() where id = $1 and tenant_id = $2`, [
        id,
        tenantId,
      ]);
      const input: RetainInput = {
        ...replacement,
        agentId: replacement.agentId ?? old.agentId,
        meta: { ...(replacement.meta ?? {}), supersedes: id },
      };
      return await MemoryStore.insertRow(tx, tenantId, input, withVector);
    });
  }

  /**
   * Hybrid recall (§5.4). Runs the voices it can — pgvector cosine when an
   * embedding is supplied and the column exists, FTS when the query is not
   * blank — over the same scope predicate, then fuses.
   *
   * With NEITHER voice available (blank query, no embedding) this returns the
   * newest rows in scope instead of nothing, so a bare `recall` still shows the
   * agent what it knows. Those hits carry `voices: {}` and are ordered
   * newest-first; their `score` is the recency+importance part of the formula
   * and is informational, since there is no relevance signal to rank by.
   */
  async search(input: SearchInput): Promise<MemoryHit[]> {
    const limit = Math.max(1, Math.floor(input.limit ?? DEFAULT_SEARCH_LIMIT));
    const candidates = Math.min(MAX_CANDIDATES, limit * 3);
    const query = input.query.trim();
    const embedding = input.queryEmbedding;
    const useVector = embedding !== undefined && embedding.length > 0 && (await this.supportsVectors());
    const useFts = query !== "";

    if (!useVector && !useFts) {
      const rows = await this.list({
        scope: input.scope,
        limit,
        ...(input.kinds ? { kinds: input.kinds } : {}),
      });
      const nowMs = (input.now ?? new Date()).getTime();
      const wRecency = input.weights?.recency ?? DEFAULT_WEIGHTS.recency;
      const wImportance = input.weights?.importance ?? DEFAULT_WEIGHTS.importance;
      return rows.map((row) => ({
        ...row,
        score:
          wRecency * recencyDecay(row.recordedAt, nowMs) +
          wImportance * (clampImportance(row.importance) / 10),
        voices: {},
      }));
    }

    const [vectorRows, ftsRows] = await Promise.all([
      useVector && embedding
        ? this.vectorVoice(input, embedding, candidates)
        : Promise.resolve(undefined),
      useFts ? this.ftsVoice(input, query, candidates) : Promise.resolve(undefined),
    ]);

    return fuseHits(
      {
        ...(vectorRows ? { vector: vectorRows } : {}),
        ...(ftsRows ? { fts: ftsRows } : {}),
      },
      {
        limit,
        ...(input.weights ? { weights: input.weights } : {}),
        ...(input.now ? { now: input.now } : {}),
      },
    );
  }

  /** Newest-first listing over the same scope predicate (CLI/audit surface). */
  async list(opts: {
    scope: RecallScope;
    limit?: number;
    includeInvalid?: boolean;
    kinds?: MemoryKind[];
  }): Promise<MemoryRow[]> {
    const limit = Math.max(1, Math.floor(opts.limit ?? DEFAULT_LIST_LIMIT));
    const params: unknown[] = [this.tenantId];
    const where = scopePredicate(opts.scope, params, {
      ...(opts.includeInvalid ? { includeInvalid: true } : {}),
    });
    const kinds = kindsClause(opts.kinds, params);
    params.push(limit);
    const rows = await this.db.query<MemoryRowRaw>(
      `select ${COLUMNS} from memories
       where tenant_id = $1 and ${where}${kinds}
       order by recorded_at desc, id desc
       limit $${params.length}`,
      params,
    );
    return rows.map(mapRow);
  }

  /**
   * Current rows in scope recorded strictly after a tuple watermark, OLDEST
   * first — the sleep worker's reflect batch (DESIGN.md §5.3 item 3, slice 6).
   *
   * The opposite direction from {@link MemoryStore.list} on purpose: a
   * consolidation pass consumes the plane forwards, so its watermark advances
   * monotonically and the pass is resumable from the log alone. See
   * {@link MemorySinceInput} for why the cursor is `(recorded_at, id)` and not
   * a timestamp.
   *
   * `(..., id) > ($ts, $id)` is a ROW comparison, which is lexicographic over
   * the tuple — "later, or the same instant with a larger id".
   *
   * BOTH SIDES ARE TRUNCATED TO MILLISECONDS, and that is load-bearing.
   * `recorded_at` is timestamptz, i.e. MICROSECOND precision, but the watermark
   * a caller hands back came out of {@link MemoryRow.recordedAt} — postgres.js
   * parses a timestamp with `new Date(text)`, which DROPS the sub-millisecond
   * digits. Compared against the raw column, the boundary row's own
   * `recorded_at` (…123456) is still strictly greater than the watermark it
   * produced (…123000), so the pass re-reads it — and since `recorded_at`
   * defaults to `now()`, which is the TRANSACTION's start time, every row a
   * batch's last transaction wrote comes back too. The cursor would never
   * advance past such a group. `date_trunc('milliseconds', ...)` on the column
   * puts both sides in the same precision, which makes the cut exact.
   *
   * The ORDER BY uses the SAME expression, deliberately. Ordering by the raw
   * column while cutting on the truncated one lets two rows inside one
   * millisecond disagree about which came first, and a row on the wrong side of
   * that disagreement is skipped FOREVER — losing a row is worse than re-reading
   * one, so the two expressions have to be identical.
   *
   * `id` is `text`, so its half of both the comparison and the ORDER BY is
   * collation-dependent; they use the SAME collation, so the order is total and
   * consistent, but a test must not assume it matches a JS `.sort()` (the CI
   * image is glibc en_US, alpine is C).
   *
   * Scope goes through {@link scopePredicate} like every other read (§5.1);
   * `visibilities` and `excludeSources` can only narrow it further — the latter
   * is what stops the reflect pass reading its own insights back as fresh
   * material (see {@link MemorySinceInput}).
   */
  async since(input: MemorySinceInput): Promise<MemoryRow[]> {
    // Clamped, never trusted: a non-finite limit would bind NaN as the LIMIT
    // and take the whole statement down with a parse error naming no column.
    const limit = Number.isFinite(input.limit) ? Math.max(1, Math.floor(input.limit)) : 1;
    const params: unknown[] = [this.tenantId];
    const where = scopePredicate(input.scope, params);
    const kinds = kindsClause(input.kinds, params);
    const visibilities = visibilitiesClause(input.visibilities, params);
    const excluded = excludeSourcesClause(input.excludeSources, params);
    let after = "";
    if (input.after) {
      params.push(input.after.recordedAt);
      const ts = `$${params.length}::timestamptz`;
      params.push(input.after.id);
      after = ` and (${RECORDED_MS}, id) > (${ts}, $${params.length}::text)`;
    }
    params.push(limit);
    const rows = await this.db.query<MemoryRowRaw>(
      `select ${COLUMNS} from memories
       where tenant_id = $1 and ${where}${kinds}${visibilities}${excluded}${after}
       order by ${RECORDED_MS} asc, id asc
       limit $${params.length}`,
      params,
    );
    return rows.map(mapRow);
  }

  /**
   * Rows whose id starts with `prefix`, newest first — the "short id" lookup a
   * human surface needs (`pinky memory show 3f2a`).
   *
   * Resolution is SQL, not a scan of the last N rows: a 2-year-old memory is
   * exactly the one somebody is most likely to name by prefix, and it is
   * precisely the one a `list(limit: 200)`-and-filter would never see.
   *
   * `prefix` must be lowercase hex/dashes, 4..36 chars, or this throws. That
   * character class is the guard: the value is concatenated into a LIKE
   * pattern, where `%` and `_` are wildcards — `id like '%' || '%'` would
   * return the whole plane and a caller checking for "exactly one match" would
   * silently act on an arbitrary row. Nothing is escaped, because nothing that
   * needs escaping can get through.
   *
   * `limit` defaults to 2 so the caller can distinguish a unique match from an
   * ambiguous prefix in a single round trip.
   */
  async findByIdPrefix(
    prefix: string,
    opts: { scope: RecallScope; includeInvalid?: boolean; limit?: number },
  ): Promise<MemoryRow[]> {
    if (typeof prefix !== "string" || !ID_PREFIX_RE.test(prefix)) {
      throw new Error(
        `memory: id prefix must be 4..36 characters of lowercase hex or "-", got ${JSON.stringify(prefix)}`,
      );
    }
    const limit = Math.max(1, Math.floor(opts.limit ?? DEFAULT_ID_PREFIX_LIMIT));
    const params: unknown[] = [this.tenantId];
    const where = scopePredicate(opts.scope, params, {
      ...(opts.includeInvalid ? { includeInvalid: true } : {}),
    });
    params.push(prefix);
    const like = `id like $${params.length} || '%'`;
    params.push(limit);
    const rows = await this.db.query<MemoryRowRaw>(
      `select ${COLUMNS} from memories
       where tenant_id = $1 and ${where} and ${like}
       order by recorded_at desc, id desc
       limit $${params.length}`,
      params,
    );
    return rows.map(mapRow);
  }

  /**
   * Cosine-nearest candidates. `embedding is not null` is not decoration: rows
   * retained while the embedder was unavailable would otherwise sort as
   * maximally distant noise ahead of nothing.
   */
  private async vectorVoice(
    input: SearchInput,
    embedding: number[],
    candidates: number,
  ): Promise<MemoryRow[]> {
    const params: unknown[] = [this.tenantId];
    const where = scopePredicate(input.scope, params);
    const kinds = kindsClause(input.kinds, params);
    params.push(vectorLiteral(embedding));
    const vec = `$${params.length}::vector`;
    params.push(candidates);
    const rows = await this.db.query<MemoryRowRaw>(
      `select ${COLUMNS} from memories
       where tenant_id = $1 and ${where}${kinds} and embedding is not null
       order by embedding <=> ${vec}
       limit $${params.length}`,
      params,
    );
    return rows.map(mapRow);
  }

  /**
   * Lexical candidates. `websearch_to_tsquery` (not `to_tsquery`) because the
   * query text comes from an LLM or a human and must not be able to raise a
   * syntax error; it also gives quoted phrases and `-negation` for free. The
   * 'english' configuration MUST match schema/0005_memory_fts.sql's generated
   * column, or `@@` matches nothing at all.
   */
  private async ftsVoice(
    input: SearchInput,
    query: string,
    candidates: number,
  ): Promise<MemoryRow[]> {
    const params: unknown[] = [this.tenantId];
    const where = scopePredicate(input.scope, params);
    const kinds = kindsClause(input.kinds, params);
    params.push(query);
    const q = `websearch_to_tsquery('english', $${params.length})`;
    params.push(candidates);
    const rows = await this.db.query<MemoryRowRaw>(
      `select ${COLUMNS} from memories
       where tenant_id = $1 and ${where}${kinds} and tsv @@ ${q}
       order by ts_rank_cd(tsv, ${q}) desc, recorded_at desc
       limit $${params.length}`,
      params,
    );
    return rows.map(mapRow);
  }

  /**
   * An embedding is stored only if there is one, a column to put it in, and it
   * is the width that column accepts (see {@link vectorDimensions} — a mismatch
   * is dropped and warned about, never thrown, so the memory itself survives).
   */
  private async wantsVector(embedding: number[] | undefined): Promise<boolean> {
    if (!embedding || embedding.length === 0) return false;
    if (!(await this.supportsVectors())) return false;
    if (embedding.length !== this.vectorDimensions) {
      this.onWarning(
        `memory: dropping a ${embedding.length}-dimension embedding — ` +
          `memories.embedding is vector(${this.vectorDimensions}). The row is stored WITHOUT ` +
          `a vector (FTS recall only); check memory.embeddingModel.`,
      );
      return false;
    }
    return true;
  }

  /**
   * The single INSERT both retain() and update() use. Static and Db-agnostic so
   * update() can run it on its transaction handle.
   */
  private static async insertRow(
    db: Db,
    tenantId: string,
    input: RetainInput,
    withVector: boolean,
  ): Promise<MemoryRow> {
    const kind = assertKind(input.kind);
    const visibility = assertVisibility(input.visibility);
    const text = input.text?.trim();
    if (!text) throw new Error("memory: text must be a non-empty string");
    const importance = normalizeImportance(input.importance);
    if (!input.agentId || input.agentId.trim() === "") {
      throw new Error("memory: agentId must be a non-empty string");
    }
    // A 'channel'/'user' row without its subject id is invisible to every
    // scope predicate — it would be written and never read again.
    if (visibility === "channel" && !input.channelId) {
      throw new Error("memory: visibility 'channel' requires channelId");
    }
    if (visibility === "user" && !input.userId) {
      throw new Error("memory: visibility 'user' requires userId");
    }

    const id = crypto.randomUUID();
    const cols = [
      "id",
      "tenant_id",
      "agent_id",
      "visibility",
      "user_id",
      "channel_id",
      "kind",
      "text",
      "importance",
      "meta",
    ];
    // `meta` PLAIN, never JSON.stringify (pg.ts JSONB CONTRACT).
    const params: unknown[] = [
      id,
      tenantId,
      input.agentId,
      visibility,
      input.userId ?? null,
      input.channelId ?? null,
      kind,
      text,
      importance,
      jsonbParam(input.meta ?? {}),
    ];
    const values = params.map((_, i) => `$${i + 1}`);
    if (withVector && input.embedding) {
      cols.push("embedding");
      params.push(vectorLiteral(input.embedding));
      values.push(`$${params.length}::vector`);
      cols.push("embedding_model");
      params.push(input.embeddingModel ?? null);
      values.push(`$${params.length}`);
    }

    const row = await db.queryOne<MemoryRowRaw>(
      `insert into memories (${cols.join(", ")}) values (${values.join(", ")})
       returning ${COLUMNS}`,
      params,
    );
    if (!row) throw new Error("memory retain: insert returned no row");
    return mapRow(row);
  }
}
