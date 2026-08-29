/**
 * The deferred-tool catalog (slice 9 — MCP + deferred tools).
 *
 * Tool schemas render at prefix position 0 of every request, so a growing or
 * changing tool list both bloats every turn and invalidates every provider
 * cache tier. The rule of this slice is that LOADING A TOOL NEVER REWRITES THE
 * HEADER: the header holds the always-on tools plus three fixed meta-tools
 * (`tool_search`, `tool_describe`, `tool_call`), and everything else lives in
 * this table and reaches the model as an ordinary tool result. This store is
 * the index those meta-tools read.
 *
 * Four rules, mirroring the memory plane (packages/core/src/memory.ts):
 *
 * 1. THE TENANT IS STATED, NOT ASSUMED. The Db handed to the constructor
 *    SHOULD be withTenant()-wrapped, and the store additionally writes
 *    `tenant_id` on every row and repeats it in every WHERE. Belt and braces
 *    matters more here than in the memory plane, because `tool_catalog` has NO
 *    row-level-security policy in this slice (schema/0006_tool_catalog.sql
 *    explains why, and flags the follow-up): the application predicate is
 *    currently the only fence.
 *
 * 2. INVALIDATE, NEVER DELETE. Nothing here issues a DELETE. A tool that
 *    disappears from its server is stamped with `removed_at`; a tool that
 *    comes back has the stamp cleared. Current truth is `removed_at is null`,
 *    and every read says so. A withdrawn row stays queryable, which is what
 *    lets a name from an old continuity document or a replayed event log still
 *    be explained.
 *
 * 3. A GENERATION IS ONE TRANSACTION. {@link ToolCatalogStore.replaceServer}
 *    upserts the new set and stamps the vanished rows inside a single
 *    transaction, so a reader never sees a server with half its tools. The
 *    generation is also what `config_hash` records: McpManager trusts the
 *    catalog on start while the hash still matches the configured server, so
 *    request 1 can see a server's tools before the server has answered.
 *
 * 4. SEARCH IS ONE VOICE, AND THE ARGUMENTS ARE PART OF IT. The vocabulary a
 *    model searches with ("repository", "issue number", "sha") is usually in
 *    the argument schema rather than in the one-line tool description, so
 *    {@link argText} flattens the JSON Schema's property names and
 *    descriptions into `arg_text` and the generated `tsv` covers
 *    name + description + arg_text. `websearch_to_tsquery` (never
 *    `to_tsquery`) because the query text comes from an LLM and must not be
 *    able to raise a syntax error. The 'english' configuration MUST match
 *    schema/0006_tool_catalog.sql's generated column or `@@` matches nothing.
 *
 * ORDERING IS `collate "C"` EVERYWHERE. Tool names are ASCII
 * (`[A-Za-z0-9_-]` plus the `mcp__` separators), and the servers this project
 * runs on disagree about text ordering — the pgvector image is Debian/glibc
 * en_US, which ignores `-` and `_` at the first level, while postgres:16-alpine
 * is C. Sorting under the C collation makes the SQL order a code-unit compare,
 * i.e. exactly what a JS `a < b` sort produces, on both images. (Not a
 * hypothetical: a real CI flake, fixed in da33d0e.)
 *
 * `parameters` obeys pg.ts's JSONB CONTRACT: plain values, never
 * JSON.stringify — see schema/0004_jsonb_repair.rerun.sql for what
 * pre-encoding costs.
 *
 * `CatalogHit` and `CatalogEntry` below are structurally identical to the ones
 * in packages/runtime/src/types.ts and are assignable to them in both
 * directions, so this class satisfies that file's `ToolCatalogView` without
 * implementing it by name. They are re-declared here rather than imported
 * because the dependency direction is `core <- runtime` — core may not import
 * from runtime — and duplicating four fields is cheaper than inverting that.
 * Keep the two in step. `CatalogRecord` is deliberately core-only: it carries
 * the bookkeeping columns (rawName, configHash, timestamps) that writers need
 * and the model must never see.
 */
import type { Db } from "./db";
import { jsonbParam } from "./pg";

/** Where a catalog row came from. Built-ins are ours; `mcp` rows are published
 *  by a configured server and namespaced `mcp__<server>__<raw>`. */
export type CatalogSource = "builtin" | "mcp";

/** A search result: enough to decide whether to describe the tool. */
export interface CatalogHit {
  /** Final, namespaced name — the only handle a caller has. */
  name: string;
  /** Capped to {@link CATALOG_DESCRIPTION_CAP} characters; full text via describe(). */
  description: string;
  source: CatalogSource;
  /** The settings key of the publishing MCP server; absent for a built-in. */
  server?: string;
}

/** A described tool: the hit plus the JSON Schema needed to call it. */
export interface CatalogEntry extends CatalogHit {
  /** JSON Schema for the arguments (MCP inputSchema / Tool.parameters). */
  parameters: Record<string, unknown>;
}

/**
 * A whole row, as a WRITER sees it — the entry plus the bookkeeping columns
 * `search`/`describe` deliberately hide from the model.
 *
 * This is the seam the MCP manager's trust path needs: when a server's
 * `config_hash` still matches its configured hash, the catalog is served
 * immediately and the tools it publishes have to be rebuilt from these rows,
 * `rawName` included — a `callTool` RPC must send the server's own spelling,
 * and it cannot be recovered from the namespaced name (sanitizing is lossy and
 * a long name is truncated with a hash suffix). Also the read behind
 * `pinky tools list` / `pinky mcp list`.
 */
export interface CatalogRecord extends CatalogEntry {
  /** The server's own name for the tool; absent for a built-in. */
  rawName?: string;
  /** Hash of the McpServerConfig this generation was published under. */
  configHash?: string;
  updatedAt: string;
  /** Present only on a withdrawn row (and only with `includeRemoved`). */
  removedAt?: string;
}

/** One tool as a writer supplies it. `rawName` is the server's own spelling,
 *  which `callTool` must send back; built-ins leave it unset. */
export interface CatalogToolInput {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  rawName?: string;
}

/** What {@link ToolCatalogStore.serverState} knows about a server's rows. */
export interface ServerState {
  /** The hash recorded by the last replaceServer; null if a writer omitted it. */
  configHash: string | null;
  /** Live (non-removed) rows for that server. */
  count: number;
  /** Newest `updated_at` among them, ISO 8601. */
  updatedAt: string;
}

/** What one generation cost: rows written, rows withdrawn. */
export interface ReplaceServerResult {
  upserted: number;
  removed: number;
}

/** Results per `tool_search` when the caller does not say (matches the
 *  `tools.searchLimit` default in core/src/config.ts). */
export const DEFAULT_CATALOG_SEARCH_LIMIT = 8;

/** Hard ceiling on one search. A deferred tool list exists to keep schemas out
 *  of the context; a caller that asks for 500 hits has defeated the point. */
export const MAX_CATALOG_SEARCH_LIMIT = 50;

/** Description length in a {@link CatalogHit}, ellipsis included. */
export const CATALOG_DESCRIPTION_CAP = 200;

/**
 * Ceiling on the generated `arg_text`. A deeply documented schema can carry
 * kilobytes of prose per tool; past a few thousand characters it stops adding
 * discriminating lexemes and starts costing index size on every row.
 */
export const ARG_TEXT_MAX = 4000;

/**
 * Rows per INSERT. A server with hundreds of tools would otherwise bind
 * thousands of parameters in one statement; chunking keeps each statement
 * ordinary (all chunks still run inside the one transaction).
 */
const UPSERT_CHUNK = 200;

/** Columns every read projects. `tsv` is never selected: it is not useful in
 *  JS, and `parameters` is only worth its bytes in describe()/entries(). */
const HIT_COLUMNS = `name, source, server, description`;
const ENTRY_COLUMNS = `${HIT_COLUMNS}, parameters`;
const RECORD_COLUMNS = `${ENTRY_COLUMNS}, raw_name, config_hash, updated_at, removed_at`;

interface CatalogRowRaw {
  name: string;
  source: string;
  server: string | null;
  description: string | null;
  /** jsonb. `string` only for legacy doubly-encoded rows — see toParameters(). */
  parameters?: Record<string, unknown> | string | null;
  raw_name?: string | null;
  config_hash?: string | null;
  updated_at?: Date | string | null;
  removed_at?: Date | string | null;
}

/**
 * timestamptz -> ISO string. postgres.js hands back a Date; a FakeDb (and a
 * driver with `types.date` off) hands back text. Anything unparseable passes
 * through verbatim rather than becoming "Invalid Date".
 */
function toIso(value: Date | string): string {
  if (value instanceof Date) return value.toISOString();
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString();
}

/**
 * jsonb -> object. The `string` branch is tolerance for legacy doubly-encoded
 * rows (the bug 0004_jsonb_repair.rerun.sql exists to undo), exactly as
 * MemoryStore.mapRow and EventStore.mapRow do for their own jsonb columns. A
 * schema that will not parse degrades to `{}` — an empty schema is a tool that
 * takes no arguments, which is wrong but callable, where a throw would make
 * one bad row poison every search.
 */
function toParameters(value: Record<string, unknown> | string | null | undefined): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
    return {};
  }
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  return {};
}

/**
 * Truncate for a search hit: at most {@link CATALOG_DESCRIPTION_CAP}
 * characters INCLUDING the ellipsis, so a caller budgeting the tool_search
 * result can multiply by the limit and be right. The "…" is the signal that
 * `tool_describe` has more.
 */
export function capDescription(text: string, max: number = CATALOG_DESCRIPTION_CAP): string {
  const cap = Math.max(1, Math.floor(max));
  if (text.length <= cap) return text;
  return `${text.slice(0, cap - 1)}…`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Flatten a JSON Schema into the words a model would search for.
 *
 * Top-level `properties` contribute each property NAME and, when present, its
 * `description` string; one further level is descended — a nested object's own
 * `properties`, and an array's `items` schema — because that is where composite
 * arguments (`filter.owner`, `commits[].sha`) put their vocabulary. Deeper than
 * that the words stop discriminating between tools and start being noise in
 * every row's tsv.
 *
 * Pure, total, and order-preserving: schema property order in, space-joined
 * tokens out, whitespace normalized (the value is concatenated into a
 * generated tsvector, and a tidy column is a readable one). Anything that is
 * not an object — a boolean schema, null, a JSON string — yields "". Never
 * throws: this runs over whatever an arbitrary MCP server published.
 */
export function argText(schema: unknown, opts?: { maxLength?: number }): string {
  const max = Math.max(0, Math.floor(opts?.maxLength ?? ARG_TEXT_MAX));
  const tokens: string[] = [];

  const push = (value: unknown): void => {
    if (typeof value !== "string") return;
    const cleaned = value.replace(/\s+/g, " ").trim();
    if (cleaned) tokens.push(cleaned);
  };

  const walk = (node: unknown, depth: number): void => {
    if (!isRecord(node) || depth > 1) return;
    const properties = node["properties"];
    if (!isRecord(properties)) return;
    for (const [key, value] of Object.entries(properties)) {
      push(key);
      if (isRecord(value)) {
        push(value["description"]);
        // One level down: an object's own properties, and an array's item
        // schema (whose properties are the interesting part).
        walk(value, depth + 1);
        const items = value["items"];
        if (isRecord(items)) {
          push(items["description"]);
          walk(items, depth + 1);
        }
      }
    }
  };

  walk(schema, 0);
  const text = tokens.join(" ");
  return text.length > max ? text.slice(0, max) : text;
}

/**
 * Read/write access to the deferred-tool catalog for one tenant.
 *
 * Structurally satisfies `ToolCatalogView` from packages/runtime/src/types.ts
 * (search + describe), which is how the runtime's meta-tools consume it
 * without core depending on runtime.
 */
export class ToolCatalogStore {
  private db: Db;
  private tenantId: string;

  constructor(db: Db, tenantId: string) {
    if (!tenantId || tenantId.trim() === "") {
      throw new Error("ToolCatalogStore: tenantId must be a non-empty string");
    }
    this.db = db;
    this.tenantId = tenantId;
  }

  /**
   * Publish one server's generation: upsert every tool it now offers and stamp
   * `removed_at` on the rows of that server that are no longer in the set —
   * ONE transaction, so no reader ever sees a server with half its tools.
   *
   * Returning tools have their `removed_at` cleared rather than being
   * re-inserted, so a server that flaps keeps its history instead of churning
   * rows. An EMPTY `tools` list is a legitimate generation ("this server now
   * offers nothing") and withdraws everything; it is not treated as a mistake,
   * because the caller — McpManager — only calls this after a successful
   * `tools/list`, and never during an outage.
   *
   * `configHash` is the trust token: on the next start, a server whose
   * configured hash still matches this one is believed without waiting for it
   * to connect.
   *
   * Duplicate names within one generation collapse to the FIRST occurrence in
   * the caller's list (the name sort is stable, so ties keep input order).
   * Postgres would
   * otherwise reject the whole statement — `ON CONFLICT DO UPDATE cannot
   * affect row a second time` — and lose a server's entire sync to one
   * collision in a sanitized name.
   */
  async replaceServer(
    server: string,
    configHash: string | null,
    tools: CatalogToolInput[],
  ): Promise<ReplaceServerResult> {
    const name = assertServer(server);
    const rows = normalizeInputs(tools);
    const tenantId = this.tenantId;
    const names = rows.map((r) => r.name);
    return await this.db.tx(async (tx) => {
      const upserted = await upsertRows(tx, tenantId, rows, "mcp", name, configHash);
      // `not (name = any($3))` with an empty array is `not false` = true, so an
      // empty generation correctly withdraws every row. Restricted to
      // `server = $2`: built-ins have a null server and are never touched here.
      const removed = await tx.query<{ name: string }>(
        `update tool_catalog set removed_at = now(), updated_at = now()
         where tenant_id = $1 and server = $2 and removed_at is null
           and not (name = any($3::text[]))
         returning name`,
        [tenantId, name, names],
      );
      return { upserted, removed: removed.length };
    });
  }

  /**
   * Register the process's built-in tools. Upsert only — unlike
   * {@link replaceServer} this does NOT withdraw built-ins that are absent
   * from the list, because the built-in set is per SURFACE, not per
   * deployment: `pinky prompt` registers `bash` and `pinky headless` (without
   * `--shell`) does not, so a generational replace would have two concurrent
   * processes stamping and clearing each other's rows forever. Built-ins are
   * retired by a code change, and a stale one is corrected the next time the
   * surface that owns it runs.
   */
  async upsertBuiltins(tools: CatalogToolInput[]): Promise<number> {
    const rows = normalizeInputs(tools);
    if (rows.length === 0) return 0;
    return await upsertRows(this.db, this.tenantId, rows, "builtin", null, null);
  }

  /**
   * Find tools. A non-blank query runs the FTS voice ranked by `ts_rank_cd`;
   * a blank one is a plain name-ordered listing, so `tool_search` with no
   * argument still shows the model what exists rather than nothing.
   *
   * Descriptions come back capped ({@link capDescription}) — this result is
   * appended to a conversation, and an MCP server's description field is
   * unbounded prose.
   */
  async search(query: string, limit?: number): Promise<CatalogHit[]> {
    const n = clampLimit(limit);
    const q = (query ?? "").trim();
    const params: unknown[] = [this.tenantId];
    let sql: string;
    if (q === "") {
      params.push(n);
      sql = `select ${HIT_COLUMNS} from tool_catalog
             where tenant_id = $1 and removed_at is null
             order by name collate "C"
             limit $${params.length}`;
    } else {
      params.push(q);
      // Bound once, referenced twice: the same $n in the filter and the rank.
      const tsq = `websearch_to_tsquery('english', $${params.length})`;
      params.push(n);
      sql = `select ${HIT_COLUMNS} from tool_catalog
             where tenant_id = $1 and removed_at is null and tsv @@ ${tsq}
             order by ts_rank_cd(tsv, ${tsq}) desc, name collate "C"
             limit $${params.length}`;
    }
    const rows = await this.db.query<CatalogRowRaw>(sql, params);
    return rows.map((row) => toHit(row));
  }

  /**
   * The full entry for one name, or null when it is unknown OR withdrawn.
   * A removed tool is deliberately indistinguishable from an absent one to
   * this caller: `tool_describe` exists to produce a schema the model can call
   * with, and a schema for a tool that is gone is a trap.
   */
  async describe(name: string): Promise<CatalogEntry | null> {
    if (typeof name !== "string" || name.trim() === "") return null;
    const row = await this.db.queryOne<CatalogRowRaw>(
      `select ${ENTRY_COLUMNS} from tool_catalog
       where tenant_id = $1 and name = $2 and removed_at is null`,
      [this.tenantId, name],
    );
    if (!row) return null;
    return { ...toHit(row, { cap: false }), parameters: toParameters(row.parameters) };
  }

  /**
   * Whole rows, C-collated — descriptions UNCAPPED and `rawName` included.
   *
   * Not a model-facing read: this is how a caller rebuilds real tools from the
   * catalog (the MCP manager's config-hash trust path, which serves a server's
   * tools before that server has answered) and how a human surface lists what
   * is registered. `search` stays the model's door precisely because it caps
   * descriptions and hides the bookkeeping.
   */
  async entries(opts?: {
    source?: CatalogSource;
    server?: string;
    includeRemoved?: boolean;
  }): Promise<CatalogRecord[]> {
    const params: unknown[] = [this.tenantId];
    let where = `tenant_id = $1`;
    if (opts?.source) {
      params.push(assertSource(opts.source));
      where += ` and source = $${params.length}`;
    }
    if (opts?.server) {
      params.push(opts.server);
      where += ` and server = $${params.length}`;
    }
    if (!opts?.includeRemoved) where += ` and removed_at is null`;
    const rows = await this.db.query<CatalogRowRaw>(
      `select ${RECORD_COLUMNS} from tool_catalog where ${where} order by name collate "C"`,
      params,
    );
    return rows.map((row) => ({
      ...toHit(row, { cap: false }),
      parameters: toParameters(row.parameters),
      ...(row.raw_name ? { rawName: row.raw_name } : {}),
      ...(row.config_hash ? { configHash: row.config_hash } : {}),
      updatedAt: row.updated_at ? toIso(row.updated_at) : new Date(0).toISOString(),
      ...(row.removed_at ? { removedAt: toIso(row.removed_at) } : {}),
    }));
  }

  /**
   * Names only, C-collated — the cheap listing the partition step and
   * `pinky tools list` want. Live rows unless `includeRemoved`.
   */
  async listNames(opts?: {
    source?: CatalogSource;
    server?: string;
    includeRemoved?: boolean;
  }): Promise<string[]> {
    const params: unknown[] = [this.tenantId];
    let where = `tenant_id = $1`;
    if (opts?.source) {
      params.push(assertSource(opts.source));
      where += ` and source = $${params.length}`;
    }
    if (opts?.server) {
      params.push(opts.server);
      where += ` and server = $${params.length}`;
    }
    if (!opts?.includeRemoved) where += ` and removed_at is null`;
    const rows = await this.db.query<{ name: string }>(
      `select name from tool_catalog where ${where} order by name collate "C"`,
      params,
    );
    return rows.map((r) => r.name);
  }

  /**
   * What the catalog currently holds for one server, or null when it holds
   * nothing live. This is the config-hash trust probe: McpManager compares
   * `configHash` with the hash of the configured `McpServerConfig` and, on a
   * match, serves the catalog immediately while connecting in the background.
   * "Nothing live" therefore has to be null rather than a zero-count row — a
   * server with no rows must be waited for, not trusted.
   *
   * `array_agg(... order by ...)` rather than `group by config_hash`: every
   * row of a generation carries the same hash, but a half-migrated table might
   * not, and one deterministic answer beats an arbitrary group.
   */
  async serverState(server: string): Promise<ServerState | null> {
    const name = assertServer(server);
    const row = await this.db.queryOne<{
      config_hash: string | null;
      count: number | string | null;
      updated_at: Date | string | null;
    }>(
      `select (array_agg(config_hash order by updated_at desc, name collate "C"))[1] as config_hash,
              count(*) as count,
              max(updated_at) as updated_at
         from tool_catalog
        where tenant_id = $1 and server = $2 and removed_at is null`,
      [this.tenantId, name],
    );
    // count(*) is bigint and postgres.js hands bigints back as STRINGS (the
    // same trap as events.seq / toSeq) — coerce, never compare raw.
    const count = Number(row?.count ?? 0);
    if (!row || !Number.isFinite(count) || count <= 0) return null;
    return {
      configHash: row.config_hash ?? null,
      count,
      updatedAt: row.updated_at ? toIso(row.updated_at) : new Date(0).toISOString(),
    };
  }
}

/** Row -> hit. `server` is spread conditionally: exactOptionalPropertyTypes
 *  distinguishes "absent" from "explicitly undefined". */
function toHit(row: CatalogRowRaw, opts?: { cap?: boolean }): CatalogHit {
  const description = row.description ?? "";
  return {
    name: row.name,
    description: opts?.cap === false ? description : capDescription(description),
    source: row.source === "mcp" ? "mcp" : "builtin",
    ...(row.server ? { server: row.server } : {}),
  };
}

function assertServer(server: string): string {
  if (typeof server !== "string" || server.trim() === "") {
    throw new Error("tool catalog: server must be a non-empty string");
  }
  return server;
}

function assertSource(source: string): CatalogSource {
  if (source !== "builtin" && source !== "mcp") {
    throw new Error(`tool catalog: unknown source ${JSON.stringify(source)}`);
  }
  return source;
}

interface NormalizedInput {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  argText: string;
  rawName: string | null;
}

/**
 * Validate, dedupe and SORT one write's inputs. Sorting by code unit here (not
 * at the call sites) is what makes a generation's SQL byte-identical run to
 * run, which is what makes the tests deterministic and a diff of two syncs
 * readable.
 */
function normalizeInputs(tools: CatalogToolInput[]): NormalizedInput[] {
  if (!Array.isArray(tools)) throw new Error("tool catalog: tools must be an array");
  const byName = new Map<string, NormalizedInput>();
  const sorted = [...tools].sort((a, b) => compareNames(a?.name, b?.name));
  for (const tool of sorted) {
    const name = tool?.name;
    if (typeof name !== "string" || name.trim() === "") {
      throw new Error("tool catalog: every tool needs a non-empty name");
    }
    if (byName.has(name)) continue; // first occurrence wins; see replaceServer()
    const parameters = isRecord(tool.parameters) ? tool.parameters : {};
    byName.set(name, {
      name,
      description: typeof tool.description === "string" ? tool.description : "",
      parameters,
      argText: argText(parameters),
      rawName: typeof tool.rawName === "string" && tool.rawName !== "" ? tool.rawName : null,
    });
  }
  return Array.from(byName.values());
}

/** Code-unit compare — locale-independent, and the JS twin of `collate "C"`. */
function compareNames(a: string | undefined, b: string | undefined): number {
  const x = a ?? "";
  const y = b ?? "";
  return x < y ? -1 : x > y ? 1 : 0;
}

/**
 * The single upsert both write paths use. Db-agnostic so replaceServer() can
 * run it on its transaction handle.
 *
 * `removed_at = null` in the DO UPDATE is the "it came back" half of rule 2:
 * a returning tool is un-withdrawn in place, keeping its row and its history.
 *
 * `parameters` is bound PLAIN via jsonbParam (pg.ts JSONB CONTRACT) — a
 * JSON.stringify here would store a jsonb *string* and make every
 * `jsonb_typeof(parameters) = 'object'` reader wrong.
 */
async function upsertRows(
  db: Db,
  tenantId: string,
  rows: NormalizedInput[],
  source: CatalogSource,
  server: string | null,
  configHash: string | null,
): Promise<number> {
  let written = 0;
  for (let i = 0; i < rows.length; i += UPSERT_CHUNK) {
    const chunk = rows.slice(i, i + UPSERT_CHUNK);
    const params: unknown[] = [];
    const tuples = chunk.map((row) => {
      const start = params.length;
      params.push(
        tenantId,
        row.name,
        source,
        server,
        row.rawName,
        row.description,
        jsonbParam(row.parameters),
        row.argText,
        configHash,
      );
      const placeholders = [];
      for (let n = start + 1; n <= params.length; n += 1) placeholders.push(`$${n}`);
      return `(${placeholders.join(", ")})`;
    });
    const returned = await db.query<{ name: string }>(
      `insert into tool_catalog
         (tenant_id, name, source, server, raw_name, description, parameters, arg_text, config_hash)
       values ${tuples.join(", ")}
       on conflict (tenant_id, name) do update set
         source = excluded.source,
         server = excluded.server,
         raw_name = excluded.raw_name,
         description = excluded.description,
         parameters = excluded.parameters,
         arg_text = excluded.arg_text,
         config_hash = excluded.config_hash,
         updated_at = now(),
         removed_at = null
       returning name`,
      params,
    );
    written += returned.length;
  }
  return written;
}

/** 1..{@link MAX_CATALOG_SEARCH_LIMIT}, default {@link DEFAULT_CATALOG_SEARCH_LIMIT}. */
function clampLimit(limit: number | undefined): number {
  const n = Math.floor(limit ?? DEFAULT_CATALOG_SEARCH_LIMIT);
  if (!Number.isFinite(n)) return DEFAULT_CATALOG_SEARCH_LIMIT;
  return Math.min(MAX_CATALOG_SEARCH_LIMIT, Math.max(1, n));
}
