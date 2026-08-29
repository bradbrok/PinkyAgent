/**
 * postgres.js-backed implementation of the Db interface (packages/core/src/db.ts).
 *
 * Params always flow positionally ($1, $2, ...) — vector param values
 * (number[] for pgvector) are passed through verbatim so callers decide the
 * wire format (e.g. the string form '[1,2,3]').
 *
 * JSONB CONTRACT — jsonb params take PLAIN VALUES; never pre-stringify.
 * Pass the object/array/string/number itself. postgres.js Describes the
 * statement, learns the parameter is jsonb, and applies its own serializer
 * (JSON.stringify) exactly once. A caller that hands in `JSON.stringify(x)`
 * gets that text encoded a SECOND time, so the row lands as a jsonb *string*:
 * `jsonb_typeof(data)` is 'string', `data->>'type'` is NULL, and the column
 * cannot be queried or indexed as JSON at all. (That is the bug
 * schema/0004_jsonb_repair.rerun.sql exists to undo; an explicit `$1::jsonb`
 * cast does NOT help — the value is already a JSON string by then.)
 *
 * A few JS types do not survive the plain-value rule: postgres.js pre-declares
 * a wire type for booleans, Dates, bigints and Buffers, and Postgres refuses
 * to coerce those to jsonb ("column is of type jsonb but expression is of type
 * boolean"). Route jsonb params through {@link jsonbParam}, which hands the
 * JSON-meaningful ones over as a toJSON carrier so they take the same
 * single-encoding path as everything else.
 *
 * Pooling note (DESIGN.md §5.1): postgres.js pins one connection for the whole
 * of a `begin` block, so a transaction-local GUC set at its head — which is
 * how withTenant() (./tenant.ts) applies the RLS tenant scope — holds for
 * every statement in that block and is discarded at COMMIT. Do not put a
 * statement-level pooler (pgbouncer in statement mode) under this.
 */
import postgres from "postgres";
import type { Db } from "./db";

/**
 * Minimal structural shape of a postgres.js `Sql`/`TransactionSql`. Both
 * expose `unsafe`; the root handle additionally exposes `begin`/`end`. Using a
 * structural type keeps us decoupled from postgres.js generics and lets
 * tests' FakeDb implement the same surface.
 */
interface Client {
  unsafe(text: string, params?: unknown[]): Promise<unknown>;
  begin?: <T>(fn: (tx: Client) => Promise<T>) => Promise<T>;
  end?: (opts?: { timeout?: number }) => Promise<void>;
}

function scopedDb(client: Client, isRoot: boolean): Db {
  const db: Db = {
    async query<T = Record<string, unknown>>(text: string, params?: unknown[]): Promise<T[]> {
      // No params → simple protocol (allows multi-statement migrations).
      const rows = (params === undefined || params.length === 0)
        ? await client.unsafe(text)
        : await client.unsafe(text, params);
      return rows as T[];
    },
    async queryOne<T = Record<string, unknown>>(text: string, params?: unknown[]): Promise<T | null> {
      const rows = await db.query<T>(text, params);
      return rows[0] ?? null;
    },
    async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
      if (client.begin) {
        return await client.begin((txClient: Client) => fn(scopedDb(txClient, false)));
      }
      // Nested tx request on a tx-scoped client without begin: reuse in
      // place. No nested BEGIN is issued, so a GUC set by the outer tx stays
      // in force (withTenant relies on this).
      return await fn(scopedDb(client, false));
    },
    async close(): Promise<void> {
      if (isRoot && client.end) await client.end({ timeout: 5 });
    },
  };
  return db;
}

/**
 * Normalize one value for a **jsonb** bind parameter (see the JSONB CONTRACT
 * above). Objects, arrays, strings and numbers are returned untouched — they
 * already serialize correctly. Booleans, Dates and bigints are wrapped in a
 * `toJSON` carrier: postgres.js pre-declares a wire type for those
 * (bool/timestamptz/int8) and the server will not cast them to jsonb, while an
 * ordinary object is left "unspecified" and serialized with the column's own
 * jsonb serializer.
 *
 * A Buffer is deliberately NOT wrapped. It has no meaningful JSON form (it
 * would silently store `{"type":"Buffer","data":[...]}`), so binding one to a
 * jsonb column should fail loudly at the server, which it does.
 *
 * Deliberately NOT applied inside query(): pg.ts cannot tell a jsonb param
 * from a `where flag = $1` bool param, and wrapping the latter would corrupt
 * it. jsonb write sites opt in explicitly (settings.ts, event-store.ts).
 */
export function jsonbParam(value: unknown): unknown {
  if (typeof value === "boolean" || typeof value === "bigint" || value instanceof Date) {
    return { toJSON: () => value };
  }
  return value;
}

/** Create a pooled Db. Call close() to end the pool. */
export function createDb(url: string, opts?: { max?: number }): Db {
  const sql = postgres(url, {
    max: opts?.max ?? 10,
    // Server NOTICEs (migration DO-blocks use RAISE NOTICE) print as one line
    // instead of postgres.js's default full-object dump.
    onnotice: (n) => console.log(`notice: ${n.message}`),
  }) as unknown as Client;
  return scopedDb(sql, true);
}
