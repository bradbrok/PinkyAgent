/**
 * Tenant scoping for row-level security (DESIGN.md §5.1).
 *
 * Postgres RLS on `memories` keys off a session variable, `pinky.tenant_id`
 * (see packages/core/schema/0001_init.sql and 0003_rls.sql). Something has to
 * set it, and it has to be set on the *same* connection that runs the query --
 * which is why DESIGN.md says "transaction pooling only, never statement
 * pooling with RLS GUCs".
 *
 * `withTenant(db, tenantId)` returns a Db that guarantees exactly that. It is
 * a wrapper, not a new interface: anything already typed `Db` keeps working.
 *
 *   const scoped = withTenant(db, settings.tenantId);
 *   await scoped.query("select * from memories");   // sees only that tenant
 *
 * Mechanics:
 *   - tx()      opens the underlying transaction and issues
 *               `select set_config('pinky.tenant_id', $1, true)` as its first
 *               statement, before the callback runs.
 *   - query()   and queryOne() are wrapped in such a transaction of their own,
 *   /queryOne() so a one-off statement is scoped too.
 *   - close()   delegates to the wrapped Db.
 *
 * The `true` third argument makes the setting *transaction-local*: it is
 * discarded at COMMIT/ROLLBACK, so a pooled connection handed to the next
 * caller carries no residue. That is the whole reason a bare `query()` has to
 * open a transaction — outside one there is nothing to scope the GUC to.
 *
 * One Postgres subtlety the policy depends on: after the first transaction
 * has set it, the discarded GUC reverts to the EMPTY STRING rather than to
 * NULL, so an un-scoped query on a recycled connection reads ''. The policy
 * therefore wraps the read in nullif(..., '') (0003_rls.sql) and treats both
 * spellings of "unset" as "no tenant" — zero rows, fail closed.
 *
 * Nesting is safe. packages/core/src/pg.ts reuses a tx-scoped client in place
 * rather than issuing a nested BEGIN, and the wrapper below does not re-issue
 * set_config inside an already-scoped transaction.
 *
 * Caveat worth knowing: wrapping a Db that is *already* transaction-scoped
 * sets the GUC for the remainder of that outer transaction, not just for the
 * wrapped calls. Wrap the root Db.
 */
import type { Db } from "./db";

const SET_TENANT = `select set_config('pinky.tenant_id', $1, true)`;

/**
 * Wrap `db` so every statement runs with `pinky.tenant_id` set to `tenantId`.
 *
 * This is a scoping tool, not an authorization check: enforcement lives in the
 * database. It only bites when the connection is a NOSUPERUSER/NOBYPASSRLS
 * role such as `pinky_app` — as `postgres`, RLS is bypassed and this wrapper
 * is a no-op with extra round-trips.
 */
export function withTenant(db: Db, tenantId: string): Db {
  if (!tenantId || tenantId.trim() === "") {
    // An empty GUC matches no row, so this would fail closed and look like
    // data loss. Refuse loudly instead.
    throw new Error("withTenant: tenantId must be a non-empty string");
  }

  const root: Db = {
    async query<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T[]> {
      return db.tx(async (tx) => {
        await tx.query(SET_TENANT, [tenantId]);
        return tx.query<T>(sql, params);
      });
    },
    async queryOne<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T | null> {
      const rows = await root.query<T>(sql, params);
      return rows[0] ?? null;
    },
    async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
      return db.tx(async (inner) => {
        await inner.query(SET_TENANT, [tenantId]);
        return fn(scopedTx(inner));
      });
    },
    async close(): Promise<void> {
      await db.close();
    },
  };
  return root;
}

/**
 * Inside a transaction the GUC is already set, so statements pass straight
 * through and a nested tx() does not re-issue set_config.
 */
function scopedTx(tx: Db): Db {
  const scoped: Db = {
    async query<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T[]> {
      return tx.query<T>(sql, params);
    },
    async queryOne<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T | null> {
      return tx.queryOne<T>(sql, params);
    },
    async tx<T>(fn: (inner: Db) => Promise<T>): Promise<T> {
      return tx.tx((inner) => fn(scopedTx(inner)));
    },
    async close(): Promise<void> {
      await tx.close();
    },
  };
  return scoped;
}
