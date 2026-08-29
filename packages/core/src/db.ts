/**
 * Minimal Postgres surface used across packages. Implemented with postgres.js
 * in packages/core/src/pg.ts; fakes in tests.
 */
export interface Db {
  query<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T[]>;
  queryOne<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T | null>;
  /** Run fn inside a transaction; pass a tx-scoped Db to fn. */
  tx<T>(fn: (tx: Db) => Promise<T>): Promise<T>;
  close(): Promise<void>;
}
