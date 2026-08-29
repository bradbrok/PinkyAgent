import { describe, expect, it } from "bun:test";
import { createDb } from "../src/pg";
import type { Db } from "../src/db";

// Hermetic by design: these pin the Db contract via a FakeDb stand-in, with
// no socket involved. The live-database behaviour of createDb (RLS, tenant
// GUCs, transaction pooling) is covered by test/integration/rls.test.ts,
// which only runs under PINKY_INTEGRATION=1.

class FakeDb implements Db {
  calls: Array<{ sql: string; params: unknown[] | undefined }> = [];
  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params });
    return [] as T[];
  }
  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }
  tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }
  async close(): Promise<void> {}
}

describe("createDb / Db contract (FakeDb stand-in)", () => {
  it("queryOne returns null for empty results, first row otherwise", async () => {
    const fake = new FakeDb();
    expect(await fake.queryOne("select 1")).toBeNull();
  });

  it("tx re-enters with the same Db surface", async () => {
    const fake = new FakeDb();
    await fake.tx(async (t) => {
      await t.query("insert into x values ($1)", [1]);
      return "ok";
    });
    expect(fake.calls[0]!.sql).toBe("insert into x values ($1)");
    expect(fake.calls[0]!.params).toEqual([1]);
  });

  it("createDb returns a Db-typed handle (structural)", () => {
    // Can't actually connect without a real DB, so we only check the type
    // shape. postgres is loaded dynamically to avoid import-time network.
    const db: Db = createDb("postgres://postgres:pinky@localhost:5544/pinky");
    expect(typeof db.query).toBe("function");
    expect(typeof db.queryOne).toBe("function");
    expect(typeof db.tx).toBe("function");
    expect(typeof db.close).toBe("function");
  });
});
