import { describe, expect, it } from "bun:test";
import { withTenant } from "../src/tenant";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
  /** transaction depth at the time of the call; 0 = outside any tx. */
  depth: number;
}

/**
 * Records every statement and the transaction nesting depth it ran at, so a
 * test can assert "set_config was the first statement inside the tx".
 * tx() mirrors pg.ts: a nested tx reuses the same scope, it does not re-BEGIN.
 */
class FakeDb implements Db {
  calls: Call[] = [];
  begins = 0;
  closed = 0;
  private depth = 0;

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params, depth: this.depth });
    return [] as T[];
  }
  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }
  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    if (this.depth === 0) this.begins++;
    this.depth++;
    try {
      return await fn(this);
    } finally {
      this.depth--;
    }
  }
  async close(): Promise<void> {
    this.closed++;
  }
}

const SET_CONFIG = /set_config\('pinky\.tenant_id', \$1, true\)/;

describe("withTenant", () => {
  it("runs a plain query inside a tx, set_config first", async () => {
    const fake = new FakeDb();
    await withTenant(fake, "acme").query("select * from memories");

    expect(fake.begins).toBe(1);
    expect(fake.calls).toHaveLength(2);
    expect(fake.calls[0]!.sql).toMatch(SET_CONFIG);
    expect(fake.calls[0]!.params).toEqual(["acme"]);
    expect(fake.calls[0]!.depth).toBe(1);
    expect(fake.calls[1]!.sql).toBe("select * from memories");
    expect(fake.calls[1]!.depth).toBe(1);
  });

  it("scopes queryOne the same way", async () => {
    const fake = new FakeDb();
    const row = await withTenant(fake, "acme").queryOne("select 1", [7]);

    expect(row).toBeNull();
    expect(fake.calls.map((c) => c.sql)).toHaveLength(2);
    expect(fake.calls[0]!.sql).toMatch(SET_CONFIG);
    expect(fake.calls[1]!.params).toEqual([7]);
  });

  it("sets the GUC before the tx callback body runs", async () => {
    const fake = new FakeDb();
    const seen: string[] = [];
    const result = await withTenant(fake, "acme").tx(async (tx) => {
      seen.push(fake.calls.map((c) => c.sql).join("|"));
      await tx.query("insert into memories (id) values ($1)", ["m1"]);
      return "done";
    });

    expect(result).toBe("done");
    expect(seen[0]).toMatch(SET_CONFIG); // already issued when fn was entered
    expect(fake.calls[0]!.sql).toMatch(SET_CONFIG);
    expect(fake.calls[1]!.sql).toBe("insert into memories (id) values ($1)");
    expect(fake.begins).toBe(1);
  });

  it("does not re-open or re-set inside a nested tx", async () => {
    const fake = new FakeDb();
    await withTenant(fake, "acme").tx(async (tx) => {
      await tx.tx(async (inner) => {
        await inner.query("select 1");
      });
    });

    expect(fake.begins).toBe(1);
    const setConfigs = fake.calls.filter((c) => SET_CONFIG.test(c.sql));
    expect(setConfigs).toHaveLength(1);
    expect(fake.calls.map((c) => c.sql)[1]).toBe("select 1");
  });

  it("passes the tenant id as a bound parameter, never interpolated", async () => {
    const fake = new FakeDb();
    await withTenant(fake, "'; drop table memories; --").query("select 1");

    expect(fake.calls[0]!.sql).not.toContain("drop table");
    expect(fake.calls[0]!.params).toEqual(["'; drop table memories; --"]);
  });

  it("close() delegates to the wrapped Db", async () => {
    const fake = new FakeDb();
    await withTenant(fake, "acme").close();
    expect(fake.closed).toBe(1);
  });

  it("rejects an empty tenant id (would silently match nothing)", () => {
    const fake = new FakeDb();
    expect(() => withTenant(fake, "")).toThrow(/non-empty/);
    expect(() => withTenant(fake, "   ")).toThrow(/non-empty/);
  });

  it("is still a Db (no interface change)", () => {
    const scoped: Db = withTenant(new FakeDb(), "acme");
    expect(typeof scoped.query).toBe("function");
    expect(typeof scoped.queryOne).toBe("function");
    expect(typeof scoped.tx).toBe("function");
    expect(typeof scoped.close).toBe("function");
  });
});
