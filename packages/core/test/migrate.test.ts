import { describe, expect, it } from "bun:test";
import { migrate } from "../src/migrate";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
}

class FakeDb implements Db {
  calls: Call[] = [];
  private script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>;

  constructor(script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>) {
    this.script = script;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params });
    for (const s of this.script) {
      if (s.pattern.test(sql)) return s.respond(params) as T[];
    }
    throw new Error(`FakeDb: no script for SQL: ${sql}`);
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

const tmpDir = () => `/tmp/pinky-migrate-test-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const makeTmp = async (files: Record<string, string>) => {
  const dir = tmpDir();
  await Bun.$`mkdir -p ${dir}`;
  for (const [name, content] of Object.entries(files)) {
    await Bun.write(`${dir}/${name}`, content);
  }
  return dir;
};
const rmTmp = async (dir: string) => Bun.$`rm -rf ${dir}`;

const ONE_TABLE = "create table if not exists a (id int);";
const OTHER_TABLE = "create table if not exists b (id int);";
const RERUN_SQL = "do $$ begin perform 1; end $$;";

/** postgres.js surfaces SQLSTATE on `.code`; the runner keys off that. */
const pgError = (code: string, message: string) =>
  Object.assign(new Error(message), { code });

describe("migrate", () => {
  it("applies files in lexicographic order and records versions", async () => {
    const applied: string[] = [];
    const script = [
      { pattern: /select version from schema_migrations/, respond: () => [] },
      {
        pattern: /create table/i,
        respond: () => {
          // calls.back() now has the raw sql string.
          return [];
        },
      },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE, "0002_b.sql": OTHER_TABLE, "README.txt": "not sql" });
    try {
      await migrate(db, dir);
    } finally {
      await rmTmp(dir);
    }
    expect(db.calls[1]!.sql.slice(0, 30)).toBe(ONE_TABLE.slice(0, 30));
    expect(db.calls[3]!.sql.slice(0, 30)).toBe(OTHER_TABLE.slice(0, 30));
    const recorded = db.calls.filter((c) => /insert into schema_migrations/i.test(c.sql)).length;
    expect(recorded).toBe(2);
  });

  it("is idempotent: already-applied versions are skipped", async () => {
    const script = [
      { pattern: /select version from schema_migrations/, respond: () => [{ version: 1 }] },
      { pattern: /create table/i, respond: () => [{ marker: 1 }] },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE });
    try {
      await migrate(db, dir);
    } finally {
      await rmTmp(dir);
    }
    expect(db.calls.length).toBe(1);
    expect(db.calls[0]!.sql).toMatch(/select version/);
  });

  it("tolerates missing schema_migrations table on first run (42P01 only)", async () => {
    const script = [
      {
        pattern: /select version from schema_migrations/,
        respond: () => {
          throw pgError("42P01", 'relation "schema_migrations" does not exist');
        },
      },
      { pattern: /create table/i, respond: () => [] },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE });
    try {
      await migrate(db, dir);
    } finally {
      await rmTmp(dir);
    }
    const recorded = db.calls.filter((c) => /insert into schema_migrations/i.test(c.sql)).length;
    expect(recorded).toBe(1);
  });

  it("rethrows anything that is not 42P01 instead of migrating blind", async () => {
    // A refused connection used to be swallowed, and every migration was then
    // re-applied against a database we could not read.
    const script = [
      {
        pattern: /select version from schema_migrations/,
        respond: () => {
          throw pgError("ECONNREFUSED", "connect ECONNREFUSED 127.0.0.1:5544");
        },
      },
      { pattern: /create table/i, respond: () => [] },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE });
    try {
      await expect(migrate(db, dir)).rejects.toThrow(/cannot read schema_migrations/);
    } finally {
      await rmTmp(dir);
    }
    // Nothing was applied.
    expect(db.calls).toHaveLength(1);
  });

  it("rethrows a permission error (42501) with the SQLSTATE in the message", async () => {
    const script = [
      {
        pattern: /select version from schema_migrations/,
        respond: () => {
          throw pgError("42501", "permission denied for table schema_migrations");
        },
      },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE });
    try {
      await expect(migrate(db, dir)).rejects.toThrow(/SQLSTATE 42501/);
    } finally {
      await rmTmp(dir);
    }
  });

  it("rethrows a non-SQLSTATE failure too", async () => {
    const script = [
      {
        pattern: /select version from schema_migrations/,
        respond: () => {
          throw new Error("socket hang up");
        },
      },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE });
    try {
      await expect(migrate(db, dir)).rejects.toThrow(/socket hang up/);
    } finally {
      await rmTmp(dir);
    }
  });
});

describe("migrate: .rerun.sql convention", () => {
  it("runs a rerun file every time and never records a version for it", async () => {
    // Versions 1 and 2 are already recorded; the rerun file must still run.
    const script = [
      { pattern: /select version from schema_migrations/, respond: () => [{ version: 1 }, { version: 2 }] },
      { pattern: /do \$\$/, respond: () => [] },
      { pattern: /create table/i, respond: () => [] },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({ "0001_a.sql": ONE_TABLE, "0002_embeddings.rerun.sql": RERUN_SQL });
    try {
      await migrate(db, dir);
      await migrate(db, dir); // second pass: still runs
    } finally {
      await rmTmp(dir);
    }
    const reruns = db.calls.filter((c) => c.sql === RERUN_SQL);
    expect(reruns).toHaveLength(2);
    const recorded = db.calls.filter((c) => /insert into schema_migrations/i.test(c.sql));
    expect(recorded).toHaveLength(0);
  });

  it("orders rerun files by their numeric prefix alongside one-shots", async () => {
    const script = [
      { pattern: /select version from schema_migrations/, respond: () => [] },
      { pattern: /do \$\$/, respond: () => [] },
      { pattern: /create table/i, respond: () => [] },
      { pattern: /insert into schema_migrations/, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const dir = await makeTmp({
      "0001_a.sql": ONE_TABLE,
      "0002_embeddings.rerun.sql": RERUN_SQL,
      "0003_b.sql": OTHER_TABLE,
    });
    try {
      await migrate(db, dir);
    } finally {
      await rmTmp(dir);
    }
    const sql = db.calls.map((c) => c.sql);
    expect(sql[0]).toMatch(/select version/);
    expect(sql[1]).toBe(ONE_TABLE);
    expect(sql[2]).toMatch(/insert into schema_migrations/);
    expect(sql[3]).toBe(RERUN_SQL);
    expect(sql[4]).toBe(OTHER_TABLE);
    expect(sql[5]).toMatch(/insert into schema_migrations/);
    // A rerun file's number is not a version: 1 and 3 recorded, never 2.
    const versions = db.calls
      .filter((c) => /insert into schema_migrations/i.test(c.sql))
      .map((c) => c.params?.[0]);
    expect(versions).toEqual([1, 3]);
  });

  it("the shipped schema dir uses the convention for embeddings", async () => {
    const script = [
      { pattern: /select version from schema_migrations/, respond: () => [{ version: 1 }] },
      { pattern: /./, respond: () => [] },
    ];
    const db = new FakeDb(script);
    const schemaDir = new URL("../schema", import.meta.url).pathname;
    await migrate(db, schemaDir);
    // 0002_embeddings.rerun.sql re-attempts the pgvector column on every run.
    expect(db.calls.some((c) => /add column if not exists embedding/.test(c.sql))).toBe(true);
    // ...and is never recorded as a version.
    const versions = db.calls
      .filter((c) => /insert into schema_migrations/i.test(c.sql))
      .map((c) => c.params?.[0]);
    expect(versions).not.toContain(2);
  });
});
