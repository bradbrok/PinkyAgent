/**
 * migrate() against a THROWAWAY database (packages/core/schema).
 *
 * The unit test for the runner uses a fake Db, so it proves the file-ordering
 * logic and nothing about the SQL. This creates a real, empty database, runs
 * the real migrations into it, checks what landed, runs them again to prove
 * the one-shots are not re-applied, and drops it. Nothing here touches the
 * developer's `pinky` database.
 *
 * Needs a SUPERUSER connection (CREATE DATABASE, and 0003's CREATE ROLE), so
 * it skips itself with a message when the configured connection is not one.
 * Skipped entirely unless PINKY_INTEGRATION=1.
 */
import { afterAll, describe, expect, it } from "bun:test";
import { loadEnvConfig } from "../../src/config";
import { createDb } from "../../src/pg";
import { migrate } from "../../src/migrate";
import type { Db } from "../../src/db";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const SCHEMA_DIR = new URL("../../schema", import.meta.url).pathname;

/** Migrations are the privileged path (DDL + CREATE ROLE): prefer the admin
 *  url, fall back to the app one — which is what a single-url dev setup has. */
const ADMIN_URL = process.env.DATABASE_ADMIN_URL ?? loadEnvConfig().databaseUrl;

/** Same server, but the always-present maintenance database: you cannot create
 *  or drop a database from a connection that is inside one of them. */
function withDatabase(url: string, name: string): string {
  const u = new URL(url);
  u.pathname = `/${name}`;
  return u.toString();
}

/** Probe for superuser before any suite is defined, so the skip is graceful. */
async function probeSuperuser(): Promise<{ ok: boolean; role: string }> {
  const probe = createDb(withDatabase(ADMIN_URL, "postgres"), { max: 1 });
  try {
    const row = await probe.queryOne<{ role: string; superuser: boolean }>(
      `select current_user as role, rolsuper as superuser
         from pg_roles where rolname = current_user`,
    );
    return { ok: row?.superuser === true, role: row?.role ?? "unknown" };
  } finally {
    await probe.close();
  }
}

// Connection failures are NOT swallowed: with PINKY_INTEGRATION=1 a database
// that cannot be reached is a real failure, not a reason to go quiet.
const probe = ENABLED ? await probeSuperuser() : { ok: false, role: "" };
if (ENABLED && !probe.ok) {
  console.warn(
    `[migrate.test] skipped: ${ADMIN_URL.replace(/:[^:@/]*@/, ":***@")} connects as "${probe.role}", ` +
      `which is not a superuser. CREATE DATABASE and 0003's CREATE ROLE both need one — ` +
      `point DATABASE_ADMIN_URL at the postgres superuser to run this file.`,
  );
}
const suite = ENABLED && probe.ok ? describe : describe.skip;

const DB_NAME = `pinky_it_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;

suite("migrate() on a throwaway database", () => {
  let maintenance: Db | undefined;
  let fresh: Db | undefined;

  afterAll(async () => {
    if (fresh) await fresh.close();
    if (maintenance) {
      // `with (force)` terminates stragglers; without it a leaked pool
      // connection would leave the throwaway database behind forever.
      await maintenance.query(`drop database if exists ${DB_NAME} with (force)`);
      await maintenance.close();
    }
  });

  it("creates the database, migrates it, and records exactly the one-shot versions", async () => {
    maintenance = createDb(withDatabase(ADMIN_URL, "postgres"), { max: 1 });
    // Name is generated from a uuid (hex only), so interpolation is safe —
    // CREATE DATABASE takes no bind parameters.
    await maintenance.query(`create database ${DB_NAME}`);

    fresh = createDb(withDatabase(ADMIN_URL, DB_NAME), { max: 2 });
    await migrate(fresh, SCHEMA_DIR);

    const rows = await fresh.query<{ version: number }>(
      `select version from schema_migrations order by version`,
    );
    // 0001, 0003, 0005 and 0006 are one-shots; 0002 and 0004 are `.rerun.sql`
    // and are deliberately never recorded, so they are re-attempted on every
    // migrate.
    expect(rows.map((r) => Number(r.version))).toEqual([1, 3, 5, 6]);
  });

  it("leaves memories with RLS enabled AND forced", async () => {
    const t = await fresh!.queryOne<{ relrowsecurity: boolean; relforcerowsecurity: boolean }>(
      `select relrowsecurity, relforcerowsecurity from pg_class where relname = 'memories'`,
    );
    expect(t?.relrowsecurity).toBe(true);
    expect(t?.relforcerowsecurity).toBe(true);
  });

  it("creates every table the app needs", async () => {
    const rows = await fresh!.query<{ tablename: string }>(
      `select tablename from pg_tables where schemaname = 'public' order by tablename`,
    );
    expect(rows.map((r) => r.tablename)).toEqual([
      "a2a_messages",
      "events",
      "ingress_dedup",
      "memories",
      "schema_migrations",
      "settings",
      "threads",
      "tool_catalog",
    ]);
  });

  it("a second migrate is a no-op: no version re-applied, no error", async () => {
    const before = await fresh!.query<{ version: number; applied_at: string }>(
      `select version, applied_at from schema_migrations order by version`,
    );

    // The `.rerun.sql` file executes again here; on an image without pgvector
    // it must raise a NOTICE and succeed, not throw.
    await migrate(fresh!, SCHEMA_DIR);

    const after = await fresh!.query<{ version: number; applied_at: string }>(
      `select version, applied_at from schema_migrations order by version`,
    );
    expect(after.map((r) => Number(r.version))).toEqual([1, 3, 5, 6]);
    // Identical timestamps prove the one-shots were skipped, not re-run.
    expect(after.map((r) => String(r.applied_at))).toEqual(before.map((r) => String(r.applied_at)));
  });

  it("the re-runnable embeddings migration matches the server's pgvector support", async () => {
    const count = async (sql: string): Promise<number> =>
      Number((await fresh!.queryOne<{ n: number }>(sql))?.n ?? -1);

    const extension = await count(`select count(*)::int as n from pg_extension where extname = 'vector'`);
    const column = await count(
      `select count(*)::int as n from information_schema.columns
         where table_name = 'memories' and column_name = 'embedding'`,
    );
    const index = await count(
      `select count(*)::int as n from pg_indexes where indexname = 'memories_embedding_idx'`,
    );

    if (extension > 0) {
      // pgvector/pgvector:pg16 — the image CI uses, precisely so this branch is
      // exercised somewhere: 0002_embeddings.rerun.sql adds the column and the
      // HNSW index the moment the extension is available.
      expect(column).toBe(1);
      expect(index).toBe(1);
    } else {
      // postgres:16-alpine — what docker-compose.yml runs locally. The rerun
      // file takes its "pgvector unavailable" branch, raises a NOTICE, and the
      // rest of the schema still migrates.
      expect(column).toBe(0);
      expect(index).toBe(0);
    }
  });

  it("0005 adds the FTS column and index on EVERY image, pgvector or not", async () => {
    // The counterpart to the test above: the vector voice is conditional, the
    // lexical voice is not. 0005_memory_fts.sql needs nothing but stock
    // Postgres, so a database with no `embedding` column must still come out
    // of migrate() with a queryable tsv — otherwise recall on postgres:16-alpine
    // has no relevance signal at all.
    const tsv = await fresh!.queryOne<{ data_type: string; generated: string }>(
      `select data_type, is_generated as generated from information_schema.columns
        where table_name = 'memories' and column_name = 'tsv'`,
    );
    expect(tsv?.data_type).toBe("tsvector");
    expect(tsv?.generated).toBe("ALWAYS");

    const gin = await fresh!.queryOne<{ indexdef: string }>(
      `select indexdef from pg_indexes where indexname = 'memories_tsv_idx'`,
    );
    expect(gin?.indexdef ?? "").toMatch(/using gin/i);

    const model = await fresh!.queryOne<{ data_type: string }>(
      `select data_type from information_schema.columns
        where table_name = 'memories' and column_name = 'embedding_model'`,
    );
    expect(model?.data_type).toBe("text");

    // The generated column is really generated: writing `text` fills `tsv`,
    // and the query side uses the same 'english' configuration it was built
    // with (a mismatch matches nothing, silently).
    await fresh!.query(
      `insert into memories (id, tenant_id, agent_id, visibility, kind, text)
       values ('mig-tsv-probe', 'mig-tsv', 'a', 'tenant', 'semantic',
               'the kangaroo deployed on Thursday')`,
    );
    const hit = await fresh!.queryOne<{ id: string }>(
      `select id from memories
        where id = 'mig-tsv-probe' and tsv @@ websearch_to_tsquery('english', $1)`,
      ["deploy kangaroo"],
    );
    expect(hit?.id).toBe("mig-tsv-probe");
    await fresh!.query(`delete from memories where id = 'mig-tsv-probe'`);
  });

  it("the app role exists and can read the migrated tables", async () => {
    // 0003 creates pinky_app cluster-wide and grants it the public schema.
    const role = await fresh!.queryOne<{ superuser: boolean; bypassrls: boolean }>(
      `select rolsuper as superuser, rolbypassrls as bypassrls
         from pg_roles where rolname = 'pinky_app'`,
    );
    expect(role).not.toBeNull();
    expect(role?.superuser).toBe(false);
    expect(role?.bypassrls).toBe(false);

    const granted = await fresh!.queryOne<{ ok: boolean }>(
      `select has_table_privilege('pinky_app', 'events', 'select') as ok`,
    );
    expect(granted?.ok).toBe(true);
  });
});
