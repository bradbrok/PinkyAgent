/**
 * Live-database proof that row-level security actually isolates tenants
 * (DESIGN.md §5.1: "a missing WHERE cannot leak").
 *
 * Skipped unless PINKY_INTEGRATION=1, so `bun test` stays hermetic:
 *
 *   docker compose up -d postgres
 *   PINKY_INTEGRATION=1 bun test packages/core/test/integration
 *
 * Both connections come from loadEnvConfig(), never a literal port — local dev
 * is 5544, CI is 5432 (.github/workflows/ci.yml):
 *   DATABASE_ADMIN_URL  superuser; runs the migrations (falls back to
 *                       DATABASE_URL, which is the local default).
 *   PINKY_TEST_APP_URL  the unprivileged pinky_app role the app connects as.
 *                       When unset it is derived from DATABASE_URL by swapping
 *                       in pinky_app's credentials, so a stock local checkout
 *                       (whose DATABASE_URL is still the superuser) works with
 *                       no extra setup.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { loadEnvConfig } from "../../src/config";
import { createDb } from "../../src/pg";
import { migrate } from "../../src/migrate";
import { withTenant } from "../../src/tenant";
import type { Db } from "../../src/db";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

/** DATABASE_URL with pinky_app's dev credentials (0003_rls.sql, docker-compose)
 *  substituted for whatever role it names. Host, port and database are kept. */
function asAppRole(url: string): string {
  const u = new URL(url);
  u.username = "pinky_app";
  u.password = "pinky";
  return u.toString();
}

const ENV = loadEnvConfig();
const ADMIN_URL = ENV.databaseAdminUrl;
const APP_URL = process.env.PINKY_TEST_APP_URL ?? asAppRole(ENV.databaseUrl);
const SCHEMA_DIR = new URL("../../schema", import.meta.url).pathname;

const TENANT_A = "rls-test-A";
const TENANT_B = "rls-test-B";
const ID_A = "rls-test-mem-A";
const ID_B_FORGED = "rls-test-mem-forged";

const INSERT = `insert into memories (id, tenant_id, agent_id, visibility, kind, text)
                values ($1, $2, 'pinky', 'tenant', 'semantic', $3)`;

suite("RLS tenant isolation (live postgres)", () => {
  let admin: Db;
  let app: Db;

  beforeAll(async () => {
    admin = createDb(ADMIN_URL, { max: 2 });
    // Creates pinky_app, FORCEs RLS, installs the policies.
    await migrate(admin, SCHEMA_DIR);
    await admin.query(`delete from memories where id like 'rls-test-%'`);
    app = createDb(APP_URL, { max: 4 });
  });

  afterAll(async () => {
    if (app) await app.close();
    if (admin) {
      await admin.query(`delete from memories where id like 'rls-test-%'`);
      await admin.close();
    }
  });

  it("connects as pinky_app, which may not bypass RLS", async () => {
    const who = await app.queryOne<{ role: string; superuser: boolean; bypassrls: boolean }>(
      `select current_user as role, rolsuper as superuser, rolbypassrls as bypassrls
         from pg_roles where rolname = current_user`,
    );
    expect(who?.role).toBe("pinky_app");
    expect(who?.superuser).toBe(false);
    expect(who?.bypassrls).toBe(false);
  });

  it("memories has RLS enabled AND forced", async () => {
    const t = await admin.queryOne<{ relrowsecurity: boolean; relforcerowsecurity: boolean }>(
      `select relrowsecurity, relforcerowsecurity from pg_class where relname = 'memories'`,
    );
    expect(t?.relrowsecurity).toBe(true);
    expect(t?.relforcerowsecurity).toBe(true);
  });

  it("withTenant(A) writes a row and reads it back", async () => {
    const a = withTenant(app, TENANT_A);
    await a.query(INSERT, [ID_A, TENANT_A, "tenant A memory"]);

    const rows = await a.query<{ id: string; tenant_id: string }>(
      `select id, tenant_id from memories where id = $1`,
      [ID_A],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]!.tenant_id).toBe(TENANT_A);
  });

  it("withTenant(B) cannot see tenant A's row — even with no WHERE tenant_id", async () => {
    const b = withTenant(app, TENANT_B);
    const byId = await b.query(`select id from memories where id = $1`, [ID_A]);
    expect(byId).toHaveLength(0);

    // The whole point: the app forgot the tenant predicate and still leaks nothing.
    const all = await b.query<{ id: string }>(`select id from memories`);
    expect(all.map((r) => r.id)).not.toContain(ID_A);
  });

  it("an un-scoped connection (GUC unset) sees zero rows — fail closed", async () => {
    const rows = await app.query(`select id from memories`);
    expect(rows).toHaveLength(0);
  });

  it("the tenant GUC does not survive its transaction", async () => {
    await withTenant(app, TENANT_A).query(`select 1`);
    const after = await app.queryOne<{ v: string | null }>(
      `select current_setting('pinky.tenant_id', true) as v`,
    );
    // Postgres quirk that the policy has to defend against: a custom GUC that
    // has been set once reverts to the EMPTY STRING at COMMIT, not to NULL.
    // Either spelling of "unset" is fine, a surviving tenant id is not.
    expect(after?.v ?? "").toBe("");
  });

  it("a row stamped tenant_id = '' is not visible to an un-scoped connection", async () => {
    // Guards the nullif() in the policy: without it the reverted-to-empty GUC
    // would match such a row on every pooled connection.
    await admin.query(INSERT, ["rls-test-mem-empty", "", "empty tenant"]);
    try {
      await withTenant(app, TENANT_A).query(`select 1`); // ensure GUC = '' afterwards
      const rows = await app.query(`select id from memories`);
      expect(rows).toHaveLength(0);
    } finally {
      await admin.query(`delete from memories where id = 'rls-test-mem-empty'`);
    }
  });

  it("withTenant(B) cannot insert a row stamped tenant A (WITH CHECK)", async () => {
    const b = withTenant(app, TENANT_B);
    let code: string | undefined;
    try {
      await b.query(INSERT, [ID_B_FORGED, TENANT_A, "forged"]);
      throw new Error("insert with a mismatched tenant_id was allowed");
    } catch (err) {
      code = (err as { code?: string }).code;
      expect(String((err as Error).message)).toMatch(/row-level security|was allowed/);
    }
    expect(code).toBe("42501"); // insufficient_privilege

    const stillGone = await admin.query(`select id from memories where id = $1`, [ID_B_FORGED]);
    expect(stillGone).toHaveLength(0);
  });

  it("withTenant(B) may write its own row, and A still cannot see it", async () => {
    const b = withTenant(app, TENANT_B);
    await b.query(INSERT, ["rls-test-mem-B", TENANT_B, "tenant B memory"]);

    const seenByB = await b.query(`select id from memories`);
    expect(seenByB.map((r) => (r as { id: string }).id)).toEqual(["rls-test-mem-B"]);

    const seenByA = await withTenant(app, TENANT_A).query<{ id: string }>(`select id from memories`);
    expect(seenByA.map((r) => r.id)).toEqual([ID_A]);
  });

  it("scoping holds across statements inside one withTenant tx", async () => {
    const rows = await withTenant(app, TENANT_A).tx(async (tx) => {
      await tx.query(`select 1`);
      return tx.query<{ id: string }>(`select id from memories`);
    });
    expect(rows.map((r) => r.id)).toEqual([ID_A]);
  });

  it("DOCUMENTED LIMIT: a superuser bypasses RLS and sees every tenant", async () => {
    // FORCE ROW LEVEL SECURITY subjects the table *owner* to the policy, but
    // nothing subjects a superuser. This is why the app must not connect as
    // `postgres` — see .env.example (DATABASE_URL vs DATABASE_ADMIN_URL).
    const rows = await admin.query<{ tenant_id: string }>(
      `select tenant_id from memories where id like 'rls-test-%' order by tenant_id`,
    );
    expect(rows.map((r) => r.tenant_id)).toEqual([TENANT_A, TENANT_B]);
  });
});
