/**
 * ToolCatalogStore against a live database (slice 9 — MCP + deferred tools).
 *
 * The unit suite (packages/core/test/tool-catalog.test.ts) drives the store
 * through a FakeDb that records SQL text, so it proves the shape of the
 * statements and nothing about whether Postgres accepts them. Everything that
 * only exists on the server is therefore untested until it runs here:
 *
 *   - the generated `tsv` over name || description || arg_text (0006), and
 *     `websearch_to_tsquery('english', $1)` matching it,
 *   - that the FTS parser splits `mcp__gh__create_issue` into mcp/gh/creat/issu,
 *     which is the whole reason a namespaced name is searchable at all,
 *   - `insert ... on conflict (tenant_id, name) do update`, including
 *     `removed_at = null` un-withdrawing a tool that came back,
 *   - `not (name = any($3::text[]))` — and its empty-array edge, where
 *     `not false` correctly withdraws everything,
 *   - `order by name collate "C"` agreeing with a JS code-unit sort on BOTH
 *     images (glibc en_US on pgvector, C on alpine — the da33d0e flake),
 *   - `count(*)` arriving as a bigint STRING,
 *   - the JSONB CONTRACT (a doubly-encoded schema stores a jsonb *string*),
 *   - replaceServer()'s "one transaction" claim, and
 *   - that `pinky_app` can actually reach the table (0003's default privileges
 *     plus 0006's explicit grant).
 *
 * NO PGVECTOR DEPENDENCY: unlike the memory suite this file has one branch and
 * runs identically on postgres:16-alpine (5544) and pgvector/pgvector:pg16
 * (5545, CI).
 *
 * Skipped unless PINKY_INTEGRATION=1. Connections come from loadEnvConfig(),
 * never a literal port. The store runs as the unprivileged `pinky_app` role —
 * `tool_catalog` has NO row-level-security policy in this slice (deliberate:
 * it holds tool schemas, not user data; see the header of
 * schema/0006_tool_catalog.sql and the follow-up it flags), so the app role is
 * here to prove the GRANT, not the isolation. Tenant isolation is the store's
 * own predicate, and this file asserts it as such.
 *
 * Every row is stamped with a run-unique tenant id starting `it-toolcat-`, so
 * cleanup is a scoped DELETE and two concurrent runs cannot see each other.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { loadEnvConfig } from "../../src/config";
import { createDb } from "../../src/pg";
import { migrate } from "../../src/migrate";
import { withTenant } from "../../src/tenant";
import { ToolCatalogStore, argText } from "../../src/tool-catalog";
import type { Db } from "../../src/db";

const ENABLED = process.env.PINKY_INTEGRATION === "1";

/** DATABASE_URL with pinky_app's dev credentials substituted (0003_rls.sql,
 *  docker-compose.yml), exactly as the memory/rls suites do it. */
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

const RUN = crypto.randomUUID().slice(0, 8);
const PREFIX = "it-toolcat-";
const TENANT = `${PREFIX}${RUN}`;
const TENANT_B = `${PREFIX}b-${RUN}`;

/** Each group gets its own server, so one group's generational replace can
 *  never withdraw another's rows and a search can assert on exact sets. */
const server = (group: string): string => `srv-${group}-${RUN}`;
const mcpName = (group: string, raw: string): string => `mcp__${server(group)}__${raw}`;

/** Code-unit sort — the JS twin of `order by name collate "C"`. Comparisons
 *  are always made on this side, never left to the server's collation. */
function sorted(names: string[]): string[] {
  return [...names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

const suite = ENABLED ? describe : describe.skip;

suite("ToolCatalogStore (live postgres)", () => {
  let admin: Db;
  let app: Db;
  let store: ToolCatalogStore;
  let storeB: ToolCatalogStore;

  const purge = async (): Promise<void> => {
    // Superuser: clears every tenant this file may have created, including a
    // crashed previous run. The ONLY delete anywhere near this table.
    await admin.query(`delete from tool_catalog where tenant_id like $1`, [`${PREFIX}%`]);
  };

  /** Raw row read as the superuser — the columns the store never selects. */
  const raw = async (
    name: string,
  ): Promise<{
    removed_at: Date | null;
    config_hash: string | null;
    arg_text: string;
    param_type: string;
    source: string;
    server: string | null;
    raw_name: string | null;
  } | null> =>
    await admin.queryOne(
      `select removed_at, config_hash, arg_text, jsonb_typeof(parameters) as param_type,
              source, server, raw_name
         from tool_catalog where tenant_id = $1 and name = $2`,
      [TENANT, name],
    );

  beforeAll(async () => {
    admin = createDb(ADMIN_URL, { max: 2 });
    // 0006 is what creates the table; on a virgin CI database nothing else has
    // run it yet. Idempotent — every other integration file does the same.
    await migrate(admin, SCHEMA_DIR);
    await purge();
    app = createDb(APP_URL, { max: 8 });
    store = new ToolCatalogStore(withTenant(app, TENANT), TENANT);
    storeB = new ToolCatalogStore(withTenant(app, TENANT_B), TENANT_B);
  });

  afterAll(async () => {
    if (app) await app.close();
    if (admin) {
      await purge();
      await admin.close();
    }
  });

  // --- the migration itself --------------------------------------------------

  it("runs as pinky_app, which 0006 granted access to the catalog", async () => {
    const who = await app.queryOne<{ role: string; superuser: boolean }>(
      `select current_user as role, rolsuper as superuser
         from pg_roles where rolname = current_user`,
    );
    expect(who?.role).toBe("pinky_app");
    expect(who?.superuser).toBe(false);
    // The grant is the point: without it every catalog read is a 42501.
    const reachable = await app.query(`select 1 from tool_catalog limit 1`);
    expect(Array.isArray(reachable)).toBe(true);
  });

  it("has NO row-level security in this slice — deliberate, and the follow-up is flagged", async () => {
    const rls = await admin.queryOne<{ enabled: boolean }>(
      `select relrowsecurity as enabled from pg_class where relname = 'tool_catalog'`,
    );
    // If someone adds a policy, this test fails loudly and the header of
    // schema/0006_tool_catalog.sql (plus the store's rule 1) needs rewriting —
    // that is the intended signal, not an obstacle.
    expect(rls?.enabled).toBe(false);
  });

  it("0006 gave tool_catalog a generated tsv over name, description and arg_text", async () => {
    const col = await admin.queryOne<{
      data_type: string;
      generated: string;
      expression: string;
    }>(
      `select data_type, is_generated as generated, generation_expression as expression
         from information_schema.columns
        where table_name = 'tool_catalog' and column_name = 'tsv'
          and table_schema = current_schema()`,
    );
    expect(col?.data_type).toBe("tsvector");
    expect(col?.generated).toBe("ALWAYS");
    // 'english' here must equal 'english' in the store's websearch_to_tsquery.
    expect(col?.expression).toContain("english");
    expect(col?.expression).toContain("name");
    expect(col?.expression).toContain("description");
    expect(col?.expression).toContain("arg_text");

    const indexes = await admin.query<{ indexname: string; indexdef: string }>(
      `select indexname, indexdef from pg_indexes where tablename = 'tool_catalog'`,
    );
    const tsvIdx = indexes.find((i) => i.indexname === "tool_catalog_tsv_idx");
    expect(tsvIdx?.indexdef).toContain("gin");
    expect(indexes.some((i) => i.indexname === "tool_catalog_server_idx")).toBe(true);
  });

  it("the FTS parser splits a namespaced name, which is what makes it searchable", async () => {
    // mcp__gh__create_issue -> mcp / gh / creat / issu. If this ever stops
    // being true, searching by tool name silently returns nothing.
    const row = await admin.queryOne<{ hit: boolean }>(
      `select to_tsvector('english', $1) @@ websearch_to_tsquery('english', $2) as hit`,
      ["mcp__gh__create_issue", "issue"],
    );
    expect(row?.hit).toBe(true);
  });

  // --- a generation ----------------------------------------------------------

  it("writes a server generation and finds a tool by a word only its ARGUMENT schema uses", async () => {
    const g = "args";
    const schema = {
      type: "object",
      properties: {
        owner: { type: "string", description: "The repository owner login" },
        page: { type: "integer" },
      },
    };
    const result = await store.replaceServer(server(g), "hash-1", [
      { name: mcpName(g, "list_pulls"), rawName: "list_pulls", description: "Lists them", parameters: schema },
      { name: mcpName(g, "quiet"), rawName: "quiet", description: "Nothing to see" },
    ]);
    expect(result).toEqual({ upserted: 2, removed: 0 });

    // "login" appears in NO name and NO description — only in an argument
    // description, i.e. only in arg_text. That is the column earning its keep.
    const hits = await store.search("login", 10);
    expect(hits.map((h) => h.name)).toEqual([mcpName(g, "list_pulls")]);
    expect(hits[0]?.source).toBe("mcp");
    expect(hits[0]?.server).toBe(server(g));

    const stored = await raw(mcpName(g, "list_pulls"));
    expect(stored?.arg_text).toBe(argText(schema));
    expect(stored?.raw_name).toBe("list_pulls");
    expect(stored?.config_hash).toBe("hash-1");
    expect(stored?.removed_at).toBeNull();
  });

  it("DEFECT: parameters must be stored as a jsonb OBJECT, never a jsonb string", async () => {
    const g = "jsonb";
    const schema = { type: "object", properties: { path: { type: "string" } }, required: ["path"] };
    await store.replaceServer(server(g), "h", [
      { name: mcpName(g, "read"), rawName: "read", parameters: schema },
    ]);

    // JSON.stringify at the write site would make jsonb_typeof 'string' and
    // parameters->>'type' NULL — the bug 0004_jsonb_repair.rerun.sql exists to
    // undo, and one the FakeDb suite can only half-see.
    const row = await admin.queryOne<{ param_type: string; type_field: string | null; n: number }>(
      `select jsonb_typeof(parameters) as param_type,
              parameters->>'type' as type_field,
              jsonb_array_length(parameters->'required') as n
         from tool_catalog where tenant_id = $1 and name = $2`,
      [TENANT, mcpName(g, "read")],
    );
    expect(row?.param_type).toBe("object");
    expect(row?.type_field).toBe("object");
    expect(Number(row?.n)).toBe(1);

    // ...and it round-trips through describe() unchanged.
    expect((await store.describe(mcpName(g, "read")))?.parameters).toEqual(schema);
  });

  it("withdraws a vanished tool, keeps the survivor, and un-withdraws one that comes back", async () => {
    const g = "gen";
    const a = mcpName(g, "alpha");
    const b = mcpName(g, "beta");

    await store.replaceServer(server(g), "hash-1", [
      { name: a, rawName: "alpha", description: "zonktastic alpha" },
      { name: b, rawName: "beta", description: "zonktastic beta" },
    ]);
    expect(sorted(await store.listNames({ server: server(g) }))).toEqual(sorted([a, b]));

    // Generation 2 drops `beta`.
    const second = await store.replaceServer(server(g), "hash-2", [
      { name: a, rawName: "alpha", description: "zonktastic alpha" },
    ]);
    expect(second).toEqual({ upserted: 1, removed: 1 });

    // Invalidated, not deleted: the row is still there, stamped.
    const gone = await raw(b);
    expect(gone).not.toBeNull();
    expect(gone?.removed_at).not.toBeNull();
    expect((await raw(a))?.removed_at).toBeNull();

    // And invisible to every read that matters.
    expect(await store.describe(b)).toBeNull();
    expect(await store.listNames({ server: server(g) })).toEqual([a]);
    expect((await store.search("zonktastic", 10)).map((h) => h.name)).toEqual([a]);
    expect(await store.listNames({ server: server(g), includeRemoved: true })).toEqual(
      sorted([a, b]),
    );

    // Generation 3 brings it back: the stamp is cleared in place, no new row.
    await store.replaceServer(server(g), "hash-3", [
      { name: a, rawName: "alpha", description: "zonktastic alpha" },
      { name: b, rawName: "beta", description: "zonktastic beta, restored" },
    ]);
    expect((await raw(b))?.removed_at).toBeNull();
    expect((await store.describe(b))?.description).toBe("zonktastic beta, restored");
    const rows = await admin.query(
      `select 1 from tool_catalog where tenant_id = $1 and name = $2`,
      [TENANT, b],
    );
    expect(rows).toHaveLength(1);
  });

  it("treats an empty generation as 'this server offers nothing' and withdraws every row", async () => {
    const g = "empty";
    const a = mcpName(g, "one");
    await store.replaceServer(server(g), "h", [{ name: a, rawName: "one" }]);

    // The `not (name = any('{}'))` edge: `not false` is true, so every live row
    // of that server matches.
    const result = await store.replaceServer(server(g), "h", []);
    expect(result).toEqual({ upserted: 0, removed: 1 });
    expect((await raw(a))?.removed_at).not.toBeNull();
    expect(await store.listNames({ server: server(g) })).toEqual([]);
  });

  it("does not touch another server's rows, or another tenant's", async () => {
    const g = "fence";
    const mine = mcpName(g, "mine");
    await store.replaceServer(server(g), "h", [{ name: mine, rawName: "mine" }]);
    await storeB.replaceServer(server(g), "h", [
      { name: mine, rawName: "mine" },
      { name: mcpName(g, "theirs"), rawName: "theirs" },
    ]);

    // Tenant B replacing its generation with nothing leaves tenant A alone —
    // this table has no RLS, so the store's own tenant predicate IS the fence.
    await storeB.replaceServer(server(g), "h", []);
    expect(await store.listNames({ server: server(g) })).toEqual([mine]);
    expect(await storeB.listNames({ server: server(g) })).toEqual([]);

    // And a different server of the same tenant is untouched by the above.
    expect(await store.listNames({ server: server("gen") })).not.toHaveLength(0);
  });

  // --- reads -----------------------------------------------------------------

  it("orders names under collate \"C\", which agrees with a JS code-unit sort", async () => {
    const g = "order";
    // Names chosen to disagree under glibc en_US (which ignores '-' and '_' at
    // the first level) but not under C: Zed sorts before a, and '-' before 'a'.
    const names = [
      `mcp__${server(g)}__Zebra`,
      `mcp__${server(g)}__a-b`,
      `mcp__${server(g)}__ab`,
      `mcp__${server(g)}__a_b`,
    ];
    await store.replaceServer(
      server(g),
      "h",
      names.map((name) => ({ name, rawName: name })),
    );
    const listed = await store.listNames({ server: server(g) });
    expect(listed).toEqual(sorted(names));
    // The blank-query search takes the same order.
    const blank = await store.search("", 50);
    const mine = blank.map((h) => h.name).filter((n) => n.startsWith(`mcp__${server(g)}__`));
    expect(mine).toEqual(sorted(names));
  });

  it("caps a search description at 200 chars but describe() returns the whole thing", async () => {
    const g = "cap";
    const long = `quixotically ${"x".repeat(400)}`;
    await store.replaceServer(server(g), "h", [
      { name: mcpName(g, "verbose"), rawName: "verbose", description: long },
    ]);

    const hits = await store.search("quixotically", 5);
    expect(hits).toHaveLength(1);
    expect(hits[0]?.description).toHaveLength(200);
    expect(hits[0]?.description.endsWith("…")).toBe(true);
    expect((await store.describe(mcpName(g, "verbose")))?.description).toBe(long);
  });

  it("ranks by ts_rank_cd and honours the limit", async () => {
    const g = "rank";
    await store.replaceServer(server(g), "h", [
      { name: mcpName(g, "one"), rawName: "one", description: "flibbertigibbet flibbertigibbet flibbertigibbet" },
      { name: mcpName(g, "two"), rawName: "two", description: "flibbertigibbet mentioned once" },
      { name: mcpName(g, "three"), rawName: "three", description: "unrelated" },
    ]);
    const hits = await store.search("flibbertigibbet", 10);
    expect(hits.map((h) => h.name)).toEqual([mcpName(g, "one"), mcpName(g, "two")]);
    expect(await store.search("flibbertigibbet", 1)).toHaveLength(1);
    // A query that matches nothing is an empty list, not an error.
    expect(await store.search("zzzzznothingmatchesthis", 10)).toEqual([]);
  });

  it("never lets websearch_to_tsquery raise on adversarial query text", async () => {
    // to_tsquery would throw a syntax error on any of these; websearch_ does
    // not, which is why the query text from an LLM goes through it.
    for (const q of ["&&& |", "a & (b", `"unbalanced`, "-", "!!!"]) {
      expect(Array.isArray(await store.search(q, 5))).toBe(true);
    }
  });

  // --- built-ins and server state -------------------------------------------

  it("upserts built-ins without a server, and never withdraws the ones a surface omits", async () => {
    const g = "builtin";
    const readFile = `read_file-${g}-${RUN}`;
    const bash = `bash-${g}-${RUN}`;
    expect(
      await store.upsertBuiltins([
        { name: readFile, description: "reads a file", parameters: { properties: { path: { description: "Absolute path" } } } },
        { name: bash, description: "runs a shell command" },
      ]),
    ).toBe(2);

    const row = await raw(readFile);
    expect(row?.source).toBe("builtin");
    expect(row?.server).toBeNull();
    expect(row?.raw_name).toBeNull();
    expect(row?.config_hash).toBeNull();

    // A second surface without `bash` (headless, no --shell) must not retire it.
    await store.upsertBuiltins([{ name: readFile, description: "reads a file" }]);
    expect((await raw(bash))?.removed_at).toBeNull();

    const builtins = await store.listNames({ source: "builtin" });
    expect(sorted(builtins)).toEqual(sorted([readFile, bash]));
    expect(builtins.some((n) => n.startsWith("mcp__"))).toBe(false);
  });

  it("entries() hands a writer back rawName + configHash, which is the trust path's whole point", async () => {
    const g = "entries";
    const schema = { type: "object", properties: { ref: { description: "A git ref" } } };
    const long = `${"w".repeat(400)}`;
    await store.replaceServer(server(g), "hash-xyz", [
      { name: mcpName(g, "checkout"), rawName: "checkout", description: long, parameters: schema },
    ]);

    // McpManager rebuilds real tools from these rows before the server has
    // answered; a capped description or a missing rawName would make the
    // rebuilt tool wrong in a way nothing else notices.
    const rows = await store.entries({ server: server(g) });
    expect(rows).toHaveLength(1);
    expect(rows[0]?.name).toBe(mcpName(g, "checkout"));
    expect(rows[0]?.rawName).toBe("checkout");
    expect(rows[0]?.configHash).toBe("hash-xyz");
    expect(rows[0]?.description).toBe(long);
    expect(rows[0]?.parameters).toEqual(schema);
    expect(Number.isNaN(Date.parse(rows[0]?.updatedAt ?? ""))).toBe(false);
    expect("removedAt" in (rows[0] ?? {})).toBe(false);

    await store.replaceServer(server(g), "hash-xyz", []);
    expect(await store.entries({ server: server(g) })).toEqual([]);
    const withRemoved = await store.entries({ server: server(g), includeRemoved: true });
    expect(withRemoved[0]?.removedAt).toBeTruthy();
  });

  it("serverState is the config-hash trust probe: hash, live count, newest updated_at", async () => {
    const g = "state";
    await store.replaceServer(server(g), "hash-abc", [
      { name: mcpName(g, "a"), rawName: "a" },
      { name: mcpName(g, "b"), rawName: "b" },
    ]);

    const state = await store.serverState(server(g));
    expect(state?.configHash).toBe("hash-abc");
    // count(*) is bigint — postgres.js hands it back as a string.
    expect(state?.count).toBe(2);
    expect(typeof state?.count).toBe("number");
    expect(Number.isNaN(Date.parse(state?.updatedAt ?? ""))).toBe(false);

    // A hash change is what makes McpManager stop trusting the catalog.
    await store.replaceServer(server(g), "hash-def", [{ name: mcpName(g, "a"), rawName: "a" }]);
    const after = await store.serverState(server(g));
    expect(after?.configHash).toBe("hash-def");
    expect(after?.count).toBe(1);

    // Unknown server, and a server whose rows are all withdrawn, are both null:
    // "nothing live" must be waited for, never trusted.
    expect(await store.serverState(`srv-nothing-${RUN}`)).toBeNull();
    await store.replaceServer(server(g), "hash-def", []);
    expect(await store.serverState(server(g))).toBeNull();
  });
});
