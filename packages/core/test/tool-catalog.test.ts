/**
 * Deferred-tool catalog unit tests (slice 9).
 *
 * Two halves, the same split as the memory-plane suite:
 *   - the SQL the store emits, recorded by a FakeDb, so the tenant fence, the
 *     `removed_at is null` predicate, the "one transaction" claim of
 *     replaceServer and the JSONB CONTRACT are asserted without a database;
 *   - argText()/capDescription() as pure functions, which is where the search
 *     vocabulary and the context budget actually live.
 * Whether Postgres accepts any of these statements is the integration suite's
 * question (test/integration/tool-catalog.test.ts).
 */
import { describe, expect, it } from "bun:test";
import {
  ARG_TEXT_MAX,
  CATALOG_DESCRIPTION_CAP,
  DEFAULT_CATALOG_SEARCH_LIMIT,
  MAX_CATALOG_SEARCH_LIMIT,
  ToolCatalogStore,
  argText,
  capDescription,
} from "../src/tool-catalog";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
  txDepth: number;
}

type Route = { pattern: RegExp; respond: (params?: unknown[]) => unknown[] };

class FakeDb implements Db {
  calls: Call[] = [];
  /** "begin"/"commit"/"rollback" for the OUTERMOST tx only, in order. */
  txLog: string[] = [];
  private routes: Route[];
  private txDepth = 0;

  constructor(routes: Route[]) {
    this.routes = routes;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params, txDepth: this.txDepth });
    for (const r of this.routes) {
      if (r.pattern.test(sql)) return r.respond(params) as T[];
    }
    throw new Error(`FakeDb: no route for SQL: ${sql}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    const outermost = this.txDepth === 0;
    if (outermost) this.txLog.push("begin");
    this.txDepth += 1;
    try {
      const out = await fn(this);
      if (outermost) this.txLog.push("commit");
      return out;
    } catch (err) {
      if (outermost) this.txLog.push("rollback");
      throw err;
    } finally {
      this.txDepth -= 1;
    }
  }

  async close(): Promise<void> {}

  find(pattern: RegExp): Call | undefined {
    return this.calls.find((c) => pattern.test(c.sql));
  }
  all(pattern: RegExp): Call[] {
    return this.calls.filter((c) => pattern.test(c.sql));
  }
}

const TENANT = "t1";

/** Columns per row in the upsert, in the order upsertRows() binds them. */
const UPSERT_ARITY = 9;

/** The upsert echoes back one `name` per tuple, as `returning name` does. */
const upsertRoute: Route = {
  pattern: /insert into tool_catalog/i,
  respond: (params) => {
    const rows: { name: string }[] = [];
    const n = (params?.length ?? 0) / UPSERT_ARITY;
    for (let i = 0; i < n; i += 1) rows.push({ name: String(params?.[i * UPSERT_ARITY + 1]) });
    return rows;
  },
};

/** Bind values for the i-th row of an upsert statement. */
function tuple(call: Call | undefined, i: number): unknown[] {
  return (call?.params ?? []).slice(i * UPSERT_ARITY, (i + 1) * UPSERT_ARITY);
}

function hitRow(over: Record<string, unknown> = {}): Record<string, unknown> {
  return { name: "alpha", source: "builtin", server: null, description: "does a thing", ...over };
}

// --- replaceServer -----------------------------------------------------------

describe("ToolCatalogStore.replaceServer", () => {
  const removedRoute = (names: string[]): Route => ({
    pattern: /update tool_catalog set removed_at/i,
    respond: () => names.map((name) => ({ name })),
  });

  it("upserts and withdraws inside ONE transaction", async () => {
    const db = new FakeDb([upsertRoute, removedRoute(["mcp__gh__gone"])]);
    const store = new ToolCatalogStore(db, TENANT);

    const result = await store.replaceServer("gh", "hash-1", [
      { name: "mcp__gh__b", rawName: "b", description: "second" },
      { name: "mcp__gh__a", rawName: "a", description: "first" },
    ]);

    expect(db.txLog).toEqual(["begin", "commit"]);
    // Both statements ran inside it.
    expect(db.calls.every((c) => c.txDepth > 0)).toBe(true);
    expect(result).toEqual({ upserted: 2, removed: 1 });
  });

  it("writes name-sorted rows with the mcp source, server, raw name and config hash", async () => {
    const db = new FakeDb([upsertRoute, removedRoute([])]);
    const store = new ToolCatalogStore(db, TENANT);

    await store.replaceServer("gh", "hash-1", [
      { name: "mcp__gh__b", rawName: "b", description: "second" },
      { name: "mcp__gh__a", rawName: "a", description: "first", parameters: { type: "object" } },
    ]);

    const insert = db.find(/insert into tool_catalog/i);
    expect(insert?.sql).toContain("on conflict (tenant_id, name) do update");
    // The "it came back" half of invalidate-never-delete.
    expect(insert?.sql).toContain("removed_at = null");
    // Sorted by code unit, so a generation's SQL is byte-stable.
    expect(tuple(insert, 0)).toEqual([
      TENANT,
      "mcp__gh__a",
      "mcp",
      "gh",
      "a",
      "first",
      { type: "object" },
      "",
      "hash-1",
    ]);
    expect(tuple(insert, 1)).toEqual([
      TENANT,
      "mcp__gh__b",
      "mcp",
      "gh",
      "b",
      "second",
      {},
      "",
      "hash-1",
    ]);
  });

  it("DEFECT GUARD: parameters bind as a plain object, never a JSON string", async () => {
    const db = new FakeDb([upsertRoute, removedRoute([])]);
    const store = new ToolCatalogStore(db, TENANT);
    const schema = { type: "object", properties: { path: { type: "string" } } };

    await store.replaceServer("fs", "h", [{ name: "mcp__fs__read", parameters: schema }]);

    // pg.ts JSONB CONTRACT: postgres.js serializes once. A pre-stringified
    // param lands as a jsonb *string* (see 0004_jsonb_repair.rerun.sql).
    const bound = tuple(db.find(/insert into tool_catalog/i), 0)[6];
    expect(typeof bound).toBe("object");
    expect(bound).toEqual(schema);
  });

  it("stamps removed_at only on that server's live rows that are not in the new set", async () => {
    const db = new FakeDb([upsertRoute, removedRoute(["mcp__gh__old"])]);
    const store = new ToolCatalogStore(db, TENANT);

    await store.replaceServer("gh", "hash-1", [
      { name: "mcp__gh__b" },
      { name: "mcp__gh__a" },
    ]);

    const update = db.find(/update tool_catalog set removed_at/i);
    expect(update?.sql).toContain("where tenant_id = $1 and server = $2 and removed_at is null");
    expect(update?.sql).toContain("not (name = any($3::text[]))");
    // Never a DELETE, here or anywhere.
    expect(db.calls.some((c) => /delete/i.test(c.sql))).toBe(false);
    expect(update?.params).toEqual([TENANT, "gh", ["mcp__gh__a", "mcp__gh__b"]]);
  });

  it("treats an empty generation as 'this server offers nothing' — no insert, everything withdrawn", async () => {
    const db = new FakeDb([upsertRoute, removedRoute(["mcp__gh__a", "mcp__gh__b"])]);
    const store = new ToolCatalogStore(db, TENANT);

    const result = await store.replaceServer("gh", "hash-2", []);

    expect(db.all(/insert into tool_catalog/i)).toHaveLength(0);
    // `not (name = any('{}'))` is `not false` = true: every live row matches.
    expect(db.find(/update tool_catalog set removed_at/i)?.params).toEqual([TENANT, "gh", []]);
    expect(result).toEqual({ upserted: 0, removed: 2 });
  });

  it("collapses a duplicate name to its first occurrence instead of losing the sync", async () => {
    const db = new FakeDb([upsertRoute, removedRoute([])]);
    const store = new ToolCatalogStore(db, TENANT);

    // Postgres would reject the whole statement: "ON CONFLICT DO UPDATE
    // command cannot affect row a second time".
    const result = await store.replaceServer("gh", "h", [
      { name: "mcp__gh__dup", description: "kept: first in the caller's list" },
      { name: "mcp__gh__dup", description: "dropped" },
    ]);

    const insert = db.find(/insert into tool_catalog/i);
    expect(insert?.params).toHaveLength(UPSERT_ARITY);
    expect(tuple(insert, 0)[5]).toBe("kept: first in the caller's list");
    expect(result.upserted).toBe(1);
  });

  it("rejects an empty server name and a nameless tool", async () => {
    const db = new FakeDb([upsertRoute, removedRoute([])]);
    const store = new ToolCatalogStore(db, TENANT);
    await expect(store.replaceServer("", "h", [])).rejects.toThrow(/non-empty string/);
    await expect(
      store.replaceServer("gh", "h", [{ name: "" }]),
    ).rejects.toThrow(/non-empty name/);
  });
});

// --- upsertBuiltins ----------------------------------------------------------

describe("ToolCatalogStore.upsertBuiltins", () => {
  it("writes builtin rows with a null server, raw name and config hash", async () => {
    const db = new FakeDb([upsertRoute]);
    const store = new ToolCatalogStore(db, TENANT);

    const written = await store.upsertBuiltins([
      { name: "write_file", description: "writes" },
      { name: "read_file", description: "reads", parameters: { properties: { path: {} } } },
    ]);

    expect(written).toBe(2);
    const insert = db.find(/insert into tool_catalog/i);
    expect(tuple(insert, 0)).toEqual([TENANT, "read_file", "builtin", null, null, "reads", { properties: { path: {} } }, "path", null]);
    expect(tuple(insert, 1)).toEqual([TENANT, "write_file", "builtin", null, null, "writes", {}, "", null]);
  });

  it("never withdraws absent built-ins — the built-in set is per surface, not per deployment", async () => {
    const db = new FakeDb([upsertRoute]);
    const store = new ToolCatalogStore(db, TENANT);
    // `pinky prompt` registers bash, `pinky headless` does not; a generational
    // replace would have the two flapping each other's rows forever.
    await store.upsertBuiltins([{ name: "read_file" }]);
    expect(db.all(/update tool_catalog set removed_at/i)).toHaveLength(0);
  });

  it("issues no statement at all for an empty list", async () => {
    const db = new FakeDb([upsertRoute]);
    const store = new ToolCatalogStore(db, TENANT);
    expect(await store.upsertBuiltins([])).toBe(0);
    expect(db.calls).toHaveLength(0);
  });
});

// --- search ------------------------------------------------------------------

describe("ToolCatalogStore.search", () => {
  const searchRoute = (rows: Record<string, unknown>[]): Route => ({
    pattern: /from tool_catalog/i,
    respond: () => rows,
  });

  it("runs the FTS voice with websearch_to_tsquery and ranks by ts_rank_cd", async () => {
    const db = new FakeDb([searchRoute([hitRow()])]);
    const store = new ToolCatalogStore(db, TENANT);

    await store.search("create issue", 5);

    const call = db.find(/from tool_catalog/i);
    // 'english' must match schema/0006_tool_catalog.sql's generated column.
    expect(call?.sql).toContain("websearch_to_tsquery('english', $2)");
    expect(call?.sql).toContain("tsv @@ websearch_to_tsquery('english', $2)");
    expect(call?.sql).toContain("ts_rank_cd(tsv, websearch_to_tsquery('english', $2)) desc");
    expect(call?.sql).toContain("where tenant_id = $1 and removed_at is null");
    // C collation: the JS twin of a code-unit sort, identical on alpine (C)
    // and the pgvector image (glibc en_US).
    expect(call?.sql).toContain(`name collate "C"`);
    expect(call?.params).toEqual([TENANT, "create issue", 5]);
  });

  it("degrades a blank query to a name-ordered listing rather than nothing", async () => {
    const db = new FakeDb([searchRoute([hitRow()])]);
    const store = new ToolCatalogStore(db, TENANT);

    await store.search("   ");

    const call = db.find(/from tool_catalog/i);
    expect(call?.sql).not.toContain("tsv @@");
    expect(call?.sql).toContain(`order by name collate "C"`);
    expect(call?.params).toEqual([TENANT, DEFAULT_CATALOG_SEARCH_LIMIT]);
  });

  it("clamps the limit to 1..MAX", async () => {
    const db = new FakeDb([searchRoute([])]);
    const store = new ToolCatalogStore(db, TENANT);
    await store.search("", 0);
    await store.search("", 10_000);
    await store.search("", 3.7);
    const limits = db.all(/from tool_catalog/i).map((c) => c.params?.[1]);
    expect(limits).toEqual([1, MAX_CATALOG_SEARCH_LIMIT, 3]);
  });

  it("caps the description at 200 characters with an ellipsis, and leaves short ones alone", async () => {
    const long = "x".repeat(400);
    const db = new FakeDb([
      searchRoute([
        hitRow({ name: "long", description: long }),
        hitRow({ name: "short", description: "brief" }),
      ]),
    ]);
    const store = new ToolCatalogStore(db, TENANT);

    const hits = await store.search("x");
    expect(hits[0]?.description).toHaveLength(CATALOG_DESCRIPTION_CAP);
    expect(hits[0]?.description.endsWith("…")).toBe(true);
    expect(hits[1]?.description).toBe("brief");
  });

  it("carries `server` only for mcp rows", async () => {
    const db = new FakeDb([
      searchRoute([
        hitRow({ name: "read_file" }),
        hitRow({ name: "mcp__gh__issue", source: "mcp", server: "gh" }),
      ]),
    ]);
    const store = new ToolCatalogStore(db, TENANT);

    const hits = await store.search("x");
    expect(hits[0]).toEqual({ name: "read_file", description: "does a thing", source: "builtin" });
    expect("server" in (hits[0] ?? {})).toBe(false);
    expect(hits[1]?.server).toBe("gh");
    expect(hits[1]?.source).toBe("mcp");
  });
});

// --- describe / listNames / serverState --------------------------------------

describe("ToolCatalogStore.describe", () => {
  it("selects the schema, hides withdrawn rows, and does NOT cap the description", async () => {
    const long = "y".repeat(400);
    const db = new FakeDb([
      {
        pattern: /from tool_catalog/i,
        respond: () => [
          {
            name: "mcp__gh__issue",
            source: "mcp",
            server: "gh",
            description: long,
            parameters: { type: "object" },
          },
        ],
      },
    ]);
    const store = new ToolCatalogStore(db, TENANT);

    const entry = await store.describe("mcp__gh__issue");

    const call = db.find(/from tool_catalog/i);
    expect(call?.sql).toContain("parameters");
    expect(call?.sql).toContain("removed_at is null");
    expect(call?.params).toEqual([TENANT, "mcp__gh__issue"]);
    expect(entry?.description).toBe(long);
    expect(entry?.parameters).toEqual({ type: "object" });
  });

  it("returns null for an unknown name, and without querying for a blank one", async () => {
    const db = new FakeDb([{ pattern: /from tool_catalog/i, respond: () => [] }]);
    const store = new ToolCatalogStore(db, TENANT);
    expect(await store.describe("nope")).toBeNull();
    expect(await store.describe("  ")).toBeNull();
    expect(db.calls).toHaveLength(1);
  });

  it("tolerates a legacy doubly-encoded parameters column", async () => {
    const db = new FakeDb([
      {
        pattern: /from tool_catalog/i,
        respond: () => [
          {
            name: "t",
            source: "mcp",
            server: "gh",
            description: "",
            parameters: JSON.stringify({ type: "object" }),
          },
        ],
      },
    ]);
    const store = new ToolCatalogStore(db, TENANT);
    expect((await store.describe("t"))?.parameters).toEqual({ type: "object" });
  });
});

describe("ToolCatalogStore.entries", () => {
  const entriesRoute: Route = {
    pattern: /raw_name, config_hash, updated_at, removed_at from tool_catalog/i,
    respond: () => [
      {
        name: "mcp__gh__issue",
        source: "mcp",
        server: "gh",
        description: "z".repeat(400),
        parameters: { type: "object" },
        raw_name: "issue",
        config_hash: "hash-1",
        updated_at: new Date("2026-08-29T10:00:00.000Z"),
        removed_at: null,
      },
      {
        name: "read_file",
        source: "builtin",
        server: null,
        description: "reads",
        parameters: {},
        raw_name: null,
        config_hash: null,
        updated_at: new Date("2026-08-29T10:00:00.000Z"),
        removed_at: new Date("2026-08-29T11:00:00.000Z"),
      },
    ],
  };

  it("returns the bookkeeping columns uncapped — this is the writer's read, not the model's", async () => {
    const db = new FakeDb([entriesRoute]);
    const store = new ToolCatalogStore(db, TENANT);

    const rows = await store.entries({ server: "gh", includeRemoved: true });

    const call = db.find(/from tool_catalog/i);
    expect(call?.sql).toContain("raw_name");
    expect(call?.sql).toContain("config_hash");
    expect(call?.sql).toContain(`order by name collate "C"`);
    expect(call?.sql).not.toContain("removed_at is null");
    expect(call?.params).toEqual([TENANT, "gh"]);

    // rawName is why this method exists: callTool must send the server's own
    // spelling, and it cannot be recovered from the namespaced name.
    expect(rows[0]?.rawName).toBe("issue");
    expect(rows[0]?.configHash).toBe("hash-1");
    expect(rows[0]?.description).toHaveLength(400);
    expect(rows[0]?.updatedAt).toBe("2026-08-29T10:00:00.000Z");
    expect("removedAt" in (rows[0] ?? {})).toBe(false);
    // Null columns are absent, not undefined (exactOptionalPropertyTypes).
    expect("rawName" in (rows[1] ?? {})).toBe(false);
    expect("configHash" in (rows[1] ?? {})).toBe(false);
    expect(rows[1]?.removedAt).toBe("2026-08-29T11:00:00.000Z");
  });

  it("hides withdrawn rows by default", async () => {
    const db = new FakeDb([entriesRoute]);
    const store = new ToolCatalogStore(db, TENANT);
    await store.entries({ source: "builtin" });
    const call = db.find(/from tool_catalog/i);
    expect(call?.sql).toContain("source = $2");
    expect(call?.sql).toContain("removed_at is null");
  });
});

describe("ToolCatalogStore.listNames", () => {
  const namesRoute: Route = {
    pattern: /select name from tool_catalog/i,
    respond: () => [{ name: "a" }, { name: "b" }],
  };

  it("filters by source and server, live rows only, C-collated", async () => {
    const db = new FakeDb([namesRoute]);
    const store = new ToolCatalogStore(db, TENANT);

    expect(await store.listNames({ source: "mcp", server: "gh" })).toEqual(["a", "b"]);
    const call = db.find(/select name from tool_catalog/i);
    expect(call?.sql).toContain("source = $2");
    expect(call?.sql).toContain("server = $3");
    expect(call?.sql).toContain("removed_at is null");
    expect(call?.sql).toContain(`order by name collate "C"`);
    expect(call?.params).toEqual([TENANT, "mcp", "gh"]);
  });

  it("drops the liveness predicate for includeRemoved", async () => {
    const db = new FakeDb([namesRoute]);
    const store = new ToolCatalogStore(db, TENANT);
    await store.listNames({ includeRemoved: true });
    expect(db.find(/select name from tool_catalog/i)?.sql).not.toContain("removed_at is null");
  });
});

describe("ToolCatalogStore.serverState", () => {
  const stateRoute = (row: Record<string, unknown> | null): Route => ({
    pattern: /array_agg\(config_hash/i,
    respond: () => (row ? [row] : []),
  });

  it("coerces the bigint count, which postgres.js hands back as a string", async () => {
    const at = new Date("2026-08-29T10:00:00.000Z");
    const db = new FakeDb([stateRoute({ config_hash: "hash-1", count: "3", updated_at: at })]);
    const store = new ToolCatalogStore(db, TENANT);

    const state = await store.serverState("gh");

    expect(state).toEqual({ configHash: "hash-1", count: 3, updatedAt: at.toISOString() });
    expect(db.find(/array_agg/i)?.params).toEqual([TENANT, "gh"]);
    expect(db.find(/array_agg/i)?.sql).toContain("removed_at is null");
  });

  it("is null when the server has no live rows, so a cold catalog is never trusted", async () => {
    const db = new FakeDb([stateRoute({ config_hash: null, count: "0", updated_at: null })]);
    const store = new ToolCatalogStore(db, TENANT);
    expect(await store.serverState("gh")).toBeNull();
  });
});

describe("ToolCatalogStore construction", () => {
  it("refuses an empty tenant id", () => {
    const db = new FakeDb([]);
    expect(() => new ToolCatalogStore(db, "")).toThrow(/tenantId/);
    expect(() => new ToolCatalogStore(db, "   ")).toThrow(/tenantId/);
  });
});

// --- pure helpers ------------------------------------------------------------

describe("argText", () => {
  it("flattens top-level property names and their descriptions in schema order", () => {
    expect(
      argText({
        type: "object",
        properties: {
          owner: { type: "string", description: "Repository owner" },
          repo: { type: "string", description: "Repository name" },
          draft: { type: "boolean" },
        },
      }),
    ).toBe("owner Repository owner repo Repository name draft");
  });

  it("descends one level into a nested object", () => {
    expect(
      argText({
        properties: {
          filter: {
            type: "object",
            description: "Search filter",
            properties: { since: { description: "ISO date" }, labels: {} },
          },
        },
      }),
    ).toBe("filter Search filter since ISO date labels");
  });

  it("descends into an array's item schema", () => {
    expect(
      argText({
        properties: {
          commits: {
            type: "array",
            items: { description: "One commit", properties: { sha: { description: "Commit sha" } } },
          },
        },
      }),
    ).toBe("commits One commit sha Commit sha");
  });

  it("stops after one level of nesting", () => {
    expect(
      argText({
        properties: {
          a: { properties: { b: { properties: { deep: { description: "unreachable" } } } } },
        },
      }),
    ).toBe("a b");
  });

  it("normalizes whitespace so the stored column is one tidy line", () => {
    expect(argText({ properties: { p: { description: "line one\n  line   two\t" } } })).toBe(
      "p line one line two",
    );
  });

  it("is total: anything that is not a schema with properties yields an empty string", () => {
    expect(argText(undefined)).toBe("");
    expect(argText(null)).toBe("");
    expect(argText(true)).toBe("");
    expect(argText("string schema")).toBe("");
    expect(argText([1, 2, 3])).toBe("");
    expect(argText({ type: "object" })).toBe("");
    expect(argText({ properties: "not an object" })).toBe("");
    expect(argText({ properties: { a: "not a schema" } })).toBe("a");
  });

  it("caps its own length so one prolix schema cannot bloat every row's tsv", () => {
    const long = argText({ properties: { p: { description: "z".repeat(ARG_TEXT_MAX * 2) } } });
    expect(long).toHaveLength(ARG_TEXT_MAX);
    expect(argText({ properties: { p: { description: "abcdef" } } }, { maxLength: 4 })).toBe("p ab");
  });
});

describe("capDescription", () => {
  it("keeps the total length at the cap, ellipsis included", () => {
    const capped = capDescription("a".repeat(500));
    expect(capped).toHaveLength(CATALOG_DESCRIPTION_CAP);
    expect(capped.endsWith("…")).toBe(true);
    expect(capDescription("short")).toBe("short");
    expect(capDescription("abcdef", 3)).toBe("ab…");
  });
});
