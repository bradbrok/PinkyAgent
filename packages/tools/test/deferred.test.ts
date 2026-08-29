/**
 * tool_search / tool_describe / tool_call — the model's door to the catalog
 * (slice 9, DESIGN.md §9).
 *
 * What these tests guard: the three tools are registered on EVERY surface, so
 * the two shapes that matter are "there is a catalog" and "there is none", and
 * the second one has to be a clean, readable isError rather than a crash — the
 * same contract the memory and settings tools keep without their planes. The
 * alternative (registering them only when a catalog exists) would make the
 * request header depend on whether an MCP server was up at boot, which is the
 * cache churn this whole slice exists to avoid.
 */
import { describe, expect, test } from "bun:test";
import { DEFAULT_SETTINGS, MAX_TOOL_SEARCH_LIMIT } from "@pinky/core";
import type { SettingsSnapshot } from "@pinky/core";
import type {
  CatalogEntry,
  CatalogHit,
  DeferredTools,
  ToolContext,
  ToolResult,
} from "@pinky/runtime";
import { ToolCallTool, ToolDescribeTool, ToolSearchTool } from "../src/deferred";
import { makeCtx } from "./helpers";

// ---------------------------------------------------------------------------
// A DeferredTools double: records what the tools asked for.
// ---------------------------------------------------------------------------

interface FakeDeferred extends DeferredTools {
  searches: { query: string; limit: number }[];
  describes: string[];
  calls: { name: string; args: Record<string, unknown> }[];
}

function makeDeferred(
  opts: {
    hits?: CatalogHit[];
    entries?: CatalogEntry[];
    result?: ToolResult;
    searchFails?: string;
    describeFails?: string;
    callThrows?: string;
  } = {},
): FakeDeferred {
  const searches: { query: string; limit: number }[] = [];
  const describes: string[] = [];
  const calls: { name: string; args: Record<string, unknown> }[] = [];
  return {
    searches,
    describes,
    calls,
    catalog: {
      async search(query, limit) {
        searches.push({ query, limit });
        if (opts.searchFails) throw new Error(opts.searchFails);
        return opts.hits ?? [];
      },
      async describe(name) {
        describes.push(name);
        if (opts.describeFails) throw new Error(opts.describeFails);
        return opts.entries?.find((e) => e.name === name) ?? null;
      },
    },
    async call(name, args) {
      calls.push({ name, args });
      if (opts.callThrows) throw new Error(opts.callThrows);
      return opts.result ?? { text: `${name} ran` };
    },
  };
}

/** ctx with a deferred plane bolted on (helpers.makeCtx has no seam for it). */
function ctxWith(deferred?: FakeDeferred, settings?: SettingsSnapshot): ToolContext {
  const base = makeCtx("/tmp", settings ? { settings } : {});
  return deferred ? { ...base, deferred } : base;
}

function settings(searchLimit: number): SettingsSnapshot {
  return { ...DEFAULT_SETTINGS, tools: { ...DEFAULT_SETTINGS.tools, searchLimit } };
}

const hit = (name: string, description = `${name} description`): CatalogHit => ({
  name,
  description,
  source: "mcp",
  server: "srv",
});

const entry = (name: string): CatalogEntry => ({
  ...hit(name, "Create an issue on a repository."),
  parameters: {
    type: "object",
    properties: { repo: { type: "string" }, title: { type: "string" } },
    required: ["repo", "title"],
  },
});

// ---------------------------------------------------------------------------
// No catalog on this surface
// ---------------------------------------------------------------------------

describe("meta-tools without ctx.deferred", () => {
  test.each([
    ["tool_search", () => new ToolSearchTool(), { query: "anything" }],
    ["tool_describe", () => new ToolDescribeTool(), { name: "anything" }],
    ["tool_call", () => new ToolCallTool(), { name: "anything", args: {} }],
  ])("%s degrades to a clean error", async (name, make, args) => {
    const res = await make().execute(args as Record<string, unknown>, ctxWith());
    expect(res.isError).toBe(true);
    expect(res.text).toBe(`${name}: no deferred tools on this surface`);
  });
});

// ---------------------------------------------------------------------------
// tool_search
// ---------------------------------------------------------------------------

describe("tool_search", () => {
  test("renders a numbered list and the follow-up hint", async () => {
    const deferred = makeDeferred({ hits: [hit("mcp__srv__a"), hit("mcp__srv__b")] });
    const res = await new ToolSearchTool().execute({ query: "issues" }, ctxWith(deferred));
    expect(res.isError).toBeUndefined();
    expect(res.text).toBe(
      [
        "1. mcp__srv__a — mcp__srv__a description",
        "2. mcp__srv__b — mcp__srv__b description",
        "",
        "tool_describe <name> for its schema; tool_call to run it.",
      ].join("\n"),
    );
    expect(deferred.searches).toEqual([{ query: "issues", limit: 8 }]);
  });

  test("the default page size comes from settings.tools.searchLimit", async () => {
    const deferred = makeDeferred();
    await new ToolSearchTool().execute({ query: "x" }, ctxWith(deferred, settings(3)));
    expect(deferred.searches[0]!.limit).toBe(3);
  });

  test("falls back to 8 when the run carries no settings snapshot", async () => {
    const deferred = makeDeferred();
    await new ToolSearchTool().execute({ query: "x" }, ctxWith(deferred));
    expect(deferred.searches[0]!.limit).toBe(8);
  });

  test("an explicit limit overrides the setting", async () => {
    const deferred = makeDeferred();
    await new ToolSearchTool().execute({ query: "x", limit: 2 }, ctxWith(deferred, settings(30)));
    expect(deferred.searches[0]!.limit).toBe(2);
  });

  test.each([
    [0, "between 1 and"],
    [MAX_TOOL_SEARCH_LIMIT + 1, "between 1 and"],
    [2.5, "between 1 and"],
    ["5", "between 1 and"],
  ])("rejects limit %p", async (limit, fragment) => {
    const deferred = makeDeferred();
    const res = await new ToolSearchTool().execute({ query: "x", limit }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain(fragment);
    expect(deferred.searches).toHaveLength(0);
  });

  test("a non-string query is refused; an empty one lists the catalog", async () => {
    const deferred = makeDeferred({ hits: [hit("read")] });
    const bad = await new ToolSearchTool().execute({ query: 7 }, ctxWith(deferred));
    expect(bad.isError).toBe(true);
    expect(bad.text).toContain("'query' must be a string");

    const listing = await new ToolSearchTool().execute({ query: "  " }, ctxWith(deferred));
    expect(listing.isError).toBeUndefined();
    expect(deferred.searches).toEqual([{ query: "", limit: 8 }]);
  });

  test("no matches says so, quoting the query, without an empty list", async () => {
    const deferred = makeDeferred({ hits: [] });
    const res = await new ToolSearchTool().execute({ query: "unicorns" }, ctxWith(deferred));
    expect(res.isError).toBeUndefined();
    expect(res.text).toContain("no matches");
    expect(res.text).toContain("unicorns");
    expect(res.text).not.toContain("tool_describe <name>");
  });

  test("an empty catalog reads as empty rather than as a failed search", async () => {
    const res = await new ToolSearchTool().execute({ query: "" }, ctxWith(makeDeferred()));
    expect(res.text).toContain("the tool catalog is empty");
  });

  test("a catalog failure is an isError, not a throw", async () => {
    const deferred = makeDeferred({ searchFails: "db down" });
    const res = await new ToolSearchTool().execute({ query: "x" }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain("catalog search failed: db down");
  });
});

// ---------------------------------------------------------------------------
// tool_describe
// ---------------------------------------------------------------------------

describe("tool_describe", () => {
  test("prints the description, the server, and a fenced JSON schema", async () => {
    const deferred = makeDeferred({ entries: [entry("mcp__srv__create_issue")] });
    const res = await new ToolDescribeTool().execute(
      { name: "mcp__srv__create_issue" },
      ctxWith(deferred),
    );
    expect(res.isError).toBeUndefined();
    expect(res.text).toContain("mcp__srv__create_issue (mcp server: srv)");
    expect(res.text).toContain("Create an issue on a repository.");
    expect(res.text).toContain("```json");
    expect(res.text).toContain('"required"');
    expect(res.text).toContain('Run it with tool_call: { "name": "mcp__srv__create_issue"');
    // The schema is real JSON, not a paraphrase: the model has to copy it.
    const json = res.text.split("```json\n")[1]!.split("\n```")[0]!;
    expect(JSON.parse(json)).toEqual(entry("mcp__srv__create_issue").parameters);
  });

  test("omits the server line for a built-in", async () => {
    const local: CatalogEntry = {
      name: "archive",
      description: "Archive a thread.",
      source: "builtin",
      parameters: { type: "object" },
    };
    const res = await new ToolDescribeTool().execute(
      { name: "archive" },
      ctxWith(makeDeferred({ entries: [local] })),
    );
    expect(res.text.startsWith("archive\nArchive a thread.")).toBe(true);
  });

  test("an unknown name points back at tool_search", async () => {
    const deferred = makeDeferred({ entries: [] });
    const res = await new ToolDescribeTool().execute({ name: "nope" }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown tool 'nope'");
    expect(res.text).toContain("tool_search");
    expect(deferred.describes).toEqual(["nope"]);
  });

  test.each([[{}], [{ name: "" }], [{ name: 3 }]])("rejects %o", async (args) => {
    const res = await new ToolDescribeTool().execute(
      args as Record<string, unknown>,
      ctxWith(makeDeferred()),
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'name' must be a non-empty string");
  });

  test("a catalog failure is an isError, not a throw", async () => {
    const res = await new ToolDescribeTool().execute(
      { name: "x" },
      ctxWith(makeDeferred({ describeFails: "boom" })),
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("catalog lookup failed: boom");
  });

  test("a header tool is described with 'call it directly' instead of a tool_call recipe", async () => {
    // The catalog is tenant-wide, so it lists tools this run already has in
    // its header. Routing those through tool_call is a wasted turn.
    const deferred = makeDeferred({ entries: [entry("read")] });
    const headAware = Object.assign(deferred, { isHeadTool: (n: string) => n === "read" });
    const res = await new ToolDescribeTool().execute({ name: "read" }, ctxWith(headAware));
    expect(res.isError).toBeUndefined();
    expect(res.text).toContain("read is already in your tool list; call it directly");
    expect(res.text).not.toContain('Run it with tool_call');
  });

  test("a plane without isHeadTool still describes normally", async () => {
    const res = await new ToolDescribeTool().execute(
      { name: "mcp__srv__x" },
      ctxWith(makeDeferred({ entries: [entry("mcp__srv__x")] })),
    );
    expect(res.text).toContain("Run it with tool_call");
  });

  test("caps a hostile description at 4000 chars with a byte-counted marker", async () => {
    // A server-supplied string lands in the window and is re-read on every
    // later turn; uncapped, one describe can eat a turn's whole budget.
    const long = "x".repeat(10_000);
    const fat: CatalogEntry = { ...entry("mcp__srv__fat"), description: long };
    const res = await new ToolDescribeTool().execute(
      { name: "mcp__srv__fat" },
      ctxWith(makeDeferred({ entries: [fat] })),
    );
    expect(res.isError).toBeUndefined();
    expect(res.text).toContain("…[truncated 6000 bytes]");
    expect(res.text.length).toBeLessThan(5_000);
    expect(res.text).toContain("mcp__srv__fat (mcp server: srv)");
  });

  test("caps a hostile schema at 16 KB, on a code-point boundary", async () => {
    const huge: CatalogEntry = {
      ...entry("mcp__srv__huge"),
      // Multi-byte on purpose: a byte cut mid-code-point would decode to
      // U+FFFD and misreport what the schema said.
      parameters: { type: "object", description: "é".repeat(30_000) },
    };
    const res = await new ToolDescribeTool().execute(
      { name: "mcp__srv__huge" },
      ctxWith(makeDeferred({ entries: [huge] })),
    );
    const block = res.text.split("```json\n")[1]!.split("\n```")[0]!;
    expect(new TextEncoder().encode(block).length).toBeLessThan(16 * 1024 + 64);
    expect(block).toContain("…[truncated ");
    expect(block).not.toContain("\uFFFD");
  });

  test("leaves an ordinary description and schema untouched", async () => {
    const res = await new ToolDescribeTool().execute(
      { name: "mcp__srv__create_issue" },
      ctxWith(makeDeferred({ entries: [entry("mcp__srv__create_issue")] })),
    );
    expect(res.text).not.toContain("truncated");
  });
});

// ---------------------------------------------------------------------------
// tool_call
// ---------------------------------------------------------------------------

describe("tool_call", () => {
  test("delegates name + args to the registry and returns its result verbatim", async () => {
    const deferred = makeDeferred({ result: { text: "issue #7 created" } });
    const res = await new ToolCallTool().execute(
      { name: "mcp__srv__create_issue", args: { repo: "a/b", title: "t" } },
      ctxWith(deferred),
    );
    expect(res).toEqual({ text: "issue #7 created" });
    expect(deferred.calls).toEqual([
      { name: "mcp__srv__create_issue", args: { repo: "a/b", title: "t" } },
    ]);
  });

  test("passes an isError result straight through (the registry owns the wording)", async () => {
    const deferred = makeDeferred({ result: { text: "invalid arguments for x", isError: true } });
    const res = await new ToolCallTool().execute({ name: "x", args: {} }, ctxWith(deferred));
    expect(res).toEqual({ text: "invalid arguments for x", isError: true });
  });

  test("a missing args object means no arguments, not an error", async () => {
    const deferred = makeDeferred();
    const res = await new ToolCallTool().execute({ name: "x" }, ctxWith(deferred));
    expect(res.isError).toBeUndefined();
    expect(deferred.calls).toEqual([{ name: "x", args: {} }]);
  });

  test.each([[[]], ["{}"], [null], [7]])("refuses args %p", async (args) => {
    const deferred = makeDeferred();
    const res = await new ToolCallTool().execute({ name: "x", args }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'args' must be a JSON object");
    expect(deferred.calls).toHaveLength(0);
  });

  test("refuses a missing or empty name before touching the registry", async () => {
    const deferred = makeDeferred();
    const res = await new ToolCallTool().execute({ args: {} }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'name' must be a non-empty string");
    expect(deferred.calls).toHaveLength(0);
  });

  test("a throwing DeferredTools implementation still yields an isError", async () => {
    const deferred = makeDeferred({ callThrows: "registry exploded" });
    const res = await new ToolCallTool().execute({ name: "x", args: {} }, ctxWith(deferred));
    expect(res.isError).toBe(true);
    expect(res.text).toContain("registry exploded");
  });

  test("hands the tool the same ctx it received", async () => {
    const seen: ToolContext[] = [];
    const deferred = makeDeferred();
    const spy: DeferredTools = {
      catalog: deferred.catalog,
      async call(_n, _a, c) {
        seen.push(c);
        return { text: "ok" };
      },
    };
    const ctx = { ...makeCtx("/tmp"), deferred: spy };
    await new ToolCallTool().execute({ name: "x", args: {} }, ctx);
    expect(seen[0]).toBe(ctx);
  });
});

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

describe("meta-tool schemas", () => {
  test("are small, fixed, and describe themselves in one first line", () => {
    for (const t of [new ToolSearchTool(), new ToolDescribeTool(), new ToolCallTool()]) {
      const params = t.parameters as { type: string; properties: Record<string, unknown> };
      expect(params.type).toBe("object");
      expect(Object.keys(params.properties).length).toBeLessThanOrEqual(2);
      // The first line is what the system prompt renders next to the name.
      expect(t.description.split("\n")[0]!.length).toBeGreaterThan(20);
    }
  });

  test("tool_search caps its limit at the same ceiling settings validation uses", () => {
    const params = new ToolSearchTool().parameters as {
      properties: { limit: { maximum: number } };
    };
    expect(params.properties.limit.maximum).toBe(MAX_TOOL_SEARCH_LIMIT);
  });
});
