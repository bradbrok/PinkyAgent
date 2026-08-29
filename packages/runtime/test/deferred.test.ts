/**
 * Deferred-tool plane tests (slice 9, DESIGN.md §9).
 *
 * Two invariants under all of this:
 *   - the HEAD partition is a cache key, so it must be deterministic (sorted
 *     by code unit, meta-tools pinned) and must never depend on which servers
 *     happen to be reachable;
 *   - a deferred call NEVER throws. It is reached through a tool result, so
 *     every failure has to come back as text the model can act on — and a bad
 *     argument comes back with the schema attached, which is what saves the
 *     round trip through tool_describe.
 */
import { describe, expect, test } from "bun:test";
import type { Db, SettingsSnapshot, ThreadRef } from "@pinky/core";
import { DeferredToolRegistry, META_TOOL_NAMES, partitionTools, validateArgs } from "../src/deferred";
import { SHED_CONTEXT_TOOL_NAME } from "../src/continuity";
import { buildSystemPrompt } from "../src/system-prompt";
import type { CatalogEntry, CatalogHit, Tool, ToolCatalogView, ToolContext } from "../src/types";

// ---------------------------------------------------------------------------
// Doubles
// ---------------------------------------------------------------------------

const unusedDb: Db = {
  query: async () => [],
  queryOne: async () => null,
  tx: async (fn) => fn(unusedDb),
  close: async () => {},
};

const THREAD: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

const ctx: ToolContext = {
  cwd: "/tmp",
  db: unusedDb,
  thread: THREAD,
  emit: async () => {},
  agentId: "pinky",
};

function tool(name: string, over: Partial<Tool> = {}): Tool {
  return {
    name,
    description: `${name} description`,
    parameters: { type: "object", properties: {} },
    execute: async () => ({ text: `${name} ran` }),
    ...over,
  };
}

type ToolsCfg = SettingsSnapshot["tools"];

function cfg(over: Partial<ToolsCfg> = {}): ToolsCfg {
  return {
    defaultMode: { builtin: "always", mcp: "deferred" },
    alwaysOn: [],
    deferred: [],
    searchLimit: 8,
    ...over,
  };
}

/** Everything that starts with `mcp__` came from a server; the rest is local. */
const sourceOf = (name: string): "builtin" | "mcp" =>
  name.startsWith("mcp__") ? "mcp" : "builtin";

class FakeCatalog implements ToolCatalogView {
  readonly searches: { query: string; limit: number }[] = [];
  readonly describes: string[] = [];
  constructor(
    private readonly entries: CatalogEntry[] = [],
    private readonly failure?: Error,
  ) {}

  async search(query: string, limit: number): Promise<CatalogHit[]> {
    this.searches.push({ query, limit });
    if (this.failure) throw this.failure;
    return this.entries.slice(0, limit).map(({ parameters: _p, ...hit }) => hit);
  }

  async describe(name: string): Promise<CatalogEntry | null> {
    this.describes.push(name);
    if (this.failure) throw this.failure;
    return this.entries.find((e) => e.name === name) ?? null;
  }
}

function entry(name: string, parameters: Record<string, unknown>): CatalogEntry {
  return { name, description: `${name} description`, source: "mcp", server: "srv", parameters };
}

const names = (tools: Tool[]): string[] => tools.map((t) => t.name);

// ---------------------------------------------------------------------------
// partitionTools
// ---------------------------------------------------------------------------

describe("partitionTools", () => {
  test("built-ins default into the header, MCP tools into the catalog", () => {
    const { head, deferred } = partitionTools(
      [tool("read"), tool("mcp__srv__create_issue"), tool("grep")],
      cfg(),
      sourceOf,
    );
    expect(names(head)).toEqual(["grep", "read"]);
    expect(names(deferred)).toEqual(["mcp__srv__create_issue"]);
  });

  test("defaultMode is per source, both directions", () => {
    const tools = [tool("read"), tool("mcp__srv__a")];
    const flipped = partitionTools(
      tools,
      cfg({ defaultMode: { builtin: "deferred", mcp: "always" } }),
      sourceOf,
    );
    expect(names(flipped.head)).toEqual(["mcp__srv__a"]);
    expect(names(flipped.deferred)).toEqual(["read"]);
  });

  test("alwaysOn beats deferred beats defaultMode", () => {
    const tools = [tool("read"), tool("grep"), tool("mcp__srv__a"), tool("mcp__srv__b")];
    const { head, deferred } = partitionTools(
      tools,
      // `read` is in BOTH lists: alwaysOn wins (settings validation rejects
      // that combination, but the partition must not depend on it having).
      cfg({ alwaysOn: ["read", "mcp__srv__a"], deferred: ["read", "grep"] }),
      sourceOf,
    );
    expect(names(head)).toEqual(["mcp__srv__a", "read"]);
    expect(names(deferred)).toEqual(["grep", "mcp__srv__b"]);
  });

  test("meta-tools and shed_context stay in the header even when deferred names them", () => {
    const pinned = [...META_TOOL_NAMES, SHED_CONTEXT_TOOL_NAME];
    const { head, deferred } = partitionTools(
      pinned.map((n) => tool(n)),
      cfg({ deferred: [...pinned], defaultMode: { builtin: "deferred", mcp: "deferred" } }),
      sourceOf,
    );
    expect(names(head).sort()).toEqual([...pinned].sort());
    expect(deferred).toHaveLength(0);
  });

  test("each side is sorted by code unit, not locale", () => {
    // "Zebra" < "alpha" < "mcp-tool" < "mcp_tool" by code unit; a locale
    // collation would put Zebra next to alpha and ignore the hyphen.
    const tools = [tool("mcp_tool"), tool("alpha"), tool("Zebra"), tool("mcp-tool")];
    const { head } = partitionTools(tools, cfg(), sourceOf);
    expect(names(head)).toEqual(["Zebra", "alpha", "mcp-tool", "mcp_tool"]);
  });

  test("is a pure split: nothing added, nothing dropped, input untouched", () => {
    const tools = [tool("read"), tool("mcp__srv__a"), tool("grep")];
    const before = names(tools);
    const { head, deferred } = partitionTools(tools, cfg({ deferred: ["grep"] }), sourceOf);
    expect([...names(head), ...names(deferred)].sort()).toEqual([...before].sort());
    expect(names(tools)).toEqual(before);
  });

  test("a settings name matching no tool is ignored", () => {
    const { head, deferred } = partitionTools(
      [tool("read")],
      cfg({ alwaysOn: ["mcp__gone__tool"], deferred: ["mcp__also_gone__tool"] }),
      sourceOf,
    );
    expect(names(head)).toEqual(["read"]);
    expect(deferred).toHaveLength(0);
  });

  test("an empty tool set partitions into two empty sides", () => {
    expect(partitionTools([], cfg(), sourceOf)).toEqual({ head: [], deferred: [] });
  });
});

// ---------------------------------------------------------------------------
// validateArgs
// ---------------------------------------------------------------------------

describe("validateArgs", () => {
  const schema = {
    type: "object",
    properties: {
      title: { type: "string" },
      count: { type: "integer" },
      ratio: { type: "number" },
      draft: { type: "boolean" },
      labels: { type: "array" },
      meta: { type: "object" },
      nothing: { type: "null" },
      either: { type: ["string", "number"] },
      mode: { type: "string", enum: ["fast", "slow"] },
      loose: {},
    },
    required: ["title"],
    additionalProperties: false,
  };

  test("accepts a well-formed payload", () => {
    expect(
      validateArgs(schema, {
        title: "t",
        count: 3,
        ratio: 1.5,
        draft: false,
        labels: [],
        meta: {},
        nothing: null,
        either: 7,
        mode: "fast",
        loose: { anything: true },
      }),
    ).toEqual({ ok: true });
  });

  test("reports a missing required property", () => {
    const out = validateArgs(schema, { count: 1 });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toEqual(['missing required property "title"']);
  });

  test("an explicit undefined does not satisfy required", () => {
    const out = validateArgs(schema, { title: undefined });
    expect(out.ok).toBe(false);
  });

  test.each([
    [{ title: 1 }, 'property "title" must be string, got number'],
    [{ title: "t", count: 1.5 }, 'property "count" must be integer, got number'],
    [{ title: "t", ratio: "1.5" }, 'property "ratio" must be number, got string'],
    [{ title: "t", draft: "yes" }, 'property "draft" must be boolean, got string'],
    [{ title: "t", labels: {} }, 'property "labels" must be array, got object'],
    [{ title: "t", meta: [] }, 'property "meta" must be object, got array'],
    [{ title: "t", nothing: 0 }, 'property "nothing" must be null, got number'],
    [{ title: "t", either: true }, 'property "either" must be string or number, got boolean'],
  ])("rejects %o", (args, problem) => {
    const out = validateArgs(schema, args as Record<string, unknown>);
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toContain(problem);
  });

  test("an integer satisfies number, and null is not object", () => {
    expect(validateArgs(schema, { title: "t", ratio: 3 })).toEqual({ ok: true });
    const out = validateArgs(schema, { title: "t", meta: null });
    expect(out.ok).toBe(false);
  });

  test("enforces enum on top-level properties", () => {
    const out = validateArgs(schema, { title: "t", mode: "medium" });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toEqual(['property "mode" must be one of "fast", "slow"']);
  });

  test("a type error suppresses the enum complaint for the same property", () => {
    const out = validateArgs(schema, { title: "t", mode: 3 });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toHaveLength(1);
    expect(out.problems[0]).toContain("must be string");
  });

  test("additionalProperties:false rejects an invented property", () => {
    const out = validateArgs(schema, { title: "t", invented: 1 });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems[0]).toContain('unexpected property "invented"');
    expect(out.problems[0]).toContain("title");
  });

  test("extra properties are allowed when additionalProperties is not false", () => {
    const open = { type: "object", properties: { a: { type: "string" } } };
    expect(validateArgs(open, { a: "x", b: 2 })).toEqual({ ok: true });
  });

  test("collects every problem at once", () => {
    const out = validateArgs(schema, { count: "no", invented: true });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toHaveLength(3);
  });

  test("a schema with nothing to check accepts anything", () => {
    expect(validateArgs({}, { anything: 1 })).toEqual({ ok: true });
    expect(validateArgs({ type: "object" }, { anything: 1 })).toEqual({ ok: true });
    expect(validateArgs(undefined, { anything: 1 })).toEqual({ ok: true });
    expect(validateArgs(null, {})).toEqual({ ok: true });
    expect(validateArgs("not a schema", {})).toEqual({ ok: true });
  });

  test("unknown or unusable type keywords are skipped, not failed", () => {
    // "any" is not a JSON Schema type; deeper keywords are the tool's business.
    const odd = {
      type: "object",
      properties: { a: { type: "any" }, b: { type: ["weird"] }, c: { minimum: 5 } },
    };
    expect(validateArgs(odd, { a: 1, b: "x", c: 0 })).toEqual({ ok: true });
  });

  test("patternProperties: a key matching a pattern is not 'unexpected'", () => {
    // additionalProperties:false + patternProperties is a normal pairing;
    // enforcing against `properties` alone rejects arguments the schema
    // explicitly allows — a false negative that blocks a valid call.
    const patterned = {
      type: "object",
      properties: { name: { type: "string" } },
      patternProperties: { "^x-": { type: "string" } },
      additionalProperties: false,
    };
    expect(validateArgs(patterned, { name: "n", "x-trace": "abc" })).toEqual({ ok: true });
    const out = validateArgs(patterned, { name: "n", nope: 1 });
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems[0]).toContain('unexpected property "nope"');
  });

  test("patternProperties: an unparseable pattern stops the check rather than failing valid args", () => {
    const broken = {
      type: "object",
      properties: {},
      patternProperties: { "([": { type: "string" } },
      additionalProperties: false,
    };
    expect(validateArgs(broken, { anything: 1 })).toEqual({ ok: true });
  });

  test("enum members compare by canonical JSON, so object key order does not matter", () => {
    const objectEnum = {
      type: "object",
      properties: { where: { type: "object", enum: [{ a: 1, b: { c: 2, d: 3 } }] } },
    };
    expect(validateArgs(objectEnum, { where: { b: { d: 3, c: 2 }, a: 1 } })).toEqual({ ok: true });
    const out = validateArgs(objectEnum, { where: { a: 1, b: { c: 2, d: 4 } } });
    expect(out.ok).toBe(false);
  });

  test("enum members compare array order strictly (order is meaning there)", () => {
    const arrayEnum = { type: "object", properties: { xs: { type: "array", enum: [[1, 2]] } } };
    expect(validateArgs(arrayEnum, { xs: [1, 2] })).toEqual({ ok: true });
    expect(validateArgs(arrayEnum, { xs: [2, 1] }).ok).toBe(false);
  });

  test("nested shape is not checked (deliberately shallow)", () => {
    const nested = {
      type: "object",
      properties: { outer: { type: "object", properties: { inner: { type: "string" } } } },
    };
    expect(validateArgs(nested, { outer: { inner: 42 } })).toEqual({ ok: true });
  });
});

// ---------------------------------------------------------------------------
// DeferredToolRegistry
// ---------------------------------------------------------------------------

const SCHEMA = {
  type: "object",
  properties: { title: { type: "string" } },
  required: ["title"],
  additionalProperties: false,
};

function registry(
  opts: { entries?: CatalogEntry[]; tools?: Tool[]; failure?: Error; headNames?: string[] } = {},
): { reg: DeferredToolRegistry; catalog: FakeCatalog } {
  const catalog = new FakeCatalog(opts.entries ?? [entry("mcp__srv__file", SCHEMA)], opts.failure);
  const map = new Map<string, Tool>();
  for (const t of opts.tools ?? []) map.set(t.name, t);
  return {
    reg: new DeferredToolRegistry({
      catalog,
      tools: map,
      ...(opts.headNames ? { headNames: new Set(opts.headNames) } : {}),
    }),
    catalog,
  };
}

const builtinEntry = (name: string): CatalogEntry => ({
  name,
  description: `${name} description`,
  source: "builtin",
  parameters: { type: "object" },
});

describe("DeferredToolRegistry", () => {
  test("executes a known tool with valid args and passes the context through", async () => {
    const seen: ToolContext[] = [];
    const target = tool("mcp__srv__file", {
      execute: async (args, c) => {
        seen.push(c);
        return { text: `ran with ${JSON.stringify(args)}` };
      },
    });
    const { reg } = registry({ tools: [target] });

    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(res.text).toBe('ran with {"title":"x"}');
    expect(seen[0]).toBe(ctx);
    expect(reg.has("mcp__srv__file")).toBe(true);
  });

  test("an unknown name points at tool_search", async () => {
    const { reg } = registry({ tools: [] });
    const res = await reg.call("nope", {}, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown deferred tool nope");
    expect(res.text).toContain("tool_search");
    expect(reg.has("nope")).toBe(false);
  });

  test("a catalog name with no executable reads as temporary, not as 'no such tool'", async () => {
    // The catalog outlives a connection on purpose: a server going down must
    // not delete its tools from the model's world.
    const { reg } = registry({ tools: [] });
    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("not executable right now");
    expect(res.text).toContain("offline");
    expect(res.text).not.toContain("unknown deferred tool");
  });

  test("a HEADER tool is not 'offline': it says call it directly", async () => {
    // The catalog is tenant-wide and upsertBuiltins never withdraws a row, so
    // every always-on built-in is catalogued here too. Answering "its server
    // is offline" for `read` is false and invites retries forever.
    const { reg } = registry({
      entries: [builtinEntry("read")],
      tools: [],
      headNames: ["read", "grep"],
    });
    const res = await reg.call("read", {}, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toBe("read is already in your tool list; call it directly, not through tool_call");
    expect(res.text).not.toContain("offline");
    expect(reg.isHeadTool("read")).toBe(true);
    expect(reg.isHeadTool("mcp__srv__file")).toBe(false);
  });

  test("a header tool wins over the offline wording even when it came from a server", async () => {
    // An MCP tool forced into the header by `tools.alwaysOn` is runnable
    // directly; it is not in the deferred map, and it is certainly not down.
    const { reg } = registry({ tools: [], headNames: ["mcp__srv__file"] });
    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res.text).toContain("already in your tool list");
  });

  test("a built-in that is NOT on this surface says so, and does not invite a retry", async () => {
    // `bash` is catalogued by the one surface that runs with --shell. Here it
    // is simply absent, and "try again shortly" would be a lie.
    const { reg } = registry({ entries: [builtinEntry("bash")], tools: [], headNames: ["read"] });
    const res = await reg.call("bash", { command: "ls" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("bash is not available on this surface");
    expect(res.text).toContain("tool_search");
    expect(res.text).not.toContain("offline");
    expect(res.text).not.toContain("Try again");
  });

  test("without headNames the built-in case still avoids the offline claim", async () => {
    const { reg } = registry({ entries: [builtinEntry("read")], tools: [] });
    const res = await reg.call("read", {}, ctx);
    expect(res.text).toContain("not available on this surface");
    expect(reg.isHeadTool("read")).toBe(false);
  });

  test("bad arguments come back WITH the full schema (no describe round trip)", async () => {
    const { reg } = registry({ tools: [tool("mcp__srv__file")] });
    const res = await reg.call("mcp__srv__file", { nope: 1 }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain('missing required property "title"');
    expect(res.text).toContain('unexpected property "nope"');
    expect(res.text).toContain("```json");
    expect(res.text).toContain('"required"');
    expect(res.text).toContain('"additionalProperties": false');
    expect(res.text).toContain("tool_call again");
  });

  test("validates against the CATALOG schema, which is what tool_describe showed", async () => {
    // The executable's own parameters are a fallback, not the source of truth:
    // the model corrected itself against what the catalog printed.
    const { reg } = registry({
      entries: [entry("mcp__srv__file", SCHEMA)],
      tools: [tool("mcp__srv__file", { parameters: { type: "object" } })],
    });
    const res = await reg.call("mcp__srv__file", {}, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain('missing required property "title"');
  });

  test("falls back to the tool's own schema when the catalog has not caught up", async () => {
    const { reg } = registry({
      entries: [],
      tools: [tool("fresh", { parameters: SCHEMA })],
    });
    const bad = await reg.call("fresh", {}, ctx);
    expect(bad.isError).toBe(true);
    expect(bad.text).toContain('missing required property "title"');
    const good = await reg.call("fresh", { title: "t" }, ctx);
    expect(good.text).toBe("fresh ran");
  });

  test("a throwing tool becomes an isError, never a throw", async () => {
    const { reg } = registry({
      tools: [
        tool("mcp__srv__file", {
          execute: async () => {
            throw new Error("socket hung up");
          },
        }),
      ],
    });
    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toBe("mcp__srv__file threw: socket hung up");
  });

  test("a throwing catalog becomes an isError too", async () => {
    const { reg } = registry({ failure: new Error("db down"), tools: [tool("mcp__srv__file")] });
    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("tool catalog unavailable: db down");
  });

  test("an isError from the tool passes through unchanged", async () => {
    const { reg } = registry({
      tools: [tool("mcp__srv__file", { execute: async () => ({ text: "server said no", isError: true }) })],
    });
    const res = await reg.call("mcp__srv__file", { title: "x" }, ctx);
    expect(res).toEqual({ text: "server said no", isError: true });
  });

  test("exposes the catalog for the meta-tools to search", () => {
    const catalog = new FakeCatalog([]);
    const reg = new DeferredToolRegistry({ catalog, tools: new Map() });
    expect(reg.catalog).toBe(catalog);
  });
});

// ---------------------------------------------------------------------------
// The system prompt's one static sentence (§4.5/§9 cached prefix)
// ---------------------------------------------------------------------------

describe("buildSystemPrompt deferred-tools line", () => {
  const DEFERRED_LINE =
    "Some tools are deferred rather than listed here: `tool_search` finds them, " +
    "`tool_describe` shows a schema, `tool_call` runs one. Record the ones you are using " +
    "in your continuity document's working set.";

  test("names the meta-tools once, under ## Tools", () => {
    const prompt = buildSystemPrompt({ agentId: "pinky", nodeId: "node1", tools: [] });
    expect(prompt).toContain(DEFERRED_LINE);
    const toolsSection = prompt.slice(prompt.indexOf("## Tools"));
    expect(toolsSection).toContain(DEFERRED_LINE);
  });

  test("is byte-identical no matter what the catalog holds", () => {
    // buildSystemPrompt takes no catalog at all — that IS the guarantee. Two
    // runs with the same head tools and different deferred catalogs cannot
    // produce different prefixes, because the catalog is not an input.
    const head = [{ name: "read", description: "Read a file.", parameters: {} }];
    const a = buildSystemPrompt({ agentId: "pinky", nodeId: "node1", tools: head });
    const b = buildSystemPrompt({ agentId: "pinky", nodeId: "node1", tools: head });
    expect(a).toBe(b);
    expect(a).toContain("- read: Read a file.");
  });

  test("mentions no catalog tool name and no count", () => {
    const prompt = buildSystemPrompt({ agentId: "pinky", nodeId: "node1", tools: [] });
    expect(prompt).not.toContain("mcp__");
    expect(prompt).not.toMatch(/\d+ (deferred )?tools/);
  });
});
