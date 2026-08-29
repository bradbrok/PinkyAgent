/**
 * The three meta-tools: the model's door to every tool that is NOT in the
 * request header (slice 9, DESIGN.md §9).
 *
 * `tool_search` finds a tool, `tool_describe` shows its schema, `tool_call`
 * runs it. Their own schemas are tiny and never change, which is the whole
 * design: the header holds always-on tools plus these three and stays
 * byte-identical from wake to wake, while the catalog behind them can grow to
 * hundreds of tools and churn as MCP servers come and go. A tool "loaded" this
 * way arrives as an ordinary tool result — appended to the conversation, never
 * a rewrite of the cached prefix (§4.5/§9: masked, not mutated).
 *
 * All three are registered unconditionally (see ./index.ts) and degrade to a
 * clean isError without `ctx.deferred`, exactly like the memory and settings
 * tools do without their planes. A surface with no catalog pays three unusable
 * tool descriptions; what it does NOT pay is a header whose contents depend on
 * whether a server happened to be up when the process started.
 */
import { MAX_TOOL_SEARCH_LIMIT } from "@pinky/core";
import type { CatalogHit, Tool, ToolContext, ToolResult } from "@pinky/runtime";

/** Same answer from all three, so "this surface has no catalog" is one string. */
const NO_DEFERRED = "no deferred tools on this surface";

/** Fallback page size when the run carries no settings snapshot. */
const DEFAULT_SEARCH_LIMIT = 8;

/**
 * Size caps for `tool_describe` output.
 *
 * A catalog row's `description` and `parameters` are whatever an MCP server
 * sent us, and they land in the conversation as a tool result — i.e. in the
 * context window, on every subsequent turn of the window, and in the token
 * bill. A server with a megabyte description would otherwise blow one turn's
 * budget (and could shove the pressure ladder into a forced shed) with a
 * single describe. Cap both, and say so, so the model knows it is looking at a
 * truncated schema rather than a complete one.
 */
const MAX_DESCRIPTION_CHARS = 4_000;
const MAX_SCHEMA_BYTES = 16 * 1024;

const UTF8 = new TextEncoder();

function byteLength(text: string): number {
  return UTF8.encode(text).length;
}

function truncationNote(bytes: number): string {
  return `\n…[truncated ${bytes} bytes]`;
}

/** Cap by characters; the note is in bytes, which is what actually costs. */
function capChars(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + truncationNote(byteLength(text.slice(max)));
}

/**
 * Cap by UTF-8 bytes, cutting on a code-point boundary (a split code point
 * would come back as U+FFFD and, in a JSON block, as a lie about the schema's
 * contents). The result is deliberately not re-parseable — a truncated schema
 * IS incomplete, and the note says so rather than pretending otherwise.
 */
function capBytes(text: string, max: number): string {
  const total = byteLength(text);
  if (total <= max) return text;
  let kept = "";
  let used = 0;
  for (const ch of text) {
    const size = byteLength(ch);
    if (used + size > max) break;
    kept += ch;
    used += size;
  }
  return kept + truncationNote(total - used);
}

function fail(text: string): ToolResult {
  return { text, isError: true };
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A required non-empty string argument, or the error to hand back. */
function requireText(args: Record<string, unknown>, key: string): string | { error: string } {
  const value = args[key];
  if (typeof value !== "string" || value.trim() === "") {
    return { error: `'${key}' must be a non-empty string` };
  }
  return value.trim();
}

function isError(v: unknown): v is { error: string } {
  return typeof v === "object" && v !== null && typeof (v as { error?: unknown }).error === "string";
}

/**
 * Whether the deferred plane says this name is already in the request header.
 *
 * Feature-detected rather than required: `DeferredTools` is the cross-package
 * contract (runtime/types.ts) and only `DeferredToolRegistry` knows the run's
 * head partition. An implementation without the method simply gets no note.
 */
function isHeadTool(ctx: ToolContext, name: string): boolean {
  const probe = ctx.deferred as { isHeadTool?: (n: string) => boolean } | undefined;
  return typeof probe?.isHeadTool === "function" && probe.isHeadTool(name) === true;
}

// ---------------------------------------------------------------------------
// tool_search
// ---------------------------------------------------------------------------

export class ToolSearchTool implements Tool {
  readonly name = "tool_search";
  readonly description =
    "Find tools that are not listed above — most tools live in a searchable catalog, not in this list.\n" +
    "Search by what you want to do (\"create a github issue\", \"read a spreadsheet\"), not by a name you " +
    "guessed. Results are name + one-line description; tool_describe shows a schema and tool_call runs one. " +
    "An empty query lists the catalog by name.";
  readonly parameters = {
    type: "object",
    properties: {
      query: {
        type: "string",
        description:
          'What the tool should do, in words — e.g. "send a slack message". Empty lists everything.',
      },
      limit: {
        type: "integer",
        minimum: 1,
        maximum: MAX_TOOL_SEARCH_LIMIT,
        description: "How many results to return. Defaults to the configured page size.",
      },
    },
    required: ["query"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (!ctx.deferred) return fail(`${this.name}: ${NO_DEFERRED}`);

    // An empty string is a legitimate query here (it lists the catalog), so
    // this is a type check, not requireText.
    if (args.query !== undefined && typeof args.query !== "string") {
      return fail(`${this.name}: 'query' must be a string`);
    }
    const query = typeof args.query === "string" ? args.query.trim() : "";

    let limit = ctx.settings?.tools.searchLimit ?? DEFAULT_SEARCH_LIMIT;
    if (args.limit !== undefined) {
      if (
        typeof args.limit !== "number" ||
        !Number.isInteger(args.limit) ||
        args.limit < 1 ||
        args.limit > MAX_TOOL_SEARCH_LIMIT
      ) {
        return fail(`${this.name}: 'limit' must be an integer between 1 and ${MAX_TOOL_SEARCH_LIMIT}`);
      }
      limit = args.limit;
    }

    let hits: CatalogHit[];
    try {
      hits = await ctx.deferred.catalog.search(query, limit);
    } catch (err) {
      return fail(`${this.name}: catalog search failed: ${message(err)}`);
    }

    if (hits.length === 0) {
      return {
        text: query
          ? `no matches for "${query}". Try different words, or an empty query to list every tool.`
          : "no matches: the tool catalog is empty on this surface.",
      };
    }

    const lines = hits.map((hit, i) => `${i + 1}. ${hit.name} — ${hit.description}`);
    lines.push("");
    lines.push("tool_describe <name> for its schema; tool_call to run it.");
    return { text: lines.join("\n") };
  }
}

// ---------------------------------------------------------------------------
// tool_describe
// ---------------------------------------------------------------------------

export class ToolDescribeTool implements Tool {
  readonly name = "tool_describe";
  readonly description =
    "Show one catalog tool's full description and argument schema, so you can call it correctly.\n" +
    "Use it on a name tool_search returned. The schema is JSON Schema, the same shape as the tools listed " +
    "above; pass its properties as the `args` object of tool_call.";
  readonly parameters = {
    type: "object",
    properties: {
      name: {
        type: "string",
        description: "The exact tool name from tool_search, e.g. \"mcp__github__create_issue\".",
      },
    },
    required: ["name"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (!ctx.deferred) return fail(`${this.name}: ${NO_DEFERRED}`);

    const name = requireText(args, "name");
    if (isError(name)) return fail(`${this.name}: ${name.error}`);

    try {
      const entry = await ctx.deferred.catalog.describe(name);
      if (!entry) {
        return fail(`${this.name}: unknown tool '${name}'; use tool_search to find the right name`);
      }
      const origin = entry.server ? ` (mcp server: ${entry.server})` : "";
      // The catalog is tenant-wide, so it also lists tools that are in THIS
      // run's header. Describing one is fine; calling it through tool_call is
      // a detour the model should not take, and it costs a wasted turn to
      // learn that from the error instead.
      const closing = isHeadTool(ctx, entry.name)
        ? `${entry.name} is already in your tool list; call it directly, not through tool_call.`
        : `Run it with tool_call: { "name": "${entry.name}", "args": { ... } }`;
      return {
        text:
          `${entry.name}${origin}\n${capChars(entry.description, MAX_DESCRIPTION_CHARS)}\n\n` +
          "```json\n" +
          `${capBytes(JSON.stringify(entry.parameters, null, 2), MAX_SCHEMA_BYTES)}\n` +
          "```\n" +
          closing,
      };
    } catch (err) {
      return fail(`${this.name}: catalog lookup failed: ${message(err)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// tool_call
// ---------------------------------------------------------------------------

export class ToolCallTool implements Tool {
  readonly name = "tool_call";
  readonly description =
    "Run a catalog tool by name with an arguments object.\n" +
    "The name comes from tool_search; the arguments follow the schema tool_describe printed. Arguments that " +
    "do not fit the schema come back as an error containing the schema, so you can correct them without " +
    "another describe. The result is whatever that tool returns.";
  readonly parameters = {
    type: "object",
    properties: {
      name: {
        type: "string",
        description: "The exact tool name from tool_search.",
      },
      args: {
        type: "object",
        description:
          "The tool's arguments, matching its schema. Pass {} for a tool that takes none.",
      },
    },
    required: ["name", "args"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (!ctx.deferred) return fail(`${this.name}: ${NO_DEFERRED}`);

    const name = requireText(args, "name");
    if (isError(name)) return fail(`${this.name}: ${name.error}`);

    const raw = args.args;
    if (raw !== undefined && (typeof raw !== "object" || raw === null || Array.isArray(raw))) {
      return fail(`${this.name}: 'args' must be a JSON object (use {} for a tool with no arguments)`);
    }
    // Missing `args` means "no arguments" rather than an error: the schema
    // requires it, but a tool that takes none is exactly where a model forgets.
    const toolArgs = (raw as Record<string, unknown> | undefined) ?? {};

    try {
      return await ctx.deferred.call(name, toolArgs, ctx);
    } catch (err) {
      // The registry folds throws into isError itself; this is the belt for a
      // custom DeferredTools implementation that does not.
      return fail(`${this.name}: ${name} failed: ${message(err)}`);
    }
  }
}
