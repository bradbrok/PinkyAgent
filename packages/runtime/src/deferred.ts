/**
 * The deferred-tool plane (slice 9): which tools live in the request header,
 * and how the ones that do not get executed.
 *
 * Why this exists: tool schemas render at prefix position 0 (tools → system →
 * messages), so the tool list IS the head of every provider cache key. A
 * hundred MCP tools in the header is a bill paid on every single request, and
 * — worse — a list that CHANGES (a server reconnects, an agent enables a tool)
 * invalidates every cache tier for the whole thread. DESIGN.md §4.5/§9 already
 * says the tool set is masked, never mutated mid-window; this module is the
 * same rule applied to the catalog: loading a tool never rewrites the header,
 * a loaded schema is APPENDED to the conversation as an ordinary tool result.
 *
 * Two pieces:
 *   - `partitionTools` — pure split of a tool array into `head` (rendered in
 *     the request) and `deferred` (reachable only through the meta-tools),
 *     decided by the settings overlay so a human (or a delegated agent, P8
 *     revised) chooses what the header costs.
 *   - `DeferredToolRegistry` — the dispatcher behind `tool_call`: catalog
 *     lookup, a minimal argument check whose failure hands the model the full
 *     schema, then execution with every throw folded into an isError.
 */
import { canonicalizeArgs } from "@pinky/core";
import type { SettingsSnapshot } from "@pinky/core";
import { SHED_CONTEXT_TOOL_NAME } from "./continuity";
import type { CatalogEntry, DeferredTools, Tool, ToolCatalogView, ToolContext, ToolResult } from "./types";

/**
 * The three fixed meta-tools. They are the ONLY door to the catalog, so they
 * are always in the header and can never be deferred — deferring the search
 * tool would leave the model with a door it cannot find. `shed_context` is
 * always in the header for the same structural reason (the hard boundary
 * forces a call to it by name; DESIGN.md §4.1).
 */
export const META_TOOL_NAMES = ["tool_search", "tool_describe", "tool_call"] as const;

/** Names no setting can move out of the header. */
const PINNED_TO_HEAD: ReadonlySet<string> = new Set<string>([
  ...META_TOOL_NAMES,
  SHED_CONTEXT_TOOL_NAME,
]);

/** Where a tool came from; picks which `defaultMode` applies. */
export type ToolSource = "builtin" | "mcp";

export interface ToolPartition {
  /** Rendered in the request's `tools` array — the cached prefix. */
  head: Tool[];
  /** Reachable only via tool_search / tool_describe / tool_call. */
  deferred: Tool[];
}

/**
 * Code-unit order, not locale order. `localeCompare` disagrees with itself
 * across servers and ICU builds (the collation flake in `da33d0e`), and this
 * order is a cache key: two runs that sort the same tools differently send two
 * different prefixes.
 */
function byName(a: Tool, b: Tool): number {
  return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
}

/**
 * Split tools into the header set and the catalog set (pure).
 *
 * Precedence, highest first:
 *   1. the pinned names above (meta-tools + shed_context) — always head;
 *   2. `tools.alwaysOn` — an exact name a human/agent forced into the header;
 *   3. `tools.deferred` — an exact name forced out of it;
 *   4. `tools.defaultMode[source]` — built-ins default to "always" (they are a
 *      handful and every surface uses them), MCP tools to "deferred" (there
 *      may be hundreds and they change under us).
 *
 * A name listed in the settings that matches no tool here is simply ignored:
 * the settings overlay is written against a catalog that may include servers
 * this surface has not connected, and a stale name must not be an error.
 */
export function partitionTools(
  tools: Tool[],
  cfg: SettingsSnapshot["tools"],
  sourceOf: (name: string) => ToolSource,
): ToolPartition {
  const alwaysOn = new Set(cfg.alwaysOn);
  const deferredNames = new Set(cfg.deferred);
  const head: Tool[] = [];
  const deferred: Tool[] = [];

  for (const tool of tools) {
    let inHead: boolean;
    if (PINNED_TO_HEAD.has(tool.name)) inHead = true;
    else if (alwaysOn.has(tool.name)) inHead = true;
    else if (deferredNames.has(tool.name)) inHead = false;
    else inHead = cfg.defaultMode[sourceOf(tool.name)] === "always";
    (inHead ? head : deferred).push(tool);
  }

  head.sort(byName);
  deferred.sort(byName);
  return { head, deferred };
}

// ---------------------------------------------------------------------------
// Minimal argument validation
// ---------------------------------------------------------------------------

export type ArgsValidation = { ok: true } | { ok: false; problems: string[] };

/** JSON Schema type names this checker understands; anything else is skipped. */
const KNOWN_TYPES = new Set(["string", "number", "integer", "boolean", "array", "object", "null"]);

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** The JSON Schema type names a value satisfies. */
function typesOf(value: unknown): string[] {
  if (value === null) return ["null"];
  if (Array.isArray(value)) return ["array"];
  switch (typeof value) {
    case "string":
      return ["string"];
    case "boolean":
      return ["boolean"];
    case "number":
      return Number.isInteger(value) ? ["number", "integer"] : ["number"];
    case "object":
      return ["object"];
    default:
      return [];
  }
}

/** `type` as a list of names we can check, or null when there is nothing to check. */
function declaredTypes(schema: Record<string, unknown>): string[] | null {
  const raw = schema["type"];
  const list = typeof raw === "string" ? [raw] : Array.isArray(raw) ? raw : null;
  if (list === null) return null;
  const usable = list.filter((t): t is string => typeof t === "string" && KNOWN_TYPES.has(t));
  return usable.length > 0 ? usable : null;
}

/**
 * Enum membership. Objects compare by CANONICAL JSON (keys sorted recursively,
 * core's own canonicalizer) rather than raw JSON: a model that writes
 * `{b:1,a:2}` where the schema's enum lists `{a:2,b:1}` has written the same
 * value, and rejecting it would be a false negative that blocks a valid call.
 * Postgres reorders jsonb keys on the way out anyway (see canonicalizeArgs),
 * so a schema round-tripped through the catalog cannot be trusted to hold the
 * author's key order either.
 */
function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  return JSON.stringify(canonicalizeArgs(a)) === JSON.stringify(canonicalizeArgs(b));
}

/**
 * Compiled `patternProperties` keys, or null when the schema has none.
 * An uncompilable pattern returns null too: "I cannot tell which keys this
 * schema allows" must degrade to allowing them, never to rejecting a valid
 * call (JSON Schema regexes are ECMA-262, but a server can still ship junk).
 */
function patternKeys(schema: Record<string, unknown>): RegExp[] | null {
  const patterns = schema["patternProperties"];
  if (!isRecord(patterns)) return null;
  const keys = Object.keys(patterns);
  if (keys.length === 0) return null;
  const out: RegExp[] = [];
  for (const key of keys) {
    try {
      out.push(new RegExp(key, "u"));
    } catch {
      try {
        out.push(new RegExp(key));
      } catch {
        return null; // unparseable: stop enforcing additionalProperties
      }
    }
  }
  return out;
}

/**
 * A deliberately small JSON-Schema check: required keys, top-level property
 * types, top-level enums, and `additionalProperties: false`.
 *
 * Not a validator — the repo has exactly one runtime dependency and this is
 * not the place to spend the second one. It exists to catch the mistakes a
 * model actually makes against a schema it has just read (a missing required
 * key, a number typed as a string, an invented property, a value outside an
 * enum) so `tool_call` can answer with the schema instead of forwarding
 * nonsense to somebody else's server. Anything deeper — nested objects,
 * formats, oneOf — is the tool's own business: it is closer to the semantics
 * and its error message will be better than ours.
 */
export function validateArgs(schema: unknown, args: Record<string, unknown>): ArgsValidation {
  if (!isRecord(schema)) return { ok: true };
  const problems: string[] = [];

  const required = schema["required"];
  if (Array.isArray(required)) {
    for (const key of required) {
      if (typeof key !== "string") continue;
      if (!(key in args) || args[key] === undefined) {
        problems.push(`missing required property "${key}"`);
      }
    }
  }

  const properties = isRecord(schema["properties"]) ? schema["properties"] : {};

  if (schema["additionalProperties"] === false) {
    const allowed = Object.keys(properties);
    // `patternProperties` names keys by regex, so `properties` is NOT the whole
    // allow-list: enforcing against it alone rejects arguments the schema
    // explicitly permits. Match the patterns; if they are unusable, skip the
    // check entirely rather than guess.
    const patterns = patternKeys(schema);
    const unusablePatterns = patterns === null && schema["patternProperties"] !== undefined;
    if (!unusablePatterns) {
      for (const key of Object.keys(args)) {
        if (allowed.includes(key)) continue;
        if (patterns?.some((re) => re.test(key))) continue;
        problems.push(
          `unexpected property "${key}" (this tool accepts only: ${allowed.join(", ") || "(no properties)"})`,
        );
      }
    }
  }

  for (const [key, value] of Object.entries(args)) {
    if (value === undefined) continue;
    const propSchema = properties[key];
    if (!isRecord(propSchema)) continue;

    const types = declaredTypes(propSchema);
    if (types !== null) {
      const actual = typesOf(value);
      if (!types.some((t) => actual.includes(t))) {
        problems.push(
          `property "${key}" must be ${types.join(" or ")}, got ${actual[0] ?? typeof value}`,
        );
        continue; // an enum complaint on top of a type error is just noise
      }
    }

    const enumValues = propSchema["enum"];
    if (Array.isArray(enumValues) && !enumValues.some((candidate) => sameValue(candidate, value))) {
      problems.push(
        `property "${key}" must be one of ${enumValues.map((v) => JSON.stringify(v)).join(", ")}`,
      );
    }
  }

  return problems.length === 0 ? { ok: true } : { ok: false, problems };
}

// ---------------------------------------------------------------------------
// The registry behind tool_call
// ---------------------------------------------------------------------------

export interface DeferredToolRegistryOptions {
  /** Names, descriptions and schemas — the Postgres catalog in production. */
  catalog: ToolCatalogView;
  /** Executable tools keyed by catalog name (deferred built-ins + MCP tools). */
  tools: Map<string, Tool>;
  /**
   * The names in THIS run's header partition (`partitionTools().head`).
   *
   * The catalog is tenant-wide and never withdraws a built-in — `bash` stays
   * catalogued from the one surface that runs with `--shell`, and every
   * always-on tool is in there too. Without this set, `tool_call("read")` and
   * `tool_call("bash")` both come back as "its server is offline", which is
   * false twice over: one is a tool the model already has, the other is a tool
   * this surface deliberately does not have. Both invite retries that can
   * never succeed. Optional; absent means the older, vaguer wording.
   */
  headNames?: ReadonlySet<string>;
}

function fail(text: string): ToolResult {
  return { text, isError: true };
}

function fence(schema: unknown): string {
  return "```json\n" + JSON.stringify(schema, null, 2) + "\n```";
}

/**
 * Dispatch for tools that are NOT in the header (DESIGN.md §9).
 *
 * The catalog and the executable map are two different things on purpose. The
 * catalog is durable — it survives an MCP server being down, because a tool
 * list that flaps with a connection would make the model's world flicker. The
 * map is whatever this process can actually run right now. So a name can be
 * known and unrunnable, and that has to read as a temporary condition ("its
 * server is offline") rather than "no such tool", or the model will stop
 * looking for it forever.
 *
 * Nothing here throws. A deferred tool is reached through a tool result, and a
 * tool result's failure mode is text the model can act on.
 */
export class DeferredToolRegistry implements DeferredTools {
  readonly catalog: ToolCatalogView;
  private readonly tools: Map<string, Tool>;
  private readonly headNames: ReadonlySet<string>;

  constructor(opts: DeferredToolRegistryOptions) {
    this.catalog = opts.catalog;
    this.tools = opts.tools;
    this.headNames = opts.headNames ?? new Set<string>();
  }

  /** Whether this process can execute `name` right now. */
  has(name: string): boolean {
    return this.tools.has(name);
  }

  /**
   * Whether `name` is already in this run's request header (so the model
   * should call it directly). Structural, not part of the `DeferredTools`
   * interface: `tool_describe` feature-detects it to add a one-line note.
   */
  isHeadTool(name: string): boolean {
    return this.headNames.has(name);
  }

  async call(name: string, args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    let entry: CatalogEntry | null = null;
    try {
      entry = await this.catalog.describe(name);
    } catch (err) {
      return fail(`tool catalog unavailable: ${message(err)}`);
    }

    const tool = this.tools.get(name);
    if (!entry && !tool) {
      return fail(`unknown deferred tool ${name}; use tool_search to find the tool you want`);
    }
    if (!tool) {
      // Three different "not runnable" cases, three different next moves. The
      // offline wording is only ever right for an MCP tool: a built-in is
      // either already in the header (call it directly) or genuinely absent
      // from this surface (never retry) — and the catalog outlives both, since
      // upsertBuiltins never withdraws a row another surface wrote.
      if (this.headNames.has(name)) {
        return fail(
          `${name} is already in your tool list; call it directly, not through tool_call`,
        );
      }
      if (entry && entry.source === "mcp") {
        return fail(
          `${name} is in the tool catalog but is not executable right now — its server is offline or ` +
            "has not finished connecting. Try again shortly, or tool_search for another tool.",
        );
      }
      return fail(
        `${name} is not available on this surface (the tool catalog lists it for other surfaces). ` +
          "Use tool_search to find something you can run here.",
      );
    }

    // Prefer the catalog's schema (what tool_describe showed the model); fall
    // back to the tool's own when the catalog has not caught up with it yet.
    const schema = entry?.parameters ?? tool.parameters;

    const check = validateArgs(schema, args);
    if (!check.ok) {
      // The full schema, inline. A failed call that answers "wrong arguments"
      // costs a tool_describe round trip to fix; one that answers with the
      // schema costs nothing but the tokens already being spent on the error.
      return fail(
        `invalid arguments for ${name}: ${check.problems.join("; ")}\n\n` +
          `schema for ${name}:\n${fence(schema)}\n\nCall tool_call again with corrected args.`,
      );
    }

    try {
      return await tool.execute(args, ctx);
    } catch (err) {
      return fail(`${name} threw: ${message(err)}`);
    }
  }
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
