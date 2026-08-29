/**
 * Pure naming + hashing for the MCP plane (slice 9).
 *
 * Nothing here touches the network, the clock or the database: every function
 * is a deterministic transform, because two different processes (the manager
 * syncing a catalog and a CLI printing it) MUST agree on the final name of a
 * tool without talking to each other.
 *
 * Three rules the rest of the package leans on:
 *
 * 1. **A tool's final name is `mcp__<server>__<raw>`** — `<server>` is the
 *    settings key (validated by core's settings layer), `<raw>` the server's
 *    own spelling sanitized to what every provider accepts. Providers cap tool
 *    names at 64 characters, so an over-long name is truncated and stamped
 *    with 8 hex of sha256(server + NUL + raw): collisions stop being possible
 *    the moment two raw names differ, and the result is still deterministic.
 * 2. **Sort order is code-unit compare on the final name.** Never
 *    `localeCompare`: `tools/list` results SHOULD be in deterministic order
 *    "to improve LLM prompt cache hit rates" (MCP 2026-07-28), and a
 *    locale-sensitive sort makes the order depend on the host's ICU data — the
 *    same class of bug as the Postgres collation flake in `da33d0e`.
 * 3. **A config hash is taken with `${ENV}` placeholders UNRESOLVED.** The
 *    hash is the catalog's trust token ("are these rows still describing the
 *    server I am configured to talk to?"). Hashing resolved secrets would put
 *    a token's value in the database, and would invalidate a whole catalog
 *    generation every time an unrelated credential rotated.
 */
import { createHash } from "node:crypto";
import type { McpServerConfig } from "@pinky/core";

/** Provider-side limit on a tool name (Anthropic/OpenAI both cap at 64). */
export const MAX_TOOL_NAME_LENGTH = 64;
/** Every MCP tool name starts with this; built-ins never do. */
export const MCP_NAME_PREFIX = "mcp__";
/** Separator between the server key and the raw tool name. */
export const MCP_NAME_SEPARATOR = "__";
/** Hex characters of sha256 appended when a name has to be truncated. */
export const NAME_HASH_CHARS = 8;
/**
 * Server keys, as core's settings validator spells them. Repeated here (not
 * imported) so this module stays free of a runtime dependency on the settings
 * layer — it is compared, never used to validate a write.
 */
export const SERVER_KEY_PATTERN = /^[a-z0-9][a-z0-9_-]{0,31}$/;

/**
 * ... and one rule the character class alone cannot express: a key may not
 * contain `__`, because `__` is the field separator in `mcp__<server>__<raw>`.
 * A key like `a__b` makes `mcp__a__b__tool` parse two ways, and
 * {@link splitMcpToolName} — which is how `call()` decides WHICH SERVER to
 * dispatch to — would silently route to the wrong one. Banning the sequence
 * is what makes the split exact rather than a documented guess.
 */
export const SERVER_KEY_FORBIDDEN = MCP_NAME_SEPARATOR;

/** Characters a provider will accept verbatim in a tool name. */
const SAFE_NAME_CHAR = /[^A-Za-z0-9_-]/g;

/** True when `key` is a legal MCP server key (the settings map's key). */
export function isValidServerKey(key: string): boolean {
  return SERVER_KEY_PATTERN.test(key) && !key.includes(SERVER_KEY_FORBIDDEN);
}

/**
 * The server's own tool name, reduced to `[A-Za-z0-9_-]`. Anything else — a
 * dot, a slash, a space, a non-ASCII letter — becomes `_`, one `_` per
 * character so the transform stays length-preserving and therefore reversible
 * enough to eyeball. An empty result becomes `_` (a nameless tool is still a
 * tool, and `mcp__srv__` must not be a valid handle).
 */
export function sanitizeRawToolName(raw: string): string {
  const sanitized = raw.replace(SAFE_NAME_CHAR, "_");
  return sanitized === "" ? "_" : sanitized;
}

/**
 * `mcp__<server>__<sanitized raw>`, truncated with a hash suffix when that
 * exceeds {@link MAX_TOOL_NAME_LENGTH}.
 *
 * The hash is taken over `server + "\0" + raw` — the ORIGINAL raw name, not
 * the sanitized one, so two tools that sanitize to the same string still get
 * different suffixes.
 */
export function mcpToolName(server: string, raw: string): string {
  assertServerKey(server);
  const full = `${MCP_NAME_PREFIX}${server}${MCP_NAME_SEPARATOR}${sanitizeRawToolName(raw)}`;
  if (full.length <= MAX_TOOL_NAME_LENGTH) return full;
  return hashedMcpToolName(server, raw);
}

/** The disambiguated spelling: always truncated-to-fit and stamped with 8 hex
 *  of sha256(server + NUL + raw). Used for an over-long name AND for the
 *  second of two raw names that sanitize to the same string — one mechanism,
 *  one guarantee (distinct raw names, distinct final names). */
export function hashedMcpToolName(server: string, raw: string): string {
  assertServerKey(server);
  const full = `${MCP_NAME_PREFIX}${server}${MCP_NAME_SEPARATOR}${sanitizeRawToolName(raw)}`;
  const suffix = `_${createHash("sha256").update(`${server}\0${raw}`).digest("hex").slice(0, NAME_HASH_CHARS)}`;
  const keep = Math.min(full.length, MAX_TOOL_NAME_LENGTH - suffix.length);
  return full.slice(0, keep) + suffix;
}

/** A bad server key is a programming error, not input: `start()` filters the
 *  settings map before anything reaches here, so a throw means a caller went
 *  around it. */
function assertServerKey(server: string): void {
  if (!isValidServerKey(server)) {
    throw new Error(
      `Invalid MCP server key ${JSON.stringify(server)}: expected ${SERVER_KEY_PATTERN} and no ${JSON.stringify(SERVER_KEY_FORBIDDEN)}`,
    );
  }
}

/**
 * Final names for one server's whole `tools/list`, with collisions resolved.
 *
 * Sanitizing is many-to-one (`a.b` and `a_b` both become `a_b`), so a single
 * generation can produce two identical final names. Dropping the second is
 * what the code used to do, and the model never learns the tool exists — it
 * simply is not in the catalog. Instead the loser gets the hashed spelling,
 * the same mechanism truncation already uses.
 *
 * Deterministic and order-independent: the raw names are sorted by code unit
 * first, so which of a colliding pair keeps the plain name does not depend on
 * the order the server happened to list them in (and therefore does not churn
 * the tool header between syncs).
 */
export function mcpToolNames(server: string, raws: string[]): { raw: string; name: string }[] {
  assertServerKey(server);
  const taken = new Set<string>();
  const seenRaw = new Set<string>();
  const out: { raw: string; name: string }[] = [];
  for (const raw of [...raws].sort(compareNames)) {
    // Two identical RAW names (a malformed server listing a tool twice) are
    // one tool: there is nothing left to distinguish them by, and hashing
    // would mint a second handle for the same call.
    if (seenRaw.has(raw)) continue;
    seenRaw.add(raw);
    let name = mcpToolName(server, raw);
    if (taken.has(name)) name = hashedMcpToolName(server, raw);
    if (taken.has(name)) continue;
    taken.add(name);
    out.push({ raw, name });
  }
  return out;
}

/**
 * The inverse, as far as an inverse exists: `mcp__srv__raw` -> `{ server:
 * "srv", raw: "raw" }`, and `null` for anything that is not an MCP name.
 *
 * TWO CAVEATS, both by construction, which is why the manager keeps its own
 * name -> raw map and only falls back to this:
 *
 *  - `raw` here is the SANITIZED spelling, and a truncated name has lost its
 *    tail entirely. `callTool` must send the server's own spelling, so the
 *    authority is the catalog row (`raw_name`), never this function.
 *  - the split is at the FIRST `__` after the prefix, which is EXACT because
 *    {@link isValidServerKey} bans `__` inside a key. `server` is therefore
 *    always the real one; only `raw` is the lossy half.
 */
export function splitMcpToolName(name: string): { server: string; raw: string } | null {
  if (!name.startsWith(MCP_NAME_PREFIX)) return null;
  const rest = name.slice(MCP_NAME_PREFIX.length);
  const at = rest.indexOf(MCP_NAME_SEPARATOR);
  if (at <= 0) return null;
  const server = rest.slice(0, at);
  const raw = rest.slice(at + MCP_NAME_SEPARATOR.length);
  if (server === "" || raw === "") return null;
  return { server, raw };
}

/**
 * Code-unit ordering — the single sort used for tool names everywhere in this
 * package. `<`/`>` on strings compare UTF-16 code units, which is exactly the
 * locale-independent order we want (and what Postgres `collate "C"` gives the
 * catalog on the other side of the wire).
 */
export function compareNames(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * The same object rebuilt with every object's keys in code-unit order,
 * recursively; arrays keep their order (an array's order is data).
 *
 * This is not cosmetic. A tool's `parameters` goes VERBATIM onto the wire as
 * the provider's `input_schema`, and the two sources it can come from disagree
 * about key order: Postgres `jsonb` re-sorts object keys (length, then bytes)
 * at every level, while a live `tools/list` preserves the server's spelling.
 * Without this, run 1 (served from the catalog) and run 2 (served from the
 * sync) send different BYTES for the same schema — a changed prefix at
 * position 0, i.e. a full provider-cache invalidation on every other wake.
 * Canonicalizing both paths makes the header stable.
 */
export function sortObjectKeysDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortObjectKeysDeep);
  if (typeof value !== "object" || value === null) return value;
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(record).sort(compareNames)) {
    const member = record[key];
    if (member === undefined) continue;
    out[key] = sortObjectKeysDeep(member);
  }
  return out;
}

/**
 * Canonical JSON: object keys sorted by code unit, `undefined` members
 * dropped, arrays left in order (an array's order is data). Used only as the
 * pre-image of {@link hashServerConfig}, so it does not need to be pretty —
 * it needs to be the same on every machine.
 */
export function canonicalJson(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const parts: string[] = [];
    for (const key of Object.keys(record).sort(compareNames)) {
      const member = record[key];
      if (member === undefined) continue;
      parts.push(`${JSON.stringify(key)}:${canonicalJson(member)}`);
    }
    return `{${parts.join(",")}}`;
  }
  if (typeof value === "undefined") return "null";
  return JSON.stringify(value) ?? "null";
}

/**
 * sha256 of the canonical JSON of a server's config, WITH `${ENV}`
 * placeholders left unresolved (see the module header). Stable across process
 * restarts, key order and credential rotation; changes the moment the command,
 * arguments, url or the NAME of an env placeholder changes.
 */
export function hashServerConfig(config: McpServerConfig): string {
  return createHash("sha256").update(canonicalJson(config)).digest("hex");
}

/** `${NAME}` — the whole value, not a substring: a half-interpolated secret is
 *  a footgun, and the settings validator documents the value form. */
const ENV_PLACEHOLDER = /^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$/;

/**
 * Resolve `${ENV_NAME}` placeholders in an env/headers record at connect time.
 * A placeholder naming a variable that is not set resolves to `""` rather than
 * throwing: a missing credential is the server's error to report (it will say
 * "unauthorized"), not a reason for the whole manager to fail to start, and a
 * literal `${FOO}` on the wire is strictly worse than an empty string.
 *
 * Values that are not a placeholder pass through verbatim.
 */
export function resolveEnvPlaceholders(
  record: Record<string, string> | undefined,
  env: Record<string, string | undefined>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(record ?? {})) {
    const match = ENV_PLACEHOLDER.exec(value);
    out[key] = match ? (env[match[1] as string] ?? "") : value;
  }
  return out;
}
