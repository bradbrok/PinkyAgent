/**
 * Settings store (DESIGN.md P8, revised: "DB is the only config; agents
 * reconfigure themselves only where a human delegated it"). The human CLI is
 * the default write path; the runtime gets a validated `SettingsSnapshot` per
 * run, and its only mutation path is the `settings_set` tool, gated on
 * `selfConfig` (see isSelfConfigWritable below).
 *
 * TWO POSTURES, ONE VALIDATOR — the rule this file turns on:
 *
 *   `set()` is STRICT — the human (or the tool) sees the error where they
 *   typed it, and nothing lands in the table until the merged candidate
 *   validates.
 *
 *   `load()` PRUNES AND WARNS — a bad row can never brick a wake. Rows whose
 *   key names no setting are dropped, `selfConfig.allowedKeys` entries that
 *   name nothing real are filtered out, and a leaf whose *value* is wrong
 *   falls back to its default. Each repair is reported once per store through
 *   `onWarning` (default `console.warn`, i.e. **stderr** — stdout is the
 *   headless JSONL protocol and must stay protocol-only).
 *
 * That asymmetry is the whole point. Config lives only in the database
 * precisely so a malformed value cannot stop a process from starting; a
 * `load()` that threw on row content handed that failure mode straight back —
 * one junk row wedged `load()`, `config get`, every per-run reload, AND
 * `config set <any other key>` (which re-validates the scope first), so
 * nothing could clear it. `unset()` is the escape hatch that does not care
 * whether the key it removes was ever legal.
 *
 * Overlay: global < channel:<id> < agent:<id> — later scopes override. Only
 * the scopes the caller asks for are applied: `load({ scopes: [...] })` never
 * lets one channel's row leak into another channel's (or an unrelated agent's)
 * snapshot, and `load()` with no args is defaults + `global`. Two scopes of the
 * same class (two channel:* entries, say) apply in list order — later wins.
 *
 * Row values are jsonb, written as PLAIN values (pg.ts's JSONB CONTRACT):
 * `set("global", "model", "openrouter/x/y")` stores the jsonb string
 * `"openrouter/x/y"`, not the doubly-encoded `"\"openrouter/x/y\""` that a
 * pre-stringified param produces. There is deliberately NO parse-on-read
 * tolerance here, unlike EventStore.mapRow: a settings value may legitimately
 * BE a string ("model"), so "does this text parse as JSON?" cannot tell a
 * healthy row from a damaged one without guessing. Legacy double-encoded rows
 * are repaired in the database instead, by
 * schema/0004_jsonb_repair.rerun.sql.
 *
 * Keys are top-level settings field names ("tenantId",
 * "model", "context", "replyGate", "memory", "selfConfig") or dotted sub-paths
 * ("context.advisoryFraction") — the loader materializes a snapshot by merging
 * each row into the tree, so "context" as a whole replaces the sub-tree while
 * dotted keys merge granularly. Within one scope rows apply in key order, so
 * "context" lands before "context.advisoryFraction" and the dotted key refines
 * the sub-tree the parent key just replaced.
 */
import type { Db } from "./db";
import { jsonbParam } from "./pg";
import { DEFAULT_SETTINGS, type SettingsSnapshot } from "./config";

/** Overlay class: global (0) < channel:<id> (1) < agent:<id> (2). */
const SCOPE_ORDER = (scope: string): number =>
  scope === "global" ? 0 : scope.startsWith("channel:") ? 1 : 2;

const SCOPE_PREFIXES = ["channel:", "agent:"] as const;

/** Throw unless `scope` is "global", "channel:<id>" or "agent:<id>". */
export function assertScope(scope: unknown): string {
  const bad = (why: string): never => {
    throw new Error(
      `Invalid settings scope ${show(scope)}: ${why} ` +
        `(expected "global", "channel:<id>" or "agent:<id>")`,
    );
  };
  if (typeof scope !== "string" || scope.trim() === "") return bad("must be a non-empty string");
  if (scope === "global") return scope;
  for (const prefix of SCOPE_PREFIXES) {
    if (!scope.startsWith(prefix)) continue;
    if (scope.slice(prefix.length).trim() === "") return bad(`"${prefix}" needs a non-empty id`);
    return scope;
  }
  return bad("unknown scope class");
}

/** Throw unless `key` is a non-empty dotted path with non-empty segments. */
export function assertKey(key: unknown): string {
  if (typeof key !== "string" || key.trim() === "" || key.split(".").some((p) => p.trim() === "")) {
    throw new Error(
      `Invalid settings key ${show(key)}: expected a field name or dotted sub-path ` +
        `(e.g. "model", "context.advisoryFraction")`,
    );
  }
  return key;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function show(v: unknown): string {
  if (v === undefined) return "undefined";
  if (typeof v === "number" && !Number.isFinite(v)) return String(v);
  try {
    return JSON.stringify(v) ?? String(v);
  } catch {
    return String(v);
  }
}

function isFraction(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v > 0 && v <= 1;
}

function isPositiveInt(v: unknown): v is number {
  return Number.isInteger(v) && (v as number) > 0;
}

/** Mirrors runtime splitModel(): a non-empty provider prefix before the first
 *  "/", and a non-empty model id after it. Shared by `model` and
 *  `memory.embeddingModel` so the two ids stay parseable by the same code. */
function hasProviderPrefix(id: string): boolean {
  const slash = id.indexOf("/");
  return slash > 0 && slash !== id.length - 1;
}

/**
 * Floor for `context.approxWindowTokens`. Below roughly a thousand tokens the
 * pressure ladder is not a ladder: the system prompt alone clears
 * `hardFraction`, so every run opens already over the hard threshold and sheds
 * context before it has any.
 */
export const MIN_APPROX_WINDOW_TOKENS = 1000;

/**
 * Minimum `hardFraction - advisoryFraction`. The advisory notice exists to buy
 * the model a turn or two to write a continuity document *before* the forced
 * shed; a gap of 0.001 fires both in the same breath, which is the same as
 * having no advisory rung at all.
 */
export const MIN_FRACTION_GAP = 0.05;

/** Float slack, so an exactly-0.05 gap written as 0.7/0.75 is not rejected. */
const GAP_EPSILON = 1e-9;

// Hand-written key lists (no schema library — this repo has one runtime dep).
// DEFAULT_SETTINGS is the base *and* the source of truth for "what exists".
const TOP_KEYS = Object.keys(DEFAULT_SETTINGS);
const CONTEXT_KEYS = Object.keys(DEFAULT_SETTINGS.context);
const REPLY_GATE_KEYS = Object.keys(DEFAULT_SETTINGS.replyGate);
const MEMORY_KEYS = Object.keys(DEFAULT_SETTINGS.memory);
const SELF_CONFIG_BLOCK_KEYS = Object.keys(DEFAULT_SETTINGS.selfConfig);

/** Walk DEFAULT_SETTINGS into the dotted paths a settings row may address. */
function collectKeys(node: Record<string, unknown>, prefix = ""): string[] {
  const out: string[] = [];
  for (const [key, value] of Object.entries(node)) {
    const path = prefix === "" ? key : `${prefix}.${key}`;
    out.push(path);
    if (isRecord(value)) out.push(...collectKeys(value, path));
  }
  return out;
}

/**
 * Every key an allow-list entry (or a `pinky config set`) may name: top-level
 * field names plus one dotted path per nested field, derived from
 * DEFAULT_SETTINGS so a field added there is delegable the day it exists.
 * (DESIGN.md P8, revised.)
 */
export const SELF_CONFIG_KEYS: string[] = collectKeys(
  DEFAULT_SETTINGS as unknown as Record<string, unknown>,
);

const KNOWN_PATHS = new Set(SELF_CONFIG_KEYS);

/** True for a key a settings row may legally address ("model", "memory",
 *  "context.hardFraction"). What `load()` prunes rows by. */
export function isKnownSettingPath(key: string): boolean {
  return KNOWN_PATHS.has(key);
}

/** Keys that have children, i.e. the ones "<key>.*" can stand for. */
const SUBTREE_KEYS: string[] = Object.entries(
  DEFAULT_SETTINGS as unknown as Record<string, unknown>,
)
  .filter(([, v]) => isRecord(v))
  .map(([k]) => k);

/**
 * Keys no allow-list can ever delegate, "*" included:
 *
 * - `tenantId` picks which tenant's rows (and which RLS partition) the agent
 *   operates in. A tool that could rewrite it could read another tenant.
 * - `selfConfig` itself: an agent that can widen its own allow-list does not
 *   have an allow-list, it has a formality.
 */
export function isImmutableSettingKey(key: string): boolean {
  return key === "tenantId" || key === "selfConfig" || key.startsWith("selfConfig.");
}

/**
 * Pure allow-list check for the agent's `settings_set` tool (DESIGN.md P8,
 * revised). Says nothing about whether `key` exists — SettingsStore.set is
 * what rejects an unknown key, and its message is what the tool surfaces.
 *
 * Entry forms:
 * - `"*"`          — everything except the immutables.
 * - `"context"`    — that key AND everything under it.
 * - `"context.*"`  — everything under it, but NOT the sub-tree as a whole
 *                    (so a delegated agent can tune one threshold without
 *                    being able to replace the whole block in one write).
 * - `"context.advisoryFraction"` — exactly that leaf.
 */
export function isSelfConfigWritable(key: string, allowedKeys: string[]): boolean {
  if (isImmutableSettingKey(key)) return false;
  for (const entry of allowedKeys) {
    if (typeof entry !== "string") continue;
    if (entry === "*") return true;
    if (entry.endsWith(".*")) {
      // "context.*" -> children of "context.", never "context" itself.
      if (key.startsWith(entry.slice(0, -1))) return true;
      continue;
    }
    if (key === entry || key.startsWith(`${entry}.`)) return true;
  }
  return false;
}

/**
 * The rule for ONE `selfConfig.allowedKeys` entry, shared by the validator
 * (which reports it, with an index label) and the loader's pruner (which drops
 * the entry and warns). Returns null when the entry is a legal delegation.
 */
function allowedEntryProblem(entry: unknown): string | null {
  if (typeof entry !== "string" || entry.trim() === "") {
    return `expected a non-empty string, got ${show(entry)}`;
  }
  // "<key>.*" delegates the sub-tree; check the key it names.
  const named = entry.endsWith(".*") ? entry.slice(0, -2) : entry;
  if (isImmutableSettingKey(named)) {
    return (
      `${show(entry)} can never be delegated to an agent ` +
      `(tenantId picks the tenant; selfConfig is the delegation itself)`
    );
  }
  if (entry === "*") return null;
  if (entry.endsWith(".*")) {
    return SUBTREE_KEYS.includes(named)
      ? null
      : `${show(entry)} names no settings sub-tree (sub-trees: ${SUBTREE_KEYS.join(", ")})`;
  }
  if (!SELF_CONFIG_KEYS.includes(entry)) {
    return (
      `${show(entry)} is not a setting key ` +
      `(expected "*", "<key>.*", or one of: ` +
      `${SELF_CONFIG_KEYS.filter((k) => !isImmutableSettingKey(k)).join(", ")})`
    );
  }
  return null;
}

/**
 * How the loader repairs one issue:
 * - `delete` — the path names no setting at all; drop it from the snapshot.
 * - `reset`  — the value is wrong; fall back to DEFAULT_SETTINGS.
 * - `filter` — `selfConfig.allowedKeys` holds entries that name nothing real;
 *              keep the legal ones.
 */
type Repair = "delete" | "reset" | "filter";

interface Issue {
  /** Dotted path the issue is about; also the repair target. */
  path: string;
  /** "<path>: <what was expected>, got <what was found>". */
  message: string;
  repair: Repair;
}

/**
 * Every rule in one place, reported rather than thrown, so the strict caller
 * (validateSettings) and the lenient one (sanitizeSettings) cannot drift.
 */
function collectIssues(candidate: Record<string, unknown>): Issue[] {
  const issues: Issue[] = [];
  const bad = (path: string, message: string, repair: Repair = "reset"): void => {
    issues.push({ path, message: `${path}: ${message}`, repair });
  };

  for (const key of Object.keys(candidate)) {
    if (!TOP_KEYS.includes(key)) {
      bad(key, `unknown setting key (known: ${TOP_KEYS.join(", ")})`, "delete");
    }
  }

  const tenantId = candidate.tenantId;
  if (typeof tenantId !== "string" || tenantId.trim() === "") {
    bad("tenantId", `expected a non-empty string, got ${show(tenantId)}`);
  }

  const model = candidate.model;
  if (typeof model !== "string" || model.trim() === "") {
    bad("model", `expected a non-empty string, got ${show(model)}`);
  } else if (!hasProviderPrefix(model)) {
    bad(
      "model",
      `expected "provider/model-id" (e.g. "openrouter/moonshotai/kimi-k2"), got ${show(model)}`,
    );
  }

  const context = candidate.context;
  if (!isRecord(context)) {
    bad("context", `expected an object, got ${show(context)}`);
  } else {
    for (const key of Object.keys(context)) {
      if (!CONTEXT_KEYS.includes(key)) {
        bad(
          `context.${key}`,
          `unknown setting key (known: ${CONTEXT_KEYS.join(", ")})`,
          "delete",
        );
      }
    }
    const advisory = context.advisoryFraction;
    const hard = context.hardFraction;
    if (!isFraction(advisory)) {
      bad("context.advisoryFraction", `expected a number in (0, 1], got ${show(advisory)}`);
    }
    if (!isFraction(hard)) {
      bad("context.hardFraction", `expected a number in (0, 1], got ${show(hard)}`);
    }
    if (isFraction(advisory) && isFraction(hard)) {
      if (advisory >= hard) {
        bad(
          "context.advisoryFraction",
          `expected to be less than context.hardFraction, got ${show(advisory)} >= ${show(hard)}`,
        );
      } else if (hard - advisory < MIN_FRACTION_GAP - GAP_EPSILON) {
        // The advisory rung exists to buy a continuity write before the forced
        // shed; too narrow a gap fires both rungs on the same turn.
        bad(
          "context.advisoryFraction",
          `expected to be at least ${MIN_FRACTION_GAP} below context.hardFraction, ` +
            `got ${show(advisory)} vs ${show(hard)}`,
        );
      }
    }
    const tokens = context.approxWindowTokens;
    if (!isPositiveInt(tokens) || tokens < MIN_APPROX_WINDOW_TOKENS) {
      bad(
        "context.approxWindowTokens",
        `expected an integer >= ${MIN_APPROX_WINDOW_TOKENS} ` +
          `(a smaller window puts every run over hardFraction before it starts), got ${show(tokens)}`,
      );
    }
  }

  const replyGate = candidate.replyGate;
  if (!isRecord(replyGate)) {
    bad("replyGate", `expected an object, got ${show(replyGate)}`);
  } else {
    for (const key of Object.keys(replyGate)) {
      if (!REPLY_GATE_KEYS.includes(key)) {
        bad(
          `replyGate.${key}`,
          `unknown setting key (known: ${REPLY_GATE_KEYS.join(", ")})`,
          "delete",
        );
      }
    }
    if (typeof replyGate.classifierEnabled !== "boolean") {
      bad(
        "replyGate.classifierEnabled",
        `expected a boolean, got ${show(replyGate.classifierEnabled)}`,
      );
    }
  }

  // Memory plane (DESIGN.md §5). "none" turns the vector voice off entirely
  // (FTS-only recall); anything else must be a "provider/model-id" the
  // runtime's createEmbedder() can split, so a typo fails here and not at the
  // first embedding request of a live run.
  const memory = candidate.memory;
  if (!isRecord(memory)) {
    bad("memory", `expected an object, got ${show(memory)}`);
  } else {
    for (const key of Object.keys(memory)) {
      if (!MEMORY_KEYS.includes(key)) {
        bad(`memory.${key}`, `unknown setting key (known: ${MEMORY_KEYS.join(", ")})`, "delete");
      }
    }
    const embeddingModel = memory.embeddingModel;
    if (typeof embeddingModel !== "string" || embeddingModel.trim() === "") {
      bad("memory.embeddingModel", `expected a non-empty string, got ${show(embeddingModel)}`);
    } else if (embeddingModel !== "none" && !hasProviderPrefix(embeddingModel)) {
      bad(
        "memory.embeddingModel",
        `expected "none" or "provider/model-id" ` +
          `(e.g. "openai/text-embedding-3-small"), got ${show(embeddingModel)}`,
      );
    }
    if (typeof memory.autoRecall !== "boolean") {
      bad("memory.autoRecall", `expected a boolean, got ${show(memory.autoRecall)}`);
    }
    if (!isPositiveInt(memory.recallLimit)) {
      bad("memory.recallLimit", `expected a positive integer, got ${show(memory.recallLimit)}`);
    }
    if (!isPositiveInt(memory.recallTokenBudget)) {
      bad(
        "memory.recallTokenBudget",
        `expected a positive integer, got ${show(memory.recallTokenBudget)}`,
      );
    }
  }

  // Human-granted self-configuration (DESIGN.md P8, revised). This block is
  // the delegation itself, so its own rules are the strictest in the file: an
  // entry that names nothing real would silently grant nothing (or, worse,
  // read as granting something), and an entry naming tenantId/selfConfig is a
  // delegation the runtime will refuse anyway — so both are rejected here,
  // where the human writing `pinky config set` still sees the message.
  const selfConfig = candidate.selfConfig;
  if (!isRecord(selfConfig)) {
    bad("selfConfig", `expected an object, got ${show(selfConfig)}`);
  } else {
    for (const key of Object.keys(selfConfig)) {
      if (!SELF_CONFIG_BLOCK_KEYS.includes(key)) {
        bad(
          `selfConfig.${key}`,
          `unknown setting key (known: ${SELF_CONFIG_BLOCK_KEYS.join(", ")})`,
          "delete",
        );
      }
    }
    if (typeof selfConfig.enabled !== "boolean") {
      bad("selfConfig.enabled", `expected a boolean, got ${show(selfConfig.enabled)}`);
    }
    const allowed = selfConfig.allowedKeys;
    if (!Array.isArray(allowed)) {
      bad("selfConfig.allowedKeys", `expected an array of strings, got ${show(allowed)}`);
    } else {
      for (let i = 0; i < allowed.length; i++) {
        const problem = allowedEntryProblem(allowed[i]);
        if (problem === null) continue;
        // The repair target is the ARRAY (drop the offending entries); the
        // message keeps the index so a human can see which one.
        issues.push({
          path: "selfConfig.allowedKeys",
          message: `selfConfig.allowedKeys[${i}]: ${problem}`,
          repair: "filter",
        });
      }
    }
  }

  return issues;
}

/**
 * Validate a materialized snapshot. Throws one Error listing *every* bad key
 * with the type it expected; returns the candidate typed on success.
 *
 * Strict on purpose — this is the `set()` path. Unknown keys are rejected at
 * both levels, so `contxt.hardFraction` or `context.hardFractoin` fail loudly
 * instead of writing a row nothing reads. `load()` uses sanitizeSettings()
 * instead: same rules, repaired rather than thrown.
 */
export function validateSettings(candidate: unknown): SettingsSnapshot {
  if (!isRecord(candidate)) {
    throw new Error(`Invalid settings: expected an object, got ${show(candidate)}`);
  }
  const issues = collectIssues(candidate);
  if (issues.length > 0) {
    throw new Error(`Invalid settings:\n  - ${issues.map((i) => i.message).join("\n  - ")}`);
  }
  return candidate as unknown as SettingsSnapshot;
}

function setPath(root: Record<string, unknown>, key: string, value: unknown): void {
  const parts = key.split(".");
  let cur = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i]!;
    const next = cur[p];
    if (next === null || typeof next !== "object" || Array.isArray(next)) {
      cur[p] = {};
    }
    cur = cur[p] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1]!] = value;
}

function getPath(root: Record<string, unknown>, key: string): unknown {
  let cur: unknown = root;
  for (const part of key.split(".")) {
    if (!isRecord(cur)) return undefined;
    cur = cur[part];
  }
  return cur;
}

function deletePath(root: Record<string, unknown>, key: string): void {
  const parts = key.split(".");
  let cur: unknown = root;
  for (let i = 0; i < parts.length - 1; i++) {
    cur = isRecord(cur) ? cur[parts[i]!] : undefined;
  }
  if (isRecord(cur)) delete cur[parts[parts.length - 1]!];
}

/** DEFAULT_SETTINGS at a dotted path, deep-cloned so the caller cannot alias it. */
function defaultAt(key: string): unknown {
  return structuredClone(getPath(DEFAULT_SETTINGS as unknown as Record<string, unknown>, key));
}

/** Cheap structural equality — the values here are jsonb, so JSON is faithful. */
function sameValue(a: unknown, b: unknown): boolean {
  return show(a) === show(b);
}

/** "context.advisoryFraction" -> "context"; "model" -> "model". */
function topLevelOf(key: string): string {
  return key.split(".")[0]!;
}

/**
 * The lenient half of the validator (see the file header). Repairs `root` in
 * place until it validates, reporting each repair through `warn`, and returns
 * it typed.
 *
 * Repairs are the smallest thing that can work: drop an unknown key, filter a
 * bad allow-list entry, or reset one leaf to its default. A leaf that is
 * ALREADY its default and still wrong is a cross-field problem (advisory vs
 * hard), so the repair escalates to the whole sub-tree in the same pass. The
 * loop is bounded and ends at "the defaults" rather than at an exception,
 * because a wake with default settings is a working agent and a throw here is
 * a dead one.
 */
function sanitizeSettings(
  root: Record<string, unknown>,
  warn: (message: string) => void,
): SettingsSnapshot {
  for (let pass = 0; pass < 4; pass++) {
    const issues = collectIssues(root);
    if (issues.length === 0) return root as unknown as SettingsSnapshot;
    const repaired = new Set<string>();
    for (const issue of issues) {
      if (issue.repair === "delete") {
        warn(`settings: ignoring ${issue.message} — the value is not used`);
        deletePath(root, issue.path);
        continue;
      }
      if (issue.repair === "filter") {
        warn(`settings: ignoring ${issue.message} — that delegation is dropped`);
        if (repaired.has(issue.path)) continue;
        repaired.add(issue.path);
        const current = getPath(root, issue.path);
        setPath(
          root,
          issue.path,
          Array.isArray(current) ? current.filter((e) => allowedEntryProblem(e) === null) : [],
        );
        continue;
      }
      // A leaf already sitting at its default cannot be repaired by resetting
      // it again: the conflict is with a sibling, so reset the sub-tree.
      const target = sameValue(getPath(root, issue.path), defaultAt(issue.path))
        ? topLevelOf(issue.path)
        : issue.path;
      warn(`settings: ${issue.message} — falling back to the default for "${target}"`);
      if (repaired.has(target)) continue;
      repaired.add(target);
      setPath(root, target, defaultAt(target));
    }
  }
  // Unreachable in practice (pass 2 resets every sub-tree that can still be
  // wrong), but "never throw on row content" has to mean never.
  warn("settings: too many bad rows to repair — falling back to the defaults");
  return structuredClone(DEFAULT_SETTINGS);
}

export interface LoadOptions {
  /**
   * Scopes to overlay on top of the defaults and `global`, e.g.
   * ["channel:slack:C123", "agent:pinky"]. Ordering across classes is fixed
   * (channel before agent); within a class, later in the list wins. Rows for
   * any other scope are never read.
   */
  scopes?: string[];
}

export interface SetOptions {
  /**
   * ALSO validate the write against this scope list — the scopes a run
   * actually loads, e.g. ["channel:slack:C1", "agent:pinky"].
   *
   * Without it, `set()` only checks `defaults + global + <target scope>`, so
   * cross-field invariants that span scopes are never checked at write time:
   * an agent with `context.*` delegated could write `hardFraction` at
   * `agent:pinky` and `advisoryFraction` at `channel:C1`, each valid on its
   * own, and invert the pressure ladder for every later run on that channel.
   * The target scope is included automatically.
   */
  validateScopes?: string[];
}

export interface SettingsStoreOptions {
  /**
   * Where `load()` reports a pruned row or a repaired value. Default
   * `console.warn` — STDERR, because stdout is the headless JSONL protocol.
   * Each distinct message is reported once per store instance, so a long-lived
   * surface reloading per run does not repeat itself every wake.
   */
  onWarning?: (message: string) => void;
}

export class SettingsStore {
  private db: Db;
  private onWarning: (message: string) => void;
  /** Messages already reported by this instance (see SettingsStoreOptions). */
  private warned = new Set<string>();

  constructor(db: Db, opts: SettingsStoreOptions = {}) {
    this.db = db;
    this.onWarning = opts.onWarning ?? ((message: string): void => console.warn(message));
  }

  private warn(message: string): void {
    if (this.warned.has(message)) return;
    this.warned.add(message);
    this.onWarning(message);
  }

  /**
   * Effective snapshot: defaults + `global` + the requested scopes, PRUNED and
   * repaired (never thrown — see the file header). Only a malformed `scopes`
   * argument, which is a programming error rather than row content, throws.
   */
  async load(opts?: LoadOptions): Promise<SettingsSnapshot> {
    return sanitizeSettings(await this.merge(opts?.scopes ?? []), (m) => this.warn(m));
  }

  /**
   * Upsert one setting. Key is a top-level field name or a dotted sub-path.
   * The candidate is overlaid onto the scope's current effective snapshot and
   * validated STRICTLY first, so a bad value is rejected and never lands in
   * the table.
   *
   * The base it validates against is the *pruned* merge: a stale row naming no
   * setting is dropped (with a warning) rather than blocking every unrelated
   * write, which is what made a single junk row unclearable. Rows with real
   * keys and wrong values are NOT repaired here — they are the human's to fix,
   * and the error names them.
   *
   * `opts.validateScopes` adds the cross-scope check (see SetOptions).
   */
  async set(scope: string, key: string, value: unknown, opts?: SetOptions): Promise<void> {
    const s = assertScope(scope);
    const k = assertKey(key);
    const candidate = await this.merge(s === "global" ? [] : [s]);
    setPath(candidate, k, value);
    validateSettings(candidate); // throws before any write

    const requested = opts?.validateScopes;
    if (requested && requested.length > 0) {
      const scopes = requested.map((x) => assertScope(x));
      if (s !== "global" && !scopes.includes(s)) scopes.push(s);
      const wide = await this.merge(scopes);
      setPath(wide, k, value);
      try {
        validateSettings(wide);
      } catch (err) {
        throw new Error(
          `${err instanceof Error ? err.message : String(err)}\n` +
            `  (checked against the scopes a run loads: ${["global", ...scopes].join(", ")})`,
        );
      }
    }

    await this.db.query(
      `insert into settings (scope, key, value) values ($1, $2, $3)
       on conflict (scope, key) do update set value = excluded.value, updated_at = now()`,
      // Plain value — the driver JSON-encodes a jsonb param exactly once.
      // jsonbParam() only rewraps the types postgres.js would otherwise tag
      // with a non-jsonb wire type (a bare `true` for replyGate.*).
      [s, k, jsonbParam(value)],
    );
  }

  /**
   * Remove one row. Returns false when there was nothing to remove.
   *
   * Deliberately unvalidated in both directions: the key does NOT have to name
   * a real setting (removing junk is the point — `pinky config unset` is how a
   * human clears a row `load()` is warning about), and the resulting snapshot
   * is not re-validated, because deleting a row can only move the scope back
   * towards the defaults.
   */
  async unset(scope: string, key: string): Promise<boolean> {
    const s = assertScope(scope);
    const k = assertKey(key);
    const rows = await this.db.query(
      `delete from settings where scope = $1 and key = $2 returning 1`,
      [s, k],
    );
    return rows.length > 0;
  }

  /**
   * Materialize defaults + `global` + `scopes` without validating VALUES, so
   * `set` can report the problem with the candidate row rather than dying on
   * state that is already broken elsewhere in the table.
   *
   * Rows whose KEY names no setting are dropped here (with a warning): they
   * can never be read by anything, and leaving them in the candidate wedged
   * every load and every unrelated write.
   */
  private async merge(scopes: string[]): Promise<Record<string, unknown>> {
    // Overlay priority within a class: order of first appearance in `scopes`.
    const priority = new Map<string, number>([["global", 0]]);
    for (const raw of scopes) {
      const scope = assertScope(raw);
      if (!priority.has(scope)) priority.set(scope, priority.size);
    }
    const wanted = [...priority.keys()];
    const rows = await this.db.query<{ scope: string; key: string; value: unknown }>(
      `select scope, key, value from settings where scope = any($1) order by scope, key`,
      [wanted],
    );
    const applicable = rows.filter((r) => {
      if (!priority.has(r.scope)) return false;
      if (isKnownSettingPath(r.key)) return true;
      this.warn(
        `settings: ignoring row ${r.scope}/${r.key} — no such setting key ` +
          `(remove it with \`pinky config unset\`)`,
      );
      return false;
    });
    applicable.sort(
      (a, b) =>
        SCOPE_ORDER(a.scope) - SCOPE_ORDER(b.scope) ||
        priority.get(a.scope)! - priority.get(b.scope)! ||
        (a.key < b.key ? -1 : a.key > b.key ? 1 : 0),
    );
    const root = structuredClone(DEFAULT_SETTINGS) as unknown as Record<string, unknown>;
    for (const r of applicable) setPath(root, r.key, r.value);
    return root;
  }
}
