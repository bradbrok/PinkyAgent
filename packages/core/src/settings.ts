/**
 * Settings store (DESIGN.md P8: "DB is the only config; agents cannot
 * self-lobotomize"). The human CLI is the only write path; the runtime gets a
 * validated `SettingsSnapshot` per wake and has no mutation path.
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
 * "model", "context", "replyGate") or dotted sub-paths
 * ("context.advisoryFraction") — the loader materializes a snapshot by merging
 * each row into the tree, so "context" as a whole replaces the sub-tree while
 * dotted keys merge granularly. Within one scope rows apply in key order, so
 * "context" lands before "context.advisoryFraction" and the dotted key refines
 * the sub-tree the parent key just replaced.
 *
 * Everything that reaches a snapshot goes through validateSettings(), on read
 * and (for the candidate row) before write: `config set context.hardFraction
 * abc` is rejected with a message naming the key instead of silently turning
 * the context-pressure ladder into NaN comparisons that are always false.
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

// Hand-written key lists (no schema library — this repo has one runtime dep).
// DEFAULT_SETTINGS is the base *and* the source of truth for "what exists".
const TOP_KEYS = Object.keys(DEFAULT_SETTINGS);
const CONTEXT_KEYS = Object.keys(DEFAULT_SETTINGS.context);
const REPLY_GATE_KEYS = Object.keys(DEFAULT_SETTINGS.replyGate);

/**
 * Validate a materialized snapshot. Throws one Error listing *every* bad key
 * with the type it expected; returns the candidate typed on success.
 *
 * Unknown keys are rejected at both levels, so `contxt.hardFraction` or
 * `context.hardFractoin` fail loudly instead of writing a row nothing reads.
 */
export function validateSettings(candidate: unknown): SettingsSnapshot {
  if (!isRecord(candidate)) {
    throw new Error(`Invalid settings: expected an object, got ${show(candidate)}`);
  }
  const issues: string[] = [];

  for (const key of Object.keys(candidate)) {
    if (!TOP_KEYS.includes(key)) {
      issues.push(`${key}: unknown setting key (known: ${TOP_KEYS.join(", ")})`);
    }
  }

  const tenantId = candidate.tenantId;
  if (typeof tenantId !== "string" || tenantId.trim() === "") {
    issues.push(`tenantId: expected a non-empty string, got ${show(tenantId)}`);
  }

  const model = candidate.model;
  if (typeof model !== "string" || model.trim() === "") {
    issues.push(`model: expected a non-empty string, got ${show(model)}`);
  } else {
    // Mirrors runtime splitModel(): a provider prefix before the first "/".
    const slash = model.indexOf("/");
    if (slash <= 0 || slash === model.length - 1) {
      issues.push(`model: expected "provider/model-id" (e.g. "openrouter/moonshotai/kimi-k2"), got ${show(model)}`);
    }
  }

  const context = candidate.context;
  if (!isRecord(context)) {
    issues.push(`context: expected an object, got ${show(context)}`);
  } else {
    for (const key of Object.keys(context)) {
      if (!CONTEXT_KEYS.includes(key)) {
        issues.push(`context.${key}: unknown setting key (known: ${CONTEXT_KEYS.join(", ")})`);
      }
    }
    const advisory = context.advisoryFraction;
    const hard = context.hardFraction;
    if (!isFraction(advisory)) {
      issues.push(`context.advisoryFraction: expected a number in (0, 1], got ${show(advisory)}`);
    }
    if (!isFraction(hard)) {
      issues.push(`context.hardFraction: expected a number in (0, 1], got ${show(hard)}`);
    }
    if (isFraction(advisory) && isFraction(hard) && advisory >= hard) {
      issues.push(
        `context.advisoryFraction: expected to be less than context.hardFraction, got ${show(advisory)} >= ${show(hard)}`,
      );
    }
    const tokens = context.approxWindowTokens;
    if (!Number.isInteger(tokens) || (tokens as number) <= 0) {
      issues.push(`context.approxWindowTokens: expected a positive integer, got ${show(tokens)}`);
    }
  }

  const replyGate = candidate.replyGate;
  if (!isRecord(replyGate)) {
    issues.push(`replyGate: expected an object, got ${show(replyGate)}`);
  } else {
    for (const key of Object.keys(replyGate)) {
      if (!REPLY_GATE_KEYS.includes(key)) {
        issues.push(`replyGate.${key}: unknown setting key (known: ${REPLY_GATE_KEYS.join(", ")})`);
      }
    }
    if (typeof replyGate.classifierEnabled !== "boolean") {
      issues.push(`replyGate.classifierEnabled: expected a boolean, got ${show(replyGate.classifierEnabled)}`);
    }
  }

  if (issues.length > 0) {
    throw new Error(`Invalid settings:\n  - ${issues.join("\n  - ")}`);
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

export interface LoadOptions {
  /**
   * Scopes to overlay on top of the defaults and `global`, e.g.
   * ["channel:slack:C123", "agent:pinky"]. Ordering across classes is fixed
   * (channel before agent); within a class, later in the list wins. Rows for
   * any other scope are never read.
   */
  scopes?: string[];
}

export class SettingsStore {
  private db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  /** Effective snapshot: defaults + `global` + the requested scopes, validated. */
  async load(opts?: LoadOptions): Promise<SettingsSnapshot> {
    return validateSettings(await this.merge(opts?.scopes ?? []));
  }

  /**
   * Upsert one setting. Key is a top-level field name or a dotted sub-path.
   * The candidate is overlaid onto the scope's current effective snapshot and
   * validated first, so a bad value is rejected and never lands in the table.
   */
  async set(scope: string, key: string, value: unknown): Promise<void> {
    const s = assertScope(scope);
    const k = assertKey(key);
    const candidate = await this.merge(s === "global" ? [] : [s]);
    setPath(candidate, k, value);
    validateSettings(candidate); // throws before any write
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
   * Materialize defaults + `global` + `scopes` without validating, so `set`
   * can report the problem with the candidate row rather than dying on state
   * that is already broken elsewhere in the table.
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
    const applicable = rows.filter((r) => priority.has(r.scope));
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
