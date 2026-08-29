/**
 * The agent-facing settings surface (DESIGN.md P8, revised: human-granted
 * self-configuration). `settings_get` reads the run's snapshot;
 * `settings_set` writes ONE key — and only under three conditions a human
 * controls.
 *
 * Why a tool and not a config file: a file is read at startup, so a malformed
 * value written by an agent is a process that will not boot. A tool call goes
 * through `SettingsStore.set`, which merges the candidate into that scope's
 * effective snapshot and validates it BEFORE the insert. A bad value is
 * therefore a tool error the model reads and retries; nothing lands, and the
 * next boot is unaffected. Config lives only in the `settings` table.
 *
 * The three gates, in order:
 *
 * 1. `selfConfig.enabled` — off by default, flipped only from the human CLI.
 * 2. `selfConfig.allowedKeys` — the human names what is delegable. Empty
 *    grants nothing, even with the switch on.
 * 3. The immutables — `tenantId` (which tenant's data this is) and
 *    `selfConfig` itself (an agent that can widen its own allow-list has
 *    none). Denied even by `"*"`.
 *
 * Scope: "agent:<id>" or "channel:<id>" only. `global` stays human-only, so a
 * self-tuning agent can never reach into another channel's or agent's config.
 * The write is nonetheless validated against the whole overlay this run reads
 * (channel + agent), because that is the snapshot the next run assembles — a
 * value that is only valid against the scope it lands in is still a broken
 * wake. `model` gets two extra refusals on top (see `refuseModelValue`).
 *
 * Every successful write emits a `config` event — audit-only, never projected
 * — so the log answers who changed what, from what, and when. And the write
 * lands in the table, not in this run's snapshot: the loop already read its
 * model and thresholds, so the change takes effect on the NEXT run (settings
 * are re-loaded per run).
 */
import { SettingsStore, isImmutableSettingKey, isSelfConfigWritable } from "@pinky/core";
import type { SettingsSnapshot } from "@pinky/core";
import { SUPPORTED_PROVIDERS } from "@pinky/runtime";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";

const SCOPES = ["agent", "channel"] as const;
type ScopeArg = (typeof SCOPES)[number];

const NO_SETTINGS = "no settings snapshot in this context (the runtime passed none)";

/** The exact commands a human runs to delegate. Shown verbatim to the model,
 *  which is how the human hears about them: the agent quotes them back. */
const ENABLE_HINT =
  "a human can enable it with `pinky config set selfConfig.enabled true` and allow keys with " +
  `\`pinky config set selfConfig.allowedKeys '["model","context.*"]'\``;

function fail(text: string): ToolResult {
  return { text, isError: true };
}

/** Read a dotted path out of the snapshot; undefined when the path is absent. */
export function readSettingPath(snapshot: SettingsSnapshot, key: string): unknown {
  let cur: unknown = snapshot;
  for (const part of key.split(".")) {
    if (cur === null || typeof cur !== "object" || Array.isArray(cur)) return undefined;
    cur = (cur as Record<string, unknown>)[part];
    if (cur === undefined) return undefined;
  }
  return cur;
}

/** One line, always present, so the model never has to guess its own powers. */
function selfConfigLine(snapshot: SettingsSnapshot): string {
  const { enabled, allowedKeys } = snapshot.selfConfig;
  const keys = allowedKeys.length > 0 ? allowedKeys.join(", ") : "(none)";
  return `self-configuration: ${enabled ? "enabled" : "disabled"}; writable keys: ${keys}`;
}

/**
 * A model that types `{"value": "0.6"}` for a numeric key gets a validation
 * failure that reads like a type error, because it is one. Auto-parsing the
 * string instead would be worse: `model` is legitimately a string, so a
 * "helpfully" parsed value would corrupt exactly the keys where quoting is
 * correct. Explain it in the error and let the model retry.
 */
function quotedJsonHint(value: unknown): string {
  if (typeof value !== "string") return "";
  try {
    if (typeof JSON.parse(value) !== "string") {
      return "\nhint: pass numbers/booleans as JSON values, not strings (0.6, not \"0.6\")";
    }
  } catch {
    // Not JSON-ish at all — an ordinary string value, nothing to warn about.
  }
  return "";
}

/**
 * Extra refusals for `model`, which is legitimately delegable but whose
 * validation in core can only check the `"provider/model-id"` *shape* — core
 * has no idea which providers this build can actually route (that lives in
 * runtime, which core must not depend on). Both cases below are values that
 * pass `validateSettings` and then hurt on the NEXT run, far from the tool
 * call that caused them:
 *
 * - `fake/*` is the scripted keyless test route (runtime/providers/fake.ts).
 *   Nothing about it is malformed; it simply turns the agent into an echo bot
 *   on its next wake. That is self-lobotomy by another name (CLAUDE.md #3), so
 *   it stays a human decision — `pinky config set` still allows it.
 * - an unknown provider throws inside `createProvider` at the start of the
 *   next run: a bricked wake instead of a tool error. Name the list now.
 *
 * Returns the refusal reason, or null when there is nothing to object to —
 * including for values whose shape is wrong, which is validateSettings' job to
 * report with its own message.
 */
export function refuseModelValue(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const slash = value.indexOf("/");
  if (slash <= 0 || slash === value.length - 1) return null; // not "a/b": shape error
  const provider = value.slice(0, slash);
  if (provider === "fake") {
    return (
      `'${value}' is a test route: fake/* replies from a script instead of a model. ` +
      `A human can set it with \`pinky config set model ${value}\`.`
    );
  }
  if (!(SUPPORTED_PROVIDERS as readonly string[]).includes(provider)) {
    return (
      `unknown provider '${provider}' in '${value}'; this build can route: ` +
      `${SUPPORTED_PROVIDERS.join(", ")}`
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// settings_get
// ---------------------------------------------------------------------------

export class SettingsGetTool implements Tool {
  readonly name = "settings_get";
  readonly description =
    "Show the settings this run is using, and which of them you are allowed to change yourself.\n" +
    "Read-only. Values come from the snapshot this run started with (stored in the database, never in a " +
    "config file). The last line says whether self-configuration is enabled and which keys a human has " +
    "delegated to you; settings_set writes those.";
  readonly parameters = {
    type: "object",
    properties: {
      key: {
        type: "string",
        description:
          'One dotted key to show, e.g. "model" or "context.advisoryFraction". Omit for everything.',
      },
    },
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    const snapshot = ctx.settings;
    if (!snapshot) return fail(`${this.name}: ${NO_SETTINGS}`);

    if (args.key !== undefined) {
      if (typeof args.key !== "string" || args.key.trim() === "") {
        return fail(`${this.name}: 'key' must be a non-empty dotted setting key`);
      }
      const key = args.key.trim();
      const value = readSettingPath(snapshot, key);
      if (value === undefined) {
        return fail(
          `${this.name}: unknown setting key '${key}' (call ${this.name} with no arguments to list them)`,
        );
      }
      return { text: `${key} = ${JSON.stringify(value, null, 2)}\n${selfConfigLine(snapshot)}` };
    }

    return { text: `${JSON.stringify(snapshot, null, 2)}\n${selfConfigLine(snapshot)}` };
  }
}

// ---------------------------------------------------------------------------
// settings_set
// ---------------------------------------------------------------------------

export class SettingsSetTool implements Tool {
  readonly name = "settings_set";
  readonly description =
    "Change one of your own settings (validated, journaled, human-allow-listed); never edits files.\n" +
    "Works only for keys a human delegated (see settings_get) and only in your agent or channel scope. " +
    "The value is validated before it is stored, so a bad one comes back as an error you can correct and " +
    "nothing is written. The change takes effect on your NEXT run — this run keeps the snapshot it " +
    "started with. tenantId and selfConfig itself can never be changed this way.";
  readonly parameters = {
    type: "object",
    properties: {
      key: {
        type: "string",
        description: 'The dotted setting key, e.g. "model" or "context.advisoryFraction".',
      },
      value: {
        description:
          'The new value as JSON: 0.6, true, "openrouter/moonshotai/kimi-k2". Numbers and booleans ' +
          "must be JSON values, not quoted strings.",
      },
      scope: {
        type: "string",
        enum: SCOPES,
        description:
          '"agent" (default) changes it wherever you run; "channel" only in this channel.',
      },
    },
    required: ["key", "value"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    const snapshot = ctx.settings;
    if (!snapshot) return fail(`${this.name}: ${NO_SETTINGS}`);

    if (typeof args.key !== "string" || args.key.trim() === "") {
      return fail(
        `${this.name}: 'key' must be a non-empty dotted setting key (e.g. "context.advisoryFraction")`,
      );
    }
    const key = args.key.trim();
    const value = args.value;
    if (value === undefined) {
      return fail(`${this.name}: 'value' is required (JSON: 0.6, true, "openrouter/x/y")`);
    }

    let scopeArg: ScopeArg = "agent";
    if (args.scope !== undefined) {
      if (typeof args.scope !== "string" || !(SCOPES as readonly string[]).includes(args.scope)) {
        return fail(
          `${this.name}: 'scope' must be "agent" or "channel" — the global scope is human-only ` +
            "(`pinky config set <key> <value>`)",
        );
      }
      scopeArg = args.scope as ScopeArg;
    }

    const selfConfig = snapshot.selfConfig;
    if (!selfConfig.enabled) {
      return fail(`${this.name}: self-configuration is disabled; ${ENABLE_HINT}`);
    }
    if (isImmutableSettingKey(key)) {
      return fail(
        `${this.name}: '${key}' can never be changed by a tool, allow-list or not ` +
          "(tenantId picks the tenant; selfConfig is the delegation itself). Ask a human to run " +
          `\`pinky config set ${key} <value>\`.`,
      );
    }
    if (!isSelfConfigWritable(key, selfConfig.allowedKeys)) {
      const patterns =
        selfConfig.allowedKeys.length > 0 ? selfConfig.allowedKeys.join(", ") : "(none)";
      return fail(
        `${this.name}: '${key}' is not delegated to you; allowed patterns: ${patterns}. ` +
          `A human can add it with \`pinky config set selfConfig.allowedKeys '["${key}"]'\`.`,
      );
    }
    // The scope name and the `by` of the journal entry both need it.
    if (!ctx.agentId) {
      return fail(`${this.name}: no agent id in this context, so there is no scope to write to`);
    }

    if (key === "model") {
      const denied = refuseModelValue(value);
      if (denied) return fail(`${this.name}: ${denied}`);
    }

    const scope = scopeArg === "channel" ? `channel:${ctx.thread.channelId}` : `agent:${ctx.agentId}`;
    const previous = readSettingPath(snapshot, key);

    try {
      // Validates the merged candidate first; a rejection writes nothing.
      //
      // `validateScopes` is the scope list THIS RUN loads (channel + agent,
      // the same overlay `settingsFor` passes), not just the scope being
      // written. Validating the target scope alone lets a cross-scope
      // invariant slip through: `context.advisoryFraction 0.8` is fine against
      // channel:c1 + defaults, but the agent scope already carries
      // `hardFraction 0.75`, so the snapshot the next run assembles would be
      // the invalid one. The write must be judged against the settings the
      // agent will actually wake up with.
      await new SettingsStore(ctx.db).set(scope, key, value, {
        validateScopes: [`channel:${ctx.thread.channelId}`, `agent:${ctx.agentId}`],
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return fail(`rejected: ${message}${quotedJsonHint(value)}`);
    }

    await ctx.emit({ type: "config", scope, key, value, previous, by: ctx.agentId });

    return {
      text:
        `set ${key} = ${JSON.stringify(value)} (${scope}); takes effect on the next run — ` +
        "the current run keeps its snapshot",
    };
  }
}
