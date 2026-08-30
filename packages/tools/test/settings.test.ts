/**
 * settings_get / settings_set — the agent's only write path into its own
 * config (DESIGN.md P8, revised: human-granted self-configuration).
 *
 * What these tests are really guarding: every refusal is a *readable tool
 * error*, never a throw and never a half-write. The whole point of doing this
 * through a tool instead of a config file is that a value the model gets
 * wrong bounces back as text it can correct, while the table (and therefore
 * the next boot) stays exactly as it was.
 */
import { describe, expect, it } from "bun:test";
import { DEFAULT_SETTINGS } from "@pinky/core";
import type { Db, SettingsSnapshot, ThreadEventData } from "@pinky/core";
import { SUPPORTED_PROVIDERS } from "@pinky/runtime";
import type { ToolContext } from "@pinky/runtime";
import { SettingsGetTool, SettingsSetTool } from "../src/settings";
import { makeCtx } from "./helpers";

interface Call {
  sql: string;
  params: unknown[] | undefined;
}

type Row = [scope: string, key: string, value: unknown];

/**
 * Minimal settings-table double, same shape as core's own FakeDb: answers the
 * store's `select scope, key, value from settings` honouring `scope = any($1)`
 * and records the upsert. Nothing here touches Postgres, so what the SQL layer
 * does with a jsonb param is asserted on the *param*, not the row.
 */
class FakeDb implements Db {
  calls: Call[] = [];
  constructor(private rows: Row[] = []) {}

  query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params });
    if (/select scope, key, value from settings/.test(sql)) {
      const wanted = (params?.[0] as string[]) ?? [];
      return Promise.resolve(
        this.rows
          .filter(([scope]) => wanted.includes(scope))
          .map(([scope, key, value]) => ({ scope, key, value })) as T[],
      );
    }
    if (/insert into settings/.test(sql)) return Promise.resolve([] as T[]);
    return Promise.reject(new Error(`FakeDb: no script for SQL: ${sql}`));
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    return (await this.query<T>(sql, params))[0] ?? null;
  }
  tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }
  close(): Promise<void> {
    return Promise.resolve();
  }
  inserts(): Call[] {
    return this.calls.filter((c) => /insert into settings/.test(c.sql));
  }
}

interface Harness {
  ctx: ToolContext;
  db: FakeDb;
  events: ThreadEventData[];
}

interface HarnessOptions {
  enabled?: boolean;
  allowedKeys?: string[];
  /** false => the runtime handed the tool no snapshot at all. */
  settings?: false;
  agentId?: string | null;
  rows?: Row[];
}

function snapshotWith(enabled: boolean, allowedKeys: string[]): SettingsSnapshot {
  const snapshot = structuredClone(DEFAULT_SETTINGS);
  snapshot.selfConfig = { enabled, allowedKeys };
  return snapshot;
}

function harness(opts: HarnessOptions = {}): Harness {
  const db = new FakeDb(opts.rows ?? []);
  const events: ThreadEventData[] = [];
  const ctx = makeCtx("/tmp/pinky-settings-tests", {
    db,
    agentId: opts.agentId === undefined ? "pinky" : opts.agentId,
    settings:
      opts.settings === false
        ? null
        : snapshotWith(opts.enabled ?? true, opts.allowedKeys ?? []),
  });
  ctx.emit = (data) => {
    events.push(data);
    return Promise.resolve();
  };
  return { ctx, db, events };
}

const get = new SettingsGetTool();
const set = new SettingsSetTool();

describe("settings_get", () => {
  it("prints the effective snapshot plus the self-configuration line", async () => {
    const { ctx } = harness({ enabled: true, allowedKeys: ["model", "context.*"] });
    const res = await get.execute({}, ctx);
    expect(res.isError).toBeUndefined();
    const [json, ...rest] = res.text.split("\nself-configuration: ");
    const printed = JSON.parse(json!) as SettingsSnapshot;
    expect(printed.model).toBe(DEFAULT_SETTINGS.model);
    expect(printed.context).toEqual(DEFAULT_SETTINGS.context);
    expect(rest.join("")).toBe("enabled; writable keys: model, context.*");
  });

  it("says disabled with no writable keys by default", async () => {
    const { ctx } = harness({ enabled: false, allowedKeys: [] });
    const res = await get.execute({}, ctx);
    expect(res.text).toContain("self-configuration: disabled; writable keys: (none)");
  });

  it("prints one key when asked", async () => {
    const { ctx } = harness();
    const res = await get.execute({ key: "context.advisoryFraction" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(res.text.split("\n")[0]).toBe(
      `context.advisoryFraction = ${DEFAULT_SETTINGS.context.advisoryFraction}`,
    );
  });

  it("refuses an unknown key instead of printing undefined", async () => {
    const { ctx } = harness();
    const res = await get.execute({ key: "context.hardFractoin" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown setting key 'context.hardFractoin'");
  });

  it("fails cleanly with no snapshot in context", async () => {
    const { ctx } = harness({ settings: false });
    const res = await get.execute({}, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("no settings snapshot in this context");
  });
});

describe("settings_set refusals", () => {
  it("fails cleanly with no snapshot in context", async () => {
    const { ctx, db } = harness({ settings: false });
    const res = await set.execute({ key: "model", value: "openrouter/x/y" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("no settings snapshot in this context");
    expect(db.inserts()).toHaveLength(0);
  });

  it("names the CLI command when self-configuration is disabled", async () => {
    const { ctx, db } = harness({ enabled: false, allowedKeys: ["*"] });
    const res = await set.execute({ key: "model", value: "openrouter/x/y" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("self-configuration is disabled");
    expect(res.text).toContain("pinky config set selfConfig.enabled true");
    expect(res.text).toContain("pinky config set selfConfig.allowedKeys");
    expect(db.inserts()).toHaveLength(0);
  });

  it("lists the allowed patterns when the key is not delegated", async () => {
    const { ctx, db } = harness({ allowedKeys: ["context.*"] });
    const res = await set.execute({ key: "model", value: "openrouter/x/y" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'model' is not delegated to you");
    expect(res.text).toContain("allowed patterns: context.*");
    expect(db.inserts()).toHaveLength(0);
  });

  it("says (none) when the allow-list is empty", async () => {
    const { ctx } = harness({ allowedKeys: [] });
    const res = await set.execute({ key: "model", value: "openrouter/x/y" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("allowed patterns: (none)");
  });

  it("refuses tenantId and selfConfig even under \"*\"", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    for (const key of ["tenantId", "selfConfig", "selfConfig.enabled", "selfConfig.allowedKeys"]) {
      const res = await set.execute({ key, value: "whatever" }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("can never be changed by a tool");
    }
    expect(db.inserts()).toHaveLength(0);
  });

  it("refuses the global scope (human-only) and any other scope word", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    for (const scope of ["global", "tenant", ""]) {
      const res = await set.execute({ key: "model", value: "openrouter/x/y", scope }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("the global scope is human-only");
    }
    expect(db.inserts()).toHaveLength(0);
  });

  it("refuses when the context has no agent id", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"], agentId: null });
    const res = await set.execute({ key: "model", value: "openrouter/x/y" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("no agent id");
    expect(db.inserts()).toHaveLength(0);
  });

  it("requires a key and a value", async () => {
    const { ctx } = harness({ allowedKeys: ["*"] });
    expect((await set.execute({ value: 1 }, ctx)).text).toContain("'key' must be a non-empty");
    expect((await set.execute({ key: "model" }, ctx)).text).toContain("'value' is required");
  });
});

describe("settings_set writes", () => {
  it("writes agent:<id> by default and journals a config event", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["context.*"] });
    const res = await set.execute({ key: "context.advisoryFraction", value: 0.6 }, ctx);

    expect(res.isError).toBeUndefined();
    expect(res.text).toContain("set context.advisoryFraction = 0.6 (agent:pinky)");
    // The loop already read its thresholds; the table is what the next run reads.
    expect(res.text).toContain("next run");

    const insert = db.inserts()[0]!;
    expect(insert.params).toEqual(["agent:pinky", "context.advisoryFraction", 0.6]);

    expect(events).toEqual([
      {
        type: "config",
        scope: "agent:pinky",
        key: "context.advisoryFraction",
        value: 0.6,
        previous: DEFAULT_SETTINGS.context.advisoryFraction,
        by: "pinky",
      },
    ]);
  });

  it("writes channel:<thread.channelId> for scope \"channel\"", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["model"] });
    const res = await set.execute(
      { key: "model", value: "openrouter/x/y", scope: "channel" },
      ctx,
    );
    expect(res.isError).toBeUndefined();
    // makeCtx's thread is c1.
    expect(db.inserts()[0]!.params).toEqual(["channel:c1", "model", "openrouter/x/y"]);
    expect((events[0] as { scope: string }).scope).toBe("channel:c1");
  });

  it("passes a bare boolean through jsonbParam, not as a boolean param", async () => {
    // Same path as the CLI: postgres.js would pre-declare the bool wire type
    // and Postgres refuses boolean -> jsonb, so the store wraps it. What must
    // survive is the JSON: `true`, encoded exactly once.
    const { ctx, db } = harness({ allowedKeys: ["replyGate.*"] });
    const res = await set.execute({ key: "replyGate.classifierEnabled", value: true }, ctx);
    expect(res.isError).toBeUndefined();
    const value = db.inserts()[0]!.params![2];
    expect(typeof value).not.toBe("boolean");
    expect(JSON.stringify(value)).toBe("true");
  });

  it("journals `previous` from the run's snapshot, not from the table", async () => {
    const { ctx, events } = harness({ allowedKeys: ["*"] });
    await set.execute({ key: "model", value: "openrouter/a/b" }, ctx);
    expect((events[0] as { previous: unknown }).previous).toBe(DEFAULT_SETTINGS.model);
  });

  it("honours a subtree pattern: children yes, whole sub-tree no", async () => {
    const { ctx, db } = harness({ allowedKeys: ["context.*"] });
    const child = await set.execute({ key: "context.approxWindowTokens", value: 90_000 }, ctx);
    expect(child.isError).toBeUndefined();

    const whole = await set.execute(
      { key: "context", value: { advisoryFraction: 0.1, hardFraction: 0.2, approxWindowTokens: 5 } },
      ctx,
    );
    expect(whole.isError).toBe(true);
    expect(whole.text).toContain("not delegated to you");
    expect(db.inserts()).toHaveLength(1);
  });

  it("validates against the whole overlay this run reads, not just the target scope", async () => {
    // DEFECT (P1.2): the agent scope already carries hardFraction 0.75, so an
    // advisoryFraction of 0.8 written to the CHANNEL scope is fine against
    // channel + defaults and broken in the snapshot the next run assembles
    // (channel < agent). Validating the target scope alone let it land.
    const { ctx, db } = harness({
      allowedKeys: ["context.*"],
      rows: [["agent:pinky", "context.hardFraction", 0.75]],
    });
    const res = await set.execute(
      { key: "context.advisoryFraction", value: 0.8, scope: "channel" },
      ctx,
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).toContain("less than context.hardFraction");
    expect(db.inserts()).toHaveLength(0);
  });

  it("asks the store for both of the run's scopes when validating", async () => {
    const { ctx, db } = harness({ allowedKeys: ["context.*"] });
    await set.execute({ key: "context.advisoryFraction", value: 0.6 }, ctx);
    // The store merges the target scope first and the run's full overlay
    // second, so what matters is that one of the reads covers both scopes.
    // makeCtx's thread is c1 and the agent is pinky.
    const selects = db.calls
      .filter((c) => /select scope, key, value from settings/.test(c.sql))
      .map((c) => (c.params?.[0] ?? []) as string[]);
    expect(
      selects.some((wanted) => wanted.includes("channel:c1") && wanted.includes("agent:pinky")),
    ).toBe(true);
  });

  it("overlays the scope's stored rows, so a write validates against them", async () => {
    // hardFraction is 0.5 in this agent's scope, so 0.6 is out of order even
    // though it is fine against the defaults.
    const { ctx, db } = harness({
      allowedKeys: ["context.*"],
      rows: [["agent:pinky", "context.hardFraction", 0.5]],
    });
    const res = await set.execute({ key: "context.advisoryFraction", value: 0.6 }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).toContain("less than context.hardFraction");
    expect(db.inserts()).toHaveLength(0);
  });
});

/**
 * `model` is delegable and `validateSettings` only checks its "a/b" shape —
 * core cannot know which providers this build routes. Both refusals below are
 * about values that validate cleanly and then bite on the NEXT run.
 */
describe("settings_set guards the model route", () => {
  it("refuses fake/* outright and names the human path", async () => {
    const { ctx, db } = harness({ allowedKeys: ["model"] });
    const res = await set.execute({ key: "model", value: "fake/echo" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("test route");
    expect(res.text).toContain("pinky config set model fake/echo");
    expect(db.inserts()).toHaveLength(0);
  });

  it("refuses fake/* even under \"*\" and in the channel scope", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute(
      { key: "model", value: "fake/retain-recall", scope: "channel" },
      ctx,
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("test route");
    expect(db.inserts()).toHaveLength(0);
  });

  it("refuses a provider this build cannot route, and lists the ones it can", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "model", value: "cohere/command-r" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown provider 'cohere'");
    for (const provider of SUPPORTED_PROVIDERS) expect(res.text).toContain(provider);
    expect(db.inserts()).toHaveLength(0);
    expect(events).toHaveLength(0);
  });

  it("lets a supported provider through", async () => {
    const { ctx, db } = harness({ allowedKeys: ["model"] });
    const res = await set.execute({ key: "model", value: "anthropic/claude-x" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(db.inserts()[0]!.params).toEqual(["agent:pinky", "model", "anthropic/claude-x"]);
  });

  it("leaves a malformed model to the store's own shape message", async () => {
    const { ctx } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "model", value: "fake" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).not.toContain("test route");
  });

  it("only guards the model key, not every string with a slash in it", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "tenantId", value: "fake/tenant" }, ctx);
    // Refused for being immutable, not by the provider guard.
    expect(res.text).toContain("can never be changed by a tool");
    expect(db.inserts()).toHaveLength(0);
  });
});

describe("settings_set rejections never land", () => {
  it("surfaces a validation failure as `rejected:` and writes nothing", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["context.*"] });
    const res = await set.execute({ key: "context.hardFraction", value: "abc" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).toContain("context.hardFraction");
    expect(db.inserts()).toHaveLength(0);
    // A refused write is not a config event: the log only records what landed.
    expect(events).toHaveLength(0);
  });

  it("surfaces an unknown key with the store's own message", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "contxt.hardFraction", value: 0.9 }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).toContain("unknown setting key");
    expect(db.inserts()).toHaveLength(0);
  });

  it("hints at JSON typing when a number arrives quoted, and does not auto-parse", async () => {
    const { ctx, db } = harness({ allowedKeys: ["context.*"] });
    const res = await set.execute({ key: "context.advisoryFraction", value: "0.6" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).toContain("hint: pass numbers/booleans as JSON values, not strings");
    expect(db.inserts()).toHaveLength(0);
  });

  it("does not hint when the value is legitimately a string", async () => {
    const { ctx } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "model", value: "kimi-k2" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("rejected:");
    expect(res.text).not.toContain("hint:");
  });
});

// ---------------------------------------------------------------------------
// slice 6: `sleep.model` is guarded exactly like `model`
// ---------------------------------------------------------------------------

/**
 * The sleep-time worker's model is delegable (it is in SELF_CONFIG_KEYS and is
 * not immutable), so `"*"` grants it — and a `fake/*` value there is worse
 * than on `model`: the worker's answers become MEMORY ROWS, which outlive the
 * setting that produced them. Same refusal, same message, one level down.
 */
describe("settings_set refuses a fake route for sleep.model too", () => {
  it("refuses fake/sleep and writes nothing", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "sleep.model", value: "fake/sleep" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("test route");
    // The hint names the key that was refused, not a generic one.
    expect(res.text).toContain("pinky config set sleep.model fake/sleep");
    expect(db.inserts()).toHaveLength(0);
    expect(events).toHaveLength(0);
  });

  it("refuses an unknown provider for sleep.model", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "sleep.model", value: "nope/whatever" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown provider 'nope'");
    expect(res.text).toContain(SUPPORTED_PROVIDERS[0]!);
    expect(db.inserts()).toHaveLength(0);
  });

  it("accepts a real provider id", async () => {
    const { ctx, db } = harness({ allowedKeys: ["sleep.*"] });
    const res = await set.execute({ key: "sleep.model", value: "anthropic/claude-x" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(db.inserts()[0]!.params).toEqual(["agent:pinky", "sleep.model", "anthropic/claude-x"]);
  });

  // "" is the documented "use the run model" value: no provider prefix, so the
  // guard has nothing to object to and the key stays writable.
  it('accepts "" (inherit the run model)', async () => {
    const { ctx, db } = harness({ allowedKeys: ["sleep.*"] });
    const res = await set.execute({ key: "sleep.model", value: "" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(db.inserts()[0]!.params).toEqual(["agent:pinky", "sleep.model", ""]);
  });
});

/**
 * The hole the keyed check had: `sleep` is a legal, non-immutable TOP-LEVEL
 * key, so writing the whole object under a `"*"` allow-list smuggled a
 * `fake/*` past a guard that only looked at `model` and `sleep.model`. A
 * parent object is the same write as its leaves.
 */
describe("settings_set walks the written value for models", () => {
  const sleepBlock = (model: string): Record<string, unknown> => ({
    ...DEFAULT_SETTINGS.sleep,
    enabled: true,
    model,
  });

  it("refuses a fake route smuggled inside a whole `sleep` object", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "sleep", value: sleepBlock("fake/echo") }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("test route");
    // Named by the path it was found at, so the human hint is runnable.
    expect(res.text).toContain("pinky config set sleep.model fake/echo");
    expect(db.inserts()).toHaveLength(0);
    expect(events).toHaveLength(0);
  });

  it("refuses an unknown provider smuggled the same way", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "sleep", value: sleepBlock("nope/whatever") }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown provider 'nope'");
    expect(db.inserts()).toHaveLength(0);
  });

  it('accepts a whole `sleep` object whose model is "" (inherit the run model)', async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const value = sleepBlock("");
    const res = await set.execute({ key: "sleep", value }, ctx);
    expect(res.isError).toBeUndefined();
    expect(db.inserts()[0]!.params).toEqual(["agent:pinky", "sleep", value]);
  });

  it("still refuses the top-level model key, and leaves other strings alone", async () => {
    const { ctx } = harness({ allowedKeys: ["*"] });
    const bad = await set.execute({ key: "model", value: "fake/echo" }, ctx);
    expect(bad.text).toContain("pinky config set model fake/echo");

    const { ctx: ctx2, db } = harness({ allowedKeys: ["*"] });
    const ok = await set.execute(
      { key: "memory.embeddingModel", value: "openai/text-embedding-3-small" },
      ctx2,
    );
    expect(ok.isError).toBeUndefined();
    expect(db.inserts()).toHaveLength(1);
  });
});

/**
 * `memory.embeddingModel` is the other `provider/model-id` leaf, and it is the
 * one with the WORST failure: `createEmbedder` throws for an unknown provider
 * and the wiring layer rethrows anything that is not the blank-key case, so a
 * bad value here does not degrade recall — it stops the next boot of every
 * surface. It gets its own list, because the embedding routes are not the chat
 * routes.
 */
describe("settings_set guards memory.embeddingModel against its own routes", () => {
  it("refuses a fake route written directly", async () => {
    const { ctx, db, events } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "memory.embeddingModel", value: "fake/echo" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown embedding provider 'fake'");
    expect(res.text).toContain("pinky config set memory.embeddingModel fake/echo");
    expect(db.inserts()).toHaveLength(0);
    expect(events).toHaveLength(0);
  });

  it("refuses it inside a whole `memory` object", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute(
      { key: "memory", value: { ...DEFAULT_SETTINGS.memory, embeddingModel: "fake/echo" } },
      ctx,
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown embedding provider 'fake'");
    expect(db.inserts()).toHaveLength(0);
  });

  it("refuses a provider that routes chat but not embeddings", async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute(
      { key: "memory.embeddingModel", value: "anthropic/claude-x" },
      ctx,
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("unknown embedding provider 'anthropic'");
    expect(db.inserts()).toHaveLength(0);
    // The same value is perfectly fine as the CHAT model — the two leaves are
    // judged against different lists, which is the whole reason for the split.
    const { ctx: ctx2, db: db2 } = harness({ allowedKeys: ["*"] });
    expect((await set.execute({ key: "model", value: "anthropic/claude-x" }, ctx2)).isError)
      .toBeUndefined();
    expect(db2.inserts()).toHaveLength(1);
  });

  it('accepts "none" (the documented way to turn the vector voice off)', async () => {
    const { ctx, db } = harness({ allowedKeys: ["*"] });
    const res = await set.execute({ key: "memory.embeddingModel", value: "none" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(db.inserts()[0]!.params).toEqual(["agent:pinky", "memory.embeddingModel", "none"]);
  });
});
