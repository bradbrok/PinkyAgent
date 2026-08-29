import { describe, expect, it } from "bun:test";
import {
  MIN_APPROX_WINDOW_TOKENS,
  MIN_FRACTION_GAP,
  SettingsStore,
  isSelfConfigWritable,
  validateSettings,
} from "../src/settings";
import { DEFAULT_SETTINGS, type SettingsSnapshot } from "../src/config";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
}

class FakeDb implements Db {
  calls: Call[] = [];
  private script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>;

  constructor(script: Array<{ pattern: RegExp; respond: (params?: unknown[]) => unknown[] }>) {
    this.script = script;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params });
    for (const s of this.script) {
      if (s.pattern.test(sql)) return s.respond(params) as T[];
    }
    throw new Error(`FakeDb: no script for SQL: ${sql}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }
  tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }
  async close(): Promise<void> {}
}

type Row = [scope: string, key: string, value: unknown];

const recordsFrom = (entries: Row[]) =>
  entries.map(([scope, key, value]) => ({ scope, key, value }));

/** A db that hands back every row regardless of the `scope = any($1)` filter,
 *  so the store's own scope filtering is what the assertions exercise. */
const dbWithAllRows = (entries: Row[]) =>
  new FakeDb([{ pattern: /select scope, key, value from settings/, respond: () => recordsFrom(entries) }]);

/** A db that honours `where scope = any($1)`, like Postgres does. */
const dbWithScopedRows = (entries: Row[], deleted: unknown[] = []) =>
  new FakeDb([
    {
      pattern: /select scope, key, value from settings/,
      respond: (params) => {
        const wanted = (params?.[0] as string[]) ?? [];
        return recordsFrom(entries).filter((r) => wanted.includes(r.scope));
      },
    },
    { pattern: /insert into settings/, respond: () => [] },
    { pattern: /delete from settings/, respond: () => deleted },
  ]);

const selectCalls = (db: FakeDb) => db.calls.filter((c) => /^\s*select/.test(c.sql));
const insertCalls = (db: FakeDb) => db.calls.filter((c) => /insert into settings/.test(c.sql));

/** Captures the store's onWarning output instead of writing to stderr. */
function warnings(): { onWarning: (m: string) => void; messages: string[] } {
  const messages: string[] = [];
  return { messages, onWarning: (m) => messages.push(m) };
}

/** A store whose warnings are collected rather than printed. */
function loudStore(db: Db): { store: SettingsStore; messages: string[] } {
  const w = warnings();
  return { store: new SettingsStore(db, { onWarning: w.onWarning }), messages: w.messages };
}

describe("SettingsStore.load", () => {
  it("returns defaults when no rows exist", async () => {
    const db = new FakeDb([{ pattern: /select scope, key, value from settings/, respond: () => [] }]);
    const store = new SettingsStore(db);
    const snapshot = await store.load();
    expect(snapshot).toEqual(DEFAULT_SETTINGS);
  });

  it("applies scope overlay: global < channel:<id> < agent:<id>", async () => {
    const db = dbWithAllRows([
      ["global", "model", "openrouter/foo"],
      ["channel:chan1", "model", "openrouter/channel-override"],
      ["agent:agent1", "model", "openrouter/agent-override"],
    ]);
    const store = new SettingsStore(db);
    const snapshot = (await store.load({
      scopes: ["channel:chan1", "agent:agent1"],
    })) as SettingsSnapshot & Record<string, unknown>;
    expect(snapshot.model).toBe("openrouter/agent-override");
  });

  it("applies class order (channel < agent) regardless of the list order", async () => {
    const db = dbWithAllRows([
      ["channel:chan1", "model", "openrouter/channel-override"],
      ["agent:agent1", "model", "openrouter/agent-override"],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load({ scopes: ["agent:agent1", "channel:chan1"] });
    expect(snapshot.model).toBe("openrouter/agent-override");
  });

  it("ignores rows for scopes that were not requested", async () => {
    const db = dbWithAllRows([
      ["global", "model", "openrouter/global"],
      ["channel:chan1", "model", "openrouter/chan1"],
      ["channel:chan2", "model", "openrouter/chan2"],
      ["agent:other", "model", "openrouter/other-agent"],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load({ scopes: ["channel:chan1"] });
    expect(snapshot.model).toBe("openrouter/chan1");
  });

  it("load() with no args is defaults + global only", async () => {
    const db = dbWithAllRows([
      ["global", "model", "openrouter/global"],
      ["channel:chan1", "model", "openrouter/chan1"],
      ["agent:agent1", "model", "openrouter/agent1"],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load();
    expect(snapshot.model).toBe("openrouter/global");
    expect(db.calls[0]!.params).toEqual([["global"]]);
  });

  it("fetches only the requested scopes, not the whole table", async () => {
    const db = dbWithAllRows([]);
    const store = new SettingsStore(db);
    await store.load({ scopes: ["channel:slack:C123", "agent:pinky"] });
    expect(db.calls[0]!.sql).toMatch(/where scope = any\(\$1\)/);
    expect(db.calls[0]!.params).toEqual([["global", "channel:slack:C123", "agent:pinky"]]);
  });

  it("lets the later of two same-class scopes win", async () => {
    const db = dbWithAllRows([
      ["channel:chan1", "model", "openrouter/chan1"],
      ["channel:chan2", "model", "openrouter/chan2"],
    ]);
    const store = new SettingsStore(db);
    expect((await store.load({ scopes: ["channel:chan1", "channel:chan2"] })).model).toBe("openrouter/chan2");
    expect((await store.load({ scopes: ["channel:chan2", "channel:chan1"] })).model).toBe("openrouter/chan1");
  });

  it("merges dotted sub-paths into nested objects", async () => {
    const db = dbWithAllRows([
      ["global", "context.advisoryFraction", 0.5],
      ["global", "context.approxWindowTokens", 90_000],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load();
    expect(snapshot.context.advisoryFraction).toBe(0.5);
    expect(snapshot.context.approxWindowTokens).toBe(90_000);
    expect(snapshot.context.hardFraction).toBe(DEFAULT_SETTINGS.context.hardFraction);
  });

  it("overrides whole top-level objects when key targets the parent", async () => {
    const db = dbWithAllRows([
      ["global", "context", { advisoryFraction: 0.5, hardFraction: 0.99, approxWindowTokens: 42_000 }],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load();
    expect(snapshot.context.hardFraction).toBe(0.99);
  });

  it("applies a dotted key after the parent sub-tree it refines", async () => {
    const db = dbWithAllRows([
      ["global", "context.advisoryFraction", 0.6],
      ["global", "context", { advisoryFraction: 0.1, hardFraction: 0.99, approxWindowTokens: 42_000 }],
    ]);
    const store = new SettingsStore(db);
    const snapshot = await store.load();
    expect(snapshot.context).toEqual({
      advisoryFraction: 0.6,
      hardFraction: 0.99,
      approxWindowTokens: 42_000,
    });
  });

  it("rejects a malformed scope before querying", async () => {
    const db = dbWithAllRows([]);
    const store = new SettingsStore(db);
    await expect(store.load({ scopes: ["channel:"] })).rejects.toThrow(/Invalid settings scope/);
    expect(db.calls).toHaveLength(0);
  });
});

/**
 * The half of the policy that keeps the agent alive: `load()` PRUNES AND WARNS.
 *
 * A row nothing can read used to throw out of `load()`, `config get`, every
 * per-run reload AND `config set <any other key>` (which re-validates the scope
 * first) — so the one command that could have cleared it was itself wedged.
 * Config lives in the database precisely so a bad value cannot stop a process
 * from starting; these tests are that promise.
 */
describe("SettingsStore.load leniency (a bad row can never brick a run)", () => {
  it("drops a row whose key names no setting, keeping the rest of the scope", async () => {
    const db = dbWithAllRows([
      ["global", "context.hardFractoin", 0.95], // the typo that used to wedge it
      ["global", "model", "openrouter/still/works"],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.model).toBe("openrouter/still/works");
    expect(snapshot.context).toEqual(DEFAULT_SETTINGS.context);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain("context.hardFractoin");
    expect(messages[0]).toContain("no such setting key");
    expect(messages[0]).toContain("config unset");
  });

  it("drops an unknown TOP-LEVEL row and an unknown sub-key inside a good object", async () => {
    const db = dbWithAllRows([
      ["global", "replyGait", { classifierEnabled: true }],
      ["global", "context", { ...DEFAULT_SETTINGS.context, hardFractoin: 0.95 }],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    // The object row itself is legal; only the stray sub-key goes.
    expect(snapshot.context).toEqual(DEFAULT_SETTINGS.context);
    expect(snapshot.replyGate).toEqual(DEFAULT_SETTINGS.replyGate);
    expect(messages.some((m) => m.includes("replyGait"))).toBe(true);
    expect(messages.some((m) => m.includes("context.hardFractoin"))).toBe(true);
  });

  it("replaces an invalid leaf VALUE with its default and warns", async () => {
    const db = dbWithAllRows([
      ["global", "context.hardFraction", "abc"],
      ["global", "memory.recallLimit", 0],
      ["global", "model", "openrouter/kept"],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.context.hardFraction).toBe(DEFAULT_SETTINGS.context.hardFraction);
    expect(snapshot.memory.recallLimit).toBe(DEFAULT_SETTINGS.memory.recallLimit);
    expect(snapshot.model).toBe("openrouter/kept"); // untouched
    expect(messages.some((m) => /context\.hardFraction.*falling back/.test(m))).toBe(true);
    expect(messages.some((m) => /memory\.recallLimit.*falling back/.test(m))).toBe(true);
  });

  it("repairs a CROSS-FIELD conflict by falling back to the whole sub-tree", async () => {
    // hardFraction 0.5 is a fine number on its own; with the default advisory
    // of 0.7 the ladder is inverted, and resetting advisory (already its
    // default) would not help — so the sub-tree goes back to the defaults.
    const db = dbWithAllRows([["global", "context.hardFraction", 0.5]]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.context).toEqual(DEFAULT_SETTINGS.context);
    expect(messages.some((m) => m.includes(`falling back to the default for "context"`))).toBe(true);
  });

  it("filters an allow-list entry that names nothing real, keeping the good ones", async () => {
    const db = dbWithAllRows([
      ["global", "selfConfig.enabled", true],
      ["agent:pinky", "selfConfig.allowedKeys", ["context.*", "contxt.hardFraction", "tenantId", 7]],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load({ scopes: ["agent:pinky"] });

    // The delegation survives, minus the entries that cannot mean anything.
    expect(snapshot.selfConfig).toEqual({ enabled: true, allowedKeys: ["context.*"] });
    expect(messages.some((m) => m.includes("contxt.hardFraction"))).toBe(true);
    expect(messages.some((m) => m.includes("tenantId"))).toBe(true);
    expect(messages).toHaveLength(3);
  });

  it("never throws, whatever the table holds", async () => {
    const db = dbWithAllRows([
      ["global", "tenantId", 7],
      ["global", "model", ""],
      ["global", "context", "not an object"],
      ["global", "replyGate", null],
      ["global", "memory", []],
      ["global", "selfConfig", "nope"],
      ["global", "junk", { any: "thing" }],
    ]);
    const { store, messages } = loudStore(db);
    expect(await store.load()).toEqual(DEFAULT_SETTINGS);
    expect(messages.length).toBeGreaterThan(0);
  });

  it("warns once per problem per store, not once per run", async () => {
    // Long-lived surfaces reload settings on every run; repeating the same
    // stderr line every wake would train the operator to ignore it.
    const db = dbWithAllRows([["global", "context.hardFractoin", 0.95]]);
    const { store, messages } = loudStore(db);
    await store.load();
    await store.load();
    await store.load();
    expect(messages).toHaveLength(1);
  });

  it("defaults its warnings to console.warn — stderr, never the JSONL stdout", async () => {
    const original = console.warn;
    const seen: unknown[] = [];
    console.warn = (...args: unknown[]): void => {
      seen.push(args[0]);
    };
    try {
      await new SettingsStore(dbWithAllRows([["global", "nope", 1]])).load();
    } finally {
      console.warn = original;
    }
    expect(String(seen[0])).toContain("nope");
  });
});

describe("validateSettings", () => {
  const withContext = (context: Record<string, unknown>) => ({ ...DEFAULT_SETTINGS, context });
  const withMemory = (memory: Record<string, unknown>) => ({ ...DEFAULT_SETTINGS, memory });
  const withSelfConfig = (selfConfig: unknown) => ({ ...DEFAULT_SETTINGS, selfConfig });
  const allowing = (...allowedKeys: unknown[]) => withSelfConfig({ enabled: true, allowedKeys });

  it("accepts the defaults and returns them", () => {
    expect(validateSettings(structuredClone(DEFAULT_SETTINGS))).toEqual(DEFAULT_SETTINGS);
  });

  it("rejects a non-object candidate", () => {
    expect(() => validateSettings("nope")).toThrow(/expected an object/);
    expect(() => validateSettings(null)).toThrow(/expected an object/);
    expect(() => validateSettings([])).toThrow(/expected an object/);
  });

  const cases: Array<[name: string, candidate: unknown, match: RegExp]> = [
    ["empty tenantId", { ...DEFAULT_SETTINGS, tenantId: "" }, /tenantId/],
    ["non-string tenantId", { ...DEFAULT_SETTINGS, tenantId: 7 }, /tenantId/],
    ["model without a provider prefix", { ...DEFAULT_SETTINGS, model: "kimi-k2" }, /model/],
    ["model with a leading slash", { ...DEFAULT_SETTINGS, model: "/kimi-k2" }, /model/],
    ["model with a trailing slash", { ...DEFAULT_SETTINGS, model: "openrouter/" }, /model/],
    ["empty model", { ...DEFAULT_SETTINGS, model: "" }, /model/],
    [
      "string advisoryFraction",
      withContext({ ...DEFAULT_SETTINGS.context, advisoryFraction: "abc" }),
      /context\.advisoryFraction/,
    ],
    [
      "string hardFraction",
      withContext({ ...DEFAULT_SETTINGS.context, hardFraction: "abc" }),
      /context\.hardFraction/,
    ],
    [
      "hardFraction above 1",
      withContext({ ...DEFAULT_SETTINGS.context, hardFraction: 1.5 }),
      /context\.hardFraction/,
    ],
    [
      "advisoryFraction of 0",
      withContext({ ...DEFAULT_SETTINGS.context, advisoryFraction: 0 }),
      /context\.advisoryFraction/,
    ],
    [
      "advisory >= hard",
      withContext({ ...DEFAULT_SETTINGS.context, advisoryFraction: 0.95, hardFraction: 0.9 }),
      /context\.advisoryFraction: expected to be less than context\.hardFraction/,
    ],
    [
      "fractional approxWindowTokens",
      withContext({ ...DEFAULT_SETTINGS.context, approxWindowTokens: 1.5 }),
      /context\.approxWindowTokens/,
    ],
    [
      "zero approxWindowTokens",
      withContext({ ...DEFAULT_SETTINGS.context, approxWindowTokens: 0 }),
      /context\.approxWindowTokens/,
    ],
    // Floors. A "valid" number that makes the pressure ladder meaningless is
    // not valid: below ~1000 tokens the system prompt alone clears
    // hardFraction, so every run opens over the hard threshold and sheds
    // context it never had.
    [
      "an approxWindowTokens under the floor",
      withContext({ ...DEFAULT_SETTINGS.context, approxWindowTokens: 999 }),
      /context\.approxWindowTokens: expected an integer >= 1000/,
    ],
    [
      "a hard/advisory gap too narrow to act on",
      withContext({ ...DEFAULT_SETTINGS.context, advisoryFraction: 0.89, hardFraction: 0.9 }),
      /context\.advisoryFraction: expected to be at least 0\.05 below context\.hardFraction/,
    ],
    ["missing context", { ...DEFAULT_SETTINGS, context: "nope" }, /context: expected an object/],
    [
      "non-boolean classifierEnabled",
      { ...DEFAULT_SETTINGS, replyGate: { classifierEnabled: "yes" } },
      /replyGate\.classifierEnabled/,
    ],
    ["unknown top-level key", { ...DEFAULT_SETTINGS, contxt: { hardFraction: 0.9 } }, /contxt: unknown setting key/],
    [
      "unknown nested key",
      withContext({ ...DEFAULT_SETTINGS.context, hardFractoin: 0.9 }),
      /context\.hardFractoin: unknown setting key/,
    ],
    // --- memory plane (DESIGN.md §5) -------------------------------------
    ["missing memory sub-tree", { ...DEFAULT_SETTINGS, memory: "nope" }, /memory: expected an object/],
    [
      "embeddingModel without a provider prefix",
      withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "text-embedding-3-small" }),
      /memory\.embeddingModel: expected "none" or "provider\/model-id"/,
    ],
    [
      "embeddingModel with a leading slash",
      withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "/text-embedding-3-small" }),
      /memory\.embeddingModel/,
    ],
    [
      "embeddingModel with a trailing slash",
      withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "openai/" }),
      /memory\.embeddingModel/,
    ],
    [
      "empty embeddingModel",
      withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "" }),
      /memory\.embeddingModel: expected a non-empty string/,
    ],
    [
      "non-string embeddingModel",
      withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: 7 }),
      /memory\.embeddingModel: expected a non-empty string/,
    ],
    [
      "non-boolean autoRecall",
      withMemory({ ...DEFAULT_SETTINGS.memory, autoRecall: "yes" }),
      /memory\.autoRecall: expected a boolean/,
    ],
    [
      "fractional recallLimit",
      withMemory({ ...DEFAULT_SETTINGS.memory, recallLimit: 1.5 }),
      /memory\.recallLimit: expected a positive integer/,
    ],
    [
      "zero recallLimit",
      withMemory({ ...DEFAULT_SETTINGS.memory, recallLimit: 0 }),
      /memory\.recallLimit: expected a positive integer/,
    ],
    [
      "negative recallTokenBudget",
      withMemory({ ...DEFAULT_SETTINGS.memory, recallTokenBudget: -1 }),
      /memory\.recallTokenBudget: expected a positive integer/,
    ],
    [
      "string recallTokenBudget",
      withMemory({ ...DEFAULT_SETTINGS.memory, recallTokenBudget: "5000" }),
      /memory\.recallTokenBudget: expected a positive integer/,
    ],
    [
      "unknown memory key",
      withMemory({ ...DEFAULT_SETTINGS.memory, autoRecal: true }),
      /memory\.autoRecal: unknown setting key/,
    ],
    // --- self-configuration (DESIGN.md P8, revised) -----------------------
    ["missing selfConfig sub-tree", { ...DEFAULT_SETTINGS, selfConfig: "nope" }, /selfConfig: expected an object/],
    [
      "non-boolean selfConfig.enabled",
      withSelfConfig({ enabled: "yes", allowedKeys: [] }),
      /selfConfig\.enabled: expected a boolean/,
    ],
    [
      "non-array allowedKeys",
      withSelfConfig({ enabled: false, allowedKeys: "model" }),
      /selfConfig\.allowedKeys: expected an array of strings/,
    ],
    [
      "unknown key inside selfConfig",
      withSelfConfig({ enabled: false, allowedKeys: [], allowAll: true }),
      /selfConfig\.allowAll: unknown setting key/,
    ],
    [
      "non-string allow-list entry",
      allowing(7),
      /selfConfig\.allowedKeys\[0\]: expected a non-empty string/,
    ],
    [
      "empty allow-list entry",
      allowing(""),
      /selfConfig\.allowedKeys\[0\]: expected a non-empty string/,
    ],
    [
      "allow-list entry naming nothing real",
      allowing("model", "contxt.hardFraction"),
      /selfConfig\.allowedKeys\[1\]: "contxt\.hardFraction" is not a setting key/,
    ],
    [
      "subtree pattern over a leaf",
      allowing("model.*"),
      /selfConfig\.allowedKeys\[0\]: "model\.\*" names no settings sub-tree/,
    ],
    // The two the runtime would refuse anyway — rejected here, where the
    // human writing `pinky config set` still gets to read why.
    [
      "tenantId in the allow-list",
      allowing("tenantId"),
      /selfConfig\.allowedKeys\[0\]: "tenantId" can never be delegated to an agent/,
    ],
    [
      "selfConfig in the allow-list",
      allowing("context.*", "selfConfig"),
      /selfConfig\.allowedKeys\[1\]: "selfConfig" can never be delegated to an agent/,
    ],
    [
      "selfConfig subtree in the allow-list",
      allowing("selfConfig.*"),
      /selfConfig\.allowedKeys\[0\]: "selfConfig\.\*" can never be delegated to an agent/,
    ],
    [
      "selfConfig leaf in the allow-list",
      allowing("selfConfig.allowedKeys"),
      /selfConfig\.allowedKeys\[0\]: "selfConfig\.allowedKeys" can never be delegated to an agent/,
    ],
  ];

  for (const [name, candidate, match] of cases) {
    it(`rejects ${name}`, () => {
      expect(() => validateSettings(candidate)).toThrow(match);
    });
  }

  it("accepts the floors themselves, exactly (no float slop)", () => {
    expect(MIN_APPROX_WINDOW_TOKENS).toBe(1000);
    expect(MIN_FRACTION_GAP).toBe(0.05);
    const at = (context: Record<string, unknown>) => () =>
      validateSettings({ ...DEFAULT_SETTINGS, context });
    expect(at({ ...DEFAULT_SETTINGS.context, approxWindowTokens: 1000 })).not.toThrow();
    // 0.75 - 0.7 is 0.04999999999999993 in binary floating point; a gap the
    // human wrote as exactly 0.05 must not be rejected by the representation.
    expect(at({ advisoryFraction: 0.7, hardFraction: 0.75, approxWindowTokens: 1000 })).not.toThrow();
    expect(at({ advisoryFraction: 0.95, hardFraction: 1, approxWindowTokens: 1000 })).not.toThrow();
  });

  it("accepts \"none\" as the embeddingModel (FTS-only recall)", () => {
    const candidate = withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "none" });
    expect(validateSettings(candidate).memory.embeddingModel).toBe("none");
  });

  it("accepts a provider-prefixed embeddingModel with extra path segments", () => {
    const candidate = withMemory({ ...DEFAULT_SETTINGS.memory, embeddingModel: "openrouter/openai/text-embedding-3-large" });
    expect(validateSettings(candidate).memory.embeddingModel).toBe("openrouter/openai/text-embedding-3-large");
  });

  it("lists the known memory keys when rejecting an unknown one", () => {
    let message = "";
    try {
      validateSettings(withMemory({ ...DEFAULT_SETTINGS.memory, nope: 1 }));
    } catch (err) {
      message = (err as Error).message;
    }
    // Same "known: a, b, c" affordance context/replyGate give a typo'd key.
    expect(message).toContain(
      `memory.nope: unknown setting key (known: ${Object.keys(DEFAULT_SETTINGS.memory).join(", ")})`,
    );
  });

  it("accepts the delegation forms a human can write", () => {
    for (const entry of ["*", "model", "context", "context.*", "context.advisoryFraction", "memory.autoRecall"]) {
      expect(validateSettings(allowing(entry)).selfConfig.allowedKeys).toEqual([entry]);
    }
  });

  it("ships self-configuration off with an empty allow-list", () => {
    expect(validateSettings(structuredClone(DEFAULT_SETTINGS)).selfConfig).toEqual({
      enabled: false,
      allowedKeys: [],
    });
  });

  it("reports every bad key in a single error", () => {
    let message = "";
    try {
      validateSettings({
        tenantId: "",
        model: "kimi-k2",
        context: { approxWindowTokens: -1 },
        replyGate: {},
        memory: {},
      });
    } catch (err) {
      message = (err as Error).message;
    }
    for (const key of [
      "tenantId",
      "model",
      "context.advisoryFraction",
      "context.hardFraction",
      "context.approxWindowTokens",
      "replyGate.classifierEnabled",
      "memory.embeddingModel",
      "memory.autoRecall",
      "memory.recallLimit",
      "memory.recallTokenBudget",
    ]) {
      expect(message).toContain(key);
    }
  });
});

describe("SettingsStore.set", () => {
  it("upserts with the given scope + key and updated_at bump", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await store.set("channel:chan1", "context.advisoryFraction", 0.5);
    const insert = insertCalls(db)[0]!;
    // PLAIN value, not JSON.stringify(0.5): the driver encodes a jsonb param
    // once by itself (pg.ts's JSONB CONTRACT). Pre-encoding here is what made
    // every row land as a jsonb *string*.
    expect(insert.params).toEqual(["channel:chan1", "context.advisoryFraction", 0.5]);
    expect(insert.sql).toMatch(
      /on conflict \(scope, key\) do update set value = excluded\.value, updated_at = now\(\)/,
    );
  });

  it("binds object values as the plain object, not pre-encoded JSON text", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    const context = { advisoryFraction: 0.5, hardFraction: 0.8, approxWindowTokens: 1_000 };
    await store.set("global", "context", context);
    expect(insertCalls(db)[0]!.params).toEqual(["global", "context", context]);
    expect(typeof insertCalls(db)[0]!.params![2]).toBe("object");
  });

  it("binds a string value bare, so jsonb holds a string and not a quoted blob", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await store.set("global", "model", "openrouter/moonshotai/kimi-k2");
    expect(insertCalls(db)[0]!.params).toEqual([
      "global",
      "model",
      "openrouter/moonshotai/kimi-k2",
    ]);
  });

  it("hands a boolean over as a toJSON carrier that encodes to bare `true`", async () => {
    // The one exception to "pass the plain value": postgres.js pre-declares
    // the bool wire type for a JS boolean and Postgres refuses to coerce
    // boolean -> jsonb ("column is of type jsonb but expression is of type
    // boolean"), so jsonbParam() wraps it. What matters is the JSON it
    // produces: `true`, encoded exactly once.
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await store.set("global", "replyGate.classifierEnabled", true);
    const value = insertCalls(db)[0]!.params![2];
    expect(typeof value).not.toBe("boolean");
    expect(JSON.stringify(value)).toBe("true");
  });

  it("hands memory.autoRecall over as a toJSON carrier that encodes to bare `false`", async () => {
    // Same jsonbParam() path as replyGate.classifierEnabled: postgres.js would
    // pre-declare the bool wire type and Postgres refuses boolean -> jsonb.
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await store.set("global", "memory.autoRecall", false);
    const insert = insertCalls(db)[0]!;
    expect(insert.params![0]).toBe("global");
    expect(insert.params![1]).toBe("memory.autoRecall");
    const value = insert.params![2];
    expect(typeof value).not.toBe("boolean");
    expect(JSON.stringify(value)).toBe("false");
  });

  it("round-trips memory.autoRecall = false through load", async () => {
    const db = dbWithScopedRows([["global", "memory.autoRecall", false]]);
    const snapshot = await new SettingsStore(db).load();
    expect(snapshot.memory.autoRecall).toBe(false);
    expect(snapshot.memory.recallLimit).toBe(DEFAULT_SETTINGS.memory.recallLimit);
  });

  it("writes memory.embeddingModel = \"none\" bare, and rejects an unprefixed id", async () => {
    const ok = dbWithScopedRows([]);
    await new SettingsStore(ok).set("global", "memory.embeddingModel", "none");
    expect(insertCalls(ok)[0]!.params).toEqual(["global", "memory.embeddingModel", "none"]);

    const bad = dbWithScopedRows([]);
    await expect(
      new SettingsStore(bad).set("global", "memory.embeddingModel", "text-embedding-3-small"),
    ).rejects.toThrow(/memory\.embeddingModel/);
    expect(insertCalls(bad)).toHaveLength(0);
  });

  it("rejects an unknown memory key typo and issues no insert", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.set("global", "memory.recallLimt", 5)).rejects.toThrow(
      /memory\.recallLimt: unknown setting key/,
    );
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("rejects a bad value and issues no insert", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.set("global", "context.hardFraction", "abc")).rejects.toThrow(/context\.hardFraction/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("rejects an unknown key typo and issues no insert", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.set("global", "contxt.hardFraction", 0.95)).rejects.toThrow(/contxt: unknown setting key/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("rejects a malformed scope without touching the db", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.set("channel", "model", "openrouter/x")).rejects.toThrow(/Invalid settings scope/);
    await expect(store.set("agent:", "model", "openrouter/x")).rejects.toThrow(/Invalid settings scope/);
    await expect(store.set("", "model", "openrouter/x")).rejects.toThrow(/Invalid settings scope/);
    expect(db.calls).toHaveLength(0);
  });

  it("rejects an empty or malformed key without touching the db", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.set("global", "", 1)).rejects.toThrow(/Invalid settings key/);
    await expect(store.set("global", "context..hardFraction", 0.9)).rejects.toThrow(/Invalid settings key/);
    expect(db.calls).toHaveLength(0);
  });

  it("validates the candidate against that scope's own effective snapshot", async () => {
    // hardFraction is 0.9 by default, so 0.95 is out of order...
    const strict = dbWithScopedRows([]);
    await expect(new SettingsStore(strict).set("global", "context.advisoryFraction", 0.95)).rejects.toThrow(
      /less than context\.hardFraction/,
    );
    expect(insertCalls(strict)).toHaveLength(0);

    // ...but fine once that scope has already raised hardFraction.
    const relaxed = dbWithScopedRows([["global", "context.hardFraction", 0.99]]);
    await new SettingsStore(relaxed).set("global", "context.advisoryFraction", 0.93);
    expect(insertCalls(relaxed)[0]!.params).toEqual(["global", "context.advisoryFraction", 0.93]);
  });

  it("stays STRICT: a bad value is the caller's error, never a repaired default", async () => {
    // The mirror image of load()'s leniency. The human is standing right here
    // with the value they typed; silently storing something else would be the
    // worst of both worlds.
    const db = dbWithScopedRows([]);
    const { store, messages } = loudStore(db);
    await expect(store.set("global", "context.hardFraction", "abc")).rejects.toThrow(
      /context\.hardFraction: expected a number/,
    );
    expect(insertCalls(db)).toHaveLength(0);
    expect(messages).toEqual([]);
  });

  it("a stale unreadable row no longer blocks an unrelated write", async () => {
    // THE WEDGE: set() re-validates the scope first, so one junk row used to
    // reject `config set` for every OTHER key — including the writes that
    // would have fixed the configuration.
    const db = dbWithScopedRows([
      ["global", "context.hardFractoin", 0.95],
      ["agent:pinky", "not.a.setting", { some: "junk" }],
    ]);
    const { store, messages } = loudStore(db);
    await store.set("agent:pinky", "model", "openrouter/works/now");
    expect(insertCalls(db)[0]!.params).toEqual(["agent:pinky", "model", "openrouter/works/now"]);
    // The write path still SAYS what it ignored.
    expect(messages.some((m) => m.includes("context.hardFractoin"))).toBe(true);
    expect(messages.some((m) => m.includes("not.a.setting"))).toBe(true);
  });

  it("unset deletes exactly one row and reports whether there was one", async () => {
    const hit = dbWithScopedRows([], [{ "?column?": 1 }]);
    expect(await new SettingsStore(hit).unset("agent:pinky", "model")).toBe(true);
    const call = hit.calls.find((c) => /delete from settings/.test(c.sql))!;
    expect(call.sql).toMatch(/delete from settings where scope = \$1 and key = \$2 returning 1/);
    expect(call.params).toEqual(["agent:pinky", "model"]);

    const miss = dbWithScopedRows([], []);
    expect(await new SettingsStore(miss).unset("global", "model")).toBe(false);
  });

  it("unset removes a key that is NOT a setting — that is what it is for", async () => {
    // The escape hatch: `load()` warns about a junk row, and this is the
    // command that clears it. Validating the key here would refuse to remove
    // exactly the rows worth removing.
    const db = dbWithScopedRows([], [{ "?column?": 1 }]);
    expect(await new SettingsStore(db).unset("global", "context.hardFractoin")).toBe(true);
    // No read, no validation, one statement.
    expect(db.calls).toHaveLength(1);
  });

  it("unset still validates the scope and the key SHAPE", async () => {
    const db = dbWithScopedRows([]);
    const store = new SettingsStore(db);
    await expect(store.unset("channel:", "model")).rejects.toThrow(/Invalid settings scope/);
    await expect(store.unset("global", "")).rejects.toThrow(/Invalid settings key/);
    await expect(store.unset("global", "a..b")).rejects.toThrow(/Invalid settings key/);
    expect(db.calls).toHaveLength(0);
  });
});

/**
 * Cross-scope validation (the P8 hole). `set()` checks
 * `defaults + global + <target scope>`, but a RUN loads
 * `defaults + global + channel:X + agent:A` — so two writes that are each
 * valid in their own scope can compose into a snapshot no run can use.
 */
describe("SettingsStore.set with validateScopes", () => {
  const RUN_SCOPES = ["channel:c1", "agent:pinky"];

  it("rejects a write that only breaks once the run's scopes are stacked", async () => {
    // The agent lowers hardFraction to 0.75 in its own scope: fine there
    // (advisory 0.7 is the default). Then the channel raises advisory to 0.8:
    // fine against the channel's own view (hard is still 0.9 there), and
    // catastrophic for every run on that channel, where both rows apply.
    const db = dbWithScopedRows([["agent:pinky", "context.hardFraction", 0.75]]);
    const store = new SettingsStore(db);

    // Without the cross-scope check, this is a legal write.
    await store.set("channel:c1", "context.advisoryFraction", 0.8);
    expect(insertCalls(db)).toHaveLength(1);

    const guarded = dbWithScopedRows([["agent:pinky", "context.hardFraction", 0.75]]);
    await expect(
      new SettingsStore(guarded).set("channel:c1", "context.advisoryFraction", 0.8, {
        validateScopes: RUN_SCOPES,
      }),
    ).rejects.toThrow(/less than context\.hardFraction/);
    expect(insertCalls(guarded)).toHaveLength(0);
  });

  it("names the scopes it checked, so the error is actionable", async () => {
    const db = dbWithScopedRows([["agent:pinky", "context.hardFraction", 0.75]]);
    const message = await new SettingsStore(db)
      .set("channel:c1", "context.advisoryFraction", 0.8, { validateScopes: RUN_SCOPES })
      .then(() => "", (e: Error) => e.message);
    expect(message).toContain("the scopes a run loads");
    expect(message).toContain("agent:pinky");
    expect(message).toContain("channel:c1");
  });

  it("adds the target scope to the list when the caller left it out", async () => {
    const db = dbWithScopedRows([]);
    await new SettingsStore(db).set("agent:pinky", "model", "openrouter/a/b", {
      validateScopes: ["channel:c1"],
    });
    const wide = selectCalls(db)[1]!;
    expect(wide.params).toEqual([["global", "channel:c1", "agent:pinky"]]);
  });

  it("still rejects on the target scope alone, before widening", async () => {
    const db = dbWithScopedRows([]);
    await expect(
      new SettingsStore(db).set("global", "context.advisoryFraction", 0.95, {
        validateScopes: RUN_SCOPES,
      }),
    ).rejects.toThrow(/less than context\.hardFraction/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("an empty or absent list is the old behaviour: target scope only", async () => {
    const db = dbWithScopedRows([["agent:pinky", "context.hardFraction", 0.75]]);
    await new SettingsStore(db).set("channel:c1", "context.advisoryFraction", 0.8, {
      validateScopes: [],
    });
    expect(insertCalls(db)).toHaveLength(1);
    expect(selectCalls(db)).toHaveLength(1);
  });

  it("rejects a malformed scope in the list without writing", async () => {
    const db = dbWithScopedRows([]);
    await expect(
      new SettingsStore(db).set("global", "model", "openrouter/a/b", { validateScopes: ["agent:"] }),
    ).rejects.toThrow(/Invalid settings scope/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("reads only the target scope (plus global) when validating", async () => {
    const db = dbWithScopedRows([["channel:chan2", "model", "not-a-valid-model"]]);
    const store = new SettingsStore(db);
    await store.set("channel:chan1", "model", "openrouter/ok");
    expect(selectCalls(db)[0]!.params).toEqual([["global", "channel:chan1"]]);
    expect(insertCalls(db)).toHaveLength(1);
  });
});

/**
 * The allow-list matcher (DESIGN.md P8, revised). Pure, and deliberately
 * ignorant of whether a key exists: SettingsStore.set is what rejects an
 * unknown key, so this answers exactly one question — did a human delegate it?
 */
describe("isSelfConfigWritable", () => {
  const cases: Array<[key: string, allowed: string[], expected: boolean]> = [
    // "*" — everything except the immutables.
    ["model", ["*"], true],
    ["context", ["*"], true],
    ["context.advisoryFraction", ["*"], true],
    ["tenantId", ["*"], false],
    ["selfConfig", ["*"], false],
    ["selfConfig.enabled", ["*"], false],
    ["selfConfig.allowedKeys", ["*"], false],
    // Exact entry: the key itself AND everything under it.
    ["model", ["model"], true],
    ["context", ["context"], true],
    ["context.hardFraction", ["context"], true],
    ["memory.autoRecall", ["context"], false],
    // Subtree pattern: children only, never the whole sub-tree at once.
    ["context.hardFraction", ["context.*"], true],
    ["context", ["context.*"], false],
    ["memory.recallLimit", ["context.*"], false],
    // Leaf entry: that leaf and nothing beside it.
    ["context.advisoryFraction", ["context.advisoryFraction"], true],
    ["context.hardFraction", ["context.advisoryFraction"], false],
    // Prefix confusion: "context" must not match "contextual".
    ["contextual", ["context"], false],
    ["contextual", ["context.*"], false],
    // Several entries, and the empty list.
    ["memory.autoRecall", ["model", "memory.*"], true],
    ["model", [], false],
    ["model", ["*", "tenantId"], true],
    ["tenantId", ["tenantId", "*"], false],
  ];

  for (const [key, allowed, expected] of cases) {
    it(`${expected ? "allows" : "denies"} ${key} under [${allowed.join(", ")}]`, () => {
      expect(isSelfConfigWritable(key, allowed)).toBe(expected);
    });
  }
});

describe("SettingsStore.set on selfConfig (the human delegation path)", () => {
  it("writes the allow-list as a plain jsonb array", async () => {
    const db = dbWithScopedRows([]);
    await new SettingsStore(db).set("global", "selfConfig.allowedKeys", ["model", "context.*"]);
    expect(insertCalls(db)[0]!.params).toEqual([
      "global",
      "selfConfig.allowedKeys",
      ["model", "context.*"],
    ]);
  });

  it("writes the master switch through the jsonbParam boolean path", async () => {
    const db = dbWithScopedRows([]);
    await new SettingsStore(db).set("global", "selfConfig.enabled", true);
    const value = insertCalls(db)[0]!.params![2];
    expect(typeof value).not.toBe("boolean");
    expect(JSON.stringify(value)).toBe("true");
  });

  it("refuses to store a delegation of tenantId, and writes nothing", async () => {
    const db = dbWithScopedRows([]);
    await expect(
      new SettingsStore(db).set("global", "selfConfig.allowedKeys", ["tenantId"]),
    ).rejects.toThrow(/can never be delegated to an agent/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("round-trips a delegation through load", async () => {
    const db = dbWithScopedRows([
      ["global", "selfConfig.enabled", true],
      ["agent:pinky", "selfConfig.allowedKeys", ["context.*"]],
    ]);
    const snapshot = await new SettingsStore(db).load({ scopes: ["agent:pinky"] });
    expect(snapshot.selfConfig).toEqual({ enabled: true, allowedKeys: ["context.*"] });
  });
});
