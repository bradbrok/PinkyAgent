import { describe, expect, it } from "bun:test";
import {
  MAX_TOOL_SEARCH_LIMIT,
  MIN_APPROX_WINDOW_TOKENS,
  MIN_FRACTION_GAP,
  SELF_CONFIG_KEYS,
  SettingsStore,
  isImmutableSettingKey,
  isKnownSettingPath,
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
  const withTools = (tools: unknown) => ({ ...DEFAULT_SETTINGS, tools });
  const someTools = (patch: Record<string, unknown>) =>
    withTools({ ...DEFAULT_SETTINGS.tools, ...patch });
  const withServers = (servers: unknown) => ({ ...DEFAULT_SETTINGS, mcp: { servers } });

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
    // --- tool partition (slice 9) ----------------------------------------
    ["missing tools sub-tree", withTools("nope"), /tools: expected an object/],
    [
      "unknown tools key",
      someTools({ alwaysOnn: [] }),
      /tools\.alwaysOnn: unknown setting key \(known: defaultMode, alwaysOn, deferred, searchLimit\)/,
    ],
    [
      "non-object defaultMode",
      someTools({ defaultMode: "always" }),
      /tools\.defaultMode: expected an object/,
    ],
    [
      "unknown defaultMode key",
      someTools({ defaultMode: { ...DEFAULT_SETTINGS.tools.defaultMode, builtins: "always" } }),
      /tools\.defaultMode\.builtins: unknown setting key \(known: builtin, mcp\)/,
    ],
    [
      "a defaultMode that is neither always nor deferred",
      someTools({ defaultMode: { builtin: "sometimes", mcp: "deferred" } }),
      /tools\.defaultMode\.builtin: expected "always" or "deferred", got "sometimes"/,
    ],
    [
      "a missing defaultMode.mcp",
      someTools({ defaultMode: { builtin: "always" } }),
      /tools\.defaultMode\.mcp: expected "always" or "deferred", got undefined/,
    ],
    ["non-array alwaysOn", someTools({ alwaysOn: "bash" }), /tools\.alwaysOn: expected an array of tool names/],
    ["non-array deferred", someTools({ deferred: {} }), /tools\.deferred: expected an array of tool names/],
    [
      "a non-string tool name",
      someTools({ alwaysOn: ["bash", 7] }),
      /tools\.alwaysOn\[1\]: expected a tool name .*got 7/,
    ],
    [
      "an empty tool name",
      someTools({ deferred: [""] }),
      /tools\.deferred\[0\]: expected a tool name .*got ""/,
    ],
    [
      "a tool name with surrounding whitespace",
      someTools({ alwaysOn: [" bash"] }),
      /tools\.alwaysOn\[0\]: expected a tool name \(a non-empty string with no surrounding whitespace\)/,
    ],
    [
      "a duplicated tool name",
      someTools({ alwaysOn: ["bash", "recall", "bash"] }),
      /tools\.alwaysOn\[2\]: duplicate tool name "bash"/,
    ],
    // Precedence is alwaysOn > deferred, so a name in both is a line that does
    // nothing — reported on the list being ignored.
    [
      "a tool named in BOTH lists",
      someTools({ alwaysOn: ["recall"], deferred: ["bash", "recall"] }),
      /tools\.deferred\[1\]: "recall" is also in tools\.alwaysOn/,
    ],
    [
      "zero searchLimit",
      someTools({ searchLimit: 0 }),
      /tools\.searchLimit: expected an integer in \[1, 50\]/,
    ],
    ["fractional searchLimit", someTools({ searchLimit: 8.5 }), /tools\.searchLimit/],
    ["string searchLimit", someTools({ searchLimit: "8" }), /tools\.searchLimit/],
    [
      "a searchLimit past the cap",
      someTools({ searchLimit: 51 }),
      /tools\.searchLimit: expected an integer in \[1, 50\]/,
    ],
    // --- mcp servers (slice 9) --------------------------------------------
    ["missing mcp sub-tree", { ...DEFAULT_SETTINGS, mcp: "nope" }, /mcp: expected an object/],
    [
      "unknown mcp key",
      { ...DEFAULT_SETTINGS, mcp: { servers: {}, timeoutMs: 1000 } },
      /mcp\.timeoutMs: unknown setting key \(known: servers\)/,
    ],
    ["non-object servers", withServers([]), /mcp\.servers: expected an object mapping a server name/],
    [
      "an uppercase server name",
      withServers({ GitHub: { transport: "stdio", command: "x" } }),
      /mcp\.servers\.GitHub: invalid server name/,
    ],
    [
      "a server name starting with a dash",
      withServers({ "-github": { transport: "stdio", command: "x" } }),
      /mcp\.servers\.-github: invalid server name/,
    ],
    [
      "a server name over 32 characters",
      withServers({ [`a${"b".repeat(32)}`]: { transport: "stdio", command: "x" } }),
      /invalid server name/,
    ],
    // `__` is the tool-name separator: server "github__issues" + tool "create"
    // and server "github" + tool "issues__create" are the same final name, so
    // they are one tool_catalog primary key, dispatch goes to whichever server
    // the runtime reaches first, and serverState("github") stops finding its
    // own generation.
    [
      "a server name containing the tool-name separator",
      withServers({ github__issues: { transport: "stdio", command: "x" } }),
      /mcp\.servers\.github__issues: invalid server name — "__" is the tool-name separator/,
    ],
    [
      "a server name that is only a separator",
      withServers({ __: { transport: "stdio", command: "x" } }),
      /invalid server name/,
    ],
    [
      "a non-object server entry",
      withServers({ github: "npx -y server" }),
      /mcp\.servers\.github: expected an object, got "npx -y server"/,
    ],
    [
      "an unknown transport",
      withServers({ github: { transport: "sse", url: "http://x/" } }),
      /mcp\.servers\.github: expected "transport" to be "stdio" or "http", got "sse"/,
    ],
    [
      "a key from the other arm of the union",
      withServers({ github: { transport: "stdio", command: "x", url: "http://x/" } }),
      /mcp\.servers\.github: unknown key "url" for a "stdio" server \(known: transport, command, args, env, cwd\)/,
    ],
    [
      "a stdio server with no command",
      withServers({ github: { transport: "stdio" } }),
      /mcp\.servers\.github: a "stdio" server needs a non-empty "command", got undefined/,
    ],
    [
      "a stdio server with a blank command",
      withServers({ github: { transport: "stdio", command: "   " } }),
      /a "stdio" server needs a non-empty "command"/,
    ],
    [
      "non-string args",
      withServers({ github: { transport: "stdio", command: "npx", args: ["-y", 7] } }),
      /mcp\.servers\.github: "args" expects an array of strings/,
    ],
    [
      "args that are not an array",
      withServers({ github: { transport: "stdio", command: "npx", args: "-y server" } }),
      /"args" expects an array of strings/,
    ],
    [
      "a non-string env value",
      withServers({ github: { transport: "stdio", command: "npx", env: { TOKEN: 7 } } }),
      /mcp\.servers\.github: "env" expects an object of string values/,
    ],
    [
      "an empty cwd",
      withServers({ github: { transport: "stdio", command: "npx", cwd: "" } }),
      /mcp\.servers\.github: "cwd" expects a non-empty string/,
    ],
    [
      "an http server with no url",
      withServers({ remote: { transport: "http" } }),
      /mcp\.servers\.remote: an "http" server needs a non-empty "url", got undefined/,
    ],
    [
      "a url that is not absolute",
      withServers({ remote: { transport: "http", url: "/mcp" } }),
      /mcp\.servers\.remote: "url" expects an absolute http\(s\) URL, got "\/mcp"/,
    ],
    [
      "a url on a non-http scheme",
      withServers({ remote: { transport: "http", url: "ftp://example.com/mcp" } }),
      /mcp\.servers\.remote: "url" expects the http or https scheme/,
    ],
    [
      "a non-string header value",
      withServers({
        remote: { transport: "http", url: "https://example.com/mcp", headers: { Auth: null } },
      }),
      /mcp\.servers\.remote: "headers" expects an object of string values/,
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
    // mcp joins them (slice 9): a server entry is a command to run on this
    // host, so no allow-list may name it in any form.
    [
      "mcp in the allow-list",
      allowing("mcp"),
      /selfConfig\.allowedKeys\[0\]: "mcp" can never be delegated to an agent/,
    ],
    [
      "the mcp subtree in the allow-list",
      allowing("mcp.*"),
      /selfConfig\.allowedKeys\[0\]: "mcp\.\*" can never be delegated to an agent/,
    ],
    [
      "mcp.servers in the allow-list",
      allowing("tools.*", "mcp.servers"),
      /selfConfig\.allowedKeys\[1\]: "mcp\.servers" can never be delegated to an agent/,
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
    for (const entry of [
      "*",
      "model",
      "context",
      "context.*",
      "context.advisoryFraction",
      "memory.autoRecall",
      "tools",
      "tools.*",
      "tools.alwaysOn",
    ]) {
      expect(validateSettings(allowing(entry)).selfConfig.allowedKeys).toEqual([entry]);
    }
  });

  it("never offers an immutable sub-tree when it lists the delegable ones", () => {
    // "mcp.*" is refused above; advertising `mcp` in the "sub-trees: ..." hint
    // for an unrelated typo would send the human straight back into it.
    const message = String(
      (() => {
        try {
          validateSettings(allowing("tolls.*"));
        } catch (err) {
          return (err as Error).message;
        }
        return "";
      })(),
    );
    expect(message).toContain("names no settings sub-tree");
    expect(message).toContain("tools");
    expect(message).not.toContain("mcp");
    expect(message).not.toContain("selfConfig,");
  });

  // --- tool partition (slice 9) -----------------------------------------
  it("round-trips a fully populated tools block unchanged", () => {
    const tools = {
      defaultMode: { builtin: "deferred", mcp: "always" },
      alwaysOn: ["recall", "retain"],
      deferred: ["bash"],
      searchLimit: 25,
    };
    expect(validateSettings(withTools(tools)).tools).toEqual(tools as never);
  });

  it("accepts the searchLimit bounds exactly", () => {
    expect(MAX_TOOL_SEARCH_LIMIT).toBe(50);
    expect(() => validateSettings(someTools({ searchLimit: 1 }))).not.toThrow();
    expect(() => validateSettings(someTools({ searchLimit: MAX_TOOL_SEARCH_LIMIT }))).not.toThrow();
    expect(() => validateSettings(someTools({ searchLimit: MAX_TOOL_SEARCH_LIMIT + 1 }))).toThrow();
  });

  it("accepts the same name in alwaysOn on one snapshot and deferred on another", () => {
    // Only the SAME snapshot holding both is contradictory; a channel that
    // defers what another channel forces on is two ordinary decisions.
    expect(() => validateSettings(someTools({ alwaysOn: ["bash"] }))).not.toThrow();
    expect(() => validateSettings(someTools({ deferred: ["bash"] }))).not.toThrow();
  });

  it("does NOT check tool names against a live tool list", () => {
    // The catalog is runtime state: MCP servers come and go and `bash` depends
    // on --shell, so "no such tool" is not a fact this validator can know. A
    // name matching nothing is inert, which is the failure mode we want —
    // config stays writable ahead of the server that will serve it.
    const tools = someTools({ alwaysOn: ["mcp__github__create_issue"], deferred: ["no_such_tool"] });
    expect(validateSettings(tools).tools.alwaysOn).toEqual(["mcp__github__create_issue"]);
  });

  // --- mcp servers (slice 9) --------------------------------------------
  it("round-trips both transports unchanged, placeholders and all", () => {
    // `${...}` values stay literal here: they are resolved from process.env at
    // connect time, so a settings row never holds the secret itself.
    const servers = {
      github: {
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
        env: { GITHUB_TOKEN: "${GITHUB_TOKEN}" },
        cwd: "/srv/mcp",
      },
      remote: {
        transport: "http",
        url: "https://mcp.example.com/rpc",
        headers: { Authorization: "Bearer ${MCP_TOKEN}" },
      },
    };
    expect(validateSettings(withServers(servers)).mcp.servers).toEqual(servers as never);
  });

  it("accepts a minimal entry of each transport, and the empty map", () => {
    expect(() =>
      validateSettings(withServers({ a: { transport: "stdio", command: "./server" } })),
    ).not.toThrow();
    expect(() =>
      validateSettings(withServers({ "a-b_9": { transport: "http", url: "http://localhost:9/mcp" } })),
    ).not.toThrow();
    expect(validateSettings(withServers({})).mcp.servers).toEqual({});
  });

  it("keeps single _ and - legal — only the doubled underscore collides", () => {
    for (const name of ["github_issues", "github-issues", "a_b-c_9"]) {
      expect(() =>
        validateSettings(withServers({ [name]: { transport: "stdio", command: "x" } })),
      ).not.toThrow();
    }
    expect(() =>
      validateSettings(withServers({ github__issues: { transport: "stdio", command: "x" } })),
    ).toThrow(/tool-name separator/);
  });

  it("accepts a 32-character server name and refuses the 33rd character", () => {
    const ok = `a${"b".repeat(31)}`;
    expect(ok).toHaveLength(32);
    expect(() =>
      validateSettings(withServers({ [ok]: { transport: "stdio", command: "x" } })),
    ).not.toThrow();
    expect(() =>
      validateSettings(withServers({ [`${ok}c`]: { transport: "stdio", command: "x" } })),
    ).toThrow(/invalid server name/);
  });

  it("names a dotted server key without pretending it is a path", () => {
    // "a.b" can never be a legal name; the message has to be able to say so
    // without reading as `mcp.servers.a` -> `b`.
    let message = "";
    try {
      validateSettings(withServers({ "a.b": { transport: "stdio", command: "x" } }));
    } catch (err) {
      message = (err as Error).message;
    }
    expect(message).toContain('mcp.servers["a.b"]: invalid server name');
  });

  it("reports every bad server in one error, not just the first", () => {
    let message = "";
    try {
      validateSettings(
        withServers({
          good: { transport: "stdio", command: "x" },
          bad1: { transport: "stdio" },
          bad2: { transport: "http", url: "nope" },
        }),
      );
    } catch (err) {
      message = (err as Error).message;
    }
    expect(message).toContain("mcp.servers.bad1");
    expect(message).toContain("mcp.servers.bad2");
    expect(message).not.toContain("mcp.servers.good");
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
    // --- slice 9 -------------------------------------------------------
    // The tool partition IS delegable: it is what an agent tuning its own
    // context pressure should be able to reach.
    ["tools.alwaysOn", ["tools.*"], true],
    ["tools.deferred", ["tools.*"], true],
    ["tools.searchLimit", ["tools.*"], true],
    ["tools.defaultMode.mcp", ["tools.*"], true],
    ["tools", ["tools.*"], false],
    ["tools", ["tools"], true],
    ["tools.alwaysOn", ["tools"], true],
    ["tools.alwaysOn", ["*"], true],
    ["tools.alwaysOn", ["memory.*"], false],
    // MCP never is, by any route: a server entry is a command on this host.
    ["mcp", ["*"], false],
    ["mcp.servers", ["*"], false],
    ["mcp.servers.github", ["*"], false],
    ["mcp", ["mcp"], false],
    ["mcp.servers", ["mcp.*"], false],
    ["mcp.servers.github", ["mcp.servers"], false],
    ["mcp.servers", ["*", "mcp", "mcp.*", "mcp.servers"], false],
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

/**
 * Which keys EXIST is derived from DEFAULT_SETTINGS, so slice 9's two new
 * sub-trees are addressable the day they land there — with one exception the
 * walk has to get right: `mcp.servers` is an open map, so its children are
 * server names a human invents, not schema fields.
 */
describe("the key set derived from DEFAULT_SETTINGS", () => {
  it("includes every tools and mcp path", () => {
    for (const key of [
      "tools",
      "tools.defaultMode",
      "tools.defaultMode.builtin",
      "tools.defaultMode.mcp",
      "tools.alwaysOn",
      "tools.deferred",
      "tools.searchLimit",
      "mcp",
      "mcp.servers",
    ]) {
      expect(SELF_CONFIG_KEYS).toContain(key);
      expect(isKnownSettingPath(key)).toBe(true);
    }
  });

  it("does not turn array elements into keys", () => {
    expect(SELF_CONFIG_KEYS.some((k) => k.startsWith("tools.alwaysOn."))).toBe(false);
    expect(isKnownSettingPath("tools.alwaysOn.0")).toBe(false);
  });

  it("treats mcp.servers.<name> as a legal ROW key without enumerating names", () => {
    // The derived list cannot contain a server name (DEFAULT_SETTINGS ships
    // `servers: {}`), but a row keyed `mcp.servers.github` is exactly how one
    // server is added — so the pruner has to accept it, or `set` would write a
    // row `load` then silently drops.
    expect(SELF_CONFIG_KEYS.some((k) => k.startsWith("mcp.servers."))).toBe(false);
    expect(isKnownSettingPath("mcp.servers.github")).toBe(true);
    // One level only: a server config is a discriminated union, set whole.
    expect(isKnownSettingPath("mcp.servers.github.command")).toBe(false);
    expect(isKnownSettingPath("mcp.servers.")).toBe(false);
  });

  it("has no other open map: memory/context children are still fixed", () => {
    expect(isKnownSettingPath("memory.anything")).toBe(false);
    expect(isKnownSettingPath("context.anything")).toBe(false);
  });

  it("marks mcp immutable at every depth, and tools at none", () => {
    for (const key of ["mcp", "mcp.servers", "mcp.servers.github"]) {
      expect(isImmutableSettingKey(key)).toBe(true);
    }
    for (const key of ["tools", "tools.alwaysOn", "tools.defaultMode.mcp"]) {
      expect(isImmutableSettingKey(key)).toBe(false);
    }
    // Prefix confusion: "mcp" must not swallow a sibling that starts with it.
    expect(isImmutableSettingKey("mcpServers")).toBe(false);
  });
});

/**
 * Slice 9 through the store: the overlay replaces a list rather than merging
 * it, one bad server entry loses only itself, and a bad tool list falls back to
 * empty instead of taking the run down.
 */
describe("tools and mcp through load()", () => {
  it("replaces tools.alwaysOn wholesale — the overlay never merges arrays", () => {
    // A merged union would make a narrower scope unable to REMOVE anything,
    // and `alwaysOn` is the header cache key: "agent:pinky runs with exactly
    // these" has to be expressible.
    const db = dbWithAllRows([
      ["global", "tools.alwaysOn", ["recall", "retain"]],
      ["agent:pinky", "tools.alwaysOn", ["bash"]],
    ]);
    return new SettingsStore(db)
      .load({ scopes: ["agent:pinky"] })
      .then((snapshot) => expect(snapshot.tools.alwaysOn).toEqual(["bash"]));
  });

  it("lets a channel empty a list its global scope filled", async () => {
    const db = dbWithAllRows([
      ["global", "tools.deferred", ["bash"]],
      ["channel:c1", "tools.deferred", []],
    ]);
    const snapshot = await new SettingsStore(db).load({ scopes: ["channel:c1"] });
    expect(snapshot.tools.deferred).toEqual([]);
  });

  it("merges one server row into the map beside another scope's", async () => {
    const db = dbWithAllRows([
      ["global", "mcp.servers.github", { transport: "stdio", command: "npx" }],
      ["channel:c1", "mcp.servers.remote", { transport: "http", url: "https://x.example/mcp" }],
    ]);
    const snapshot = await new SettingsStore(db).load({ scopes: ["channel:c1"] });
    expect(Object.keys(snapshot.mcp.servers).sort()).toEqual(["github", "remote"]);
  });

  it("a row keyed `mcp.servers` REPLACES the map, as any parent key does", async () => {
    const db = dbWithAllRows([
      ["global", "mcp.servers.github", { transport: "stdio", command: "npx" }],
      ["agent:pinky", "mcp.servers", { remote: { transport: "http", url: "https://x.example/mcp" } }],
    ]);
    const snapshot = await new SettingsStore(db).load({ scopes: ["agent:pinky"] });
    expect(Object.keys(snapshot.mcp.servers)).toEqual(["remote"]);
  });

  it("drops ONE bad server entry and keeps the rest of the map", async () => {
    const db = dbWithAllRows([
      [
        "global",
        "mcp.servers",
        {
          github: { transport: "stdio", command: "npx" },
          broken: { transport: "stdio" },
        },
      ],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(Object.keys(snapshot.mcp.servers)).toEqual(["github"]);
    expect(messages.some((m) => m.includes("mcp.servers.broken"))).toBe(true);
    expect(messages.some((m) => m.includes("github"))).toBe(false);
  });

  it("drops a server whose NAME cannot be a name, dot and all", async () => {
    // The repair target here cannot be a dotted path (the name holds the dot),
    // which is the whole reason an issue can carry raw segments.
    const db = dbWithAllRows([
      [
        "global",
        "mcp.servers",
        {
          "a.b": { transport: "stdio", command: "npx" },
          ok: { transport: "stdio", command: "npx" },
        },
      ],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(Object.keys(snapshot.mcp.servers)).toEqual(["ok"]);
    expect(messages.some((m) => m.includes('mcp.servers["a.b"]'))).toBe(true);
  });

  it("drops a bad entry written as its own row, leaving the row's siblings", async () => {
    const db = dbWithAllRows([
      ["global", "mcp.servers.github", { transport: "stdio", command: "npx" }],
      ["global", "mcp.servers.broken", { transport: "http", url: "not a url" }],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(Object.keys(snapshot.mcp.servers)).toEqual(["github"]);
    expect(messages.some((m) => /mcp\.servers\.broken.*absolute http/.test(m))).toBe(true);
  });

  it("resets a bad tool list to empty without touching the other one", async () => {
    const db = dbWithAllRows([
      ["global", "tools.alwaysOn", ["bash", "bash"]],
      ["global", "tools.deferred", ["recall"]],
      ["global", "tools.searchLimit", 500],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.tools.alwaysOn).toEqual([]);
    expect(snapshot.tools.deferred).toEqual(["recall"]);
    expect(snapshot.tools.searchLimit).toBe(DEFAULT_SETTINGS.tools.searchLimit);
    expect(messages.some((m) => /tools\.alwaysOn\[1\].*duplicate/.test(m))).toBe(true);
    expect(messages.some((m) => /tools\.searchLimit.*falling back/.test(m))).toBe(true);
  });

  it("resets only tools.deferred when a name is in both lists", async () => {
    const db = dbWithAllRows([
      ["global", "tools.alwaysOn", ["bash"]],
      ["global", "tools.deferred", ["bash"]],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.tools.alwaysOn).toEqual(["bash"]);
    expect(snapshot.tools.deferred).toEqual([]);
    expect(messages.some((m) => m.includes("also in tools.alwaysOn"))).toBe(true);
  });

  it("prunes an unreadable row under mcp.servers, and says which", async () => {
    const db = dbWithAllRows([
      ["global", "mcp.servers.github.command", "npx"],
      ["global", "model", "openrouter/kept"],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();

    expect(snapshot.mcp.servers).toEqual({});
    expect(snapshot.model).toBe("openrouter/kept");
    expect(messages[0]).toContain("mcp.servers.github.command");
    expect(messages[0]).toContain("no such setting key");
  });

  it("never throws on a wholly bogus tools/mcp pair", async () => {
    const db = dbWithAllRows([
      ["global", "tools", "yes please"],
      ["global", "mcp", 7],
    ]);
    const { store, messages } = loudStore(db);
    const snapshot = await store.load();
    expect(snapshot.tools).toEqual(DEFAULT_SETTINGS.tools);
    expect(snapshot.mcp).toEqual(DEFAULT_SETTINGS.mcp);
    expect(messages.length).toBeGreaterThan(0);
  });
});

describe("SettingsStore.set on tools and mcp (the human write path)", () => {
  it("writes one server as a plain jsonb object under its own key", async () => {
    const db = dbWithScopedRows([]);
    const server = { transport: "stdio", command: "npx", args: ["-y", "srv"] };
    await new SettingsStore(db).set("global", "mcp.servers.github", server);
    expect(insertCalls(db)[0]!.params).toEqual(["global", "mcp.servers.github", server]);
    expect(typeof insertCalls(db)[0]!.params![2]).toBe("object");
  });

  it("round-trips that row through load — the write is not silently pruned", async () => {
    const server = { transport: "stdio", command: "npx" };
    const db = dbWithScopedRows([["global", "mcp.servers.github", server]]);
    const { store, messages } = loudStore(db);
    expect((await store.load()).mcp.servers).toEqual({ github: server } as never);
    expect(messages).toEqual([]);
  });

  it("rejects a bad server entry and issues no insert", async () => {
    const db = dbWithScopedRows([]);
    await expect(
      new SettingsStore(db).set("global", "mcp.servers.github", { transport: "stdio" }),
    ).rejects.toThrow(/mcp\.servers\.github: a "stdio" server needs a non-empty "command"/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("rejects an illegal server NAME at the key, before anything is stored", async () => {
    const db = dbWithScopedRows([]);
    await expect(
      new SettingsStore(db).set("global", "mcp.servers.GitHub", { transport: "stdio", command: "x" }),
    ).rejects.toThrow(/invalid server name/);
    expect(insertCalls(db)).toHaveLength(0);
  });

  it("rejects a searchLimit past the cap and a name in both lists", async () => {
    const over = dbWithScopedRows([]);
    await expect(new SettingsStore(over).set("global", "tools.searchLimit", 200)).rejects.toThrow(
      /tools\.searchLimit: expected an integer in \[1, 50\]/,
    );
    expect(insertCalls(over)).toHaveLength(0);

    // Cross-key, within one scope's effective snapshot: alwaysOn is already
    // stored, so the deferred write is the one that reads as doing nothing.
    const clash = dbWithScopedRows([["global", "tools.alwaysOn", ["bash"]]]);
    await expect(new SettingsStore(clash).set("global", "tools.deferred", ["bash"])).rejects.toThrow(
      /also in tools\.alwaysOn/,
    );
    expect(insertCalls(clash)).toHaveLength(0);
  });

  it("writes a tool list as a plain jsonb array", async () => {
    const db = dbWithScopedRows([]);
    await new SettingsStore(db).set("agent:pinky", "tools.alwaysOn", ["recall", "retain"]);
    expect(insertCalls(db)[0]!.params).toEqual([
      "agent:pinky",
      "tools.alwaysOn",
      ["recall", "retain"],
    ]);
  });
});
