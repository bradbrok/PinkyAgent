import { describe, expect, it } from "bun:test";
import { SettingsStore, validateSettings } from "../src/settings";
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
const dbWithScopedRows = (entries: Row[]) =>
  new FakeDb([
    {
      pattern: /select scope, key, value from settings/,
      respond: (params) => {
        const wanted = (params?.[0] as string[]) ?? [];
        return recordsFrom(entries).filter((r) => wanted.includes(r.scope));
      },
    },
    { pattern: /insert into settings/, respond: () => [] },
  ]);

const selectCalls = (db: FakeDb) => db.calls.filter((c) => /^\s*select/.test(c.sql));
const insertCalls = (db: FakeDb) => db.calls.filter((c) => /insert into settings/.test(c.sql));

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

  it("rejects a snapshot poisoned by an already-stored bad value", async () => {
    const db = dbWithAllRows([["global", "context.hardFraction", "abc"]]);
    const store = new SettingsStore(db);
    await expect(store.load()).rejects.toThrow(/context\.hardFraction/);
  });

  it("rejects a malformed scope before querying", async () => {
    const db = dbWithAllRows([]);
    const store = new SettingsStore(db);
    await expect(store.load({ scopes: ["channel:"] })).rejects.toThrow(/Invalid settings scope/);
    expect(db.calls).toHaveLength(0);
  });
});

describe("validateSettings", () => {
  const withContext = (context: Record<string, unknown>) => ({ ...DEFAULT_SETTINGS, context });

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
  ];

  for (const [name, candidate, match] of cases) {
    it(`rejects ${name}`, () => {
      expect(() => validateSettings(candidate)).toThrow(match);
    });
  }

  it("reports every bad key in a single error", () => {
    let message = "";
    try {
      validateSettings({ tenantId: "", model: "kimi-k2", context: { approxWindowTokens: -1 }, replyGate: {} });
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
    await new SettingsStore(relaxed).set("global", "context.advisoryFraction", 0.95);
    expect(insertCalls(relaxed)[0]!.params).toEqual(["global", "context.advisoryFraction", 0.95]);
  });

  it("reads only the target scope (plus global) when validating", async () => {
    const db = dbWithScopedRows([["channel:chan2", "model", "not-a-valid-model"]]);
    const store = new SettingsStore(db);
    await store.set("channel:chan1", "model", "openrouter/ok");
    expect(selectCalls(db)[0]!.params).toEqual([["global", "channel:chan1"]]);
    expect(insertCalls(db)).toHaveLength(1);
  });
});
