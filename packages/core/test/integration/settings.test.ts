/**
 * SettingsStore against real jsonb (DESIGN.md P8: the DB is the only config).
 *
 * The unit suite backs the store with an in-memory Map that hands values back
 * exactly as they went in, so the jsonb round-trip is only ever exercised
 * here — and it did not survive it. The tests below named DEFECT are the
 * regression guard for that bug (every value `set` wrote was JSON-encoded
 * TWICE and landed as a jsonb string, so `pinky config get model` printed
 * `"\"openrouter/...\""` and any object-valued row threw out of every later
 * `load()`); see the block above them for the mechanism and the fix.
 *
 * Skipped unless PINKY_INTEGRATION=1. Writes ONLY to run-unique
 * `channel:it-settings-<run>-*` / `agent:it-settings-<run>-*` scopes: `global`
 * is shared with the developer's own `pinky config set` and is never touched.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { DEFAULT_SETTINGS, loadEnvConfig } from "../../src/config";
import { createDb, jsonbParam } from "../../src/pg";
import { migrate } from "../../src/migrate";
import { SettingsStore } from "../../src/settings";
import type { Db } from "../../src/db";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const DB_URL = loadEnvConfig().databaseUrl;
const SCHEMA_DIR = new URL("../../schema", import.meta.url).pathname;

const RUN = crypto.randomUUID().slice(0, 8);
const scope = (kind: "channel" | "agent", name: string): string =>
  `${kind}:it-settings-${RUN}-${name}`;

suite("SettingsStore (live postgres, real jsonb)", () => {
  let db: Db;
  let store: SettingsStore;

  beforeAll(async () => {
    db = createDb(DB_URL, { max: 4 });
    await migrate(db, SCHEMA_DIR);
    await purge(db);
    store = new SettingsStore(db);
  });

  afterAll(async () => {
    if (!db) return;
    await purge(db);
    await db.close();
  });

  /** Scoped delete — this file's scopes plus any an interrupted run left. */
  async function purge(handle: Db): Promise<void> {
    await handle.query(
      `delete from settings where scope like 'channel:it-settings-%'
                              or scope like 'agent:it-settings-%'`,
    );
  }

  /**
   * Inspect a scope's rows with the JSON decoding done SERVER-side:
   * `jsonb_typeof` reports what actually landed in the column and `#>> '{}'`
   * renders a jsonb scalar as plain text. Independent of any client-side
   * encoding, which is exactly what is in question here.
   */
  function rowsIn(s: string): Promise<{ key: string; jtype: string; text: string }[]> {
    return db.query<{ key: string; jtype: string; text: string }>(
      `select key, jsonb_typeof(value) as jtype, coalesce(value #>> '{}', value::text) as text
         from settings where scope = $1 order by key`,
      [s],
    );
  }

  // -------------------------------------------------------------------------
  // Write path: validation and rejection. Sound today.
  // -------------------------------------------------------------------------

  it("set rejects a badly typed value and leaves the table untouched", async () => {
    const s = scope("agent", "badtype");
    await expect(store.set(s, "context.hardFraction", "abc")).rejects.toThrow(
      /context\.hardFraction: expected a number/,
    );
    expect(await rowsIn(s)).toEqual([]);
  });

  it("set rejects an unknown key and leaves the table untouched", async () => {
    const s = scope("agent", "unknownkey");
    await expect(store.set(s, "context.hardFractoin", 0.5)).rejects.toThrow(/unknown setting key/);
    expect(await rowsIn(s)).toEqual([]);
  });

  it("set rejects a CROSS-field violation and leaves the table untouched", async () => {
    const s = scope("agent", "crossfield");
    // Default hardFraction is 0.9, so an advisory of 0.95 inverts the ladder.
    await expect(store.set(s, "context.advisoryFraction", 0.95)).rejects.toThrow(
      /advisoryFraction: expected to be less than/,
    );
    expect(await rowsIn(s)).toEqual([]);
  });

  it("set rejects a bad scope and a bad key before touching the database", async () => {
    await expect(store.set("channel:", "model", "openrouter/a/b")).rejects.toThrow(
      /Invalid settings scope/,
    );
    await expect(store.set(scope("agent", "badkey"), "", "x")).rejects.toThrow(
      /Invalid settings key/,
    );
    expect(await rowsIn(scope("agent", "badkey"))).toEqual([]);
  });

  it("set is an upsert on (scope, key): two writes leave exactly one row", async () => {
    const s = scope("agent", "upsert");
    await store.set(s, "model", "openrouter/v1/x");
    await store.set(s, "model", "openrouter/v2/x");
    const rows = await rowsIn(s);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.key).toBe("model");
    // What the row CONTAINS is the jsonb block's business, not this test's.
  });

  // -------------------------------------------------------------------------
  // DEFECT (FIXED — these are its regression tests) — jsonb values used to be
  // encoded twice on the way in.
  //
  // `SettingsStore.set` binds `JSON.stringify(value)`. postgres.js learns each
  // parameter's type from the server's Describe response (here: jsonb) and
  // then applies its OWN serializer for that type, which is JSON.stringify.
  // The already-encoded text is therefore encoded a second time and stored as
  // a jsonb STRING: `set(s, "model", "openrouter/a/b")` lands
  // `"\"openrouter/a/b\""` and `jsonb_typeof` is 'string' even for objects.
  // An explicit `$1::jsonb` cast does not help — the value is already a JSON
  // string by the time the server sees it.
  //
  // Consequences on a live database:
  //   * `pinky config set model '"openrouter/moonshotai/kimi-k2"'` followed by
  //     `pinky config get` prints "\"openrouter/moonshotai/kimi-k2\"". That
  //     passes validateSettings (a non-empty string containing "/"), so it
  //     fails LATER, in splitModel(), which reads the provider as `"openrouter`
  //     and throws "Model must be provider/model-id" / "Unsupported provider".
  //   * any object-, number- or boolean-valued row makes validateSettings
  //     THROW on read, so every subsequent load() — gateway startup, each wake,
  //     `pinky prompt` — dies with `context: expected an object, got "{...}"`.
  //     It also poisons `set` itself, which re-validates the scope first.
  // The settings table was therefore only usable while it was EMPTY and load()
  // returned pure defaults, which is why the unit suite and `bun run smoke`
  // never noticed.
  //
  // THE FIX (src/settings.ts, src/pg.ts): hand postgres.js the PLAIN value and
  // let it serialize exactly once — `[s, k, jsonbParam(value)]` instead of
  // `[s, k, JSON.stringify(value)]`. jsonbParam() is identity for objects,
  // strings and numbers; it only rewraps a bare boolean, which postgres.js
  // would otherwise tag with the bool wire type (Postgres will not coerce
  // boolean -> jsonb at all). EventStore needed the same change
  // (event-store.test.ts), and rows already written are repaired in place by
  // schema/0004_jsonb_repair.rerun.sql.
  // -------------------------------------------------------------------------

  it("DEFECT: set stores a scalar as a jsonb scalar of the right type", async () => {
    const s = scope("channel", "scalars");
    await store.set(s, "model", "openrouter/moonshotai/kimi-k2-it");
    await store.set(s, "context.approxWindowTokens", 4096);

    expect(await rowsIn(s)).toEqual([
      { key: "context.approxWindowTokens", jtype: "number", text: "4096" },
      { key: "model", jtype: "string", text: "openrouter/moonshotai/kimi-k2-it" },
    ]);
  });

  it("DEFECT: set stores an object value as a jsonb object, not a quoted blob", async () => {
    const s = scope("channel", "subtree");
    await store.set(s, "context", {
      advisoryFraction: 0.5,
      hardFraction: 0.8,
      approxWindowTokens: 4096,
    });
    const [row] = await rowsIn(s);
    expect(row?.jtype).toBe("object");
    const fields = await db.queryOne<{ advisory: string | null; hard: string | null }>(
      `select value ->> 'advisoryFraction' as advisory, value ->> 'hardFraction' as hard
         from settings where scope = $1 and key = 'context'`,
      [s],
    );
    expect(fields?.advisory).toBe("0.5");
    expect(fields?.hard).toBe("0.8");
  });

  it("DEFECT: set + load round-trips a string through jsonb", async () => {
    const s = scope("channel", "roundtrip");
    await store.set(s, "model", "openrouter/moonshotai/kimi-k2-it");
    expect((await store.load({ scopes: [s] })).model).toBe("openrouter/moonshotai/kimi-k2-it");
  });

  it("DEFECT: a subtree value is replaced wholesale, then a dotted key refines it", async () => {
    const s = scope("channel", "dotted");
    // "context" sorts before "context.advisoryFraction", so within one scope
    // the sub-tree lands first and the dotted key refines what it wrote.
    await store.set(s, "context", {
      advisoryFraction: 0.5,
      hardFraction: 0.8,
      approxWindowTokens: 4096,
    });
    expect((await store.load({ scopes: [s] })).context).toEqual({
      advisoryFraction: 0.5,
      hardFraction: 0.8,
      approxWindowTokens: 4096,
    });

    await store.set(s, "context.advisoryFraction", 0.6);
    expect((await store.load({ scopes: [s] })).context).toEqual({
      advisoryFraction: 0.6,
      hardFraction: 0.8,
      approxWindowTokens: 4096,
    });
  });

  it("DEFECT: booleans round-trip as booleans", async () => {
    const s = scope("agent", "bool");
    await store.set(s, "replyGate.classifierEnabled", true);
    expect((await store.load({ scopes: [s] })).replyGate.classifierEnabled).toBe(true);
  });

  it("DEFECT: agent scope beats channel scope, whatever the list order", async () => {
    const ch = scope("channel", "prec");
    const ag = scope("agent", "prec");
    await store.set(ch, "model", "openrouter/from-channel/x");
    await store.set(ag, "model", "openrouter/from-agent/x");

    expect((await store.load({ scopes: [ch, ag] })).model).toBe("openrouter/from-agent/x");
    // Class order is fixed (global < channel < agent): reversing the list must
    // not let the channel row win.
    expect((await store.load({ scopes: [ag, ch] })).model).toBe("openrouter/from-agent/x");
    expect((await store.load({ scopes: [ch] })).model).toBe("openrouter/from-channel/x");
  });

  it("DEFECT: an unrelated channel's row never leaks into this snapshot", async () => {
    const mine = scope("channel", "mine");
    const theirs = scope("channel", "theirs");
    await store.set(mine, "model", "openrouter/mine/x");
    await store.set(theirs, "model", "openrouter/theirs/x");
    expect((await store.load({ scopes: [mine] })).model).toBe("openrouter/mine/x");
  });

  // -------------------------------------------------------------------------
  // DEFECT (FIXED) — one unreadable row used to wedge the whole scope.
  //
  // `validateSettings` rejected unknown keys on READ, and `set()` validated a
  // merged candidate that still contained the offending row. So a single row
  // whose key named no setting — a typo, a key removed by a later version, a
  // hand-written INSERT — made `load()`, `pinky config get`, every per-run
  // reload AND `pinky config set <any other key>` throw. Nothing could clear
  // it from inside the product; you needed psql.
  //
  // The fix is the load()/set() asymmetry documented in src/settings.ts:
  // load() prunes and warns, set() stays strict, and unset() removes a row
  // without caring whether its key was ever legal.
  // -------------------------------------------------------------------------

  it("DEFECT: a row naming no setting cannot brick load(), and unset clears it", async () => {
    const s = scope("channel", "junkrow");
    // Straight past the store, the way a typo in an older version left it.
    await db.query(`insert into settings (scope, key, value) values ($1, $2, $3)`, [
      s,
      "context.hardFractoin",
      jsonbParam(0.95),
    ]);

    const warnings: string[] = [];
    const lenient = new SettingsStore(db, { onWarning: (m) => warnings.push(m) });

    // 1. load() survives, with the defaults for the sub-tree it could not read.
    const snapshot = await lenient.load({ scopes: [s] });
    expect(snapshot.context).toEqual(DEFAULT_SETTINGS.context);
    expect(warnings.some((m) => m.includes("context.hardFractoin"))).toBe(true);

    // 2. an unrelated write still works — this is the half that used to make
    //    the wedge permanent.
    await lenient.set(s, "model", "openrouter/still/writable");
    expect((await lenient.load({ scopes: [s] })).model).toBe("openrouter/still/writable");

    // 3. and the human can actually remove it.
    expect(await lenient.unset(s, "context.hardFractoin")).toBe(true);
    expect(await lenient.unset(s, "context.hardFractoin")).toBe(false);
    expect((await rowsIn(s)).map((r) => r.key)).toEqual(["model"]);
  });

  it("DEFECT: a row with a real key and a broken VALUE degrades to the default", async () => {
    const s = scope("channel", "badvalue");
    await db.query(`insert into settings (scope, key, value) values ($1, $2, $3)`, [
      s,
      "context.hardFraction",
      jsonbParam("abc"),
    ]);
    const warnings: string[] = [];
    const lenient = new SettingsStore(db, { onWarning: (m) => warnings.push(m) });

    const snapshot = await lenient.load({ scopes: [s] });
    expect(snapshot.context.hardFraction).toBe(DEFAULT_SETTINGS.context.hardFraction);
    expect(warnings.some((m) => /context\.hardFraction/.test(m))).toBe(true);

    // Writing the key properly is the repair, and it is accepted.
    await lenient.set(s, "context.hardFraction", 0.95);
    expect((await lenient.load({ scopes: [s] })).context.hardFraction).toBe(0.95);
  });

  it("set(validateScopes) catches an inversion that spans two scopes", async () => {
    // Each write is valid in its own scope; together they invert the ladder
    // for every run that loads both. Only the cross-scope check sees it.
    const ch = scope("channel", "cross");
    const ag = scope("agent", "cross");
    // Valid in the agent's own view: the default advisory is 0.7.
    await store.set(ag, "context.hardFraction", 0.75);

    await expect(
      store.set(ch, "context.advisoryFraction", 0.8, { validateScopes: [ch, ag] }),
    ).rejects.toThrow(/less than context\.hardFraction/);
    expect(await rowsIn(ch)).toEqual([]);

    // Without the run's scopes the very same write is accepted — that IS the
    // hole: the channel's own view still has the default hardFraction 0.9.
    await store.set(ch, "context.advisoryFraction", 0.8);
    expect((await rowsIn(ch)).map((r) => r.key)).toEqual(["context.advisoryFraction"]);

    // What a run then gets: not a crash (load() repairs), but not the operator's
    // intent either — the advisory they wrote is gone. Refusing the write is
    // the only place that outcome can actually be prevented.
    const warnings: string[] = [];
    const lenient = new SettingsStore(db, { onWarning: (m) => warnings.push(m) });
    const run = await lenient.load({ scopes: [ch, ag] });
    expect(run.context.hardFraction).toBe(0.75);
    expect(run.context.advisoryFraction).toBe(DEFAULT_SETTINGS.context.advisoryFraction);
    expect(warnings.some((m) => m.includes("context.advisoryFraction"))).toBe(true);
  });

  it("DEFECT: a second set in a scope that already holds an object row succeeds", async () => {
    const s = scope("channel", "second-set");
    await store.set(s, "context", {
      advisoryFraction: 0.5,
      hardFraction: 0.8,
      approxWindowTokens: 4096,
    });
    await store.set(s, "model", "openrouter/still/works");
    expect((await rowsIn(s)).map((r) => r.key)).toEqual(["context", "model"]);
  });
});
