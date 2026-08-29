/**
 * MemoryStore against a live database (DESIGN.md §5).
 *
 * The unit suite (packages/core/test/memory.test.ts) drives the store through
 * a FakeDb that records SQL text, so it proves the shape of the statements and
 * nothing about whether Postgres accepts them. Everything that only exists on
 * the server is therefore untested until it runs here:
 *
 *   - the generated `tsv` column and `websearch_to_tsquery` (0005_memory_fts),
 *   - `kind = any($n::text[])`, which needs the cast to unify with a text column,
 *   - `meta || $n::jsonb` merging an invalidation reason into existing meta,
 *   - the JSONB CONTRACT (a doubly-encoded meta stores a jsonb *string*),
 *   - update()'s "one transaction" claim, proven by making the INSERT fail,
 *   - RLS as the app role, and
 *   - the vector voice: `embedding <=> $n::vector` with the '[…]' text form.
 *
 * TWO SERVERS, TWO BRANCHES. docker-compose.yml runs postgres:16-alpine on
 * 5544 (no pgvector: `supportsVectors()` is false, retain drops embeddings,
 * search falls back to FTS) and, behind the `vector` profile,
 * pgvector/pgvector:pg16 on 5545 (both voices). Neither branch is reachable on
 * the other image, so a full check is one run per server:
 *
 *   bun run db:up && bun run migrate && bun run test:integration
 *   bun run db:up:vector
 *   DATABASE_URL=postgres://postgres:pinky@localhost:5545/pinky \
 *   DATABASE_ADMIN_URL=postgres://postgres:pinky@localhost:5545/pinky \
 *     bun run migrate
 *   bun run test:integration:vector
 *
 * CI runs the pgvector image, so the vector suite below is the CI half and the
 * "no pgvector" suite is the local half; each asserts that the *other* branch
 * is genuinely unavailable rather than silently skipped.
 *
 * Skipped unless PINKY_INTEGRATION=1. Connections come from loadEnvConfig(),
 * never a literal port (local 5544/5545, CI 5432). Reads and writes go through
 * the unprivileged `pinky_app` role — derived from DATABASE_URL the way
 * rls.test.ts does it — because a superuser bypasses RLS and would make the
 * tenant assertions meaningless. The admin handle is used only for migrate,
 * for cleanup, and to look at columns the store deliberately never selects
 * (`embedding`, `meta`'s jsonb type).
 *
 * Every row this file writes is stamped with a run-unique tenant id starting
 * `it-memory-`, so cleanup is a scoped DELETE and two concurrent runs cannot
 * see each other.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { loadEnvConfig } from "../../src/config";
import { createDb } from "../../src/pg";
import { migrate } from "../../src/migrate";
import { withTenant } from "../../src/tenant";
import { MemoryStore } from "../../src/memory";
import type { Db } from "../../src/db";
import type { MemoryRow, RecallScope } from "../../src/memory";

const ENABLED = process.env.PINKY_INTEGRATION === "1";

/** DATABASE_URL with pinky_app's dev credentials substituted (0003_rls.sql,
 *  docker-compose.yml). Host, port and database are kept, so a stock checkout
 *  whose DATABASE_URL is still the superuser tests the RLS path anyway. */
function asAppRole(url: string): string {
  const u = new URL(url);
  u.username = "pinky_app";
  u.password = "pinky";
  return u.toString();
}

const ENV = loadEnvConfig();
const ADMIN_URL = ENV.databaseAdminUrl;
const APP_URL = process.env.PINKY_TEST_APP_URL ?? asAppRole(ENV.databaseUrl);
const SCHEMA_DIR = new URL("../../schema", import.meta.url).pathname;

const RUN = crypto.randomUUID().slice(0, 8);
const PREFIX = "it-memory-";
const TENANT = `${PREFIX}${RUN}`;
const TENANT_B = `${PREFIX}b-${RUN}`;
const CHANNEL = `slack:C${RUN}`;
const OTHER_CHANNEL = `slack:C${RUN}-other`;
const USER = `slack:U${RUN}`;

/** Each group of tests gets its own agent id: the scope predicate keys on
 *  agent_id, so groups are invisible to one another and a search can assert on
 *  exact result sets instead of "contains". */
const agent = (group: string): string => `pinky-${group}-${RUN}`;

function scope(agentId: string, over: Partial<RecallScope> = {}): RecallScope {
  return { agentId, includeUser: false, includePrivate: false, ...over };
}

/** Await a call that must reject, and hand back the error (with its SQLSTATE
 *  when Postgres raised it) instead of an unhelpful "expected to throw". */
async function rejection(p: Promise<unknown>): Promise<Error & { code?: string }> {
  try {
    await p;
  } catch (err) {
    return err as Error & { code?: string };
  }
  throw new Error("expected the call to reject, but it resolved");
}

/**
 * Does this SERVER have pgvector? Deliberately probed on pg_extension rather
 * than on `memories.embedding`: the point of the vector suite is that when the
 * extension is there, MemoryStore.supportsVectors() must agree — a migration
 * that failed to add the column has to fail a test, not quietly skip one.
 *
 * Runs the migrations first (idempotent; the other integration files do the
 * same in beforeAll) because 0001/0002 are what CREATE EXTENSION.
 */
async function probeVectorSupport(): Promise<boolean> {
  const admin = createDb(ADMIN_URL, { max: 1 });
  try {
    await migrate(admin, SCHEMA_DIR);
    const row = await admin.queryOne<{ n: number }>(
      `select count(*)::int as n from pg_extension where extname = 'vector'`,
    );
    return Number(row?.n ?? 0) > 0;
  } finally {
    await admin.close();
  }
}

const HAS_PGVECTOR = ENABLED ? await probeVectorSupport() : false;

const suite = ENABLED ? describe : describe.skip;
/** Only on a pgvector server. */
const vectorSuite = ENABLED && HAS_PGVECTOR ? describe : describe.skip;
/** Only on a server WITHOUT pgvector (postgres:16-alpine, i.e. local compose). */
const plainSuite = ENABLED && !HAS_PGVECTOR ? describe : describe.skip;

const DIMENSIONS = 1536;

/** A 1536-d vector with the given (index, value) pairs and zeros elsewhere.
 *  Values are exact in float4 — pgvector's element type — so `embedding::text`
 *  round-trips them verbatim and the wire-format assertion can be exact. */
function vector(spec: [number, number][]): number[] {
  const v = new Array<number>(DIMENSIONS).fill(0);
  for (const [index, value] of spec) v[index] = value;
  return v;
}

// Cosine distance from V_A: A = 0, B = 1 - 0.7071 = 0.2929, C = 1. Distinct,
// so the vector voice has a deterministic order with no ties to break.
const V_A = vector([[0, 1]]);
const V_B = vector([
  [0, 0.5],
  [1, 0.5],
]);
const V_C = vector([[1, 1]]);

suite("MemoryStore (live postgres)", () => {
  let admin: Db;
  let app: Db;
  let store: MemoryStore;
  let storeB: MemoryStore;

  const purge = async (): Promise<void> => {
    // Superuser: bypasses RLS, so one statement clears every tenant this file
    // may have created (including a crashed previous run).
    await admin.query(`delete from memories where tenant_id like $1`, [`${PREFIX}%`]);
  };

  beforeAll(async () => {
    admin = createDb(ADMIN_URL, { max: 2 });
    // Migrations already ran in probeVectorSupport(); this connection only
    // does setup, cleanup and the raw column reads.
    await purge();
    app = createDb(APP_URL, { max: 8 });
    store = new MemoryStore(withTenant(app, TENANT), TENANT);
    storeB = new MemoryStore(withTenant(app, TENANT_B), TENANT_B);
  });

  afterAll(async () => {
    if (app) await app.close();
    if (admin) {
      await purge();
      await admin.close();
    }
  });

  it("runs as pinky_app, so RLS actually applies", async () => {
    const who = await app.queryOne<{ role: string; superuser: boolean; bypassrls: boolean }>(
      `select current_user as role, rolsuper as superuser, rolbypassrls as bypassrls
         from pg_roles where rolname = current_user`,
    );
    expect(who?.role).toBe("pinky_app");
    expect(who?.superuser).toBe(false);
    expect(who?.bypassrls).toBe(false);
  });

  it("0005 gave memories a generated tsv column with a GIN index", async () => {
    const col = await admin.queryOne<{ data_type: string; generated: string }>(
      `select data_type, is_generated as generated from information_schema.columns
        where table_name = 'memories' and column_name = 'tsv'`,
    );
    expect(col?.data_type).toBe("tsvector");
    expect(col?.generated).toBe("ALWAYS");

    const idx = await admin.queryOne<{ indexdef: string }>(
      `select indexdef from pg_indexes where indexname = 'memories_tsv_idx'`,
    );
    expect(idx?.indexdef ?? "").toMatch(/using gin/i);

    const model = await admin.queryOne(
      `select 1 from information_schema.columns
        where table_name = 'memories' and column_name = 'embedding_model'`,
    );
    expect(model).not.toBeNull();
  });

  // -------------------------------------------------------------------------
  describe("retain -> get", () => {
    const AGENT = agent("rt");
    let row: MemoryRow;

    beforeAll(async () => {
      row = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "  Brad prefers terse answers without preamble  ",
        importance: 7,
        meta: { source: { channelId: CHANNEL }, retainedBy: AGENT, count: 3 },
      });
    });

    it("round-trips every field, with ISO timestamps and a numeric importance", async () => {
      expect(row.id).toMatch(/^[0-9a-f-]{36}$/);
      expect(row.tenantId).toBe(TENANT);
      expect(row.agentId).toBe(AGENT);
      expect(row.visibility).toBe("tenant");
      expect(row.kind).toBe("semantic");
      // insertRow trims before storing.
      expect(row.text).toBe("Brad prefers terse answers without preamble");
      expect(typeof row.importance).toBe("number");
      expect(row.importance).toBe(7);
      expect(row.userId).toBeNull();
      expect(row.channelId).toBeNull();
      expect(row.validTo).toBeNull();
      expect(row.embeddingModel).toBeNull();
      // Timestamptz arrives as a Date; the store must hand back an ISO string.
      expect(typeof row.recordedAt).toBe("string");
      expect(row.recordedAt).toMatch(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z$/);
      expect(new Date(row.recordedAt).getTime()).toBeGreaterThan(Date.now() - 300_000);
      expect(typeof row.validFrom).toBe("string");
      expect(row.validFrom).toMatch(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z$/);

      const fetched = await store.get(row.id);
      expect(fetched).toEqual(row);
    });

    it("stores meta as a jsonb OBJECT, not a doubly-encoded string", async () => {
      // The failure this guards is invisible from JS: JSON.stringify-ing the
      // param stores a jsonb *string*, `meta->>'retainedBy'` goes NULL, and
      // mapRow's legacy branch would still hand back a plausible object.
      const raw = await admin.queryOne<{ t: string; by: string | null; n: string | null }>(
        `select jsonb_typeof(meta) as t, meta->>'retainedBy' as by,
                meta#>>'{source,channelId}' as n
           from memories where id = $1`,
        [row.id],
      );
      expect(raw?.t).toBe("object");
      expect(raw?.by).toBe(AGENT);
      expect(raw?.n).toBe(CHANNEL);
      expect(row.meta).toEqual({
        source: { channelId: CHANNEL },
        retainedBy: AGENT,
        count: 3,
      });
    });

    it("get() returns null for an id this tenant does not have", async () => {
      expect(await store.get(crypto.randomUUID())).toBeNull();
    });

    it("refuses invalid input before it reaches the server", async () => {
      const base = { agentId: AGENT, visibility: "tenant", kind: "semantic" } as const;
      expect((await rejection(store.retain({ ...base, text: "x", importance: 11 }))).message).toMatch(
        /importance must be an integer 1\.\.10/,
      );
      expect((await rejection(store.retain({ ...base, text: "   " }))).message).toMatch(
        /text must be a non-empty string/,
      );
      expect(
        (
          await rejection(
            store.retain({ agentId: AGENT, visibility: "channel", kind: "semantic", text: "x" }),
          )
        ).message,
      ).toMatch(/visibility 'channel' requires channelId/);
    });

    it("the importance CHECK is a real constraint underneath, not only a TS guard", async () => {
      // Belt and braces: the store validates, and the schema would too.
      const err = await rejection(
        withTenant(app, TENANT).query(
          `insert into memories (id, tenant_id, agent_id, visibility, kind, text, importance)
           values ($1, $2, $3, 'tenant', 'semantic', 'x', 11)`,
          [crypto.randomUUID(), TENANT, AGENT],
        ),
      );
      expect(err.code).toBe("23514"); // check_violation
    });
  });

  // -------------------------------------------------------------------------
  describe("FTS voice (generated tsv + websearch_to_tsquery)", () => {
    const AGENT = agent("fts");
    const s = scope(AGENT);
    let terse: MemoryRow;
    let deploy: MemoryRow;
    let commit: MemoryRow;

    beforeAll(async () => {
      const write = (kind: "semantic" | "episodic" | "procedural", text: string) =>
        store.retain({ agentId: AGENT, visibility: "tenant", kind, text });
      terse = await write("semantic", "Brad prefers terse answers without preamble");
      deploy = await write("episodic", "The deploy failed on a missing DATABASE_URL env var");
      commit = await write("procedural", "Run bun test before every commit");
    });

    it("finds the one matching row, ranked by the fts voice", async () => {
      const hits = await store.search({ scope: s, query: "deploy failed" });
      expect(hits.map((h) => h.id)).toEqual([deploy.id]);
      expect(hits[0]!.voices.fts).toBe(1);
      expect(hits[0]!.voices.vector).toBeUndefined();
      expect(hits[0]!.score).toBeGreaterThan(0);
      // English stemming is live: the query said "failed", the row says the same
      // stem, and a word from another row must not match.
      expect((await store.search({ scope: s, query: "preamble" })).map((h) => h.id)).toEqual([
        terse.id,
      ]);
    });

    it("a stopword-only query returns nothing and does not throw", async () => {
      // websearch_to_tsquery('english', 'the') is the EMPTY tsquery, which
      // matches no row; the interesting part is that it is not an error.
      expect(await store.search({ scope: s, query: "the" })).toEqual([]);
    });

    it("tsquery punctuation is inert — websearch_to_tsquery, never to_tsquery", async () => {
      // Every one of these is a syntax error for to_tsquery(). Recall queries
      // come from an LLM or a human, so they must never be able to raise one.
      for (const q of [
        `"terse answers"`,
        "deploy & commit",
        "deploy | commit",
        "deploy:*",
        "!deploy",
        "deploy!!! & | : ( )",
        "'quoted' \"double\" <-> ampersand & pipe |",
        "-preamble",
      ]) {
        const hits = await store.search({ scope: s, query: q });
        expect(Array.isArray(hits)).toBe(true);
      }
    });

    it("a blank query with no embedding degrades to a newest-first listing", async () => {
      const hits = await store.search({ scope: s, query: "" });
      expect(hits.map((h) => h.id)).toEqual([commit.id, deploy.id, terse.id]);
      // No relevance signal, so no voice ranked them.
      for (const h of hits) expect(h.voices).toEqual({});
      expect(hits[0]!.score).toBeGreaterThan(0);
    });

    it("kinds filters both voices — `kind = any($n::text[])` on a live text column", async () => {
      const listed = await store.list({ scope: s, kinds: ["procedural"] });
      expect(listed.map((r) => r.id)).toEqual([commit.id]);

      const searched = await store.search({ scope: s, query: "deploy failed", kinds: ["semantic"] });
      expect(searched).toEqual([]);

      const both = await store.search({
        scope: s,
        query: "",
        kinds: ["episodic", "procedural"],
      });
      expect(both.map((h) => h.id)).toEqual([commit.id, deploy.id]);
    });

    it("limit cuts the result set", async () => {
      const hits = await store.search({ scope: s, query: "", limit: 2 });
      expect(hits).toHaveLength(2);
      expect(await store.list({ scope: s, limit: 1 })).toHaveLength(1);
    });
  });

  // -------------------------------------------------------------------------
  describe("scope predicate on real rows (§5.1)", () => {
    const AGENT = agent("scope");
    let tenantRow: MemoryRow;
    let globalRow: MemoryRow;
    let channelRow: MemoryRow;
    let userRow: MemoryRow;
    let privateRow: MemoryRow;

    beforeAll(async () => {
      tenantRow = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "scope tenant row",
      });
      globalRow = await store.retain({
        agentId: AGENT,
        visibility: "global",
        kind: "semantic",
        text: "scope global row",
      });
      channelRow = await store.retain({
        agentId: AGENT,
        visibility: "channel",
        channelId: CHANNEL,
        kind: "episodic",
        text: "scope channel row",
      });
      userRow = await store.retain({
        agentId: AGENT,
        visibility: "user",
        userId: USER,
        kind: "semantic",
        text: "scope user row",
      });
      privateRow = await store.retain({
        agentId: AGENT,
        visibility: "private",
        kind: "procedural",
        text: "scope private row",
      });
    });

    const ids = (rows: MemoryRow[]): string[] => rows.map((r) => r.id).sort();

    it("a shared channel sees tenant + global + that channel only", async () => {
      const rows = await store.list({ scope: scope(AGENT, { channelId: CHANNEL }) });
      expect(ids(rows)).toEqual([tenantRow.id, globalRow.id, channelRow.id].sort());
    });

    it("another channel cannot see the first channel's row", async () => {
      const rows = await store.list({ scope: scope(AGENT, { channelId: OTHER_CHANNEL }) });
      expect(ids(rows)).toEqual([tenantRow.id, globalRow.id].sort());
      expect(ids(rows)).not.toContain(channelRow.id);
    });

    it("user rows need includeUser AND the matching userId", async () => {
      const withoutFlag = await store.list({ scope: scope(AGENT, { userId: USER }) });
      expect(ids(withoutFlag)).not.toContain(userRow.id);

      const withFlag = await store.list({
        scope: scope(AGENT, { userId: USER, includeUser: true }),
      });
      expect(ids(withFlag)).toEqual([tenantRow.id, globalRow.id, userRow.id].sort());

      const otherUser = await store.list({
        scope: scope(AGENT, { userId: `${USER}-someone-else`, includeUser: true }),
      });
      expect(ids(otherUser)).not.toContain(userRow.id);
    });

    it("private rows are never projected without includePrivate", async () => {
      const shared = await store.list({ scope: scope(AGENT, { channelId: CHANNEL }) });
      expect(ids(shared)).not.toContain(privateRow.id);

      const trusted = await store.list({ scope: scope(AGENT, { includePrivate: true }) });
      expect(ids(trusted)).toEqual([tenantRow.id, globalRow.id, privateRow.id].sort());
    });

    it("the predicate applies to search as well as list", async () => {
      const hidden = await store.search({ scope: scope(AGENT), query: "scope channel row" });
      expect(hidden).toEqual([]);

      const visible = await store.search({
        scope: scope(AGENT, { channelId: CHANNEL }),
        query: "scope channel row",
      });
      expect(visible.map((h) => h.id)).toEqual([channelRow.id]);
    });

    it("another agent's rows are invisible even inside the same tenant", async () => {
      const rows = await store.list({
        scope: scope(agent("scope-other"), { channelId: CHANNEL, includePrivate: true }),
      });
      expect(rows).toEqual([]);
    });
  });

  // -------------------------------------------------------------------------
  describe("invalidate (§5.2: never DELETE)", () => {
    const AGENT = agent("inval");
    const s = scope(AGENT);
    let row: MemoryRow;

    beforeAll(async () => {
      row = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "the standup is at nine wombat",
        meta: { origin: "seed", nested: { keep: true } },
      });
    });

    it("stamps valid_to, merges the reason into meta, and is idempotent", async () => {
      expect(await store.invalidate(row.id, { reason: "standup moved" })).toBe(true);
      // Second call finds no `valid_to is null` row: false, and the original
      // invalidation time is not rewritten.
      expect(await store.invalidate(row.id, { reason: "again" })).toBe(false);

      const after = await store.get(row.id);
      expect(after?.validTo).not.toBeNull();
      expect(after?.validTo).toMatch(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z$/);
      // `meta || $n::jsonb` MERGES; the original keys must survive.
      expect(after?.meta).toEqual({
        origin: "seed",
        nested: { keep: true },
        invalidatedReason: "standup moved",
      });

      const raw = await admin.queryOne<{ t: string; reason: string | null }>(
        `select jsonb_typeof(meta) as t, meta->>'invalidatedReason' as reason
           from memories where id = $1`,
        [row.id],
      );
      expect(raw?.t).toBe("object");
      expect(raw?.reason).toBe("standup moved");
    });

    it("the row still exists — it is history, not a deletion", async () => {
      const raw = await admin.queryOne(`select id from memories where id = $1`, [row.id]);
      expect(raw).not.toBeNull();
    });

    it("an invalidated row is gone from list and search, unless includeInvalid", async () => {
      expect(await store.list({ scope: s })).toEqual([]);
      expect(await store.search({ scope: s, query: "wombat" })).toEqual([]);
      expect(await store.search({ scope: s, query: "" })).toEqual([]);

      const history = await store.list({ scope: s, includeInvalid: true });
      expect(history.map((r) => r.id)).toEqual([row.id]);
    });

    it("invalidate() on an unknown id is false, not an error", async () => {
      expect(await store.invalidate(crypto.randomUUID())).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  describe("update = invalidate + insert in ONE transaction (§5.2)", () => {
    const AGENT = agent("upd");
    const s = scope(AGENT);
    let original: MemoryRow;
    let replacement: MemoryRow;

    beforeAll(async () => {
      original = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "Brad prefers dark mode",
        importance: 4,
        meta: { origin: "seed" },
      });
      replacement = await store.update(original.id, {
        visibility: "tenant",
        kind: "semantic",
        text: "Brad prefers light mode",
        importance: 6,
        meta: { origin: "correction" },
      });
    });

    it("retires the old row and inserts the successor with meta.supersedes", async () => {
      expect(replacement.id).not.toBe(original.id);
      expect(replacement.agentId).toBe(AGENT); // carried from the old row
      expect(replacement.importance).toBe(6);
      expect(replacement.validTo).toBeNull();
      expect(replacement.meta).toEqual({ origin: "correction", supersedes: original.id });

      const old = await store.get(original.id);
      expect(old?.validTo).not.toBeNull();

      // Current truth is exactly one row.
      const current = await store.list({ scope: s });
      expect(current.map((r) => r.id)).toEqual([replacement.id]);
      const history = await store.list({ scope: s, includeInvalid: true });
      expect(history.map((r) => r.id).sort()).toEqual([replacement.id, original.id].sort());
    });

    it("refuses to supersede a row that is unknown or already retired", async () => {
      expect((await rejection(store.update(original.id, {
        visibility: "tenant",
        kind: "semantic",
        text: "third opinion",
      }))).message).toMatch(/already invalidated/);

      expect((await rejection(store.update(crypto.randomUUID(), {
        visibility: "tenant",
        kind: "semantic",
        text: "orphan",
      }))).message).toMatch(/not found/);
    });

    it("DEFECT: a failing INSERT rolls the invalidation back — no half-applied update", async () => {
      // The rollback trigger has to be something the store does NOT pre-validate,
      // or the transaction never opens: importance 11 and a bad kind are both
      // rejected in TS. A NUL inside a jsonb string is not — it reaches the
      // server and is rejected there (SQLSTATE 22P05), inside the transaction,
      // after `valid_to` has already been stamped. If update() were two
      // statements instead of one transaction, the row would be left retired
      // with no successor: the fact silently disappears from recall.
      const victim = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "the release train leaves on Thursday",
        meta: { origin: "seed" },
      });

      const err = await rejection(
        store.update(victim.id, {
          visibility: "tenant",
          kind: "semantic",
          text: "the release train leaves on Friday",
          meta: { poison: `a${String.fromCharCode(0)}b` },
        }),
      );
      expect(err.code).toBe("22P05");
      expect(err.message).toMatch(/unsupported Unicode escape sequence/);

      // Rolled back: the old row is still current truth...
      const after = await store.get(victim.id);
      expect(after?.validTo).toBeNull();
      expect(after?.meta).toEqual({ origin: "seed" });
      // ...and the replacement never landed.
      const current = await store.list({ scope: s, includeInvalid: true });
      expect(current.map((r) => r.text)).not.toContain("the release train leaves on Friday");
      expect(current.filter((r) => r.id === victim.id && r.validTo === null)).toHaveLength(1);
    });
  });

  // -------------------------------------------------------------------------
  describe("findByIdPrefix: short ids resolved in SQL, not by scanning", () => {
    const AGENT = agent("prefix");
    const s = scope(AGENT);
    const FILLERS = 250;
    let needle: MemoryRow;

    beforeAll(async () => {
      needle = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "the pager rotation moved to Tuesdays",
      });
      // Backdate it behind every filler: the row somebody names by short id is
      // exactly the OLD one, and "list the newest 200 and filter in JS" is
      // structurally incapable of finding it.
      await admin.query(`update memories set recorded_at = now() - interval '400 minutes' where id = $1`, [
        needle.id,
      ]);
      await admin.query(
        `insert into memories (id, tenant_id, agent_id, visibility, kind, text, recorded_at)
         select gen_random_uuid()::text, $1, $2, 'tenant', 'semantic', 'filler ' || g,
                now() - (g || ' minutes')::interval
           from generate_series(1, $3::int) g`,
        [TENANT, AGENT, FILLERS],
      );
      // Two rows sharing a prefix, so ambiguity is a real case and not a story.
      for (const suffix of ["0001", "0002"]) {
        await admin.query(
          `insert into memories (id, tenant_id, agent_id, visibility, kind, text)
           values ($1, $2, $3, 'tenant', 'semantic', $4)`,
          [`deadbeef-0000-0000-0000-00000000${suffix}`, TENANT, AGENT, `twin ${suffix}`],
        );
      }
    });

    it("finds a row buried under 250 newer ones", async () => {
      const found = await store.findByIdPrefix(needle.id.slice(0, 8), { scope: s });
      expect(found.map((r) => r.id)).toEqual([needle.id]);
      expect(found[0]!.text).toBe("the pager rotation moved to Tuesdays");

      // The approach this replaces, for contrast.
      const listed = await store.list({ scope: s, limit: 200 });
      expect(listed).toHaveLength(200);
      expect(listed.map((r) => r.id)).not.toContain(needle.id);
    });

    it("returns two rows for an ambiguous prefix, so the caller can say so", async () => {
      const hits = await store.findByIdPrefix("deadbeef", { scope: s });
      expect(hits).toHaveLength(2); // the default limit is 2, on purpose
      expect(hits.map((r) => r.text).sort()).toEqual(["twin 0001", "twin 0002"]);
      expect(await store.findByIdPrefix("deadbeef-0000-0000-0000-000000000001", { scope: s })).toHaveLength(1);
      expect(await store.findByIdPrefix("0000dead", { scope: s })).toEqual([]);
    });

    it("obeys the scope predicate, the tenant, and valid_to", async () => {
      // Another agent's rows are not addressable by prefix either.
      expect(
        await store.findByIdPrefix(needle.id.slice(0, 8), { scope: scope(agent("prefix-other")) }),
      ).toEqual([]);
      expect(await storeB.findByIdPrefix(needle.id.slice(0, 8), { scope: s })).toEqual([]);

      const twin = "deadbeef-0000-0000-0000-000000000001";
      expect(await store.invalidate(twin, { reason: "prefix test" })).toBe(true);
      expect((await store.findByIdPrefix("deadbeef", { scope: s })).map((r) => r.id)).toEqual([
        "deadbeef-0000-0000-0000-000000000002",
      ]);
      expect(
        (await store.findByIdPrefix("deadbeef", { scope: s, includeInvalid: true })).map((r) => r.id).sort(),
      ).toEqual([twin, "deadbeef-0000-0000-0000-000000000002"]);
    });

    it("a LIKE wildcard never reaches the query", async () => {
      // `%` would make `id like '%' || '%'` match the whole plane, and a
      // caller checking "exactly one row" would act on an arbitrary memory.
      // The character class is the guard — nothing downstream escapes anything.
      expect((await rejection(store.findByIdPrefix("%", { scope: s }))).message).toMatch(
        /id prefix must be/,
      );
      expect((await rejection(store.findByIdPrefix("dead%", { scope: s }))).message).toMatch(
        /id prefix must be/,
      );
      expect((await rejection(store.findByIdPrefix("dead_eef", { scope: s }))).message).toMatch(
        /id prefix must be/,
      );
    });
  });

  // -------------------------------------------------------------------------
  describe("RLS: one store per tenant, and nothing crosses", () => {
    const AGENT = agent("rls");
    const s = scope(AGENT);
    let mine: MemoryRow;

    beforeAll(async () => {
      mine = await store.retain({
        agentId: AGENT,
        visibility: "global",
        kind: "semantic",
        text: "tenant A armadillo secret",
      });
    });

    it("tenant B sees none of tenant A's rows — even 'global' ones", async () => {
      // 'global' visibility means "every channel and user of THIS tenant"
      // (§5.1); it is not cross-tenant in v1.
      expect(await storeB.list({ scope: s })).toEqual([]);
      expect(await storeB.search({ scope: s, query: "armadillo" })).toEqual([]);
      expect(await storeB.get(mine.id)).toBeNull();
    });

    it("tenant B's own row is invisible to tenant A", async () => {
      const theirs = await storeB.retain({
        agentId: AGENT,
        visibility: "global",
        kind: "semantic",
        text: "tenant B armadillo secret",
      });
      expect((await store.list({ scope: s })).map((r) => r.id)).toEqual([mine.id]);
      expect(await store.get(theirs.id)).toBeNull();
      expect((await storeB.list({ scope: s })).map((r) => r.id)).toEqual([theirs.id]);
    });

    it("DOCUMENTED LIMIT: the admin (superuser) connection sees both tenants", async () => {
      // FORCE ROW LEVEL SECURITY binds the table owner, not a superuser. This
      // is why the app must connect as pinky_app — and why every store query
      // repeats `tenant_id = $1` as well as relying on the policy.
      const rows = await admin.query<{ tenant_id: string }>(
        `select distinct tenant_id from memories
          where text like '%armadillo secret'`,
      );
      // Sort on the JS side, not with `order by`: the server's collation is
      // the image's (glibc en_US on pgvector/pgvector, C on alpine) and glibc
      // ignores the hyphens in these ids at the first level, so a DB-ordered
      // list only matches a code-point sort for some run ids. Flaky in CI once.
      expect(rows.map((r) => r.tenant_id).sort()).toEqual([TENANT, TENANT_B].sort());
    });
  });

  // -------------------------------------------------------------------------
  // The vector voice. THIS is the half no test has ever executed: the local
  // compose image has no pgvector, so `embedding <=> $n::vector` and the HNSW
  // index only run on pgvector/pgvector:pg16 (CI, or `bun run db:up:vector`).
  // -------------------------------------------------------------------------
  vectorSuite("vector voice (pgvector present)", () => {
    const AGENT = agent("vec");
    const s = scope(AGENT);
    const MODEL = "fake/it-1536";
    let alpha: MemoryRow;
    let beta: MemoryRow;
    let gamma: MemoryRow;
    let delta: MemoryRow;

    beforeAll(async () => {
      alpha = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "alpha kangaroo doctrine",
        embedding: V_A,
        embeddingModel: MODEL,
      });
      beta = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "beta platypus doctrine",
        embedding: V_B,
        embeddingModel: MODEL,
      });
      gamma = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "gamma armadillo doctrine",
        embedding: V_C,
        embeddingModel: MODEL,
      });
      // No embedding: only the lexical voice can ever reach it.
      delta = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "delta wombat doctrine",
      });
    });

    it("supportsVectors() agrees with the server (pg_extension says yes)", async () => {
      // The whole point of the gate: if the extension is installed but
      // 0002_embeddings.rerun.sql did not add the column, this FAILS rather
      // than skipping the suite and reporting green.
      expect(await store.supportsVectors()).toBe(true);
      const col = await admin.queryOne<{ udt: string }>(
        `select udt_name as udt from information_schema.columns
          where table_name = 'memories' and column_name = 'embedding'`,
      );
      expect(col?.udt).toBe("vector");
      const idx = await admin.queryOne<{ indexdef: string }>(
        `select indexdef from pg_indexes where indexname = 'memories_embedding_idx'`,
      );
      expect(idx?.indexdef ?? "").toMatch(/using hnsw/i);
    });

    it("ranks by cosine distance: the identical vector is rank 1", async () => {
      const hits = await store.search({ scope: s, query: "", queryEmbedding: V_A });
      expect(hits.map((h) => h.id)).toEqual([alpha.id, beta.id, gamma.id]);
      expect(hits[0]!.voices.vector).toBe(1);
      expect(hits[0]!.voices.fts).toBeUndefined();
      expect(hits[1]!.voices.vector).toBe(2);
      expect(hits[2]!.voices.vector).toBe(3);
      // A query nearer beta reorders the voice — the distance is real, not
      // insertion order dressed up.
      const nearBeta = await store.search({ scope: s, query: "", queryEmbedding: V_B });
      expect(nearBeta[0]!.id).toBe(beta.id);
    });

    it("a row with no embedding is excluded from the vector voice but found by FTS", async () => {
      const vectorOnly = await store.search({ scope: s, query: "", queryEmbedding: V_A });
      expect(vectorOnly.map((h) => h.id)).not.toContain(delta.id);

      const lexical = await store.search({ scope: s, query: "wombat", queryEmbedding: V_A });
      const hit = lexical.find((h) => h.id === delta.id);
      expect(hit).toBeDefined();
      expect(hit!.voices.fts).toBe(1);
      expect(hit!.voices.vector).toBeUndefined();
    });

    it("a row ranked by BOTH voices outranks one ranked by a single voice", async () => {
      // "platypus" matches beta alone; V_A puts alpha at vector rank 1 and beta
      // at 2. RRF: beta = 1/61 + 1/62 > alpha = 1/61, so beta leads.
      const hits = await store.search({ scope: s, query: "platypus", queryEmbedding: V_A });
      expect(hits[0]!.id).toBe(beta.id);
      expect(hits[0]!.voices).toEqual({ vector: 2, fts: 1 });
      expect(hits[1]!.id).toBe(alpha.id);
      expect(hits[1]!.voices).toEqual({ vector: 1 });
      expect(hits[0]!.score).toBeGreaterThan(hits[1]!.score);
    });

    it("records embedding_model, and the store never selects the embedding itself", async () => {
      expect(alpha.embeddingModel).toBe(MODEL);
      expect((await store.get(alpha.id))?.embeddingModel).toBe(MODEL);
      expect(delta.embeddingModel).toBeNull();
      const raw = await admin.queryOne<{ m: string | null }>(
        `select embedding_model as m from memories where id = $1`,
        [delta.id],
      );
      expect(raw?.m).toBeNull();
      // MemoryRow has no embedding field: 1536 floats per row are useless in JS
      // and would dominate every recall payload.
      expect(Object.keys(alpha)).not.toContain("embedding");
    });

    it("DEFECT: the param is the '[…]' TEXT form — a JS number[] would not survive", async () => {
      // What actually landed in the column, byte for byte.
      const raw = await admin.queryOne<{ v: string }>(
        `select embedding::text as v from memories where id = $1`,
        [beta.id],
      );
      const parts = (raw?.v ?? "").replace(/^\[|\]$/g, "").split(",");
      expect(parts).toHaveLength(DIMENSIONS);
      expect(parts[0]).toBe("0.5");
      expect(parts[1]).toBe("0.5");
      expect(parts[2]).toBe("0");
      expect(raw?.v.startsWith("[0.5,0.5,0,")).toBe(true);

      // And the alternative the store deliberately avoids: postgres.js binds a
      // JS array as an array type, which the server will not read as a vector.
      const err = await rejection(app.query(`select $1::vector as v`, [[1, 2, 3]]));
      expect(err.message).toMatch(/invalid input syntax for type vector|cannot be cast|does not exist/);
      // The text form the store builds is accepted.
      const ok = await app.queryOne<{ v: string }>(`select $1::vector::text as v`, ["[1,2,3]"]);
      expect(ok?.v).toBe("[1,2,3]");
    });

    it("DEFECT: a wrong-width embedding is DROPPED, and the memory still lands", async () => {
      // What this used to be: the insert threw (22000, "expected 1536
      // dimensions") and the memory was lost. `memory.embeddingModel` is a
      // setting, so "openai/text-embedding-3-large" (3072) is a legal thing
      // for a human to write — and every retain after that write silently
      // failed. The text is worth more than the vector: keep the row, drop the
      // vector, warn, and let FTS recall it.
      const warnings: string[] = [];
      const lenient = new MemoryStore(withTenant(app, TENANT), TENANT, {
        onWarning: (m) => warnings.push(m),
      });
      expect(lenient.vectorDimensions).toBe(DIMENSIONS);

      const row = await lenient.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "three dimensional truth",
        embedding: [1, 2, 3],
        embeddingModel: "openai/text-embedding-3-large",
      });

      expect(row.id).toMatch(/^[0-9a-f-]{36}$/);
      expect(row.embeddingModel).toBeNull();
      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toContain("vector(1536)");

      // The column really is empty, and the row is recallable by FTS.
      const raw = await admin.queryOne<{ v: string | null }>(
        `select embedding::text as v from memories where id = $1`,
        [row.id],
      );
      expect(raw?.v).toBeNull();
      expect((await store.search({ scope: s, query: "dimensional" })).map((h) => h.id)).toContain(
        row.id,
      );
    });

    it("update() re-embeds inside the transaction: the vector param works on a tx handle", async () => {
      // insertRow() runs on the transaction handle here, not the pool handle —
      // a different code path for the `$n::vector` bind, and the one the
      // memory_edit tool uses when it recomputes an embedding.
      const before = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "epsilon numbat doctrine",
        embedding: V_C,
        embeddingModel: MODEL,
      });
      const after = await store.update(before.id, {
        visibility: "tenant",
        kind: "semantic",
        text: "epsilon numbat doctrine, revised",
        embedding: V_A,
        embeddingModel: MODEL,
      });
      expect(after.embeddingModel).toBe(MODEL);
      expect((await store.get(before.id))?.validTo).not.toBeNull();

      const raw = await admin.queryOne<{ v: string }>(
        `select embedding::text as v from memories where id = $1`,
        [after.id],
      );
      expect(raw?.v.startsWith("[1,0,0,")).toBe(true);

      // The successor is now the nearest row to V_A, ahead of alpha, because
      // ties break newer-first and both sit at distance 0.
      const hits = await store.search({ scope: s, query: "numbat", queryEmbedding: V_A });
      expect(hits[0]!.id).toBe(after.id);
      expect(hits[0]!.voices.vector).toBeLessThanOrEqual(2);
      expect(hits.map((h) => h.id)).not.toContain(before.id);
    });

    it("a non-finite embedding is refused in TS, before the wire", async () => {
      const err = await rejection(
        store.retain({
          agentId: AGENT,
          visibility: "tenant",
          kind: "semantic",
          text: "not a number",
          embedding: vector([[0, Number.NaN]]),
        }),
      );
      expect(err.message).toMatch(/embedding must contain only finite numbers/);
    });
  });

  // -------------------------------------------------------------------------
  plainSuite("no pgvector (postgres:16-alpine): the FTS-only branch", () => {
    const AGENT = agent("novec");
    const s = scope(AGENT);
    let row: MemoryRow;

    beforeAll(async () => {
      row = await store.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "alpine kangaroo doctrine",
        embedding: V_A,
        embeddingModel: "fake/it-1536",
      });
    });

    it("supportsVectors() is false and memories has no embedding column", async () => {
      expect(await store.supportsVectors()).toBe(false);
      const col = await admin.queryOne(
        `select 1 from information_schema.columns
          where table_name = 'memories' and column_name = 'embedding'`,
      );
      expect(col).toBeNull();
    });

    it("DEFECT: retain with an embedding still succeeds — the row is stored without one", async () => {
      // The failure mode this guards: a store that always emits the embedding
      // column makes every retain throw 42703 on the image most developers run,
      // so the memory plane is simply dead locally.
      expect(row.id).toMatch(/^[0-9a-f-]{36}$/);
      expect(row.text).toBe("alpine kangaroo doctrine");
      // embeddingModel goes with the embedding: there is no vector to attribute.
      expect(row.embeddingModel).toBeNull();
      const raw = await admin.queryOne<{ m: string | null }>(
        `select embedding_model as m from memories where id = $1`,
        [row.id],
      );
      expect(raw?.m).toBeNull();
    });

    it("DEFECT: search with a queryEmbedding falls back to FTS instead of failing", async () => {
      const hits = await store.search({ scope: s, query: "kangaroo", queryEmbedding: V_A });
      expect(hits.map((h) => h.id)).toEqual([row.id]);
      expect(hits[0]!.voices.fts).toBe(1);
      expect(hits[0]!.voices.vector).toBeUndefined();
    });

    it("a queryEmbedding with a blank query degrades to the newest-first listing", async () => {
      // Both voices unavailable: still rows, not an exception and not [].
      const hits = await store.search({ scope: s, query: "", queryEmbedding: V_A });
      expect(hits.map((h) => h.id)).toEqual([row.id]);
      expect(hits[0]!.voices).toEqual({});
    });
  });
});
