/**
 * Memory plane unit tests (DESIGN.md §5).
 *
 * Two halves, deliberately:
 *   - the SQL the store emits (a FakeDb records text + params, so the scope
 *     predicate, the pgvector text literal and the JSONB CONTRACT are asserted
 *     without a database), and
 *   - fuseHits() as pure math, which is where the §5.4 ranking actually lives.
 * The queries themselves are only really exercised by the integration suite.
 */
import { describe, expect, it } from "bun:test";
import {
  DEFAULT_WEIGHTS,
  MEMORY_VECTOR_DIMENSIONS,
  MemoryStore,
  fuseHits,
  scopePredicate,
  type MemoryRow,
  type RecallScope,
} from "../src/memory";
import type { Db } from "../src/db";

interface Call {
  sql: string;
  params: unknown[] | undefined;
  txDepth: number;
}

type Route = { pattern: RegExp; respond: (params?: unknown[]) => unknown[] };

class FakeDb implements Db {
  calls: Call[] = [];
  /** "begin"/"commit"/"rollback" for the OUTERMOST tx only, in order. */
  txLog: string[] = [];
  private routes: Route[];
  private txDepth = 0;

  constructor(routes: Route[]) {
    this.routes = routes;
  }

  async query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    this.calls.push({ sql, params, txDepth: this.txDepth });
    for (const r of this.routes) {
      if (r.pattern.test(sql)) return r.respond(params) as T[];
    }
    throw new Error(`FakeDb: no route for SQL: ${sql}`);
  }

  async queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const rows = await this.query<T>(sql, params);
    return rows[0] ?? null;
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    const outermost = this.txDepth === 0;
    if (outermost) this.txLog.push("begin");
    this.txDepth += 1;
    try {
      const out = await fn(this);
      if (outermost) this.txLog.push("commit");
      return out;
    } catch (err) {
      if (outermost) this.txLog.push("rollback");
      throw err;
    } finally {
      this.txDepth -= 1;
    }
  }

  async close(): Promise<void> {}

  find(pattern: RegExp): Call | undefined {
    return this.calls.find((c) => pattern.test(c.sql));
  }
  all(pattern: RegExp): Call[] {
    return this.calls.filter((c) => pattern.test(c.sql));
  }
}

// --- fixtures ---------------------------------------------------------------

const AT = "2026-08-01T00:00:00.000Z";

/** A raw row shaped like postgres.js would hand it back. */
function rawRow(id: string, over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id,
    tenant_id: "t1",
    agent_id: "a1",
    visibility: "tenant",
    user_id: null,
    channel_id: null,
    kind: "semantic",
    text: `text ${id}`,
    importance: 5,
    valid_from: new Date(AT),
    valid_to: null,
    recorded_at: new Date(AT),
    embedding_model: null,
    meta: {},
    ...over,
  };
}

function row(id: string, over: Partial<MemoryRow> = {}): MemoryRow {
  return {
    id,
    tenantId: "t1",
    agentId: "a1",
    visibility: "tenant",
    userId: null,
    channelId: null,
    kind: "semantic",
    text: `text ${id}`,
    importance: 5,
    validFrom: AT,
    validTo: null,
    recordedAt: AT,
    embeddingModel: null,
    meta: {},
    ...over,
  };
}

const channelScope: RecallScope = {
  agentId: "a1",
  channelId: "slack:C1",
  includeUser: false,
  includePrivate: false,
};

const dmScope: RecallScope = {
  agentId: "a1",
  channelId: "slack:D1",
  userId: "u1",
  includeUser: true,
  includePrivate: false,
};

const cliScope: RecallScope = {
  agentId: "a1",
  channelId: "cli:local",
  userId: "local",
  includeUser: true,
  includePrivate: true,
};

/** A vector of the width `memories.embedding` accepts, `head` first then zeros.
 *  Anything narrower is now DROPPED rather than sent, so tests that mean "a
 *  real embedding" have to be full width. */
function wide(head: number[]): number[] {
  const v = new Array<number>(MEMORY_VECTOR_DIMENSIONS).fill(0);
  head.forEach((n, i) => {
    v[i] = n;
  });
  return v;
}

/** Collects MemoryStore's onWarning output instead of writing to stderr. */
function warnings(): { onWarning: (m: string) => void; messages: string[] } {
  const messages: string[] = [];
  return { messages, onWarning: (m) => messages.push(m) };
}

const VECTOR_YES: Route = { pattern: /information_schema\.columns/, respond: () => [{ ok: 1 }] };
const VECTOR_NO: Route = { pattern: /information_schema\.columns/, respond: () => [] };
const INSERT: Route = {
  pattern: /insert into memories/,
  respond: (params) => [rawRow(String((params as unknown[])[0])) ],
};

// --- scopePredicate ---------------------------------------------------------

describe("scopePredicate", () => {
  it("shared channel sees tenant + channel + global, never user or private", () => {
    const params: unknown[] = [];
    const sql = scopePredicate(channelScope, params);
    expect(sql).toContain("agent_id = $1");
    expect(sql).toContain("valid_to is null");
    expect(sql).toContain("visibility = 'global'");
    expect(sql).toContain("visibility = 'tenant'");
    expect(sql).toContain("(visibility = 'channel' and channel_id = $2)");
    expect(sql).not.toContain("visibility = 'user'");
    expect(sql).not.toContain("visibility = 'private'");
    expect(params).toEqual(["a1", "slack:C1"]);
  });

  it("a DM additionally sees the subject user's rows", () => {
    const params: unknown[] = [];
    const sql = scopePredicate(dmScope, params);
    expect(sql).toContain("(visibility = 'user' and user_id = $3)");
    expect(sql).not.toContain("visibility = 'private'");
    expect(params).toEqual(["a1", "slack:D1", "u1"]);
  });

  it("a trusted local surface adds the agent's private rows", () => {
    const params: unknown[] = [];
    const sql = scopePredicate(cliScope, params);
    expect(sql).toContain("visibility = 'private'");
    expect(params).toEqual(["a1", "cli:local", "local"]);
  });

  it("omits the user arm when includeUser is set but no userId is known", () => {
    const params: unknown[] = [];
    const sql = scopePredicate({ agentId: "a1", includeUser: true, includePrivate: false }, params);
    expect(sql).not.toContain("visibility = 'user'");
    expect(sql).not.toContain("visibility = 'channel'");
    expect(params).toEqual(["a1"]);
  });

  it("numbers its placeholders from the params already bound", () => {
    const params: unknown[] = ["tenant"];
    const sql = scopePredicate(channelScope, params);
    expect(sql).toContain("agent_id = $2");
    expect(sql).toContain("channel_id = $3");
    expect(params).toEqual(["tenant", "a1", "slack:C1"]);
  });

  it("includeInvalid drops the current-truth conjunct", () => {
    expect(scopePredicate(channelScope, [], { includeInvalid: true })).not.toContain("valid_to");
  });

  it("refuses an empty agentId rather than matching every agent", () => {
    expect(() => scopePredicate({ ...channelScope, agentId: " " }, [])).toThrow(/agentId/);
  });
});

// --- retain -----------------------------------------------------------------

describe("MemoryStore.retain", () => {
  it("writes tenant_id, defaults importance to 5 and passes meta as a PLAIN object", async () => {
    const db = new FakeDb([VECTOR_NO, INSERT]);
    const store = new MemoryStore(db, "t1");
    const out = await store.retain({
      agentId: "a1",
      visibility: "tenant",
      kind: "semantic",
      text: "Brad prefers terse answers",
      meta: { source: { channelId: "slack:C1" } },
    });

    const insert = db.find(/insert into memories/)!;
    const params = insert.params as unknown[];
    expect(insert.sql).toContain("tenant_id");
    expect(params[1]).toBe("t1");
    expect(params[8]).toBe(5); // importance default
    // JSONB CONTRACT: the object itself, never JSON.stringify(...).
    expect(params[9]).toEqual({ source: { channelId: "slack:C1" } });
    expect(typeof params[9]).toBe("object");
    expect(out.id).toBe(String(params[0]));
    expect(out.recordedAt).toBe(AT);
  });

  it("stores the embedding as the pgvector TEXT literal with an explicit ::vector cast", async () => {
    const db = new FakeDb([VECTOR_YES, INSERT]);
    const store = new MemoryStore(db, "t1");
    await store.retain({
      agentId: "a1",
      visibility: "tenant",
      kind: "semantic",
      text: "hi",
      embedding: wide([0.1, -0.25, 3]),
      embeddingModel: "openai/text-embedding-3-small",
    });

    const insert = db.find(/insert into memories/)!;
    expect(insert.sql).toContain("::vector");
    expect(insert.sql).toContain("embedding_model");
    const params = insert.params as unknown[];
    const literal = params[10] as string;
    expect(literal.startsWith("[0.1,-0.25,3,0,0,")).toBe(true);
    expect(literal.split(",")).toHaveLength(MEMORY_VECTOR_DIMENSIONS);
    expect(params).toContain("openai/text-embedding-3-small");
  });

  it("drops a wrong-width embedding and stores the row anyway, with a warning", async () => {
    // memory.embeddingModel is a setting, so "text-embedding-3-large" (3072) is
    // a legal thing for a human to write. Sending it to vector(1536) throws
    // 22000 and the memory the agent was trying to keep is LOST. The row is
    // worth more than the vector: store the text, warn, and recall it by FTS.
    const db = new FakeDb([VECTOR_YES, INSERT]);
    const w = warnings();
    const store = new MemoryStore(db, "t1", { onWarning: w.onWarning });
    expect(store.vectorDimensions).toBe(1536);

    const out = await store.retain({
      agentId: "a1",
      visibility: "tenant",
      kind: "semantic",
      text: "hi",
      embedding: new Array<number>(3072).fill(0.01),
      embeddingModel: "openai/text-embedding-3-large",
    });

    const insert = db.find(/insert into memories/)!;
    expect(insert.sql).not.toContain("::vector");
    expect(insert.params).toHaveLength(10);
    // No embedding => no model to attribute it to, and the row still landed.
    expect(insert.sql.slice(0, insert.sql.indexOf(")"))).not.toContain("embedding_model");
    expect(out.id).toBeTruthy();
    expect(w.messages).toHaveLength(1);
    expect(w.messages[0]).toContain("3072");
    expect(w.messages[0]).toContain("vector(1536)");
  });

  it("drops a wrong-width embedding on update() too", async () => {
    const db = new FakeDb([
      VECTOR_YES,
      { pattern: /from memories where id = \$1[\s\S]*for update/, respond: () => [rawRow("m1")] },
      { pattern: /update memories set valid_to/, respond: () => [{ id: "m1" }] },
      INSERT,
    ]);
    const w = warnings();
    await new MemoryStore(db, "t1", { onWarning: w.onWarning }).update("m1", {
      visibility: "tenant",
      kind: "semantic",
      text: "revised",
      embedding: [0.1, 0.2],
      embeddingModel: "openai/text-embedding-3-large",
    });
    expect(db.find(/insert into memories/)!.sql).not.toContain("::vector");
    expect(w.messages[0]).toContain("dropping a 2-dimension embedding");
    expect(db.txLog).toEqual(["begin", "commit"]);
  });

  it("says nothing when there is no embedding column to mismatch", async () => {
    // On postgres:16-alpine there is no vector at all; every embedding is
    // dropped for that reason and a width warning would be pure noise.
    const db = new FakeDb([VECTOR_NO, INSERT]);
    const w = warnings();
    await new MemoryStore(db, "t1", { onWarning: w.onWarning }).retain({
      agentId: "a1",
      visibility: "tenant",
      kind: "semantic",
      text: "hi",
      embedding: new Array<number>(3072).fill(0.01),
    });
    expect(w.messages).toEqual([]);
  });

  it("drops the embedding (and its model) when the column does not exist", async () => {
    const db = new FakeDb([VECTOR_NO, INSERT]);
    const store = new MemoryStore(db, "t1");
    await store.retain({
      agentId: "a1",
      visibility: "tenant",
      kind: "semantic",
      text: "hi",
      embedding: [0.1, 0.2],
      embeddingModel: "openai/text-embedding-3-small",
    });
    const insert = db.find(/insert into memories/)!;
    expect(insert.sql).not.toContain("::vector");
    // Neither column is in the INSERT list (they are still in RETURNING).
    expect(insert.sql.slice(0, insert.sql.indexOf(")"))).not.toContain("embedding");
    expect(insert.params).toHaveLength(10);
  });

  it("rejects an out-of-range importance instead of clamping it", async () => {
    const db = new FakeDb([VECTOR_NO, INSERT]);
    const store = new MemoryStore(db, "t1");
    await expect(
      store.retain({ agentId: "a1", visibility: "tenant", kind: "semantic", text: "x", importance: 11 }),
    ).rejects.toThrow(/importance/);
    await expect(
      store.retain({ agentId: "a1", visibility: "tenant", kind: "semantic", text: "x", importance: 2.5 }),
    ).rejects.toThrow(/importance/);
  });

  it("refuses a scoped row with no subject id (it would be unrecallable)", async () => {
    const db = new FakeDb([VECTOR_NO, INSERT]);
    const store = new MemoryStore(db, "t1");
    await expect(
      store.retain({ agentId: "a1", visibility: "channel", kind: "semantic", text: "x" }),
    ).rejects.toThrow(/channelId/);
    await expect(
      store.retain({ agentId: "a1", visibility: "user", kind: "semantic", text: "x" }),
    ).rejects.toThrow(/userId/);
    await expect(
      store.retain({ agentId: "a1", visibility: "tenant", kind: "nope" as never, text: "x" }),
    ).rejects.toThrow(/kind/);
    await expect(
      store.retain({ agentId: "a1", visibility: "tenant", kind: "semantic", text: "   " }),
    ).rejects.toThrow(/text/);
  });

  it("requires a non-empty tenantId at construction", () => {
    expect(() => new MemoryStore(new FakeDb([]), "")).toThrow(/tenantId/);
  });
});

// --- get / mapRow -----------------------------------------------------------

describe("MemoryStore.get", () => {
  it("maps Date and string timestamps alike to ISO, and jsonb meta to an object", async () => {
    const db = new FakeDb([
      {
        pattern: /from memories where id = \$1/,
        respond: () => [
          rawRow("m1", {
            valid_from: "2026-08-01 00:00:00+00",
            recorded_at: new Date(AT),
            valid_to: null,
            importance: "7",
            meta: { supersedes: "m0" },
          }),
        ],
      },
    ]);
    const out = await (new MemoryStore(db, "t1")).get("m1");
    expect(out).not.toBeNull();
    expect(out!.validFrom).toBe(AT);
    expect(out!.recordedAt).toBe(AT);
    expect(out!.validTo).toBeNull();
    expect(out!.importance).toBe(7);
    expect(out!.meta).toEqual({ supersedes: "m0" });
    expect(db.find(/from memories where id = \$1/)!.params).toEqual(["m1", "t1"]);
  });

  it("returns null for an unknown id", async () => {
    const db = new FakeDb([{ pattern: /from memories where id = \$1/, respond: () => [] }]);
    expect(await (new MemoryStore(db, "t1")).get("nope")).toBeNull();
  });
});

// --- invalidate -------------------------------------------------------------

describe("MemoryStore.invalidate", () => {
  it("stamps valid_to (never deletes), guarded on the row still being current", async () => {
    const db = new FakeDb([
      { pattern: /update memories set valid_to/, respond: () => [{ id: "m1" }] },
    ]);
    expect(await (new MemoryStore(db, "t1")).invalidate("m1")).toBe(true);
    const call = db.find(/update memories set valid_to/)!;
    expect(call.sql).toContain("valid_to is null");
    expect(call.sql).not.toMatch(/delete/i);
    expect(call.params).toEqual(["m1", "t1"]);
  });

  it("merges the reason into meta as a plain jsonb object", async () => {
    const db = new FakeDb([
      { pattern: /update memories set valid_to/, respond: () => [{ id: "m1" }] },
    ]);
    await (new MemoryStore(db, "t1")).invalidate("m1", { reason: "forget: superseded" });
    const call = db.find(/update memories set valid_to/)!;
    expect(call.sql).toContain("meta = meta || $3::jsonb");
    expect((call.params as unknown[])[2]).toEqual({ invalidatedReason: "forget: superseded" });
  });

  it("returns false when nothing was updated (unknown id, or already invalid)", async () => {
    const db = new FakeDb([{ pattern: /update memories set valid_to/, respond: () => [] }]);
    expect(await (new MemoryStore(db, "t1")).invalidate("m1")).toBe(false);
  });
});

// --- update -----------------------------------------------------------------

describe("MemoryStore.update", () => {
  const routes = (existing: Record<string, unknown>[]): Route[] => [
    VECTOR_NO,
    { pattern: /from memories where id = \$1[\s\S]*for update/, respond: () => existing },
    { pattern: /update memories set valid_to/, respond: () => [{ id: "m1" }] },
    INSERT,
  ];

  it("invalidates and re-inserts in ONE transaction, recording meta.supersedes", async () => {
    const db = new FakeDb(routes([rawRow("m1", { importance: 7 })]));
    const out = await (new MemoryStore(db, "t1")).update("m1", {
      visibility: "tenant",
      kind: "semantic",
      text: "Brad prefers terse answers, with code",
      importance: 7,
    });

    expect(db.txLog).toEqual(["begin", "commit"]);
    // Every statement of the swap ran inside that transaction.
    for (const c of db.all(/memories/)) {
      if (/information_schema/.test(c.sql)) continue;
      expect(c.txDepth).toBeGreaterThan(0);
    }
    const insert = db.find(/insert into memories/)!;
    expect((insert.params as unknown[])[9]).toEqual({ supersedes: "m1" });
    // agentId carried from the old row when the caller does not restate it.
    expect((insert.params as unknown[])[2]).toBe("a1");
    expect(out.id).not.toBe("m1");
  });

  it("throws (writing nothing) when the id is unknown", async () => {
    const db = new FakeDb(routes([]));
    await expect(
      (new MemoryStore(db, "t1")).update("m1", { visibility: "tenant", kind: "semantic", text: "x" }),
    ).rejects.toThrow(/not found/);
    expect(db.txLog).toEqual(["begin", "rollback"]);
    expect(db.find(/insert into memories/)).toBeUndefined();
  });

  it("throws when the row was already invalidated", async () => {
    const db = new FakeDb(routes([rawRow("m1", { valid_to: new Date(AT) })]));
    await expect(
      (new MemoryStore(db, "t1")).update("m1", { visibility: "tenant", kind: "semantic", text: "x" }),
    ).rejects.toThrow(/already invalidated/);
    expect(db.find(/insert into memories/)).toBeUndefined();
  });
});

// --- supportsVectors --------------------------------------------------------

describe("MemoryStore.supportsVectors", () => {
  it("probes information_schema exactly once and caches the answer", async () => {
    const db = new FakeDb([VECTOR_YES, INSERT]);
    const store = new MemoryStore(db, "t1");
    const [a, b] = await Promise.all([store.supportsVectors(), store.supportsVectors()]);
    expect(a).toBe(true);
    expect(b).toBe(true);
    expect(await store.supportsVectors()).toBe(true);
    expect(db.all(/information_schema/)).toHaveLength(1);
  });

  it("reports false when the column is absent (no pgvector on this image)", async () => {
    const db = new FakeDb([VECTOR_NO]);
    expect(await (new MemoryStore(db, "t1")).supportsVectors()).toBe(false);
  });

  it("asks about OUR memories table, not another schema's", async () => {
    // information_schema lists every schema the role can see. Without
    // current_schema() a stale `public.memories` (or a per-tenant sandbox copy)
    // answers for the one the search_path actually resolves to, and the vector
    // voice is enabled — or disabled — on the strength of the wrong table.
    const db = new FakeDb([VECTOR_YES]);
    await (new MemoryStore(db, "t1")).supportsVectors();
    const probe = db.find(/information_schema\.columns/)!;
    expect(probe.sql).toContain("table_schema = current_schema()");
    expect(probe.sql).toContain("table_name = 'memories'");
    expect(probe.sql).toContain("column_name = 'embedding'");
  });
});

// --- findByIdPrefix ---------------------------------------------------------

describe("MemoryStore.findByIdPrefix", () => {
  const PREFIX_ROUTE: Route = {
    pattern: /id like \$\d+ \|\| '%'/,
    respond: () => [rawRow("3f2a1111-0000-0000-0000-000000000000")],
  };

  it("resolves the prefix in SQL, fenced by the scope predicate", async () => {
    // The alternative — list(200) and filter in JS — cannot see an old row,
    // which is exactly the row somebody names by short id.
    const db = new FakeDb([PREFIX_ROUTE]);
    const rows = await (new MemoryStore(db, "t1")).findByIdPrefix("3f2a", { scope: channelScope });
    const call = db.find(/id like/)!;
    expect(call.sql).toContain("tenant_id = $1");
    expect(call.sql).toContain("valid_to is null");
    expect(call.sql).toContain("channel_id = $3");
    expect(call.sql).toContain("id like $4 || '%'");
    expect(call.sql).toContain("order by recorded_at desc");
    // The prefix is a PARAM, never interpolated, and the default limit is 2 so
    // one query answers "unique or ambiguous?".
    expect(call.params).toEqual(["t1", "a1", "slack:C1", "3f2a", 2]);
    expect(rows.map((r) => r.id)).toEqual(["3f2a1111-0000-0000-0000-000000000000"]);
  });

  it("honours limit and includeInvalid", async () => {
    const db = new FakeDb([PREFIX_ROUTE]);
    await (new MemoryStore(db, "t1")).findByIdPrefix("3f2a", {
      scope: channelScope,
      includeInvalid: true,
      limit: 5,
    });
    const call = db.find(/id like/)!;
    expect(call.sql).not.toContain("valid_to is null");
    expect(call.params).toEqual(["t1", "a1", "slack:C1", "3f2a", 5]);
  });

  it("refuses anything LIKE could read as a wildcard", async () => {
    // `%` and `_` are LIKE metacharacters. Nothing is escaped downstream — the
    // character class IS the guard — so "%" must never reach the query, or
    // `id like '%' || '%'` quietly returns the whole plane and a caller
    // checking for a unique match acts on an arbitrary row.
    const db = new FakeDb([PREFIX_ROUTE]);
    const store = new MemoryStore(db, "t1");
    for (const bad of ["%", "3f2%", "3f2_", "3f2a'", "3F2A", "abc", "", "x".repeat(37), "3f2a\\"]) {
      await expect(store.findByIdPrefix(bad, { scope: channelScope })).rejects.toThrow(
        /id prefix must be/,
      );
    }
    expect(db.calls).toHaveLength(0);
    // A full uuid is still a legal prefix of itself.
    await store.findByIdPrefix("3f2a1111-0000-0000-0000-000000000000", { scope: channelScope });
    expect(db.calls).toHaveLength(1);
  });
});

// --- search -----------------------------------------------------------------

const VECTOR_VOICE: Route = {
  pattern: /order by embedding <=>/,
  respond: () => [rawRow("v1"), rawRow("shared")],
};
const FTS_VOICE: Route = {
  pattern: /tsv @@ websearch_to_tsquery/,
  respond: () => [rawRow("shared"), rawRow("f1")],
};
const LIST_VOICE: Route = {
  pattern: /order by recorded_at desc, id desc/,
  respond: () => [rawRow("newest"), rawRow("older")],
};

describe("MemoryStore.search", () => {
  it("runs both voices over the same scope and fuses them", async () => {
    const db = new FakeDb([VECTOR_YES, VECTOR_VOICE, FTS_VOICE]);
    const hits = await (new MemoryStore(db, "t1")).search({
      scope: channelScope,
      query: "terse answers",
      queryEmbedding: [0.5, 0.5],
      limit: 10,
      now: new Date(AT),
    });

    const vec = db.find(/order by embedding <=>/)!;
    expect(vec.sql).toContain("embedding is not null");
    expect(vec.sql).toContain("::vector");
    expect(vec.params).toContain("[0.5,0.5]");
    // K = 3 x limit, capped at 60.
    expect(vec.params).toContain(30);

    const fts = db.find(/tsv @@ websearch_to_tsquery/)!;
    expect(fts.sql).toContain("ts_rank_cd");
    expect(fts.params).toContain("terse answers");
    for (const call of [vec, fts]) {
      expect(call.sql).toContain("tenant_id = $1");
      expect(call.sql).toContain("valid_to is null");
      expect(call.sql).toContain("channel_id = $3");
    }

    // "shared" was ranked by both voices, so it leads.
    expect(hits[0]!.id).toBe("shared");
    expect(hits[0]!.voices).toEqual({ vector: 2, fts: 1 });
    expect(hits.map((h) => h.id).sort()).toEqual(["f1", "shared", "v1"]);
  });

  it("caps per-voice candidates at 60 however large the limit", async () => {
    const db = new FakeDb([VECTOR_YES, FTS_VOICE]);
    await (new MemoryStore(db, "t1")).search({ scope: channelScope, query: "x", limit: 50 });
    expect(db.find(/tsv @@/)!.params).toContain(60);
  });

  it("skips the vector voice when no embedding is supplied", async () => {
    const db = new FakeDb([VECTOR_YES, FTS_VOICE]);
    const hits = await (new MemoryStore(db, "t1")).search({ scope: channelScope, query: "hello" });
    expect(db.find(/order by embedding <=>/)).toBeUndefined();
    expect(hits[0]!.voices.vector).toBeUndefined();
    expect(hits[0]!.voices.fts).toBe(1);
  });

  it("skips the vector voice when the database has no embedding column", async () => {
    const db = new FakeDb([VECTOR_NO, FTS_VOICE]);
    await (new MemoryStore(db, "t1")).search({
      scope: channelScope,
      query: "hello",
      queryEmbedding: [0.1, 0.2],
    });
    expect(db.find(/order by embedding <=>/)).toBeUndefined();
    expect(db.find(/tsv @@/)).toBeDefined();
  });

  it("skips the FTS voice when the query is blank", async () => {
    const db = new FakeDb([VECTOR_YES, VECTOR_VOICE]);
    await (new MemoryStore(db, "t1")).search({
      scope: channelScope,
      query: "   ",
      queryEmbedding: [0.1, 0.2],
    });
    expect(db.find(/tsv @@/)).toBeUndefined();
    expect(db.find(/order by embedding <=>/)).toBeDefined();
  });

  it("falls back to a newest-first listing when neither voice is available", async () => {
    const db = new FakeDb([VECTOR_NO, LIST_VOICE]);
    const hits = await (new MemoryStore(db, "t1")).search({ scope: channelScope, query: "" });
    expect(db.find(/order by recorded_at desc, id desc/)).toBeDefined();
    expect(hits.map((h) => h.id)).toEqual(["newest", "older"]);
    expect(hits[0]!.voices).toEqual({});
    expect(hits[0]!.score).toBeGreaterThan(0);
  });

  it("passes a kinds filter to every voice", async () => {
    const db = new FakeDb([VECTOR_YES, VECTOR_VOICE, FTS_VOICE]);
    await (new MemoryStore(db, "t1")).search({
      scope: channelScope,
      query: "x",
      queryEmbedding: [1],
      kinds: ["procedural", "episodic"],
    });
    for (const call of [db.find(/order by embedding <=>/)!, db.find(/tsv @@/)!]) {
      expect(call.sql).toContain("kind = any(");
      expect(call.params).toContainEqual(["procedural", "episodic"]);
    }
  });
});

// --- list -------------------------------------------------------------------

describe("MemoryStore.list", () => {
  it("is newest-first, current-truth-only by default", async () => {
    const db = new FakeDb([LIST_VOICE]);
    const rows = await (new MemoryStore(db, "t1")).list({ scope: cliScope, limit: 5 });
    const call = db.find(/order by recorded_at desc, id desc/)!;
    expect(call.sql).toContain("valid_to is null");
    expect(call.sql).toContain("visibility = 'private'");
    expect(call.params).toEqual(["t1", "a1", "cli:local", "local", 5]);
    expect(rows.map((r) => r.id)).toEqual(["newest", "older"]);
  });

  it("includeInvalid returns history too", async () => {
    const db = new FakeDb([LIST_VOICE]);
    await (new MemoryStore(db, "t1")).list({ scope: channelScope, includeInvalid: true });
    expect(db.find(/order by recorded_at desc, id desc/)!.sql).not.toContain("valid_to is null");
  });
});

// --- fuseHits (pure) --------------------------------------------------------

const HOUR = 3_600_000;
const now = new Date("2026-08-02T00:00:00.000Z");
/** Same age and importance for every row, so one term at a time is under test. */
const flat = (id: string, over: Partial<MemoryRow> = {}) =>
  row(id, { recordedAt: now.toISOString(), ...over });

describe("fuseHits", () => {
  it("returns nothing when no voice ranked anything", () => {
    expect(fuseHits({}, { now })).toEqual([]);
    expect(fuseHits({ vector: [], fts: [] }, { now })).toEqual([]);
  });

  it("keeps a single voice's order and records the 1-based rank", () => {
    const hits = fuseHits({ fts: [flat("a"), flat("b"), flat("c")] }, { now });
    expect(hits.map((h) => h.id)).toEqual(["a", "b", "c"]);
    expect(hits.map((h) => h.voices)).toEqual([{ fts: 1 }, { fts: 2 }, { fts: 3 }]);
    expect(hits[0]!.score).toBeGreaterThan(hits[2]!.score);
  });

  it("ranks a hit found by BOTH voices above one found by a single voice", () => {
    // x is the vector voice's best; y is only second there, but the FTS voice
    // also ranked it, so its summed reciprocal ranks win.
    const hits = fuseHits({ vector: [flat("x"), flat("y")], fts: [flat("y")] }, { now });
    expect(hits.map((h) => h.id)).toEqual(["y", "x"]);
    expect(hits[0]!.voices).toEqual({ vector: 2, fts: 1 });
    expect(hits[1]!.voices).toEqual({ vector: 1 });
  });

  it("normalizes relevance to the best candidate, so the leader scores its full weight", () => {
    const hits = fuseHits({ fts: [flat("a", { importance: 10 })] }, {
      now,
      weights: { relevance: 1, recency: 0, importance: 0 },
    });
    expect(hits[0]!.score).toBeCloseTo(1, 10);
  });

  it("weights: importance can overturn the relevance order", () => {
    const voices = { fts: [flat("low", { importance: 1 }), flat("high", { importance: 10 })] };
    expect(
      fuseHits(voices, { now, weights: { relevance: 1, recency: 0, importance: 0 } }).map((h) => h.id),
    ).toEqual(["low", "high"]);
    expect(
      fuseHits(voices, { now, weights: { relevance: 0, recency: 0, importance: 1 } }).map((h) => h.id),
    ).toEqual(["high", "low"]);
    // Worth knowing about the §5.4 formula: normalized RRF separates ADJACENT
    // ranks by ~1.6% (1/61 vs 1/62), which the importance term (up to 0.2 of
    // the score) easily outweighs. Near the top of the list, importance and
    // recency decide; the voices decide who is in the candidate set at all.
    expect(fuseHits(voices, { now }).map((h) => h.id)).toEqual(["high", "low"]);
  });

  it("recency decays 0.995 per hour and can overturn the relevance order", () => {
    const old = row("old", { recordedAt: new Date(now.getTime() - 500 * HOUR).toISOString() });
    const fresh = row("fresh", { recordedAt: now.toISOString() });
    const hits = fuseHits({ fts: [old, fresh] }, {
      now,
      weights: { relevance: 0, recency: 1, importance: 0 },
    });
    expect(hits.map((h) => h.id)).toEqual(["fresh", "old"]);
    expect(hits[0]!.score).toBeCloseTo(1, 10);
    expect(hits[1]!.score).toBeCloseTo(Math.pow(0.995, 500), 10);
  });

  it("a future timestamp gets no more than a fresh one (clock skew is not a boost)", () => {
    const future = row("f", { recordedAt: new Date(now.getTime() + 100 * HOUR).toISOString() });
    const hits = fuseHits({ fts: [future] }, {
      now,
      weights: { relevance: 0, recency: 1, importance: 0 },
    });
    expect(hits[0]!.score).toBeCloseTo(1, 10);
  });

  it("breaks exact score ties newest-first", () => {
    // Rank 1 in one voice each => identical rrf; relevance-only weights => a tie.
    const older = row("older", { recordedAt: new Date(now.getTime() - HOUR).toISOString() });
    const newer = row("newer", { recordedAt: now.toISOString() });
    const hits = fuseHits({ vector: [older], fts: [newer] }, {
      now,
      weights: { relevance: 1, recency: 0, importance: 0 },
    });
    expect(hits[0]!.score).toBeCloseTo(hits[1]!.score, 12);
    expect(hits.map((h) => h.id)).toEqual(["newer", "older"]);
  });

  it("cuts to the limit after scoring, not before", () => {
    const hits = fuseHits({ fts: [flat("a"), flat("b"), flat("c")] }, { now, limit: 2 });
    expect(hits.map((h) => h.id)).toEqual(["a", "b"]);
  });

  it("defaults match DESIGN §5.4 (relevance 0.6, recency 0.2, importance 0.2)", () => {
    expect(DEFAULT_WEIGHTS).toEqual({ relevance: 0.6, recency: 0.2, importance: 0.2 });
    // Best candidate, brand new, importance 10 => every term at full weight.
    const hits = fuseHits({ fts: [flat("a", { importance: 10 })] }, { now });
    expect(hits[0]!.score).toBeCloseTo(1, 10);
  });
});
