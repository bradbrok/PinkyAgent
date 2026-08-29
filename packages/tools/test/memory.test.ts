import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import type { MemoryRow, RecallScope, ThreadEventData } from "@pinky/core";
import {
  MemoryEditTool,
  RecallTool,
  RetainTool,
  allowedVisibility,
  visibleInScope,
} from "../src/memory";
import {
  DEFAULT_SCOPE,
  makeCtx,
  makeFakeEmbedder,
  makeFakeMemory,
  makeMemoryHit,
  makeMemoryRow,
  makeTmpDir,
  type FakeMemoryOptions,
} from "./helpers";
import type { ToolContext } from "@pinky/runtime";

// No memory tool touches the filesystem; one shared scratch cwd is enough.
let dir = "";
let cleanupDir: () => void = () => {};
beforeAll(() => {
  const tmp = makeTmpDir();
  dir = tmp.dir;
  cleanupDir = tmp.cleanup;
});
afterAll(() => cleanupDir());

type MemoryEvent = Extract<ThreadEventData, { type: "memory" }>;

function memoryEvents(events: ThreadEventData[]): MemoryEvent[] {
  return events.filter((e): e is MemoryEvent => e.type === "memory");
}

function setup(opts: FakeMemoryOptions & { agentId?: string } = {}) {
  const fake = makeFakeMemory(opts);
  const events: ThreadEventData[] = [];
  const ctx: ToolContext = makeCtx(dir, {
    memory: fake.context,
    agentId: opts.agentId ?? "pinky",
  });
  ctx.emit = (data) => {
    events.push(data);
    return Promise.resolve();
  };
  return { fake, ctx, events };
}

/** A ToolContext with no memory plane wired in. */
function bareCtx(): ToolContext {
  return makeCtx(dir, { agentId: "pinky" });
}

const recall = new RecallTool();
const retain = new RetainTool();
const memoryEdit = new MemoryEditTool();

// ---------------------------------------------------------------------------
// Pure scope helpers
// ---------------------------------------------------------------------------

describe("visibleInScope", () => {
  const scope: RecallScope = { ...DEFAULT_SCOPE };

  test("tenant and global rows are visible from any scope", () => {
    for (const visibility of ["tenant", "global"] as const) {
      expect(visibleInScope(makeMemoryRow({ visibility }), scope)).toBe(true);
      expect(
        visibleInScope(makeMemoryRow({ visibility }), {
          agentId: "pinky",
          includeUser: false,
          includePrivate: false,
        }),
      ).toBe(true);
    }
  });

  test("channel rows need the same channel in scope", () => {
    const row = makeMemoryRow({ visibility: "channel", channelId: "c1" });
    expect(visibleInScope(row, scope)).toBe(true);
    expect(visibleInScope(row, { ...scope, channelId: "other" })).toBe(false);
    const { channelId: _drop, ...noChannel } = scope;
    expect(visibleInScope(row, noChannel)).toBe(false);
  });

  test("user rows need includeUser and a matching userId", () => {
    const row = makeMemoryRow({ visibility: "user", userId: "u1", channelId: null });
    expect(visibleInScope(row, scope)).toBe(false); // includeUser false
    expect(visibleInScope(row, { ...scope, includeUser: true })).toBe(true);
    expect(visibleInScope(row, { ...scope, includeUser: true, userId: "u2" })).toBe(false);
    const { userId: _drop, ...noUser } = scope;
    expect(visibleInScope(row, { ...noUser, includeUser: true })).toBe(false);
  });

  test("private rows need includePrivate", () => {
    const row = makeMemoryRow({ visibility: "private" });
    expect(visibleInScope(row, scope)).toBe(false);
    expect(visibleInScope(row, { ...scope, includePrivate: true })).toBe(true);
  });

  test("another agent's row is never visible", () => {
    const row = makeMemoryRow({ visibility: "global", agentId: "other-agent" });
    expect(visibleInScope(row, { ...scope, includePrivate: true, includeUser: true })).toBe(
      false,
    );
  });

  test("an invalidated row is never visible (valid_to is current truth, DESIGN 5.2)", () => {
    const row = makeMemoryRow({ visibility: "tenant", validTo: "2026-08-20T00:00:00.000Z" });
    expect(visibleInScope(row, scope)).toBe(false);
  });
});

describe("allowedVisibility", () => {
  const scope: RecallScope = { ...DEFAULT_SCOPE };

  test("tenant and global are always writable", () => {
    expect(allowedVisibility("tenant", scope)).toBeNull();
    expect(allowedVisibility("global", scope)).toBeNull();
  });

  test("channel needs a channel in scope", () => {
    expect(allowedVisibility("channel", scope)).toBeNull();
    const { channelId: _drop, ...noChannel } = scope;
    expect(allowedVisibility("channel", noChannel)).toContain("needs a channel");
  });

  test("user needs includeUser and a subject user", () => {
    expect(allowedVisibility("user", scope)).toContain("only writable from a DM");
    expect(allowedVisibility("user", { ...scope, includeUser: true })).toBeNull();
    const { userId: _drop, ...noUser } = scope;
    expect(allowedVisibility("user", { ...noUser, includeUser: true })).toContain(
      "needs a subject user",
    );
  });

  test("private needs includePrivate", () => {
    expect(allowedVisibility("private", scope)).toContain("trusted surface");
    expect(allowedVisibility("private", { ...scope, includePrivate: true })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// recall
// ---------------------------------------------------------------------------

describe("recall", () => {
  test("absent memory context is a clean error", async () => {
    const res = await recall.execute({ query: "anything" }, bareCtx());
    expect(res.isError).toBe(true);
    expect(res.text).toContain("memory plane not enabled");
  });

  test("non-string query is an error", async () => {
    const { ctx } = setup();
    const res = await recall.execute({ query: 42 }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'query'");
  });

  test("limit outside 1..50 is an error", async () => {
    const { ctx } = setup();
    for (const limit of [0, 51, 2.5]) {
      const res = await recall.execute({ query: "q", limit }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("'limit'");
    }
  });

  test("an unknown kind is an error", async () => {
    const { ctx } = setup();
    const res = await recall.execute({ query: "q", kinds: ["semantic", "vibes"] }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'kinds'");
  });

  test("renders one line per hit: id prefix, kind, importance, date, visibility, text", async () => {
    const hit = makeMemoryHit({
      kind: "episodic",
      importance: 7,
      visibility: "tenant",
      recordedAt: "2026-08-20T11:22:33.000Z",
      text: "deploy failed on a missing env var",
    });
    const { ctx } = setup({ hits: [hit] });
    const res = await recall.execute({ query: "deploy" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(res.text).toBe(
      `- [${hit.id.slice(0, 8)}] (episodic, importance 7, 2026-08-20, tenant) deploy failed on a missing env var`,
    );
  });

  test("passes the scope, limit and kinds through to the store", async () => {
    const { fake, ctx } = setup();
    await recall.execute({ query: "  terse answers  ", limit: 3, kinds: ["procedural"] }, ctx);
    expect(fake.searches.length).toBe(1);
    const search = fake.searches[0]!;
    expect(search.scope).toBe(fake.scope);
    expect(search.query).toBe("terse answers");
    expect(search.limit).toBe(3);
    expect(search.kinds).toEqual(["procedural"]);
  });

  test("limit defaults to 10", async () => {
    const { fake, ctx } = setup();
    await recall.execute({ query: "q" }, ctx);
    expect(fake.searches[0]!.limit).toBe(10);
    expect(fake.searches[0]!.kinds).toBeUndefined();
  });

  test("embeds the query for the vector voice when an embedder is present", async () => {
    const embedder = makeFakeEmbedder({ dimensions: 4 });
    const { fake, ctx } = setup({ embedder });
    await recall.execute({ query: "brad prefers terse" }, ctx);
    expect(embedder.calls).toEqual([["brad prefers terse"]]);
    expect(fake.searches[0]!.queryEmbedding).toHaveLength(4);
  });

  test("no embedder means no vector voice and no note", async () => {
    const { fake, ctx } = setup({ hits: [makeMemoryHit()] });
    const res = await recall.execute({ query: "q" }, ctx);
    expect(fake.searches[0]!.queryEmbedding).toBeUndefined();
    expect(fake.supportsVectorsCalls).toBe(0);
    expect(res.text).not.toContain("vector search unavailable");
  });

  test("a store without the vector column skips the embedder entirely", async () => {
    const embedder = makeFakeEmbedder();
    const { fake, ctx } = setup({ embedder, supportsVectors: false });
    const res = await recall.execute({ query: "q" }, ctx);
    expect(embedder.calls).toEqual([]);
    expect(fake.searches[0]!.queryEmbedding).toBeUndefined();
    expect(res.text).not.toContain("vector search unavailable");
  });

  test("an embedder failure degrades to FTS-only and says so", async () => {
    const embedder = makeFakeEmbedder({ fail: "embeddings api 503" });
    const { fake, ctx } = setup({ embedder, hits: [makeMemoryHit()] });
    const res = await recall.execute({ query: "q" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.searches.length).toBe(1);
    expect(fake.searches[0]!.queryEmbedding).toBeUndefined();
    expect(res.text).toContain("vector search unavailable: embeddings api 503");
  });

  test("caps a runaway query instead of handing it to the FTS parser", async () => {
    // A 200k-character query is not hypothetical: a model pasting a file into
    // `recall` reaches websearch_to_tsquery, which answers `stack depth limit
    // exceeded` — a Postgres error where a search result belongs.
    const { fake, ctx, events } = setup({ hits: [makeMemoryHit({ visibility: "tenant" })] });
    const res = await recall.execute({ query: `${"x".repeat(200_000)}  ` }, ctx);

    expect(res.isError).toBeUndefined();
    expect(fake.searches[0]!.query.length).toBe(1000);
    expect(memoryEvents(events)[0]!.text!.length).toBe(1000);
    expect(res.text).toContain("query truncated to 1000 characters, from 200000");
  });

  test("a query at the cap is passed through untouched and unannotated", async () => {
    const { fake, ctx } = setup();
    const res = await recall.execute({ query: "y".repeat(1000) }, ctx);
    expect(fake.searches[0]!.query.length).toBe(1000);
    expect(res.text).not.toContain("truncated");
  });

  test("an empty result reads 'no memories matched'", async () => {
    const { ctx } = setup({ hits: [] });
    const res = await recall.execute({ query: "q" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(res.text).toBe("no memories matched");
  });

  test("emits a memory recall event with the hit ids and count", async () => {
    const hits = [makeMemoryHit(), makeMemoryHit()];
    const { ctx, events } = setup({ hits });
    await recall.execute({ query: "what do I know" }, ctx);
    const emitted = memoryEvents(events);
    expect(emitted).toEqual([
      {
        type: "memory",
        op: "recall",
        ids: [hits[0]!.id, hits[1]!.id],
        text: "what do I know",
        count: 2,
      },
    ]);
  });
});

// ---------------------------------------------------------------------------
// retain
// ---------------------------------------------------------------------------

describe("retain", () => {
  test("absent memory context is a clean error", async () => {
    const res = await retain.execute({ text: "a fact" }, bareCtx());
    expect(res.isError).toBe(true);
    expect(res.text).toContain("memory plane not enabled");
  });

  test("empty or non-string text is an error", async () => {
    const { fake, ctx } = setup();
    for (const text of ["", "   ", 7, undefined]) {
      const res = await retain.execute({ text }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("'text'");
    }
    expect(fake.retains.length).toBe(0);
  });

  test("defaults: semantic kind, importance 5, channel visibility in a channel", async () => {
    const { fake, ctx } = setup();
    const res = await retain.execute({ text: "  brad prefers terse answers  " }, ctx);
    expect(res.isError).toBeUndefined();
    const input = fake.retains[0]!;
    expect(input.kind).toBe("semantic");
    expect(input.importance).toBe(5);
    expect(input.visibility).toBe("channel");
    expect(input.channelId).toBe("c1");
    expect(input.userId).toBeUndefined();
    expect(input.text).toBe("brad prefers terse answers");
    expect(input.agentId).toBe("pinky");
    expect(res.text).toMatch(/^retained .+ \(semantic, channel\)$/);
  });

  test("defaults to tenant visibility when the scope has no channel", async () => {
    const { fake, ctx } = setup({ scope: { channelId: undefined } });
    await retain.execute({ text: "a fact" }, ctx);
    expect(fake.retains[0]!.visibility).toBe("tenant");
    expect(fake.retains[0]!.channelId).toBeUndefined();
  });

  test("user visibility without includeUser is refused", async () => {
    const { fake, ctx } = setup();
    const res = await retain.execute({ text: "a fact", visibility: "user" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("only writable from a DM");
    expect(fake.retains.length).toBe(0);
  });

  test("user visibility from a DM scope stores the subject user", async () => {
    const { fake, ctx } = setup({ scope: { includeUser: true, userId: "u9" } });
    const res = await retain.execute({ text: "a fact", visibility: "user" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.retains[0]!.userId).toBe("u9");
    expect(fake.retains[0]!.channelId).toBeUndefined();
  });

  test("private visibility without includePrivate is refused", async () => {
    const { fake, ctx } = setup();
    const res = await retain.execute({ text: "a fact", visibility: "private" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("trusted surface");
    expect(fake.retains.length).toBe(0);
  });

  test("channel visibility without a channel in scope is refused", async () => {
    const { fake, ctx } = setup({ scope: { channelId: undefined } });
    const res = await retain.execute({ text: "a fact", visibility: "channel" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("needs a channel");
    expect(fake.retains.length).toBe(0);
  });

  test("tenant and global are writable from a plain channel scope", async () => {
    const { fake, ctx } = setup();
    for (const visibility of ["tenant", "global"] as const) {
      const res = await retain.execute({ text: "a fact", visibility }, ctx);
      expect(res.isError).toBeUndefined();
    }
    expect(fake.retains.map((r) => r.visibility)).toEqual(["tenant", "global"]);
  });

  test("bad enum or importance values are errors", async () => {
    const { fake, ctx } = setup();
    const bad = [
      [{ text: "x", kind: "gossip" }, "'kind'"],
      [{ text: "x", visibility: "everyone" }, "'visibility'"],
      [{ text: "x", importance: 0 }, "'importance'"],
      [{ text: "x", importance: 11 }, "'importance'"],
      [{ text: "x", importance: 4.5 }, "'importance'"],
    ] as const;
    for (const [args, needle] of bad) {
      const res = await retain.execute({ ...args }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain(needle);
    }
    expect(fake.retains.length).toBe(0);
  });

  test("attaches the embedding and the embedding model", async () => {
    const embedder = makeFakeEmbedder({ model: "openai/text-embedding-3-small", dimensions: 4 });
    const { fake, ctx } = setup({ embedder });
    const res = await retain.execute({ text: "a fact worth keeping" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(embedder.calls).toEqual([["a fact worth keeping"]]);
    expect(fake.retains[0]!.embedding).toHaveLength(4);
    expect(fake.retains[0]!.embeddingModel).toBe("openai/text-embedding-3-small");
    expect(res.text).not.toContain("without embedding");
  });

  test("an embedder failure still retains the memory, and says so", async () => {
    const embedder = makeFakeEmbedder({ fail: "embeddings api 500" });
    const { fake, ctx, events } = setup({ embedder });
    const res = await retain.execute({ text: "must not be lost" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.retains.length).toBe(1);
    expect(fake.retains[0]!.embedding).toBeUndefined();
    expect(fake.retains[0]!.embeddingModel).toBeUndefined();
    expect(res.text).toContain("(stored without embedding: embeddings api 500)");
    expect(memoryEvents(events).length).toBe(1);
  });

  test("records provenance in meta.source and meta.retainedBy", async () => {
    const { fake, ctx } = setup({ agentId: "pinky" });
    await retain.execute({ text: "a fact" }, ctx);
    expect(fake.retains[0]!.meta).toEqual({
      source: { channelId: "c1", threadId: "thread-test" },
      retainedBy: "pinky",
    });
  });

  test("emits a memory retain event carrying the new id and text", async () => {
    const { fake, ctx, events } = setup();
    await retain.execute({ text: "a fact" }, ctx);
    const id = [...fake.rows.keys()][0]!;
    expect(memoryEvents(events)).toEqual([
      { type: "memory", op: "retain", ids: [id], text: "a fact" },
    ]);
  });
});

// ---------------------------------------------------------------------------
// memory_edit
// ---------------------------------------------------------------------------

describe("memory_edit", () => {
  function withRow(over: Partial<MemoryRow> = {}, opts: FakeMemoryOptions = {}) {
    const row = makeMemoryRow(over);
    return { row, ...setup({ ...opts, rows: [row, ...(opts.rows ?? [])] }) };
  }

  test("absent memory context is a clean error", async () => {
    const res = await memoryEdit.execute({ op: "invalidate", id: "x" }, bareCtx());
    expect(res.isError).toBe(true);
    expect(res.text).toContain("memory plane not enabled");
  });

  test("an unknown or missing op is an error", async () => {
    const { ctx } = setup();
    for (const op of ["delete", undefined, 3]) {
      const res = await memoryEdit.execute({ op, id: "x" }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("'op'");
    }
  });

  test("a missing id is an error", async () => {
    const { ctx } = setup();
    const res = await memoryEdit.execute({ op: "invalidate", id: "  " }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'id'");
  });

  test("update without text is an error", async () => {
    const { row, ctx, fake } = withRow();
    const res = await memoryEdit.execute({ op: "update", id: row.id }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("'text' is required");
    expect(fake.updates.length).toBe(0);
  });

  test("an unknown id is an error", async () => {
    const { ctx } = setup();
    const res = await memoryEdit.execute(
      { op: "invalidate", id: "ffffffff-0000-4000-8000-000000000000" },
      ctx,
    );
    expect(res.isError).toBe(true);
    expect(res.text).toContain("no memory with id");
  });

  test("a row that is not visible from this scope is refused", async () => {
    const { row, ctx, fake } = withRow({ visibility: "private" });
    const res = await memoryEdit.execute({ op: "invalidate", id: row.id }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("not visible from this scope");
    expect(fake.invalidations.length).toBe(0);
  });

  test("another agent's row is refused even when the id is known", async () => {
    const { row, ctx } = withRow({ agentId: "other-agent", visibility: "global" });
    const res = await memoryEdit.execute({ op: "invalidate", id: row.id }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("not visible from this scope");
  });

  test("an already-invalidated row is refused", async () => {
    const { row, ctx, fake } = withRow({ validTo: "2026-08-10T00:00:00.000Z" });
    const res = await memoryEdit.execute({ op: "invalidate", id: row.id }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("already invalidated");
    expect(fake.invalidations.length).toBe(0);
  });

  test("update carries kind, visibility and importance from the old row and re-embeds", async () => {
    const embedder = makeFakeEmbedder({ model: "openai/text-embedding-3-small", dimensions: 4 });
    const { row, ctx, fake } = withRow(
      { kind: "procedural", visibility: "tenant", importance: 9, channelId: null },
      { embedder },
    );
    const res = await memoryEdit.execute(
      { op: "update", id: row.id, text: "  the corrected rule  " },
      ctx,
    );
    expect(res.isError).toBeUndefined();
    expect(fake.updates.length).toBe(1);
    const { id, replacement } = fake.updates[0]!;
    expect(id).toBe(row.id);
    expect(replacement).toMatchObject({
      agentId: "pinky",
      kind: "procedural",
      visibility: "tenant",
      importance: 9,
      text: "the corrected rule",
      embeddingModel: "openai/text-embedding-3-small",
      meta: { source: { channelId: "c1", threadId: "thread-test" }, retainedBy: "pinky" },
    });
    expect(replacement.embedding).toHaveLength(4);
    expect(embedder.calls).toEqual([["the corrected rule"]]);
  });

  test("update reports old -> new and emits both ids", async () => {
    const { row, ctx, fake, events } = withRow({ visibility: "tenant" });
    const res = await memoryEdit.execute(
      { op: "update", id: row.id, text: "corrected" },
      ctx,
    );
    const newId = [...fake.rows.values()].find((r) => r.text === "corrected")!.id;
    expect(res.text).toBe(`updated ${row.id} -> ${newId} (semantic, tenant)`);
    expect(memoryEvents(events)).toEqual([
      { type: "memory", op: "update", ids: [row.id, newId], text: "corrected" },
    ]);
    // The old row is superseded, not deleted (DESIGN 5.2).
    expect(fake.rows.get(row.id)!.validTo).not.toBeNull();
  });

  test("update still succeeds when the embedder fails, and says so", async () => {
    const embedder = makeFakeEmbedder({ fail: "embeddings api 429" });
    const { row, ctx, fake } = withRow({ visibility: "tenant" }, { embedder });
    const res = await memoryEdit.execute({ op: "update", id: row.id, text: "corrected" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.updates[0]!.replacement.embedding).toBeUndefined();
    expect(res.text).toContain("(stored without embedding: embeddings api 429)");
  });

  test("invalidate records the reason and emits it", async () => {
    const { row, ctx, fake, events } = withRow({ visibility: "tenant" });
    const res = await memoryEdit.execute(
      { op: "invalidate", id: row.id, reason: "superseded by the new policy" },
      ctx,
    );
    expect(res.text).toBe(`invalidated ${row.id}`);
    expect(fake.invalidations).toEqual([
      { id: row.id, reason: "superseded by the new policy" },
    ]);
    expect(memoryEvents(events)).toEqual([
      {
        type: "memory",
        op: "invalidate",
        ids: [row.id],
        text: "superseded by the new policy",
      },
    ]);
  });

  test("invalidate without a reason is allowed", async () => {
    const { row, ctx, fake } = withRow({ visibility: "tenant" });
    const res = await memoryEdit.execute({ op: "invalidate", id: row.id }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.invalidations).toEqual([{ id: row.id }]);
  });

  test("forget prefixes the reason with 'forget:' and never deletes", async () => {
    const { row, ctx, fake, events } = withRow({ visibility: "tenant" });
    const res = await memoryEdit.execute(
      { op: "forget", id: row.id, reason: "brad asked me to" },
      ctx,
    );
    expect(res.text).toBe(`forgot ${row.id} (invalidated, not deleted)`);
    expect(fake.invalidations).toEqual([{ id: row.id, reason: "forget: brad asked me to" }]);
    expect(memoryEvents(events)[0]!.text).toBe("forget: brad asked me to");
    expect(fake.rows.has(row.id)).toBe(true);
    expect(fake.rows.get(row.id)!.validTo).not.toBeNull();
  });

  test("forget without a reason still records that it was a forget", async () => {
    const { row, ctx, fake } = withRow({ visibility: "tenant" });
    await memoryEdit.execute({ op: "forget", id: row.id }, ctx);
    expect(fake.invalidations[0]!.reason).toBe("forget: agent requested");
  });

  test("scope is checked before currency, so an unreachable id never leaks its state", async () => {
    // `get` is by id and NOT scope-fenced, so ordering these two checks the
    // other way answers "already invalidated" to a caller who was never
    // entitled to know the row exists at all.
    const { row, ctx, fake } = withRow({
      agentId: "other-agent",
      visibility: "global",
      validTo: "2026-08-10T00:00:00.000Z",
    });
    const res = await memoryEdit.execute({ op: "invalidate", id: row.id }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("not visible from this scope");
    expect(res.text).not.toContain("invalidated");
    expect(fake.invalidations.length).toBe(0);
  });

  test("resolves the 8-char id prefix that recall prints", async () => {
    const { row, ctx, fake } = withRow({ visibility: "tenant" });
    const res = await memoryEdit.execute(
      { op: "invalidate", id: row.id.slice(0, 8) },
      ctx,
    );
    expect(res.isError).toBeUndefined();
    expect(fake.invalidations).toEqual([{ id: row.id }]);
    expect(fake.prefixes).toEqual([
      { prefix: row.id.slice(0, 8), scope: fake.scope, limit: 2 },
    ]);
    // The resolution is a prefix query, not a page of `list` filtered in TS.
    expect(fake.lists).toHaveLength(0);
  });

  test("resolves a prefix for a memory far past any listing page (DEFECT: P2.3)", async () => {
    // The old resolver scanned `list({ limit: 200 })`, which is newest-first.
    // `recall` ranks by relevance, so it happily printed the id of a memory
    // sitting 250 rows back — and `memory_edit` then refused its own tool's
    // output. The needle here is the OLDEST row, behind 250 fillers.
    const needle = makeMemoryRow({
      id: "deadbeef-0000-4000-8000-000000000000",
      visibility: "tenant",
      recordedAt: "2020-01-01T00:00:00.000Z",
      text: "the memory recall surfaced from further back",
    });
    const filler = Array.from({ length: 250 }, () => makeMemoryRow({ visibility: "tenant" }));
    const { ctx, fake } = setup({ rows: [...filler, needle] });

    const res = await memoryEdit.execute({ op: "invalidate", id: "deadbeef" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.invalidations).toEqual([{ id: needle.id }]);
    // Proof it was not a scan: the needle is not in the first 200 of `list`.
    const page = await fake.context.store.list({ scope: fake.scope, limit: 200 });
    expect(page.some((r) => r.id === needle.id)).toBe(false);
  });

  test("a prefix only matches rows this scope can see", async () => {
    const hidden = makeMemoryRow({
      id: "beefbeef-0000-4000-8000-000000000000",
      visibility: "private",
    });
    const { ctx, fake } = setup({ rows: [hidden] });
    const res = await memoryEdit.execute({ op: "invalidate", id: "beefbeef" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("no memory with id beefbeef");
    expect(fake.invalidations.length).toBe(0);
  });

  test("a malformed prefix comes back as a fixable message, not a thrown query", async () => {
    // The store concatenates the prefix into a LIKE pattern, so it refuses
    // anything outside `[0-9a-f-]{4,36}` rather than escaping. That refusal is
    // a throw; the tool has to turn it into text the model can act on.
    const { ctx, fake } = setup();
    for (const id of ["not-a-uuid!", "ab"]) {
      const res = await memoryEdit.execute({ op: "invalidate", id }, ctx);
      expect(res.isError).toBe(true);
      expect(res.text).toContain("id prefix must be 4..36 characters");
    }
    expect(fake.invalidations.length).toBe(0);
  });

  test("an uppercase prefix is normalised rather than refused", async () => {
    const row = makeMemoryRow({
      id: "abc12345-0000-4000-8000-000000000000",
      visibility: "tenant",
    });
    const { ctx, fake } = setup({ rows: [row] });
    const res = await memoryEdit.execute({ op: "invalidate", id: "ABC12345" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(fake.prefixes[0]!.prefix).toBe("abc12345");
    expect(fake.invalidations).toEqual([{ id: row.id }]);
  });

  test("an ambiguous prefix is refused rather than guessed", async () => {
    const a = makeMemoryRow({ id: "abcdef12-0000-4000-8000-000000000001", visibility: "tenant" });
    const b = makeMemoryRow({ id: "abcdef12-0000-4000-8000-000000000002", visibility: "tenant" });
    const { ctx, fake } = setup({ rows: [a, b] });
    const res = await memoryEdit.execute({ op: "invalidate", id: "abcdef12" }, ctx);
    expect(res.isError).toBe(true);
    expect(res.text).toContain("ambiguous");
    expect(fake.invalidations.length).toBe(0);
  });
});
