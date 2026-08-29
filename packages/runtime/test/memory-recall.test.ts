/**
 * Auto-recall unit tests (DESIGN.md §5.4): query construction from the window,
 * the token-budgeted `<memories>` block, and the degradation ladder
 * (embedder down -> FTS-only; store down -> nothing injected, never a throw).
 *
 * The MemoryStore is a hand-rolled stub cast through `as unknown as
 * MemoryStore`: these tests are about what the loop asks for and what it does
 * with the answer, not about SQL — the store's own SQL is covered by
 * packages/core.
 */
import { describe, expect, test } from "bun:test";
import { estimateTokens } from "@pinky/core";
import type {
  MemoryHit,
  MemoryStore,
  RecallScope,
  SearchInput,
  ThreadEvent,
  ThreadEventData,
} from "@pinky/core";
import { MEMORIES_HEADER, autoRecall, recallQueryFor, renderMemoriesBlock } from "../src/memory-recall";
import type { Embedder, LlmMessage, MemoryContext } from "../src/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SCOPE: RecallScope = {
  agentId: "pinky",
  channelId: "c1",
  includeUser: false,
  includePrivate: false,
};

let nextId = 0;

function hit(partial: Partial<MemoryHit> = {}): MemoryHit {
  nextId += 1;
  return {
    id: `m${nextId}`,
    tenantId: "t1",
    agentId: "pinky",
    visibility: "channel",
    userId: null,
    channelId: "c1",
    kind: "semantic",
    text: `memory ${nextId}`,
    importance: 5,
    validFrom: "2026-08-01T00:00:00.000Z",
    validTo: null,
    recordedAt: "2026-08-01T09:30:00.000Z",
    embeddingModel: null,
    meta: {},
    score: 0.5,
    voices: { fts: 1 },
    ...partial,
  };
}

function event(data: ThreadEventData, seq = 1): ThreadEvent {
  return {
    id: `e${seq}`,
    tenantId: "t1",
    channelId: "c1",
    threadId: "th1",
    seq,
    ts: "2026-08-01T09:00:00.000Z",
    data,
  };
}

function continuityEvent(memoryHints: string[], seq = 1): ThreadEvent {
  return event(
    {
      type: "continuity",
      document: {
        goal: "ship the memory plane",
        plan: { done: [], now: "wire recall", next: [] },
        workingSet: {},
        decisions: [],
        openLoops: [],
        lessons: [],
        memoryHints,
      },
      tokensBefore: 1000,
    },
    seq,
  );
}

const user = (text: string): LlmMessage => ({ role: "user", text });

interface StubOptions {
  hits?: MemoryHit[];
  vectors?: boolean;
  searchError?: Error;
  vectorProbeError?: Error;
}

class StubStore {
  readonly searches: SearchInput[] = [];
  probes = 0;
  constructor(private readonly opts: StubOptions = {}) {}

  async supportsVectors(): Promise<boolean> {
    this.probes += 1;
    if (this.opts.vectorProbeError) throw this.opts.vectorProbeError;
    return this.opts.vectors ?? false;
  }

  async search(input: SearchInput): Promise<MemoryHit[]> {
    this.searches.push(input);
    if (this.opts.searchError) throw this.opts.searchError;
    return this.opts.hits ?? [];
  }
}

class StubEmbedder implements Embedder {
  readonly model = "fake/embed";
  readonly dimensions = 4;
  readonly calls: string[][] = [];
  constructor(private readonly fail?: Error) {}

  async embed(texts: string[]): Promise<number[][]> {
    this.calls.push(texts);
    if (this.fail) throw this.fail;
    return texts.map(() => [0.1, 0.2, 0.3, 0.4]);
  }
}

function context(store: StubStore, embedder?: Embedder): MemoryContext {
  return {
    store: store as unknown as MemoryStore,
    scope: SCOPE,
    ...(embedder ? { embedder } : {}),
  };
}

function recorder(): { emit: (d: ThreadEventData) => Promise<void>; events: ThreadEventData[] } {
  const events: ThreadEventData[] = [];
  return {
    events,
    emit: async (d) => {
      events.push(d);
    },
  };
}

// ---------------------------------------------------------------------------
// recallQueryFor
// ---------------------------------------------------------------------------

describe("recallQueryFor", () => {
  test("uses the last three user messages, author prefix stripped", () => {
    const msgs: LlmMessage[] = [
      user("[slack Brad]: one"),
      { role: "assistant", text: "ack" },
      user("[slack Brad]: two"),
      user("[cli u1]: three"),
      user("[cli u1]: four"),
    ];
    expect(recallQueryFor(msgs, [])).toBe("two three four");
  });

  test("skips harness notices and the projected continuity document", () => {
    const msgs: LlmMessage[] = [
      user("# Pinky Continuity\n**Goal:** ship it\n## Plan\n- now: wire recall"),
      user("[harness notice] context pressure: this window is filling up."),
      user("[slack Brad]: what did we decide about retries?"),
    ];
    expect(recallQueryFor(msgs, [])).toBe("what did we decide about retries?");
  });

  test("prepends the latest continuity memoryHints", () => {
    const events = [
      continuityEvent(["stale hint"], 1),
      continuityEvent(["retry policy", "kimi rate limits"], 2),
    ];
    const query = recallQueryFor([user("[cli u1]: keep going")], events);
    expect(query).toBe("retry policy kimi rate limits keep going");
  });

  test("hints survive a window with no user messages (the post-restart case)", () => {
    const msgs: LlmMessage[] = [
      user("# Pinky Continuity\n**Goal:** ship it"),
    ];
    expect(recallQueryFor(msgs, [continuityEvent(["deploy runbook"])])).toBe("deploy runbook");
  });

  test("returns empty string when the window has nothing usable", () => {
    expect(recallQueryFor([{ role: "assistant", text: "thinking" }], [])).toBe("");
    expect(recallQueryFor([], [])).toBe("");
  });

  test("caps the query at 1000 chars and collapses whitespace", () => {
    const query = recallQueryFor([user(`[cli u1]: ${"word ".repeat(600)}`)], []);
    expect(query.length).toBeLessThanOrEqual(1000);
    expect(query).not.toContain("  ");
    expect(query.startsWith("word word")).toBe(true);
  });

  test("survives a continuity document with no usable memoryHints", () => {
    // The doc comes back out of the event log: an older schema, a hand-patched
    // row or a tool that skipped validation can leave `memoryHints` missing,
    // null, or not an array. The boundary is permanent, so a throw here would
    // kill this thread on every wake from now on — never a throw, just no hints.
    const malformed = (document: unknown): ThreadEvent =>
      event({ type: "continuity", document, tokensBefore: 1000 } as ThreadEventData, 1);

    const msgs = [user("[cli u1]: keep going")];
    for (const document of [
      { goal: "g" }, // no memoryHints at all
      { goal: "g", memoryHints: null },
      { goal: "g", memoryHints: "retry policy" }, // a string, not an array
      null,
    ]) {
      expect(recallQueryFor(msgs, [malformed(document)])).toBe("keep going");
    }

    // A partly-good array keeps its strings and drops the rest.
    expect(
      recallQueryFor(msgs, [malformed({ memoryHints: ["retry policy", 7, null, "  "] })]),
    ).toBe("retry policy keep going");
  });

  test("accepts core ProjectedMessage shapes as well as LlmMessage", () => {
    // The loop hands over buildContext() output directly.
    const projected = [{ role: "user" as const, text: "[slack Brad]: hybrid retrieval" }];
    expect(recallQueryFor(projected, [])).toBe("hybrid retrieval");
  });
});

// ---------------------------------------------------------------------------
// renderMemoriesBlock
// ---------------------------------------------------------------------------

describe("renderMemoriesBlock", () => {
  test("renders one line per hit inside a <memories> block", () => {
    const block = renderMemoriesBlock(
      [
        hit({ kind: "semantic", importance: 7, recordedAt: "2026-08-01T09:30:00.000Z", text: "Brad prefers terse answers.", score: 0.9 }),
        hit({ kind: "episodic", importance: 4, recordedAt: "2026-08-20T22:00:00.000Z", text: "Deploy failed on a missing env var.", score: 0.5 }),
      ],
      5_000,
    );
    expect(block).not.toBeNull();
    expect(block!.text).toBe(
      [
        MEMORIES_HEADER,
        "<memories>",
        "- (semantic, importance 7, 2026-08-01) Brad prefers terse answers.",
        "- (episodic, importance 4, 2026-08-20) Deploy failed on a missing env var.",
        "</memories>",
      ].join("\n"),
    );
    expect(block!.used).toHaveLength(2);
    // Background context, not instructions (§5.4) — and never role "system".
    expect(block!.text.startsWith("[harness notice]")).toBe(true);
    expect(block!.text).toContain("not instructions");
  });

  test("orders by score desc regardless of input order", () => {
    const low = hit({ text: "low", score: 0.1 });
    const high = hit({ text: "high", score: 0.9 });
    const block = renderMemoriesBlock([low, high], 5_000)!;
    expect(block.used.map((h) => h.text)).toEqual(["high", "low"]);
    expect(block.text.indexOf("high")).toBeLessThan(block.text.indexOf("low"));
  });

  test("null for an empty hit list", () => {
    expect(renderMemoriesBlock([], 5_000)).toBeNull();
  });

  test("the budget cut keeps the highest-scored hits", () => {
    const hits = [
      hit({ text: `A ${"a".repeat(200)}`, score: 0.9 }),
      hit({ text: `B ${"b".repeat(200)}`, score: 0.8 }),
      hit({ text: `C ${"c".repeat(200)}`, score: 0.7 }),
    ];
    // Header alone is ~40 tokens; 110 fits roughly one 200-char line.
    const block = renderMemoriesBlock(hits, 110)!;
    expect(block.used.map((h) => h.text[0])).toEqual(["A"]);
    expect(estimateTokens([{ role: "user", text: block.text }])).toBeLessThanOrEqual(110);
  });

  test("the budget is counted the way the pressure ladder counts it", () => {
    // The block is injected as ONE user message, so its cost is core's
    // estimateTokens over that message — per-message overhead included. A
    // private chars/4 formula here would under-count every block and drift the
    // day the ladder's estimate changed.
    const one = hit({ text: "x".repeat(200), score: 0.9 });
    const exact = estimateTokens([{ role: "user", text: renderMemoriesBlock([one], 10_000)!.text }]);
    expect(renderMemoriesBlock([one], exact)).not.toBeNull();
    expect(renderMemoriesBlock([one], exact - 1)).toBeNull();
  });

  test("null when not even the first hit fits the budget", () => {
    expect(renderMemoriesBlock([hit({ text: "x".repeat(400) })], 5)).toBeNull();
  });

  test("flattens newlines so one hit stays one line", () => {
    const block = renderMemoriesBlock([hit({ text: "line one\nline two" })], 5_000)!;
    expect(block.text.split("\n")).toHaveLength(4); // header + open + 1 hit + close
    expect(block.text).toContain("line one line two");
  });
});

// ---------------------------------------------------------------------------
// autoRecall
// ---------------------------------------------------------------------------

describe("autoRecall", () => {
  const run = (memory: MemoryContext, rec: ReturnType<typeof recorder>, over: Partial<{ query: string; limit: number; tokenBudget: number; signal: AbortSignal }> = {}) =>
    autoRecall({
      memory,
      query: "retry policy",
      limit: 12,
      tokenBudget: 5_000,
      emit: rec.emit,
      ...over,
    });

  test("returns the block and journals a memory event when there are hits", async () => {
    const store = new StubStore({ hits: [hit({ text: "use jittered backoff", score: 0.9 })] });
    const rec = recorder();
    const block = await run(context(store), rec);

    expect(block).toContain("use jittered backoff");
    expect(store.searches[0]).toMatchObject({ scope: SCOPE, query: "retry policy", limit: 12 });
    expect(store.searches[0]!.queryEmbedding).toBeUndefined();
    expect(rec.events).toEqual([
      { type: "memory", op: "recall", ids: ["m" + nextId], text: "retry policy", count: 1 },
    ]);
  });

  test("no hits: no block, no event", async () => {
    const store = new StubStore({ hits: [] });
    const rec = recorder();
    expect(await run(context(store), rec)).toBeNull();
    expect(rec.events).toEqual([]);
  });

  test("embeds the query only when the store supports vectors", async () => {
    const embedder = new StubEmbedder();
    const store = new StubStore({ vectors: true, hits: [hit()] });
    await run(context(store, embedder), recorder());
    expect(embedder.calls).toEqual([["retry policy"]]);
    expect(store.searches[0]!.queryEmbedding).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  test("no vector column: the embedder is never called", async () => {
    const embedder = new StubEmbedder();
    const store = new StubStore({ vectors: false, hits: [hit()] });
    await run(context(store, embedder), recorder());
    expect(embedder.calls).toEqual([]);
    expect(store.searches[0]!.queryEmbedding).toBeUndefined();
  });

  test("an empty query skips the embedder but still searches (newest-first fallback)", async () => {
    const embedder = new StubEmbedder();
    const store = new StubStore({ vectors: true, hits: [hit()] });
    await run(context(store, embedder), recorder(), { query: "" });
    expect(embedder.calls).toEqual([]);
    expect(store.searches[0]!.query).toBe("");
  });

  test("embedder failure degrades to FTS-only and journals an error", async () => {
    const embedder = new StubEmbedder(new Error("402 no credits"));
    const store = new StubStore({ vectors: true, hits: [hit({ text: "still recalled" })] });
    const rec = recorder();
    const block = await run(context(store, embedder), rec);

    expect(block).toContain("still recalled");
    expect(store.searches[0]!.queryEmbedding).toBeUndefined();
    const err = rec.events[0] as Extract<ThreadEventData, { type: "error" }>;
    expect(err).toMatchObject({ type: "error", source: "memory", count: 1 });
    expect(err.message).toContain("402 no credits");
    expect(err.message).toContain("FTS-only");
    expect(rec.events[1]).toMatchObject({ type: "memory", op: "recall" });
  });

  test("a failing vector-support probe degrades the same way", async () => {
    const embedder = new StubEmbedder();
    const store = new StubStore({ vectorProbeError: new Error("column probe failed"), hits: [hit()] });
    const rec = recorder();
    expect(await run(context(store, embedder), rec)).not.toBeNull();
    expect(embedder.calls).toEqual([]);
    expect(rec.events[0]).toMatchObject({ type: "error", source: "memory" });
  });

  test("store failure journals an error and returns null (never throws)", async () => {
    const store = new StubStore({ searchError: new Error("connection reset") });
    const rec = recorder();
    expect(await run(context(store), rec)).toBeNull();
    expect(rec.events).toHaveLength(1);
    const err = rec.events[0] as Extract<ThreadEventData, { type: "error" }>;
    expect(err).toMatchObject({ type: "error", source: "memory", count: 1 });
    expect(err.message).toContain("recall failed");
  });

  test("the memory event counts candidates before the budget cut and ids after it", async () => {
    const hits = [
      hit({ text: `A ${"a".repeat(200)}`, score: 0.9 }),
      hit({ text: `B ${"b".repeat(200)}`, score: 0.8 }),
    ];
    const rec = recorder();
    const block = await run(context(new StubStore({ hits })), rec, { tokenBudget: 110 });
    expect(block).toContain("A a");
    expect(block).not.toContain("B b");
    const ev = rec.events[0] as Extract<ThreadEventData, { type: "memory" }>;
    expect(ev.ids).toEqual([hits[0]!.id]);
    expect(ev.count).toBe(2);
  });

  test("a throwing emit cannot take the run down with it", async () => {
    // `emit` appends to the event log — the same database a failing recall has
    // just tripped over. Journaling the error must not become the thing that
    // throws out of the loop.
    const dead = async (): Promise<void> => {
      throw new Error("append failed: connection reset");
    };

    // (1) the memory event on the happy path
    const ok = new StubStore({ hits: [hit({ text: "still returned", score: 0.9 })] });
    const block = await autoRecall({
      memory: context(ok),
      query: "retry policy",
      limit: 12,
      tokenBudget: 5_000,
      emit: dead,
    });
    expect(block).toContain("still returned");

    // (2) the error event on the store-failure path
    const broken = new StubStore({ searchError: new Error("connection reset") });
    expect(
      await autoRecall({
        memory: context(broken),
        query: "retry policy",
        limit: 12,
        tokenBudget: 5_000,
        emit: dead,
      }),
    ).toBeNull();

    // (3) the error event on the embedder-failure path — and the FTS-only
    // retry still has to happen after it.
    const embedder = new StubEmbedder(new Error("402 no credits"));
    const degraded = new StubStore({ vectors: true, hits: [hit({ text: "fts only", score: 0.9 })] });
    expect(
      await autoRecall({
        memory: context(degraded, embedder),
        query: "retry policy",
        limit: 12,
        tokenBudget: 5_000,
        emit: dead,
      }),
    ).toContain("fts only");
  });

  test("an aborted signal short-circuits without touching the store", async () => {
    const controller = new AbortController();
    controller.abort();
    const store = new StubStore({ hits: [hit()] });
    const rec = recorder();
    expect(await run(context(store), rec, { signal: controller.signal })).toBeNull();
    expect(store.searches).toHaveLength(0);
    expect(rec.events).toEqual([]);
  });
});
