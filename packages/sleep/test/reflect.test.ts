/**
 * Reflection pass (DESIGN.md §5.3 item 3 "Consolidation"; slice 6, contract §3.3).
 *
 * What is worth asserting here is not "it calls an LLM" but the four rules that
 * make consolidation safe to run unattended:
 *
 *  1. it never reads `user`/`private` rows (§5.1) — a leak there is permanent;
 *  2. it costs nothing when there is nothing to do (no call, no writes);
 *  3. its rows and its receipt commit together, receipt LAST, under the lock;
 *  4. a pass that lost the race, or failed, writes no memory rows at all.
 */
import { describe, expect, it } from "bun:test";
import type { ThreadEventData } from "@pinky/core";
import { insightVisibility, runReflectPass } from "../src/reflect";
import { REFLECT_TOOL_NAME } from "../src/schemas";
import { reflectThread } from "../src/types";
import {
  byTool,
  installReflectCursor,
  makeDeps,
  makeEmbedder,
  makeFakeMemory,
  makeMemoryRow,
  makeProvider,
  reflectReceipt,
  seedAt,
  textTurn,
  toolTurn,
  type FakeDb,
  type FakeMemoryOptions,
} from "./helpers";

const THREAD = reflectThread("t1", "pinky");

/** `count` current rows, ids m1..mN, an hour apart, oldest first (since()'s order). */
function batch(count: number, over: Parameters<typeof makeMemoryRow>[0] = {}) {
  return Array.from({ length: count }, (_, i) =>
    makeMemoryRow({
      id: `m${i + 1}`,
      text: `fact ${i + 1}`,
      visibility: "tenant",
      channelId: null,
      recordedAt: `2026-08-29T0${i}:00:00.000Z`,
      ...over,
    }),
  );
}

function reflectTurn(insights: unknown[], over: Parameters<typeof toolTurn>[2] = {}) {
  return toolTurn(REFLECT_TOOL_NAME, { insights }, over);
}

/** A harness with the reflect cursor routed and a scripted reflect answer. */
function harness(opts: {
  since?: ReturnType<typeof batch>;
  insights?: unknown[];
  turn?: ReturnType<typeof textTurn>;
  memory?: FakeMemoryOptions;
}) {
  const memory = makeFakeMemory({ since: opts.since ?? [], ...(opts.memory ?? {}) });
  const turn = opts.turn ?? reflectTurn(opts.insights ?? []);
  const provider = makeProvider(byTool({ [REFLECT_TOOL_NAME]: turn }));
  return { memory, provider };
}

function seen(db: FakeDb): ThreadEventData[] {
  return db.dataFor(THREAD);
}

describe("runReflectPass — the below-threshold gate", () => {
  it("makes no LLM call and writes nothing when the batch is short", async () => {
    const { memory, provider } = harness({ since: batch(3) });
    const { deps, db } = makeDeps({ memory, provider, settings: { reflectMinMemories: 5 } });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result).toEqual({ status: "skipped", reason: "below-threshold" });
    // The expensive half of a pass is the provider call, and the gate is
    // upstream of it: that is what makes an idle sweep free.
    expect(provider.received.length).toBe(0);
    expect(seen(db)).toEqual([]);
    expect(db.txLog).toEqual([]);
    expect(memory.retains).toEqual([]);
  });
});

describe("runReflectPass — the §5.1 scope of the batch", () => {
  it("asks for tenant/channel/global rows only, and never widens with the surface", async () => {
    const { memory, provider } = harness({ since: batch(2) });
    // Both scope flags TRUE — the trusted local surface (`pinky sleep run`).
    // The reflect read must STILL refuse user/private rows: the insight it
    // synthesizes lands at tenant visibility, where one person's fact would
    // become readable from every channel, permanently.
    const { deps, db } = makeDeps({ memory, provider, settings: { reflectBatch: 42 } });
    installReflectCursor(db);

    await runReflectPass(deps);

    expect(memory.sinces.length).toBe(1);
    expect(memory.sinces[0]).toEqual({
      // `allChannels` is the worker-only read arm (§5.1): consolidation is
      // cross-thread by definition, and extraction writes `channel` by default,
      // so without it the batch could only ever hold tenant/global rows.
      scope: { agentId: "pinky", allChannels: true, includeUser: false, includePrivate: false },
      after: null,
      limit: 42,
      visibilities: ["tenant", "channel", "global"],
      // Never consolidate the previous consolidation: insights land after the
      // watermark at tenant/channel visibility under this same agent, so
      // without this the pass reflects on its own output on the next sweep.
      excludeSources: ["sleep:reflect"],
    });
  });

  it("reads its watermark from the newest reflect receipt and resumes after it", async () => {
    const through = { recordedAt: "2026-08-28T00:00:00.000Z", id: "m0" };
    const { memory, provider } = harness({ since: batch(5) });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);
    db.seed(THREAD, [reflectReceipt({ through })]);

    const result = await runReflectPass(deps);

    expect(memory.sinces[0]?.after).toEqual(through);
    expect(result.status).toBe("done");
    if (result.status !== "done") return;
    expect(result.receipt.after).toEqual(through);
    // Receipts live on the worker's own thread — and discovery excludes
    // `sleep:` channels, so the worker can never extract from its own log.
    expect(THREAD.channelId).toBe("sleep:pinky");
    expect(seen(db).length).toBe(2); // the seeded receipt, then ours
  });
});

describe("runReflectPass — a completed pass", () => {
  it("forces the reflect tool and sends a batch payload with no personal fields", async () => {
    const rows = batch(5);
    const { memory, provider } = harness({ since: rows });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    await runReflectPass(deps);

    const call = provider.received[0];
    expect(call?.model).toBe("sleep"); // provider prefix stripped
    expect(call?.toolChoice).toEqual({ type: "tool", name: REFLECT_TOOL_NAME });
    expect(call?.tools.map((t) => t.name)).toEqual([REFLECT_TOOL_NAME]);
    const text = call?.messages[0]?.text ?? "{}";
    const payload = JSON.parse(text) as { memories: Record<string, unknown>[] };
    expect(payload.memories.length).toBe(5);
    expect(Object.keys(payload.memories[0] ?? {}).sort()).toEqual([
      "channelId",
      "id",
      "importance",
      "kind",
      "recordedAt",
      "text",
      "visibility",
    ]);
    // No `userId` key in the shape at all: the payload cannot carry a subject
    // principal even if a row somehow had one.
    expect(text).not.toContain("userId");
  });

  it("retains each insight, invalidates its supersedes with the reason, and journals the receipt LAST", async () => {
    const rows = batch(5);
    const { memory, provider } = harness({
      since: rows,
      insights: [
        { text: "insight one", importance: 8, sources: ["m1", "m2", "m3"], supersedes: ["m2"] },
      ],
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    expect(memory.retains.length).toBe(1);
    expect(memory.retains[0]).toMatchObject({
      agentId: "pinky",
      visibility: "tenant",
      kind: "semantic",
      text: "insight one",
      importance: 8,
      meta: { source: "sleep:reflect", sources: ["m1", "m2", "m3"] },
    });
    // §5.2: invalidate, never DELETE — and the reason names the successor, so
    // the consolidation is walkable forwards from the retired row.
    const newId = memory.written[0]?.id;
    expect(memory.invalidations).toEqual([
      { id: "m2", reason: `sleep:reflect consolidated into ${String(newId)}` },
    ]);

    const events = seen(db);
    expect(events.map((e) => e.type)).toEqual(["memory", "memory", "sleep"]);
    expect(events[0]).toMatchObject({ op: "retain", ids: [newId] });
    expect(events[1]).toMatchObject({ op: "invalidate", ids: ["m2"] });
    expect(events[2]).toMatchObject({
      phase: "reflect",
      after: null,
      // The last row of the batch: the next pass resumes strictly after it.
      through: { recordedAt: rows[4]?.recordedAt, id: "m5" },
      scanned: 5,
      candidates: 1,
      added: 1,
      updated: 0,
      invalidated: 1,
      noop: 0,
      model: "fake/sleep",
    });

    // ONE transaction, every write inside it: the receipt exists iff the rows
    // do. `bind()` is what put the store on that transaction.
    expect(db.txLog).toEqual(["begin", "commit"]);
    expect(db.all(/insert into events/).every((c) => c.txDepth > 0)).toBe(true);
    expect(memory.boundTo.length).toBe(1);
    expect(memory.boundTo[0]).toBe(db);
    // The lock is taken BEFORE the writes, in the same transaction.
    const lock = db.all(/from threads .* for update/)[0];
    expect(lock?.txDepth).toBeGreaterThan(0);
  });

  it("journals the provider's usage when it reported any", async () => {
    const { memory, provider } = harness({
      since: batch(5),
      turn: reflectTurn([], { usage: { input: 100, output: 20, cacheRead: 4 } }),
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    if (result.status !== "done") return;
    expect(result.receipt.usage).toEqual({ input: 100, output: 20, cacheRead: 4 });
  });

  it("journals a receipt even for zero insights, so the watermark advances", async () => {
    const { memory, provider } = harness({ since: batch(5), insights: [] });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    expect(memory.retains).toEqual([]);
    const events = seen(db);
    expect(events.map((e) => e.type)).toEqual(["sleep"]);
    expect(events[0]).toMatchObject({ candidates: 0, added: 0, through: { id: "m5" } });
    // Without this the same batch would be re-read, re-sent and re-judged on
    // every sweep, forever, for nothing.
  });

  it("counts only the invalidations that actually happened", async () => {
    const { memory, provider } = harness({
      since: batch(5),
      insights: [{ text: "i", importance: 5, sources: ["m1", "m2"], supersedes: ["m1", "m2"] }],
      memory: { invalidateReturns: false },
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    if (result.status !== "done") return;
    // Both rows were already retired by someone else. A receipt claiming two
    // invalidations would be a lie in the audit, and `pinky stats sleep` reads
    // these numbers.
    expect(result.receipt.invalidated).toBe(0);
    expect(seen(db).map((e) => e.type)).toEqual(["memory", "sleep"]);
  });

  it("embeds insights in one call and stores them without vectors when that fails", async () => {
    const insights = [{ text: "a", importance: 5, sources: ["m1"] }];
    const ok = harness({ since: batch(5), insights });
    const embedder = makeEmbedder({ dimensions: 4 });
    const first = makeDeps({ memory: ok.memory, provider: ok.provider, embedder });
    installReflectCursor(first.db);
    await runReflectPass(first.deps);
    expect(embedder.calls).toEqual([["a"]]);
    expect(ok.memory.retains[0]?.embeddingModel).toBe("fake/embed");
    expect(ok.memory.retains[0]?.embedding?.length).toBe(4);

    const broken = harness({ since: batch(5), insights });
    const second = makeDeps({
      memory: broken.memory,
      provider: broken.provider,
      embedder: makeEmbedder({ fail: "embeddings disabled: no key" }),
    });
    installReflectCursor(second.db);
    const degraded = await runReflectPass(second.deps);

    // DESIGN.md §5.5: embeddings are optional everywhere. A dead embedder costs
    // that row its vector voice, never the row.
    expect(degraded.status).toBe("done");
    expect(broken.memory.retains[0]?.embedding).toBeUndefined();
    expect(second.logs.some((l) => l.includes("embedding failed"))).toBe(true);
  });
});

describe("insightVisibility — where a synthesized row is allowed to live", () => {
  it("is tenant when no source is channel-visible", () => {
    expect(insightVisibility([])).toEqual({ visibility: "tenant" });
    expect(
      insightVisibility([
        makeMemoryRow({ visibility: "tenant", channelId: null }),
        makeMemoryRow({ visibility: "global", channelId: null }),
      ]),
    ).toEqual({ visibility: "tenant" });
  });

  it("is that channel when exactly one channel is involved, even alongside wider sources", () => {
    expect(
      insightVisibility([
        makeMemoryRow({ visibility: "channel", channelId: "c1" }),
        makeMemoryRow({ visibility: "channel", channelId: "c1" }),
      ]),
    ).toEqual({ visibility: "channel", channelId: "c1" });
    // NARROWER than the tenant row that also fed it — which leaks nothing: the
    // tenant row stays exactly where it was.
    expect(
      insightVisibility([
        makeMemoryRow({ visibility: "channel", channelId: "c1" }),
        makeMemoryRow({ visibility: "tenant", channelId: null }),
      ]),
    ).toEqual({ visibility: "channel", channelId: "c1" });
  });

  it("refuses an insight drawn from two channels", () => {
    // There is no honest placement: `tenant` publishes c1's content to c2, and
    // either channel files the other's content where it cannot be read.
    expect(
      insightVisibility([
        makeMemoryRow({ visibility: "channel", channelId: "c1" }),
        makeMemoryRow({ visibility: "channel", channelId: "c2" }),
      ]),
    ).toEqual({ drop: ["c1", "c2"] });
  });

  it("is applied to the retained row end to end", async () => {
    const rows = batch(5, { visibility: "channel", channelId: "c1" });
    const { memory, provider } = harness({
      since: rows,
      insights: [{ text: "chan", importance: 5, sources: ["m1", "m2"] }],
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    await runReflectPass(deps);

    expect(memory.retains[0]).toMatchObject({ visibility: "channel", channelId: "c1" });
  });
});

describe("runReflectPass — the §5.1 placement rules, end to end", () => {
  /** A batch built row by row, so channels can differ within it. */
  function mixed(specs: { id: string; channelId: string | null }[]) {
    return specs.map((spec, i) =>
      makeMemoryRow({
        id: spec.id,
        text: `fact ${spec.id}`,
        visibility: spec.channelId === null ? "tenant" : "channel",
        channelId: spec.channelId,
        recordedAt: `2026-08-29T0${i}:00:00.000Z`,
      }),
    );
  }

  it("drops an insight spanning two channels, counts it as a noop, and still journals a receipt", async () => {
    const rows = mixed([
      { id: "m1", channelId: "c1" },
      { id: "m2", channelId: "c2" },
      { id: "m3", channelId: null },
      { id: "m4", channelId: null },
      { id: "m5", channelId: null },
    ]);
    const { memory, provider } = harness({
      since: rows,
      insights: [{ text: "spans two channels", importance: 7, sources: ["m1", "m2"] }],
    });
    const { deps, db, logs } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    if (result.status !== "done") return;
    // Nothing written: `tenant` would publish c1's content into c2, and either
    // channel would file the other's where it cannot be read.
    expect(memory.retains).toEqual([]);
    expect(result.receipt).toMatchObject({ candidates: 1, added: 0, invalidated: 0, noop: 1 });
    // The receipt still lands, so the watermark advances and this batch is not
    // re-judged on every sweep forever.
    expect(seen(db).map((e) => e.type)).toEqual(["sleep"]);
    expect(logs.some((l) => l.includes("span channels c1, c2"))).toBe(true);
  });

  it("keeps a one-channel insight narrow even when a tenant row also fed it", async () => {
    const rows = mixed([
      { id: "m1", channelId: "c1" },
      { id: "m2", channelId: "c1" },
      { id: "m3", channelId: null },
      { id: "m4", channelId: null },
      { id: "m5", channelId: null },
    ]);
    const { memory, provider } = harness({
      since: rows,
      insights: [{ text: "about c1", importance: 6, sources: ["m1", "m3"] }],
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    expect(memory.retains[0]).toMatchObject({ visibility: "channel", channelId: "c1" });
  });

  it("refuses to retire a row outside the insight's own scope, and keeps the insight", async () => {
    const rows = mixed([
      { id: "m1", channelId: "c1" },
      { id: "m2", channelId: "c1" },
      { id: "m3", channelId: null },
      { id: "m4", channelId: null },
      { id: "m5", channelId: null },
    ]);
    const { memory, provider } = harness({
      since: rows,
      insights: [
        {
          text: "about c1",
          importance: 6,
          sources: ["m1", "m2", "m3"],
          supersedes: ["m1", "m3"],
        },
      ],
    });
    const { deps, db, logs } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("done");
    if (result.status !== "done") return;
    // m3 is tenant-wide; retiring it because a c1-scoped insight replaced it
    // would lose the fact in every OTHER channel, where the insight is not
    // readable. The insight itself is still true, so it is kept.
    expect(memory.invalidations.map((i) => i.id)).toEqual(["m1"]);
    expect(result.receipt).toMatchObject({ added: 1, invalidated: 1, noop: 0 });
    expect(seen(db).map((e) => e.type)).toEqual(["memory", "memory", "sleep"]);
    expect(logs.some((l) => l.includes("outside its scope") && l.includes("m3"))).toBe(true);
  });
});

describe("runReflectPass — the idle gate on the worker's own thread", () => {
  /** An event stamped `ageMs` before the harness's fixed clock. */
  const FIXED = new Date("2026-08-29T00:00:00.000Z");

  it("skips when the reflect thread was written to more recently than idleMs", async () => {
    const { memory, provider } = harness({
      since: batch(5),
      insights: [{ text: "i", importance: 5, sources: ["m1"] }],
    });
    const { deps, db } = makeDeps({
      memory,
      provider,
      settings: { idleMs: 600_000 },
      now: () => FIXED,
    });
    installReflectCursor(db);
    // A receipt from one minute ago: the previous pass (or its failure) is
    // still fresh.
    seedAt(db, THREAD, [reflectReceipt()], new Date(FIXED.getTime() - 60_000));

    const result = await runReflectPass(deps);

    expect(result).toEqual({ status: "skipped", reason: "not-idle" });
    // Cheapest possible refusal: no batch read, no provider call, no writes.
    // This is the ONLY backoff a failing reflect pass has — its own `error`
    // event is the newest event here, so a dead provider is refused for
    // `idleMs` instead of costing two LLM calls and one row every tick.
    expect(memory.sinces).toEqual([]);
    expect(provider.received.length).toBe(0);
    expect(db.txLog).toEqual([]);
  });

  it("runs when the newest event is older than idleMs", async () => {
    const { memory, provider } = harness({ since: batch(5) });
    const { deps, db } = makeDeps({
      memory,
      provider,
      settings: { idleMs: 600_000 },
      now: () => FIXED,
    });
    installReflectCursor(db);
    seedAt(db, THREAD, [reflectReceipt()], new Date(FIXED.getTime() - 3_600_000));

    expect((await runReflectPass(deps)).status).toBe("done");
  });

  it("treats a thread with no events at all as idle", async () => {
    const { memory, provider } = harness({ since: batch(5) });
    const { deps, db } = makeDeps({ memory, provider, settings: { idleMs: 600_000 } });
    installReflectCursor(db);

    // Nothing seeded: the agent has never reflected, which must not be
    // mistaken for "just reflected".
    expect((await runReflectPass(deps)).status).toBe("done");
  });

  it("is bypassed by ignoreIdle, and skipped entirely when idleMs is 0", async () => {
    const first = harness({ since: batch(5) });
    const bypass = makeDeps({
      memory: first.memory,
      provider: first.provider,
      settings: { idleMs: 600_000 },
      now: () => FIXED,
    });
    installReflectCursor(bypass.db);
    seedAt(bypass.db, THREAD, [reflectReceipt()], new Date(FIXED.getTime() - 1_000));

    // `pinky sleep run --now` and smoke both depend on this.
    expect((await runReflectPass(bypass.deps, { ignoreIdle: true })).status).toBe("done");

    const second = harness({ since: batch(5) });
    const ungated = makeDeps({
      memory: second.memory,
      provider: second.provider,
      settings: { idleMs: 0 },
      now: () => FIXED,
    });
    installReflectCursor(ungated.db);
    seedAt(ungated.db, THREAD, [reflectReceipt()], new Date(FIXED.getTime() - 1_000));

    expect((await runReflectPass(ungated.deps)).status).toBe("done");
    // idleMs 0 means "no gate": not even the probe is issued.
    expect(ungated.db.all(/select ts from events/).length).toBe(0);
  });
});

describe("runReflectPass — losing the race", () => {
  it("writes nothing when the watermark moved while the provider was answering", async () => {
    const memory = makeFakeMemory({ since: batch(5) });
    let db: FakeDb | undefined;
    // The competing pass commits its receipt WHILE we are talking to the
    // provider: our pre-read saw none, the in-transaction re-read sees theirs.
    const provider = makeProvider(
      byTool({
        [REFLECT_TOOL_NAME]: () => {
          db?.seed(THREAD, [
            reflectReceipt({ through: { recordedAt: "2026-08-29T09:00:00.000Z", id: "m5" } }),
          ]);
          return reflectTurn([{ text: "i", importance: 5, sources: ["m1"] }]);
        },
      }),
    );
    const h = makeDeps({ memory, provider });
    db = h.db;
    installReflectCursor(h.db);

    const result = await runReflectPass(h.deps);

    expect(result).toEqual({ status: "skipped", reason: "lost-claim" });
    expect(memory.retains).toEqual([]);
    expect(memory.invalidations).toEqual([]);
    // Only the competitor's seeded receipt is on the thread.
    expect(seen(h.db).length).toBe(1);
    expect(h.db.all(/insert into events/).length).toBe(0);
  });
});

describe("runReflectPass — failure", () => {
  it("journals exactly one error event when the model returns no tool call", async () => {
    const { memory } = harness({ since: batch(5) });
    const provider = makeProvider(byTool({ [REFLECT_TOOL_NAME]: textTurn("nope") }));
    const { deps, db, logs } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("failed");
    const events = seen(db);
    expect(events.length).toBe(1);
    expect(events[0]).toMatchObject({ type: "error", source: "sleep", count: 1 });
    expect(memory.retains).toEqual([]);
    expect(logs.some((l) => l.startsWith("[sleep] reflect failed:"))).toBe(true);
  });

  it("refuses an invalid tool call rather than writing a half-validated insight", async () => {
    // `sources` cites a row outside the batch — parseReflect must reject it.
    const { memory, provider } = harness({
      since: batch(5),
      insights: [{ text: "i", importance: 5, sources: ["not-in-batch"] }],
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("failed");
    if (result.status !== "failed") return;
    expect(result.error).toContain(REFLECT_TOOL_NAME);
    expect(memory.retains).toEqual([]);
    expect(seen(db).map((e) => e.type)).toEqual(["error"]);
  });

  it("rolls back and appends the error OUTSIDE the failed transaction", async () => {
    const { memory, provider } = harness({
      since: batch(5),
      insights: [{ text: "i", importance: 5, sources: ["m1"] }],
      memory: { failOn: { op: "retain", message: "retain exploded" } },
    });
    const { deps, db } = makeDeps({ memory, provider });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("failed");
    // begin/rollback is the pass; begin/commit is the error append, which has
    // to be its own transaction or it would roll back with the pass and the
    // failure would leave no trace at all.
    expect(db.txLog).toEqual(["begin", "rollback", "begin", "commit"]);
    expect(seen(db).map((e) => e.type)).toEqual(["error"]);
  });

  it("does not journal an error when the failure is this process shutting down", async () => {
    const controller = new AbortController();
    controller.abort();
    const { memory } = harness({ since: batch(5) });
    const provider = makeProvider(() => {
      throw new Error("aborted");
    });
    const { deps, db } = makeDeps({ memory, provider, signal: controller.signal });
    installReflectCursor(db);

    const result = await runReflectPass(deps);

    expect(result.status).toBe("failed");
    // The pool may already be closing; an `error` row about our own SIGTERM is
    // noise at best and a second failure at worst.
    expect(seen(db)).toEqual([]);
  });
});
