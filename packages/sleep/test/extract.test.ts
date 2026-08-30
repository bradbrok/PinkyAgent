/**
 * The extraction pass (slice 6, DESIGN.md §5.3 item 3).
 *
 * What these tests are really about is ORDER and ATOMICITY: read the cursor,
 * read the range, ask the model twice, then lock — re-read the cursor — write
 * — journal the receipt, all in ONE transaction. Every claim about that
 * sequence is asserted off `db.calls` (with their tx depth) and `db.txLog`,
 * because a comment saying "one transaction" is not evidence.
 *
 * The pass's SQL against a real Postgres is E2's integration suite; nothing
 * here proves anything about the database.
 */
import { describe, expect, test } from "bun:test";
import type { ThreadEventData } from "@pinky/core";
import type { AssistantTurn, CompleteOptions } from "@pinky/runtime";
import { runExtractPass } from "../src/extract";
import { DECIDE_TOOL_NAME, EXTRACT_TOOL_NAME } from "../src/schemas";
import { DECIDE_SYSTEM, EXTRACT_SYSTEM } from "../src/prompts";
import type { ExtractReceipt } from "../src/types";
import {
  FakeDb,
  NARROW_SCOPE,
  THREAD,
  assistant,
  byTool,
  ingress,
  makeDeps,
  makeEmbedder,
  makeFakeMemory,
  makeMemoryHit,
  makeProvider,
  textTurn,
  toolTurn,
} from "./helpers";
import type { DepsOverrides, FakeMemoryOptions } from "./helpers";

// ---------------------------------------------------------------------------
// Scripting helpers
// ---------------------------------------------------------------------------

interface PayloadCandidate {
  index: number;
  text: string;
  visibility: string;
  neighbors: { id: string; text: string; kind: string; importance: number; recordedAt: string }[];
}

/** The decide call's payload, as the provider received it. */
function payloadOf(opts: CompleteOptions): { candidates: PayloadCandidate[] } {
  return JSON.parse(opts.messages[0]?.text ?? "{}") as { candidates: PayloadCandidate[] };
}

const extractTurn = (
  candidates: unknown[],
  over: Partial<AssistantTurn> = {},
): AssistantTurn => toolTurn(EXTRACT_TOOL_NAME, { candidates }, over);

/** Answer every candidate with ADD. */
const decideAdd = (opts: CompleteOptions): AssistantTurn =>
  toolTurn(DECIDE_TOOL_NAME, {
    decisions: payloadOf(opts).candidates.map((c) => ({ candidate: c.index, action: "ADD" })),
  });

const CANDIDATE = { text: "Brad prefers terse answers", kind: "semantic", importance: 7, visibility: "channel" };

function harness(
  script: Record<string, AssistantTurn | ((opts: CompleteOptions) => AssistantTurn)>,
  over: DepsOverrides = {},
  memoryOpts: FakeMemoryOptions = {},
) {
  const db = over.db ?? new FakeDb();
  const memory = over.memory ?? makeFakeMemory(memoryOpts);
  return makeDeps({ ...over, db, memory, provider: makeProvider(byTool(script)) });
}

/** Every `sleep`/`extract` receipt on the thread, oldest first. */
function receipts(db: FakeDb): ExtractReceipt[] {
  return db
    .dataFor(THREAD)
    .filter((d): d is ExtractReceipt => d.type === "sleep" && d.phase === "extract");
}

// ---------------------------------------------------------------------------

describe("runExtractPass — reading", () => {
  test("reads the cursor first, then the range after it", () => {
    const { deps, db } = harness({});
    db.seed(THREAD, [{ type: "decision", action: "silent", reason: "noise" }]);
    return runExtractPass(deps, THREAD).then(() => {
      const reads = db.calls.map((c) => c.sql.replace(/\s+/g, " "));
      expect(reads[0]).toContain("data->>'toSeq'");
      expect(reads[1]).toContain("seq > $4");
    });
  });

  test("a SHORT page of audit-only events skips without writing anything", async () => {
    // Nothing is hiding beyond a short page, so there is no stall to fix and a
    // receipt here would be a treadmill: it is itself an audit event past the
    // new cursor, so the next call would write another.
    const { deps, db, provider } = harness({});
    db.seed(THREAD, [
      { type: "decision", action: "silent", reason: "noise" },
      { type: "memory", op: "retain", ids: ["m1"], text: "x" },
    ]);
    const before = db.events.length;

    expect(await runExtractPass(deps, THREAD)).toEqual({
      status: "skipped",
      reason: "no-new-events",
    });
    expect(provider.received).toHaveLength(0);
    expect(db.events).toHaveLength(before);
    expect(db.txLog).toEqual([]);
    // Calling again writes nothing either — no receipt chain.
    await runExtractPass(deps, THREAD);
    expect(db.events).toHaveLength(before);
  });

  test("a FULL page of ONLY audit-only events still moves the cursor", async () => {
    const { deps, db, provider } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([]) },
      { settings: { maxEventsPerPass: 2 } },
    );
    // `decision` and `memory` are audit-only for the worker too. One busy pass
    // can fill a whole `history` page with them (12 candidates + a receipt =
    // 13 audit events), and "no-new-events" there would park the cursor
    // forever while discovery keeps seeing real material beyond the page.
    db.seed(THREAD, [
      { type: "decision", action: "silent", reason: "noise" },
      { type: "memory", op: "retain", ids: ["m1"], text: "x" },
      // Real material BEYOND the page: this is what discovery keeps seeing,
      // and what the parked cursor would never reach.
      ingress("the material the stall would hide"),
    ]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("done");
    expect(provider.received).toHaveLength(0);
    const receipt = receipts(db).at(-1);
    expect(receipt).toMatchObject({
      fromSeq: 1,
      toSeq: 2,
      scanned: 0,
      candidates: 0,
      added: 0,
      updated: 0,
      invalidated: 0,
      noop: 0,
    });
    // Nobody was asked, so nobody counted: `usage` is absent, never 0.
    expect(receipt && "usage" in receipt).toBe(false);
    // The receipt is the only thing appended, under the ordinary claim.
    expect(db.dataFor(THREAD).slice(3).map((d) => d.type)).toEqual(["sleep"]);
    expect(db.txLog).toEqual(["begin", "commit"]);

    // And the next pass reaches the material the stall would have hidden.
    const second = await runExtractPass(deps, THREAD);
    expect(second.status).toBe("done");
    expect(provider.received[0]?.messages[0]?.text).toContain("the material the stall would hide");
  });

  test("a full page of the worker's OWN output converges — no receipt treadmill", async () => {
    // A normal pass writes up to 12 `memory` events plus its `sleep` receipt,
    // so with a page this small the NEXT page is made entirely of that output.
    // A catch-up receipt there would append one more `sleep` event, filling the
    // page again — one receipt per invocation, forever, on any surface that
    // pins a thread and calls straight in (`pinky sleep run --thread`, smoke).
    const { deps, db, provider } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      { settings: { maxEventsPerPass: 1 } },
    );
    db.seed(THREAD, [ingress("remember this")]);

    expect((await runExtractPass(deps, THREAD)).status).toBe("done");
    const afterFirst = db.events.length;
    expect(receipts(db)).toHaveLength(1);

    // Every later invocation is a full page of `memory`/`sleep` and nothing else.
    for (let i = 0; i < 8; i++) {
      expect(await runExtractPass(deps, THREAD)).toEqual({
        status: "skipped",
        reason: "no-new-events",
      });
    }
    expect(receipts(db)).toHaveLength(1);
    expect(db.events).toHaveLength(afterFirst);
    expect(db.txLog).toEqual(["begin", "commit"]); // the first pass's, and no more
    expect(provider.received).toHaveLength(2); // the first pass's two calls only
  });

  test("a FAILED pass's own error event does not feed the next pass", async () => {
    // `error` is not extractable (types.ts): otherwise the failure path would
    // self-feed — thread stays due, cursor parked, transcript growing by one
    // error line and two LLM calls per sweep, forever.
    const { deps, db, provider } = harness({});
    db.seed(THREAD, [{ type: "error", source: "sleep", message: "provider timed out", count: 1 }]);

    expect(await runExtractPass(deps, THREAD)).toEqual({
      status: "skipped",
      reason: "no-new-events",
    });
    expect(provider.received).toHaveLength(0);
  });

  test("an empty thread is also no-new-events", async () => {
    const { deps, provider } = harness({});
    expect(await runExtractPass(deps, THREAD)).toEqual({
      status: "skipped",
      reason: "no-new-events",
    });
    expect(provider.received).toHaveLength(0);
  });

  test("the cursor starts after the newest receipt's toSeq", async () => {
    const { deps, db, provider } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([]),
    });
    db.seed(THREAD, [ingress("old one"), ingress("old two")]);
    db.seed(THREAD, [
      {
        type: "sleep",
        phase: "extract",
        fromSeq: 1,
        toSeq: 2,
        scanned: 2,
        candidates: 0,
        added: 0,
        updated: 0,
        invalidated: 0,
        noop: 0,
        model: "fake/sleep",
        ms: 0,
      },
    ]);
    db.seed(THREAD, [ingress("new one")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("done");
    // Only the event after the receipt is in the transcript.
    const prompt = provider.received[0]?.messages[0]?.text ?? "";
    expect(prompt).toContain("new one");
    expect(prompt).not.toContain("old one");
    // fromSeq is cursor+1 = 3, which is the PREVIOUS receipt's own seq: an
    // audit event inside the range, consumed and never re-read.
    expect(receipts(db).at(-1)).toMatchObject({ fromSeq: 3, toSeq: 4, scanned: 1 });
  });

  test("the range covers audit events inside it, so they are never re-read", async () => {
    const { deps, db } = harness({ [EXTRACT_TOOL_NAME]: extractTurn([]) });
    db.seed(THREAD, [ingress("hi"), { type: "decision", action: "reply", reason: "asked" }]);

    await runExtractPass(deps, THREAD);

    // scanned counts the RENDERED events; the range is what was consumed.
    expect(receipts(db).at(-1)).toMatchObject({ fromSeq: 1, toSeq: 2, scanned: 1 });
  });

  test("when the char budget binds, the cursor stops where the transcript did", async () => {
    const { deps, db } = harness({ [EXTRACT_TOOL_NAME]: extractTurn([]) });
    db.seed(THREAD, [ingress("x".repeat(30_000)), ingress("this one waits for the next pass")]);

    await runExtractPass(deps, THREAD);

    const receipt = receipts(db).at(-1);
    expect(receipt).toMatchObject({ fromSeq: 1, toSeq: 1, scanned: 1 });
  });
});

describe("runExtractPass — the two calls", () => {
  test("the extract call forces its tool and sends the transcript", async () => {
    const { deps, db, provider } = harness({ [EXTRACT_TOOL_NAME]: extractTurn([]) });
    db.seed(THREAD, [ingress("remember the canary")]);

    await runExtractPass(deps, THREAD);

    const call = provider.received[0];
    expect(call?.model).toBe("sleep"); // "fake/sleep" with the routing prefix stripped
    expect(call?.system).toBe(EXTRACT_SYSTEM);
    expect(call?.tools.map((t) => t.name)).toEqual([EXTRACT_TOOL_NAME]);
    expect(call?.toolChoice).toEqual({ type: "tool", name: EXTRACT_TOOL_NAME });
    expect(call?.messages).toEqual([{ role: "user", text: "[1] user cli:u1: remember the canary" }]);
  });

  test("zero candidates: the decide call is skipped, but the receipt still lands", async () => {
    const { deps, db, provider, memory } = harness({ [EXTRACT_TOOL_NAME]: extractTurn([]) });
    db.seed(THREAD, [ingress("nothing worth keeping")]);

    const result = await runExtractPass(deps, THREAD);

    // Without the receipt the cursor would not move and every later sweep
    // would re-extract these events at full price.
    expect(provider.received).toHaveLength(1);
    expect(memory.searches).toHaveLength(0);
    expect(result).toMatchObject({ status: "done" });
    expect(receipts(db).at(-1)).toMatchObject({
      candidates: 0,
      added: 0,
      updated: 0,
      invalidated: 0,
      noop: 0,
      scanned: 1,
    });
  });

  test("the decide call forces its tool and carries index/neighbour ids", async () => {
    const neighbor = makeMemoryHit({ text: "Brad likes long answers" });
    const { deps, db, provider } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("terse please")]);

    await runExtractPass(deps, THREAD);

    const call = provider.received[1];
    expect(call?.system).toBe(DECIDE_SYSTEM);
    expect(call?.tools.map((t) => t.name)).toEqual([DECIDE_TOOL_NAME]);
    expect(call?.toolChoice).toEqual({ type: "tool", name: DECIDE_TOOL_NAME });
    const payload = payloadOf(call as CompleteOptions);
    expect(payload.candidates).toHaveLength(1);
    expect(payload.candidates[0]).toMatchObject({ index: 0, text: CANDIDATE.text, visibility: "channel" });
    expect(payload.candidates[0]?.neighbors).toEqual([
      {
        id: neighbor.id,
        text: neighbor.text,
        kind: neighbor.kind,
        importance: neighbor.importance,
        recordedAt: neighbor.recordedAt,
      },
    ]);
  });

  test("a forced tool the model ignored fails the pass", async () => {
    const { deps, db } = harness({ [EXTRACT_TOOL_NAME]: textTurn("I would rather chat") });
    db.seed(THREAD, [ingress("hi")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("failed");
    expect(result).toMatchObject({ error: expect.stringContaining("returned no call to it") });
  });

  test("an invalid tool call fails the pass, naming the field", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, importance: 99 }]),
    });
    db.seed(THREAD, [ingress("hi")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result).toMatchObject({
      status: "failed",
      error: expect.stringContaining("candidates[0].importance"),
    });
    expect(memory.retains).toHaveLength(0);
    expect(receipts(db)).toHaveLength(0);
  });
});

describe("runExtractPass — visibility (DESIGN.md §5.1)", () => {
  const userCandidate = { ...CANDIDATE, visibility: "user", userId: "brad" };

  test("a `user` candidate about a transcript author survives on a wide surface", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([userCandidate]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("terse please", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.retains[0]).toMatchObject({
      visibility: "user",
      userId: "brad",
      channelId: THREAD.channelId,
      agentId: "pinky",
    });
  });

  test("a `user` candidate about someone who never spoke is downgraded to channel", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([{ ...userCandidate, userId: "someone-else" }]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("terse please", "brad")]);

    await runExtractPass(deps, THREAD);

    // An invented userId would write a row no scope predicate ever matches.
    expect(memory.retains[0]).toMatchObject({ visibility: "channel" });
    expect(memory.retains[0]?.userId).toBeUndefined();
  });

  test("a narrow surface cannot mint a `user` row at all", async () => {
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([userCandidate]), [DECIDE_TOOL_NAME]: decideAdd },
      { scope: NARROW_SCOPE },
    );
    db.seed(THREAD, [ingress("terse please", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.retains[0]).toMatchObject({ visibility: "channel" });
    expect(memory.retains[0]?.userId).toBeUndefined();
  });

  test("a userId on a non-user candidate is dropped", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, visibility: "tenant", userId: "brad" }]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.retains[0]).toMatchObject({ visibility: "tenant", channelId: THREAD.channelId });
    expect(memory.retains[0]?.userId).toBeUndefined();
  });
});

describe("runExtractPass — neighbour scope", () => {
  test("includePrivate is always false; includeUser only with a userId", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([
        CANDIDATE,
        { ...CANDIDATE, text: "Brad is in Sydney", visibility: "user", userId: "brad" },
      ]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hello", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.searches).toHaveLength(2);
    expect(memory.searches[0]?.scope).toEqual({
      agentId: "pinky",
      channelId: THREAD.channelId,
      includeUser: false,
      includePrivate: false,
    });
    expect(memory.searches[0]?.query).toBe(CANDIDATE.text);
    expect(memory.searches[0]?.limit).toBe(10);
    expect(memory.searches[1]?.scope).toEqual({
      agentId: "pinky",
      channelId: THREAD.channelId,
      userId: "brad",
      includeUser: true,
      includePrivate: false,
    });
  });

  test("a narrow surface never widens the neighbour search either", async () => {
    const { deps, db, memory } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, visibility: "user", userId: "brad" }]),
        [DECIDE_TOOL_NAME]: decideAdd,
      },
      { scope: NARROW_SCOPE },
    );
    db.seed(THREAD, [ingress("hello", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.searches[0]?.scope.includeUser).toBe(false);
    expect(memory.searches[0]?.scope.includePrivate).toBe(false);
  });
});

describe("runExtractPass — the placement guard (DESIGN.md §5.1)", () => {
  /** UPDATE (or DELETE) candidate 0 against its first neighbour. */
  const decideAgainstNeighbor = (action: "UPDATE" | "DELETE") => (opts: CompleteOptions) =>
    toolTurn(DECIDE_TOOL_NAME, {
      decisions: [
        {
          candidate: 0,
          action,
          target: payloadOf(opts).candidates[0]?.neighbors[0]?.id,
          ...(action === "UPDATE" ? { text: "merged wording" } : {}),
        },
      ],
    });

  test("UPDATE is allowed when the target lives exactly where the candidate does", async () => {
    const neighbor = makeMemoryHit({ visibility: "channel", channelId: THREAD.channelId });
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE") },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates).toHaveLength(1);
    expect(memory.updates[0]?.replacement).toMatchObject({
      visibility: "channel",
      channelId: THREAD.channelId,
      text: "merged wording",
    });
    expect(receipts(db).at(-1)).toMatchObject({ updated: 1, noop: 0 });
  });

  test("NARROWING is refused: a channel candidate may not rewrite a tenant row", async () => {
    // scopePredicate always shows tenant/global rows, so this neighbour is a
    // routine sight — and rewriting it at channel scope would make a
    // tenant-visible fact invisible everywhere but this conversation.
    const neighbor = makeMemoryHit({ visibility: "tenant", channelId: null });
    const { deps, db, memory, logs } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE") },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates).toHaveLength(0);
    expect(memory.written).toHaveLength(0);
    expect(db.dataFor(THREAD).filter((d) => d.type === "memory")).toHaveLength(0);
    expect(receipts(db).at(-1)).toMatchObject({ updated: 0, noop: 1, candidates: 1 });
    expect(logs.join("\n")).toContain(neighbor.id);
    expect(logs.join("\n")).toContain("it is tenant but the candidate is channel c1");
  });

  test("WIDENING is refused: a tenant candidate may not rewrite a channel row", async () => {
    // The other direction, and the worse one: it republishes one
    // conversation's content tenant-wide.
    const neighbor = makeMemoryHit({ visibility: "channel", channelId: THREAD.channelId });
    const { deps, db, memory, logs } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, visibility: "tenant" }]),
        [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE"),
      },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates).toHaveLength(0);
    expect(receipts(db).at(-1)).toMatchObject({ updated: 0, noop: 1 });
    expect(logs.join("\n")).toContain("it is channel c1 but the candidate is tenant");
  });

  test("a channel row in ANOTHER channel is refused too", async () => {
    const neighbor = makeMemoryHit({ visibility: "channel", channelId: "some-other-channel" });
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE") },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates).toHaveLength(0);
    expect(receipts(db).at(-1)).toMatchObject({ noop: 1 });
  });

  test("DELETE obeys the same guard — an out-of-scope row is never retired", async () => {
    const neighbor = makeMemoryHit({ visibility: "tenant", channelId: null });
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAgainstNeighbor("DELETE") },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.invalidations).toHaveLength(0);
    expect(receipts(db).at(-1)).toMatchObject({ invalidated: 0, noop: 1 });
  });

  test("DELETE is allowed for an equally placed row", async () => {
    const neighbor = makeMemoryHit({ visibility: "channel", channelId: THREAD.channelId });
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAgainstNeighbor("DELETE") },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.invalidations).toEqual([
      { id: neighbor.id, reason: `sleep:extract contradicted by: ${CANDIDATE.text}` },
    ]);
    expect(receipts(db).at(-1)).toMatchObject({ invalidated: 1, noop: 0 });
  });

  test("a `user` candidate may only touch that same user's rows", async () => {
    const neighbor = makeMemoryHit({ visibility: "user", userId: "someone-else", channelId: THREAD.channelId });
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, visibility: "user", userId: "brad" }]),
      [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE"),
    }, {}, { hits: [neighbor] });
    db.seed(THREAD, [ingress("hi", "brad")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates).toHaveLength(0);
    expect(receipts(db).at(-1)).toMatchObject({ noop: 1 });
  });

  test("an allowed UPDATE takes its placement from the TARGET, not the candidate", async () => {
    // Belt and braces on top of the guard: an update is "the same fact, better
    // detail", so it never moves house — including the channel it was learned in.
    const neighbor = makeMemoryHit({ visibility: "tenant", channelId: "learned-elsewhere" });
    const { deps, db, memory } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([{ ...CANDIDATE, visibility: "tenant" }]),
        [DECIDE_TOOL_NAME]: decideAgainstNeighbor("UPDATE"),
      },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.updates[0]?.replacement).toMatchObject({
      visibility: "tenant",
      channelId: "learned-elsewhere",
    });
  });
});

describe("runExtractPass — apply and receipt", () => {
  const three = [
    { ...CANDIDATE, text: "new fact" },
    { ...CANDIDATE, text: "better wording" },
    { ...CANDIDATE, text: "contradicts an old one" },
  ];

  /** ADD / UPDATE / DELETE / NOOP across four candidates. */
  const decideMixed = (opts: CompleteOptions): AssistantTurn => {
    const payload = payloadOf(opts);
    return toolTurn(DECIDE_TOOL_NAME, {
      decisions: [
        { candidate: 0, action: "ADD" },
        {
          candidate: 1,
          action: "UPDATE",
          target: payload.candidates[1]?.neighbors[0]?.id,
          text: "merged wording",
        },
        { candidate: 2, action: "DELETE", target: payload.candidates[2]?.neighbors[0]?.id },
        { candidate: 3, action: "NOOP" },
      ],
    });
  };

  test("counts, writes and audit events line up, with the receipt LAST", async () => {
    const neighbor = makeMemoryHit({ text: "an older statement" });
    const { deps, db, memory } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([...three, { ...CANDIDATE, text: "already known" }]),
        [DECIDE_TOOL_NAME]: decideMixed,
      },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("a busy turn")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("done");
    expect(memory.retains).toHaveLength(1);
    expect(memory.updates).toEqual([
      expect.objectContaining({ id: neighbor.id }),
    ]);
    expect(memory.updates[0]?.replacement).toMatchObject({ text: "merged wording" });
    expect(memory.invalidations[0]).toEqual({
      id: neighbor.id,
      reason: "sleep:extract contradicted by: contradicts an old one",
    });

    // The appended batch: one audit `memory` event per write, then the receipt.
    const appended = db.dataFor(THREAD).slice(1);
    expect(appended.map((d) => d.type)).toEqual(["memory", "memory", "memory", "sleep"]);
    expect(appended.map((d) => (d.type === "memory" ? d.op : d.type))).toEqual([
      "retain",
      "update",
      "invalidate",
      "sleep",
    ]);
    expect(appended.at(-1)).toMatchObject({
      type: "sleep",
      phase: "extract",
      candidates: 4,
      added: 1,
      updated: 1,
      invalidated: 1,
      noop: 1,
      model: "fake/sleep",
      ms: 0,
    });
  });

  test("an UPDATE journals both ids; a DELETE journals its reason", async () => {
    const neighbor = makeMemoryHit({ text: "an older statement" });
    const { deps, db } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([...three, { ...CANDIDATE, text: "already known" }]),
        [DECIDE_TOOL_NAME]: decideMixed,
      },
      {},
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("a busy turn")]);

    await runExtractPass(deps, THREAD);

    const events = db.dataFor(THREAD).filter((d): d is Extract<ThreadEventData, { type: "memory" }> => d.type === "memory");
    expect(events[1]?.ids).toHaveLength(2);
    expect(events[1]?.ids[0]).toBe(neighbor.id);
    expect(events[2]).toMatchObject({
      op: "invalidate",
      ids: [neighbor.id],
      text: "sleep:extract contradicted by: contradicts an old one",
    });
  });

  test("provenance meta names the source and the range", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.retains[0]?.meta).toEqual({
      source: "sleep:extract",
      channelId: THREAD.channelId,
      threadId: THREAD.threadId,
      fromSeq: 1,
      toSeq: 1,
    });
  });

  test("an invalidate that lost a race is logged, not counted", async () => {
    const neighbor = makeMemoryHit({});
    const { deps, db, logs } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
        [DECIDE_TOOL_NAME]: (opts) =>
          toolTurn(DECIDE_TOOL_NAME, {
            decisions: [
              { candidate: 0, action: "DELETE", target: payloadOf(opts).candidates[0]?.neighbors[0]?.id },
            ],
          }),
      },
      {},
      { hits: [neighbor], invalidateReturns: false },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    // The receipt records what HAPPENED, not what was decided.
    expect(receipts(db).at(-1)).toMatchObject({ invalidated: 0, candidates: 1 });
    expect(db.dataFor(THREAD).filter((d) => d.type === "memory")).toHaveLength(0);
    expect(logs.join("\n")).toContain("already invalidated");
  });

  test("usage sums both calls; a counter nobody reported stays absent", async () => {
    const { deps, db } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE], {
        usage: { input: 10, output: 2, cacheRead: 5 },
      }),
      [DECIDE_TOOL_NAME]: (opts) => ({ ...decideAdd(opts), usage: { input: 20, output: 3 } }),
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(receipts(db).at(-1)?.usage).toEqual({ input: 30, output: 5, cacheRead: 5 });
  });

  test("usage is ABSENT when neither call reported any", async () => {
    const { deps, db } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    const receipt = receipts(db).at(-1);
    expect(receipt).toBeDefined();
    expect(receipt && "usage" in receipt).toBe(false);
  });
});

describe("runExtractPass — the claim", () => {
  test("one transaction: lock, re-read the cursor, write, append", async () => {
    const { deps, db } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(db.txLog).toEqual(["begin", "commit"]);
    const inTx = db.calls.filter((c) => c.txDepth > 0).map((c) => c.sql.replace(/\s+/g, " "));
    // The lock comes FIRST, and the cursor re-check comes after it — the whole
    // point is that the second read happens while the row is held.
    expect(inTx[0]).toContain("insert into threads");
    expect(inTx[1]).toContain("for update");
    expect(inTx[2]).toContain("data->>'toSeq'");
    expect(inTx.some((s) => s.includes("insert into events"))).toBe(true);
    // Every memory write ran on the tx handle, not the outer pool.
    expect(db.txLog).toEqual(["begin", "commit"]);
  });

  test("memory writes are bound to the transaction handle", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(memory.boundTo).toHaveLength(1);
    expect(memory.boundTo[0]).toBe(db);
  });

  test("lost claim: a concurrent pass won, so this one writes NOTHING", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);
    // The cursor is 0 on the first read (before the LLM calls) and 9 on the
    // re-read under the lock: exactly the interleaving two sweeps produce.
    let reads = 0;
    db.routes.push({
      pattern: /data->>'toSeq'/,
      respond: () => (reads++ === 0 ? [] : [{ to_seq: "9" }]),
    });

    const result = await runExtractPass(deps, THREAD);

    expect(result).toEqual({ status: "skipped", reason: "lost-claim" });
    // The transaction still commits — it just wrote nothing.
    expect(db.txLog).toEqual(["begin", "commit"]);
    expect(memory.retains).toHaveLength(0);
    expect(db.all(/insert into events/)).toHaveLength(0);
    expect(receipts(db)).toHaveLength(0);
  });
});

describe("runExtractPass — failure", () => {
  test("a store failure rolls the pass back and journals ONE error event", async () => {
    const { deps, db, memory, logs } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      {},
      { failOn: { op: "retain", message: "memories table is on fire" } },
    );
    db.seed(THREAD, [ingress("hi")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result).toEqual({ status: "failed", error: "memories table is on fire" });
    // The pass's transaction rolled back; the error event is its OWN append
    // afterwards, or it would roll back with everything else and leave no trace.
    expect(db.txLog).toEqual(["begin", "rollback", "begin", "commit"]);
    const errors = db.dataFor(THREAD).filter((d) => d.type === "error");
    expect(errors).toEqual([
      { type: "error", source: "sleep", message: "memories table is on fire", count: 1 },
    ]);
    expect(receipts(db)).toHaveLength(0);
    expect(memory.written).toHaveLength(0);
    expect(logs.join("\n")).toContain("memories table is on fire");
  });

  test("the error event is the newest event, which is what throttles the retry", async () => {
    const { deps, db } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      {},
      { failOn: { op: "retain", message: "nope" } },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    // The idle gate is the backoff (DESIGN.md §5.3 / contract §3.2 step 8).
    expect(db.dataFor(THREAD).at(-1)).toMatchObject({ type: "error", source: "sleep" });
  });

  test("an aborted pass journals no error event", async () => {
    const controller = new AbortController();
    const { deps, db } = harness(
      {
        [EXTRACT_TOOL_NAME]: () => {
          controller.abort();
          throw new Error("aborted");
        },
      },
      { signal: controller.signal },
    );
    db.seed(THREAD, [ingress("hi")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("failed");
    // Shutdown is not a broken thread, and the append would likely fail anyway
    // against a closing pool.
    expect(db.dataFor(THREAD).filter((d) => d.type === "error")).toHaveLength(0);
  });
});

describe("runExtractPass — embeddings degrade, never fail (DESIGN.md §5.5)", () => {
  test("with an embedder, one call covers every candidate and the vector is stored", async () => {
    const embedder = makeEmbedder();
    const { deps, db, memory } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE, { ...CANDIDATE, text: "a second fact" }]),
        [DECIDE_TOOL_NAME]: decideAdd,
      },
      { embedder },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(embedder.calls).toHaveLength(1);
    expect(embedder.calls[0]).toEqual([CANDIDATE.text, "a second fact"]);
    expect(memory.searches[0]?.queryEmbedding).toBeDefined();
    expect(memory.retains[0]?.embedding).toBeDefined();
    expect(memory.retains[0]?.embeddingModel).toBe("fake/embed");
  });

  test("a failing embedder logs once and the pass continues FTS-only", async () => {
    const embedder = makeEmbedder({ fail: "embedding api down" });
    const { deps, db, memory, logs } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      { embedder },
    );
    db.seed(THREAD, [ingress("hi")]);

    const result = await runExtractPass(deps, THREAD);

    expect(result.status).toBe("done");
    expect(memory.searches[0]?.queryEmbedding).toBeUndefined();
    expect(memory.retains[0]?.embedding).toBeUndefined();
    expect(logs.filter((l) => l.includes("embedding api down"))).toHaveLength(1);
  });

  test("no pgvector column: nothing is embedded, and the embedder is never called", async () => {
    const embedder = makeEmbedder();
    const { deps, db, memory } = harness(
      { [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]), [DECIDE_TOOL_NAME]: decideAdd },
      { embedder },
      { supportsVectors: false },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(embedder.calls).toHaveLength(0);
    expect(memory.retains[0]?.embedding).toBeUndefined();
  });

  test("no embedder at all is the ordinary degraded mode", async () => {
    const { deps, db, memory } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    expect((await runExtractPass(deps, THREAD)).status).toBe("done");
    expect(memory.retains[0]?.embedding).toBeUndefined();
  });

  test("an UPDATE whose wording changed is re-embedded; identical wording is not", async () => {
    const embedder = makeEmbedder();
    const neighbor = makeMemoryHit({});
    const { deps, db } = harness(
      {
        [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
        [DECIDE_TOOL_NAME]: (opts) =>
          toolTurn(DECIDE_TOOL_NAME, {
            decisions: [
              {
                candidate: 0,
                action: "UPDATE",
                target: payloadOf(opts).candidates[0]?.neighbors[0]?.id,
                text: "a different merged wording",
              },
            ],
          }),
      },
      { embedder },
      { hits: [neighbor] },
    );
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    expect(embedder.calls).toEqual([[CANDIDATE.text], ["a different merged wording"]]);
  });
});

describe("runExtractPass — nothing it writes is model-visible", () => {
  test("every appended event is an audit type (DESIGN.md §3)", async () => {
    const { deps, db } = harness({
      [EXTRACT_TOOL_NAME]: extractTurn([CANDIDATE]),
      [DECIDE_TOOL_NAME]: decideAdd,
    });
    db.seed(THREAD, [ingress("hi")]);

    await runExtractPass(deps, THREAD);

    // `memory` events with no `block` key and `sleep` receipts are both skipped
    // by buildContext, so the rendered prompt is identical before and after.
    for (const data of db.dataFor(THREAD).slice(1)) {
      expect(["memory", "sleep"]).toContain(data.type);
      if (data.type === "memory") expect("block" in data).toBe(false);
    }
  });
});

describe("assistant helper coverage", () => {
  test("assistant turns are extractable material", async () => {
    const { deps, db, provider } = harness({ [EXTRACT_TOOL_NAME]: extractTurn([]) });
    db.seed(THREAD, [
      ingress("do the thing"),
      assistant("on it", [{ id: "c1", name: "bash", args: { cmd: "ls" } }]),
      { type: "tool_result", callId: "c1", name: "bash", text: "a b c", isError: false },
    ]);

    await runExtractPass(deps, THREAD);

    expect(provider.received[0]?.messages[0]?.text).toBe(
      ["[1] user cli:u1: do the thing", "[2] assistant: on it", '  -> bash({"cmd":"ls"})', "[3] tool bash: a b c"].join("\n"),
    );
  });
});
