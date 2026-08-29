/**
 * Projection tests (DESIGN.md §3): the boundary rule and the cut-point-safety
 * rules that keep a post-restart prompt acceptable to providers (§4.5).
 */
import { describe, expect, it } from "bun:test";
import {
  buildContext,
  estimateTokens,
  latestContinuity,
  serializeContinuity,
  windowRecall,
} from "../src/projection";
import { canonicalizeArgs } from "../src/events";
import type { ContinuityDoc, ThreadEvent, ThreadEventData, ThreadRef } from "../src/events";

const ref: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

let nextSeq = 0;
function ev(data: ThreadEventData, seq?: number): ThreadEvent {
  const s = seq ?? ++nextSeq;
  nextSeq = s;
  return { ...ref, id: `e${s}`, seq: s, ts: "2026-08-28T00:00:00Z", data };
}

function ingress(text: string, seq?: number): ThreadEvent {
  return ev(
    { type: "ingress", platform: "cli", author: { platform: "cli", userId: "u1" }, text, refs: [] },
    seq,
  );
}

function assistant(text: string, calls: { id: string; name: string }[] = [], seq?: number): ThreadEvent {
  return ev(
    {
      type: "message",
      role: "assistant",
      text,
      toolCalls: calls.map((c) => ({ ...c, args: {} })),
      model: "fake/m",
    },
    seq,
  );
}

function toolResult(callId: string, text: string, seq?: number): ThreadEvent {
  return ev({ type: "tool_result", callId, name: "echo", text, isError: false }, seq);
}

function notice(text: string, seq?: number): ThreadEvent {
  return ev({ type: "notice", text }, seq);
}

/**
 * A recall event. `block` is a string (`""` included) for the loop's auto-recall
 * pass, which ALWAYS journals the key; `null` reproduces the agent-initiated
 * `recall` tool, which never writes one and so never opens a window.
 */
function recall(block: string | null, seq?: number): ThreadEvent {
  return ev(
    {
      type: "memory",
      op: "recall",
      ids: block ? ["m1"] : [],
      text: "seed query",
      count: 1,
      ...(block !== null ? { block } : {}),
    },
    seq,
  );
}

const BLOCK = "[harness notice] Recalled memories\n<memories>\n- widgets cost $4\n</memories>";

const doc: ContinuityDoc = {
  goal: "ship the continuity engine",
  plan: { done: ["read design"], now: "write the loop", next: ["tests"] },
  workingSet: { files: ["/a.ts"], urls: [] },
  decisions: [{ what: "shed runs last", why: "cut-point safety" }],
  openLoops: ["is 90% the right hard fraction?"],
  lessons: ["never mutate the system prompt mid-run"],
  memoryHints: ["continuity ladder"],
  mood: "focused",
};

function continuity(seq?: number): ThreadEvent {
  return ev({ type: "continuity", document: doc, tokensBefore: 1234 }, seq);
}

describe("serializeContinuity", () => {
  it("renders every populated section in a stable order", () => {
    const out = serializeContinuity(doc);
    expect(out.startsWith("# Pinky Continuity")).toBe(true);
    expect(out).toContain("**Goal:** ship the continuity engine");
    expect(out).toContain("- now: write the loop");
    expect(out).toContain("- file: /a.ts");
    expect(out).toContain("- shed runs last (because: cut-point safety)");
    expect(out).toContain("**Mood:** focused");
  });
});

describe("latestContinuity", () => {
  it("finds the newest continuity event", () => {
    nextSeq = 0;
    const events = [continuity(), ingress("hi"), continuity()];
    expect(latestContinuity(events)?.seq).toBe(3);
    expect(latestContinuity([ingress("hi", 1)])).toBeNull();
  });
});

describe("buildContext", () => {
  it("renders ingress, assistant, and paired tool results", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("hello"),
      assistant("", [{ id: "c1", name: "echo" }]),
      toolResult("c1", "echoed"),
      assistant("done"),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant"]);
    expect(msgs[0]!.text).toBe("[cli u1]: hello");
    expect(msgs[1]!.toolCalls?.map((c) => c.id)).toEqual(["c1"]);
    expect(msgs[2]!.toolCallId).toBe("c1");
  });

  it("drops everything before the latest continuity boundary", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("ancient history"),
      assistant("old chatter"),
      continuity(),
      ingress("fresh question"),
    ]);
    expect(msgs).toHaveLength(2);
    expect(msgs[0]!.role).toBe("user");
    expect(msgs[0]!.text).toContain("# Pinky Continuity");
    expect(msgs[1]!.text).toBe("[cli u1]: fresh question");
    expect(msgs.some((m) => m.text.includes("ancient history"))).toBe(false);
  });

  it("drops the orphan tool_result left behind by a shed, and never starts with a tool message", () => {
    // Exact event order the loop produces: assistant(tool call) -> continuity
    // (emitted during execution) -> tool_result for the shed call.
    nextSeq = 0;
    const events = [
      ingress("do the thing"),
      assistant("", [{ id: "shed1", name: "shed_context" }]),
      continuity(),
      toolResult("shed1", "continuity written"),
    ];
    const msgs = buildContext(events);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.role).toBe("user");
    expect(msgs[0]!.text).toContain("# Pinky Continuity");
  });

  it("drops an assistant tool call that has no result in the window (aborted mid-tool)", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("go"),
      assistant("thinking out loud", [{ id: "dangling", name: "echo" }]),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(msgs[1]!.toolCalls).toBeUndefined();
    expect(msgs[1]!.text).toBe("thinking out loud");
  });

  it("skips an assistant message left with neither text nor calls", () => {
    nextSeq = 0;
    const msgs = buildContext([ingress("go"), assistant("", [{ id: "dangling", name: "echo" }])]);
    expect(msgs.map((m) => m.role)).toEqual(["user"]);
  });

  // DESIGN.md §7 wake-on-message: a peer's message is the whole reason a run
  // was woken (issue #4), so it has to be IN the prompt. It is the only record
  // the run has — the mailbox row is not part of the projection.
  it("renders an a2a event as a user turn", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ev({
        type: "a2a",
        from: "weather@node2",
        to: "pinky@local",
        kind: "request",
        text: "what is the forecast?",
        msgId: "m-1",
      }),
    ]);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.role).toBe("user");
    // The address is spelled out so the model can answer the right peer.
    expect(msgs[0]!.text).toBe("[a2a request from weather@node2]: what is the forecast?");
  });

  it("interleaves a2a arrivals with ingress and drops them before a boundary", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ev({ type: "a2a", from: "old@n", to: "pinky@local", kind: "message", text: "ancient", msgId: "m-0" }),
      continuity(),
      ev({ type: "a2a", from: "weather@n", to: "pinky@local", kind: "response", text: "sunny", msgId: "m-1" }),
      assistant("thanks"),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "user", "assistant"]);
    expect(msgs[1]!.text).toBe("[a2a response from weather@n]: sunny");
    expect(msgs.some((m) => m.text.includes("ancient"))).toBe(false);
  });

  it("skips audit-only event types", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("hi"),
      ev({ type: "decision", action: "reply", reason: "addressed" }),
      ev({ type: "egress", target: { kind: "thread" }, text: "sent" }),
      ev({ type: "error", source: "tool", message: "boom", count: 1 }),
      ev({ type: "checkpoint", ref: "ck1" }),
    ]);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.text).toBe("[cli u1]: hi");
  });

  // DESIGN.md P8 (revised): a settings change the agent made through
  // `settings_set` is journaled for audit, but it is not conversation. It also
  // must not read back as an instruction on a later turn — the new value
  // reaches the agent as the next run's snapshot, not as a message.
  it("skips config events (audit-only, DESIGN.md P8 revised)", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("turn your advisory threshold down"),
      ev({
        type: "config",
        scope: "agent:pinky",
        key: "context.advisoryFraction",
        value: 0.6,
        previous: 0.7,
        by: "pinky",
      }),
    ]);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.text).toBe("[cli u1]: turn your advisory threshold down");
  });

  // DESIGN.md §5.3: memory writes are journaled for audit and never rendered.
  // A recall with no `block` injected nothing into the prompt, so it has
  // nothing to replay either.
  it("skips memory events with no injected block (audit-only, DESIGN.md §5.3)", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("what do you know about me?"),
      ev({ type: "memory", op: "recall", ids: ["m1", "m2"], text: "about me", count: 2 }),
      ev({ type: "memory", op: "retain", ids: ["m3"], text: "Brad prefers terse answers" }),
    ]);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.text).toBe("[cli u1]: what do you know about me?");
  });

  // DESIGN.md §13 (cost model): the restart accounting sits in the log right
  // after the boundary it measures, which is inside every window built from
  // that boundary. Rendering it would put the harness's own bookkeeping in
  // front of the model on every turn of the successor — and re-injecting the
  // token counts a restart was meant to shed.
  it("skips restart events (audit-only, DESIGN.md §13)", () => {
    nextSeq = 0;
    const msgs = buildContext([
      continuity(1),
      ev({
        type: "restart",
        boundarySeq: 1,
        tokensBefore: 120_000,
        tokensAfter: 3_400,
        recallTokens: 800,
        messages: 2,
      }),
      ingress("keep going"),
    ]);
    expect(msgs).toHaveLength(2);
    expect(msgs[0]!.text.startsWith("# Pinky Continuity")).toBe(true);
    expect(msgs[1]!.text).toBe("[cli u1]: keep going");
  });

  // The pressure notices are harness-authored conversation (DESIGN.md §4.1).
  // They are journaled so the next wake's projection reproduces them in the
  // same slot; a notice that lived only in the run's in-memory array would
  // vanish here and the transcript would diverge from what the provider cached.
  it("renders a notice as a user message in seq order", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("keep going"),
      notice("[harness notice] context pressure: this window is filling up."),
      assistant("shedding soon"),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "user", "assistant"]);
    expect(msgs[1]!.text).toBe("[harness notice] context pressure: this window is filling up.");
  });

  it("renders two notices in order, with no dedupe", () => {
    // Both forced attempts at the hard rung: the model saw two, so does the
    // successor. Collapsing them would shorten the transcript the provider
    // already cached.
    nextSeq = 0;
    const msgs = buildContext([notice("first"), assistant("no"), notice("final attempt")]);
    expect(msgs.map((m) => m.text)).toEqual(["first", "no", "final attempt"]);
  });

  // DESIGN.md §5.4: the <memories> block goes in at context start, which is
  // literally index 0 — ahead of the continuity document, where the loop
  // unshifted it. Replaying it from the log is what lets the loop skip a
  // second live search on the next wake (§4.5 cache alignment).
  it("hoists the window's recall block to index 0, ahead of the continuity doc", () => {
    nextSeq = 0;
    const msgs = buildContext([continuity(1), recall(BLOCK, 2), ingress("carry on", 3)]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "user", "user"]);
    expect(msgs[0]!.text).toBe(BLOCK);
    expect(msgs[1]!.text).toContain("# Pinky Continuity");
    expect(msgs[2]!.text).toBe("[cli u1]: carry on");
  });

  it("hoists it on a thread with no continuity boundary at all", () => {
    nextSeq = 0;
    const msgs = buildContext([ingress("hello", 1), recall(BLOCK, 2), assistant("hi", [], 3)]);
    expect(msgs.map((m) => m.text)).toEqual([BLOCK, "[cli u1]: hello", "hi"]);
  });

  it("renders only the FIRST recall block in a window; later ones stay audit-only", () => {
    // A mid-window recall (the agent called the `recall` tool, or a successor
    // pass wrote another) must not rewrite byte 0 of a prompt the provider has
    // already cached for this window.
    nextSeq = 0;
    const msgs = buildContext([
      continuity(1),
      recall(BLOCK, 2),
      ingress("and now?", 3),
      recall("A DIFFERENT BLOCK", 4),
    ]);
    expect(msgs).toHaveLength(3);
    expect(msgs[0]!.text).toBe(BLOCK);
    expect(msgs.some((m) => m.text === "A DIFFERENT BLOCK")).toBe(false);
  });

  it("an agent-tool recall (no `block` key) is skipped, and does not shadow a later one", () => {
    nextSeq = 0;
    const msgs = buildContext([recall(null, 1), ingress("hello", 2), recall(BLOCK, 3)]);
    expect(msgs.map((m) => m.text)).toEqual([BLOCK, "[cli u1]: hello"]);
  });

  it("an auto-recall that injected nothing opens the window with no block at all", () => {
    // `block: ""` is the loop's receipt that auto-recall already ran here. It
    // renders nothing AND it claims the window, so a later block in the same
    // window cannot be hoisted into index 0 of a prompt the provider cached.
    nextSeq = 0;
    const msgs = buildContext([recall("", 1), ingress("hello", 2), recall(BLOCK, 3)]);
    expect(msgs.map((m) => m.text)).toEqual(["[cli u1]: hello"]);
  });

  it("a pre-boundary recall block does not leak into the new window", () => {
    // The restart is exactly where a stale block must die: the successor's
    // window opens on the continuity event and gets its own recall pass.
    nextSeq = 0;
    const msgs = buildContext([
      recall("OLD BLOCK", 1),
      ingress("ancient history", 2),
      continuity(3),
      ingress("fresh question", 4),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "user"]);
    expect(msgs[0]!.text).toContain("# Pinky Continuity");
    expect(msgs.some((m) => m.text === "OLD BLOCK")).toBe(false);
  });

  it("the block is hoisted over a window whose leading tool message was trimmed", () => {
    // The shed's orphan tool_result is dropped and the trim runs BEFORE the
    // hoist, so the rule stays "never open with a tool message" and the block
    // still lands at index 0.
    nextSeq = 0;
    const msgs = buildContext([
      assistant("", [{ id: "shed1", name: "shed_context" }], 1),
      continuity(2),
      toolResult("shed1", "continuity written", 3),
      recall(BLOCK, 4),
    ]);
    expect(msgs.map((m) => m.role)).toEqual(["user", "user"]);
    expect(msgs[0]!.text).toBe(BLOCK);
    expect(msgs[1]!.text).toContain("# Pinky Continuity");
  });
});

// The loop's "has auto-recall already run in this window?" question. Same
// boundary rule as buildContext, so the two can never disagree about index 0.
describe("windowRecall", () => {
  it("returns the first auto-recall after the boundary, else null", () => {
    nextSeq = 0;
    expect(windowRecall([continuity(1), recall(BLOCK, 2), recall("later", 3)])).toEqual({
      ran: true,
      block: BLOCK,
    });
    expect(windowRecall([recall("OLD BLOCK", 1), continuity(2)])).toBeNull();
    expect(windowRecall([])).toBeNull();
  });

  it("an agent-tool recall (no `block` key) never claims the window", () => {
    // Otherwise the agent could close the loop's gate by calling `recall`, and
    // auto-recall would silently stop running for the rest of the window.
    nextSeq = 0;
    expect(windowRecall([recall(null, 1), ingress("hi", 2)])).toBeNull();
    expect(windowRecall([recall(null, 1), recall(BLOCK, 2)])).toEqual({ ran: true, block: BLOCK });
  });

  it("distinguishes 'ran and injected nothing' from 'never ran'", () => {
    // The whole point of `ran`: an empty memory plane journals `block: ""`, and
    // the loop must read that as "done here" — not as "no block yet, recall
    // again", which would unshift a block at index 0 on the next wake and cold-
    // start the provider's prefix cache for the whole thread (DESIGN.md §4.5).
    nextSeq = 0;
    expect(windowRecall([recall("", 1), ingress("hi", 2)])).toEqual({ ran: true, block: "" });
    expect(windowRecall([ingress("hi", 1)])).toBeNull();
  });

  it("carries the scope the block was built under, when the event has one", () => {
    nextSeq = 0;
    const scoped = ev(
      {
        type: "memory",
        op: "recall",
        ids: ["m1"],
        text: "seed query",
        count: 1,
        block: BLOCK,
        scope: { includeUser: true, includePrivate: true },
      },
      1,
    );
    expect(windowRecall([scoped])).toEqual({
      ran: true,
      block: BLOCK,
      scope: { includeUser: true, includePrivate: true },
    });
    // Absent on older events; the loop treats that as "unknown, replay as-is".
    expect(Object.keys(windowRecall([recall(BLOCK, 2)])!)).not.toContain("scope");
  });

  it("agrees with buildContext about index 0", () => {
    nextSeq = 0;
    const events = [continuity(1), recall(BLOCK, 2), ingress("carry on", 3)];
    expect(buildContext(events)[0]!.text).toBe(windowRecall(events)!.block);
    // And when the opening pass injected nothing, neither of them puts a
    // memories message anywhere.
    const empty = [continuity(4), recall("", 5), ingress("carry on", 6)];
    expect(windowRecall(empty)).toEqual({ ran: true, block: "" });
    expect(buildContext(empty)[0]!.text).toContain("# Pinky Continuity");
  });
});

// jsonb does not preserve object key order: it sorts keys by (length, bytes).
// The tool-call arguments the loop rendered in-run would therefore come back
// reordered on the next wake and break the provider's prefix match at that
// block (DESIGN.md §4.5). Both ends canonicalize instead.
describe("canonicalizeArgs", () => {
  it("sorts object keys, recursively, leaving arrays and primitives alone", () => {
    const out = canonicalizeArgs({
      zulu: 1,
      a: { yankee: [3, 1, 2], b: "x" },
      mm: [{ q: 1, p: 2 }],
    });
    expect(JSON.stringify(out)).toBe('{"a":{"b":"x","yankee":[3,1,2]},"mm":[{"p":2,"q":1}],"zulu":1}');
    expect(canonicalizeArgs(null)).toBeNull();
    expect(canonicalizeArgs("x")).toBe("x");
    expect(canonicalizeArgs([2, 1])).toEqual([2, 1]);
  });

  it("is what makes two key orders render the same tool_use bytes", () => {
    const rendered = (args: Record<string, unknown>, at: number): string => {
      const msgs = buildContext([
        ev({ type: "message", role: "assistant", text: "", toolCalls: [{ id: "c1", name: "edit", args }], model: "m" }, at),
        toolResult("c1", "done", at + 1),
      ]);
      return JSON.stringify(msgs[0]!.toolCalls![0]!.args);
    };
    nextSeq = 0;
    // As the model wrote it, and as jsonb hands it back (length, then bytes) —
    // note `aa` before `b` in one and after it in the other, which is where
    // sorted order and jsonb order actually disagree.
    const asSent = rendered({ zulu: 1, a: 2, mm: 3, aa: 4, b: 5 }, 1);
    const asStored = rendered({ a: 2, b: 5, aa: 4, mm: 3, zulu: 1 }, 3);
    expect(asStored).toBe(asSent);
    expect(asSent).toBe('{"a":2,"aa":4,"b":5,"mm":3,"zulu":1}');
  });
});

describe("estimateTokens", () => {
  it("grows with message size", () => {
    const small = estimateTokens([{ role: "user", text: "x" }]);
    const big = estimateTokens([{ role: "user", text: "x".repeat(400) }]);
    expect(big).toBeGreaterThan(small);
    expect(small).toBeGreaterThan(0);
  });
});
