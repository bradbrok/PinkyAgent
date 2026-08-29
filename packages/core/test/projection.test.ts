/**
 * Projection tests (DESIGN.md §3): the boundary rule and the cut-point-safety
 * rules that keep a post-restart prompt acceptable to providers (§4.5).
 */
import { describe, expect, it } from "bun:test";
import { buildContext, estimateTokens, latestContinuity, serializeContinuity } from "../src/projection";
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

  // DESIGN.md §5.3: memory writes/recalls are journaled for audit, but the
  // model sees recalled memories only via the injected <memories> block the
  // loop builds — never as a replayed event. A `memory` event rendering here
  // would duplicate that block on every later turn.
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

  it("skips memory events (audit-only, DESIGN.md §5.3)", () => {
    nextSeq = 0;
    const msgs = buildContext([
      ingress("what do you know about me?"),
      ev({ type: "memory", op: "recall", ids: ["m1", "m2"], text: "about me", count: 2 }),
      ev({ type: "memory", op: "retain", ids: ["m3"], text: "Brad prefers terse answers" }),
    ]);
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.text).toBe("[cli u1]: what do you know about me?");
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
