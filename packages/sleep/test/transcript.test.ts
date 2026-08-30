/**
 * Transcript rendering (slice 6, DESIGN.md §5.3 item 3).
 *
 * Two things are worth testing here and they are not the same thing: the LINE
 * FORMATS (what the extraction model reads) and the BUDGET RULE (what the
 * cursor is allowed to skip). The second is the one with data loss behind it.
 */
import { describe, expect, test } from "bun:test";
import type { ThreadEvent, ThreadEventData, ThreadRef } from "@pinky/core";
import {
  DEFAULT_TRANSCRIPT_CHARS,
  MAX_ARGS_CHARS,
  MAX_TOOL_RESULT_CHARS,
  renderTranscript,
} from "../src/transcript";

const THREAD: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

/** Events at seq 1..n, in order. */
function events(...datas: ThreadEventData[]): ThreadEvent[] {
  return datas.map((data, i) => ({
    ...THREAD,
    id: `e${i + 1}`,
    seq: i + 1,
    ts: "2026-08-01T00:00:00.000Z",
    data,
  }));
}

const ingress = (text: string, userId = "u1"): ThreadEventData => ({
  type: "ingress",
  platform: "cli",
  author: { platform: "cli", userId },
  text,
  refs: [],
});

describe("renderTranscript line formats", () => {
  test("ingress carries platform and userId", () => {
    const t = renderTranscript(events(ingress("hello there", "brad")));
    expect(t.text).toBe("[1] user cli:brad: hello there");
    expect(t.scanned).toBe(1);
    expect(t.toSeq).toBe(1);
  });

  test("a2a renders as a peer line", () => {
    const t = renderTranscript(
      events({ type: "a2a", from: "ada@n2", to: "pinky@n1", kind: "message", text: "ping", msgId: "m1" }),
    );
    expect(t.text).toBe("[1] peer ada@n2: ping");
  });

  test("an assistant turn renders its text then one line per tool call", () => {
    const t = renderTranscript(
      events({
        type: "message",
        role: "assistant",
        text: "checking",
        toolCalls: [
          { id: "c1", name: "read_file", args: { path: "/a" } },
          { id: "c2", name: "bash", args: { cmd: "ls" } },
        ],
        model: "anthropic/x",
      }),
    );
    expect(t.text).toBe(
      ['[1] assistant: checking', '  -> read_file({"path":"/a"})', '  -> bash({"cmd":"ls"})'].join("\n"),
    );
  });

  test("a pure tool-call turn has no trailing space after 'assistant:'", () => {
    const t = renderTranscript(
      events({
        type: "message",
        role: "assistant",
        text: "",
        toolCalls: [{ id: "c1", name: "noop", args: {} }],
        model: "anthropic/x",
      }),
    );
    expect(t.text).toBe(["[1] assistant:", "  -> noop({})"].join("\n"));
  });

  test("tool-call arguments are capped", () => {
    const t = renderTranscript(
      events({
        type: "message",
        role: "assistant",
        text: "",
        toolCalls: [{ id: "c1", name: "write", args: { body: "x".repeat(500) } }],
        model: "anthropic/x",
      }),
    );
    const argLine = t.text.split("\n")[1] ?? "";
    // "  -> write(" + <= MAX_ARGS_CHARS + ")"
    expect(argLine.length).toBe("  -> write(".length + MAX_ARGS_CHARS + 1);
    expect(argLine.endsWith("…)")).toBe(true);
  });

  test("tool_result names the tool, caps the text, and flags an error", () => {
    const ok = renderTranscript(
      events({ type: "tool_result", callId: "c1", name: "read_file", text: "contents", isError: false }),
    );
    expect(ok.text).toBe("[1] tool read_file: contents");

    const bad = renderTranscript(
      events({ type: "tool_result", callId: "c1", name: "bash", text: "y".repeat(400), isError: true }),
    );
    expect(bad.text.startsWith("[1] tool bash (error): ")).toBe(true);
    expect(bad.text.length).toBe("[1] tool bash (error): ".length + MAX_TOOL_RESULT_CHARS);
  });

  test("continuity renders goal, current step and lessons", () => {
    const t = renderTranscript(
      events({
        type: "continuity",
        tokensBefore: 100,
        document: {
          goal: "ship slice 6",
          plan: { done: ["read"], now: "build the worker", next: ["wire it"] },
          workingSet: {},
          decisions: [],
          openLoops: [],
          lessons: ["do not compact", "restart instead"],
          memoryHints: [],
        },
      }),
    );
    expect(t.text).toBe(
      "[1] continuity: goal=ship slice 6 | now=build the worker | lessons=do not compact; restart instead",
    );
  });

  test("a malformed continuity document renders blanks instead of throwing", () => {
    // A doc comes back out of the log; one bad row must not make every later
    // pass over this thread fail, or the cursor never advances again.
    const broken = { type: "continuity", tokensBefore: 0, document: {} } as unknown as ThreadEventData;
    const t = renderTranscript(events(broken));
    expect(t.text).toBe("[1] continuity: goal= | now= | lessons=");
  });

  test("a standalone `error` event renders NOTHING", () => {
    // A failed pass journals `{source: "sleep"}` on the thread it failed on.
    // Rendering it would make the failure feed itself: the thread stays due,
    // the cursor never advances, and every sweep re-pays two LLM calls to add
    // one more error line (types.ts EXTRACT_EVENT_TYPES).
    const t = renderTranscript(
      events(
        { type: "error", source: "sleep", message: "provider timed out", count: 1 },
        { type: "error", source: "bash", message: "exit 1", count: 2 },
        ingress("kept"),
      ),
    );
    expect(t.text).toBe("[3] user cli:u1: kept");
    expect(t.scanned).toBe(1);
  });

  test("a FAILED tool result still carries its failure (DESIGN.md §4.4)", () => {
    // Negative evidence reaches the worker through the tool result and through
    // a continuity doc's `lessons` — not through standalone `error` events.
    const t = renderTranscript(
      events({ type: "tool_result", callId: "c1", name: "bash", text: "exit 1", isError: true }),
    );
    expect(t.text).toBe("[1] tool bash (error): exit 1");
  });

  test("audit-only types render nothing and are not scanned", () => {
    const t = renderTranscript(
      events(
        { type: "decision", action: "reply", reason: "asked" },
        { type: "memory", op: "retain", ids: ["m1"], text: "x" },
        ingress("kept"),
      ),
    );
    expect(t.text).toBe("[3] user cli:u1: kept");
    expect(t.scanned).toBe(1);
    expect(t.toSeq).toBe(3);
  });
});

describe("renderTranscript authors", () => {
  test("distinct ingress userIds, in first-seen order", () => {
    const t = renderTranscript(
      events(ingress("a", "brad"), ingress("b", "ada"), ingress("c", "brad")),
    );
    expect(t.authors).toEqual(["brad", "ada"]);
  });

  test("only authors of INCLUDED events (the budget cut removes them too)", () => {
    // The downgrade rule keys on this list, so an author whose line did not
    // make the transcript must not license a `user`-visible memory about them.
    const t = renderTranscript(events(ingress("a".repeat(60), "brad"), ingress("b", "ada")), {
      maxChars: 70,
    });
    expect(t.authors).toEqual(["brad"]);
    expect(t.scanned).toBe(1);
  });

  test("no ingress, no authors", () => {
    const t = renderTranscript(events({ type: "error", source: "s", message: "m", count: 1 }));
    expect(t.authors).toEqual([]);
  });
});

describe("renderTranscript char budget", () => {
  test("stops at the last event that fits and reports its seq", () => {
    const t = renderTranscript(events(ingress("aaaa"), ingress("bbbb"), ingress("cccc")), {
      maxChars: 45,
    });
    // "[1] user cli:u1: aaaa" is 21 chars; two lines + newline is 43, three is 65.
    expect(t.scanned).toBe(2);
    expect(t.toSeq).toBe(2);
    expect(t.text.split("\n")).toHaveLength(2);
  });

  test("never drops OLDER lines to fit newer ones", () => {
    const t = renderTranscript(events(ingress("first"), ingress("second"), ingress("third")), {
      maxChars: 25,
    });
    // The cursor advances to toSeq, so anything dropped from the front would be
    // material the worker skipped permanently.
    expect(t.text).toContain("first");
    expect(t.text).not.toContain("third");
    expect(t.toSeq).toBe(1);
  });

  test("the first event is included even when it alone blows the budget", () => {
    // Otherwise one giant paste yields toSeq 0, the cursor never moves past it,
    // and every sweep re-reads the same event forever.
    const t = renderTranscript(events(ingress("x".repeat(500)), ingress("next")), { maxChars: 10 });
    expect(t.scanned).toBe(1);
    expect(t.toSeq).toBe(1);
    expect(t.text.length).toBeGreaterThan(10);
  });

  test("everything fits under the default budget for an ordinary window", () => {
    const t = renderTranscript(events(...Array.from({ length: 50 }, (_, i) => ingress(`m${i}`))), {});
    expect(t.scanned).toBe(50);
    expect(t.toSeq).toBe(50);
    expect(t.text.length).toBeLessThan(DEFAULT_TRANSCRIPT_CHARS);
  });

  test("an empty input is an empty transcript with toSeq 0", () => {
    const t = renderTranscript([]);
    expect(t).toEqual({ text: "", authors: [], scanned: 0, toSeq: 0 });
  });
});
