/**
 * JSONL headless protocol tests. The session takes stdin as an AsyncIterable
 * and stdout as a `write` callback, so every case here is a real end-to-end
 * session: chunks in, protocol lines out — no process, no socket, no DB.
 *
 * Two fakes carry the suite: FakeEvents (the atomic ingest sink) and a
 * hand-driven runAgent whose promise the test resolves, which is what makes
 * "the second run starts only after the first finished" observable.
 */
import { describe, expect, test } from "bun:test";
import type { ThreadEvent, ThreadEventData, ThreadRef } from "@pinky/core";
import type { AgentRunResult } from "@pinky/runtime";
import { parseCommand, runHeadless, type HeadlessRunHooks } from "../src/headless";
import type { EventSink, RawIngress } from "../src/server";

// ---------------------------------------------------------------------------
// Fakes and helpers
// ---------------------------------------------------------------------------

class FakeEvents implements EventSink {
  readonly calls: { ref: ThreadRef; externalId: string; data: ThreadEventData[] }[] = [];
  private readonly seen = new Set<string>();
  /** Make ingest reject for this external id (transaction rolled back). */
  failOn: string | null = null;

  ingest(
    ref: ThreadRef,
    externalId: string,
    data: ThreadEventData[],
  ): Promise<unknown[] | null> {
    this.calls.push({ ref, externalId, data });
    if (this.failOn === externalId) return Promise.reject(new Error("ingest exploded"));
    const key = `${ref.tenantId}:${externalId}`;
    if (this.seen.has(key)) return Promise.resolve(null);
    this.seen.add(key);
    return Promise.resolve(data.map(() => ({})));
  }
}

interface RunCall {
  thread: ThreadRef;
  batch: RawIngress[];
  hooks: HeadlessRunHooks;
  finish: (result: AgentRunResult) => void;
  explode: (err: Error) => void;
  /** Signal state the instant runAgent was entered. A run cancelled while it
   *  was still parked on the lane chain must never produce one of these. */
  abortedAtEntry: boolean;
}

/** runAgent under the test's control: every call parks until it is finished. */
class ManualRunner {
  readonly calls: RunCall[] = [];

  readonly run = (
    thread: ThreadRef,
    batch: RawIngress[],
    hooks: HeadlessRunHooks,
  ): Promise<AgentRunResult> => {
    let finish!: (result: AgentRunResult) => void;
    let explode!: (err: Error) => void;
    const promise = new Promise<AgentRunResult>((resolve, reject) => {
      finish = resolve;
      explode = reject;
    });
    this.calls.push({
      thread,
      batch,
      hooks,
      finish,
      explode,
      abortedAtEntry: hooks.signal.aborted,
    });
    return promise;
  };
}

type StdinItem = string | Uint8Array | (() => Promise<void> | void);

/**
 * stdin as a script: strings/bytes are yielded as chunks, functions are
 * barriers the generator awaits before emitting the next chunk. That is how a
 * test writes "…and only once the run is in flight, send the abort".
 */
async function* scriptedStdin(items: StdinItem[]): AsyncGenerator<string | Uint8Array> {
  for (const item of items) {
    if (typeof item === "function") {
      await item();
      continue;
    }
    yield item;
  }
}

async function waitFor(check: () => boolean, label: string, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
    await Bun.sleep(1);
  }
}

interface Session {
  out: string[];
  events: FakeEvents;
  done: Promise<void>;
  /** Parsed stdout, with the "one JSON object per line" invariant enforced. */
  lines: () => Record<string, unknown>[];
  types: () => string[];
}

function session(opts: {
  stdin: AsyncIterable<string | Uint8Array>;
  runAgent: (
    thread: ThreadRef,
    batch: RawIngress[],
    hooks: HeadlessRunHooks,
  ) => Promise<AgentRunResult>;
  events?: FakeEvents;
}): Session {
  const out: string[] = [];
  const events = opts.events ?? new FakeEvents();
  const done = runHeadless({
    tenantId: "t1",
    agentId: "pinky",
    nodeId: "local",
    defaultModel: "fake/test-model",
    events,
    runAgent: opts.runAgent,
    stdin: opts.stdin,
    write: (line) => {
      out.push(line);
    },
  });
  const lines = (): Record<string, unknown>[] =>
    out.map((line) => {
      // Every stdout write is exactly one line of JSON — that is the protocol.
      expect(line.endsWith("\n")).toBe(true);
      expect(line.slice(0, -1)).not.toContain("\n");
      return JSON.parse(line) as Record<string, unknown>;
    });
  return { out, events, done, lines, types: () => lines().map((l) => l.type as string) };
}

const threadEvent = (seq: number, text: string): ThreadEvent => ({
  id: `evt-${seq}`,
  tenantId: "t1",
  channelId: "jsonl:local",
  threadId: "main",
  seq,
  ts: "2026-08-28T00:00:00.000Z",
  data: { type: "message", role: "assistant", text, toolCalls: [], model: "fake/test-model" },
});

/** A runAgent that streams two events and one reply, then completes. */
const chattyRun = async (
  _thread: ThreadRef,
  _batch: RawIngress[],
  hooks: HeadlessRunHooks,
): Promise<AgentRunResult> => {
  hooks.onEvent(threadEvent(1, "thinking"));
  await hooks.deliver("hi back");
  hooks.onEvent(threadEvent(2, "egress"));
  return { turns: 2, stopReason: "completed" };
};

const prompt = (fields: Record<string, unknown>): string =>
  `${JSON.stringify({ type: "prompt", ...fields })}\n`;

// ---------------------------------------------------------------------------
// parseCommand
// ---------------------------------------------------------------------------

describe("parseCommand", () => {
  test("a full prompt keeps every supplied field", () => {
    const r = parseCommand(
      JSON.stringify({
        type: "prompt",
        text: "hello",
        threadId: "t",
        channelId: "c",
        id: "client-1",
        author: { userId: "brad", displayName: "Brad" },
      }),
    );
    expect(r).toEqual({
      ok: true,
      cmd: {
        type: "prompt",
        text: "hello",
        threadId: "t",
        channelId: "c",
        id: "client-1",
        // The platform tag is ours, never the client's claim.
        author: { platform: "jsonl", userId: "brad", displayName: "Brad" },
      },
    });
  });

  test("a bare prompt gets the documented defaults and no id", () => {
    const r = parseCommand('{"type":"prompt","text":"hi"}');
    expect(r).toEqual({
      ok: true,
      cmd: {
        type: "prompt",
        text: "hi",
        threadId: "main",
        channelId: "jsonl:local",
        author: { platform: "jsonl", userId: "local" },
      },
    });
    // Absent, not undefined: an `id` key would become a dedup claim.
    if (!r.ok) throw new Error("expected ok");
    expect("id" in r.cmd).toBe(false);
  });

  test("an author without displayName omits the key", () => {
    const r = parseCommand('{"type":"prompt","text":"hi","author":{"userId":"u2"}}');
    if (!r.ok || r.cmd.type !== "prompt") throw new Error("expected a prompt");
    expect(r.cmd.author).toEqual({ platform: "jsonl", userId: "u2" });
  });

  test("abort defaults to the main thread", () => {
    expect(parseCommand('{"type":"abort"}')).toEqual({
      ok: true,
      cmd: { type: "abort", threadId: "main" },
    });
    expect(parseCommand('{"type":"abort","threadId":"other"}')).toEqual({
      ok: true,
      cmd: { type: "abort", threadId: "other" },
    });
  });

  test("exit defaults to draining, not aborting", () => {
    expect(parseCommand('{"type":"exit"}')).toEqual({
      ok: true,
      cmd: { type: "exit", abort: false },
    });
    expect(parseCommand('{"type":"exit","abort":true}')).toEqual({
      ok: true,
      cmd: { type: "exit", abort: true },
    });
  });

  test.each([
    ["not json at all", "invalid JSON"],
    ["[1,2,3]", "command must be a JSON object"],
    ["42", "command must be a JSON object"],
    ["null", "command must be a JSON object"],
    ['{"text":"hi"}', "command.type must be a string"],
    ['{"type":123}', "command.type must be a string"],
    ['{"type":"nope"}', "unknown command type"],
    ['{"type":"prompt"}', "prompt.text must be a string"],
    ['{"type":"prompt","text":7}', "prompt.text must be a string"],
    ['{"type":"prompt","text":""}', "prompt.text must not be empty"],
    ['{"type":"prompt","text":"h","threadId":5}', "prompt.threadId must be a string"],
    ['{"type":"prompt","text":"h","channelId":[]}', "prompt.channelId must be a string"],
    ['{"type":"prompt","text":"h","id":9}', "prompt.id must be a string"],
    ['{"type":"prompt","text":"h","author":"brad"}', "prompt.author must be an object"],
    ['{"type":"prompt","text":"h","author":{"userId":1}}', "prompt.author.userId must be a string"],
    [
      '{"type":"prompt","text":"h","author":{"displayName":1}}',
      "prompt.author.displayName must be a string",
    ],
    ['{"type":"abort","threadId":{}}', "abort.threadId must be a string"],
    ['{"type":"exit","abort":"yes"}', "exit.abort must be a boolean"],
  ])("rejects %p", (line, expected) => {
    const r = parseCommand(line);
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("expected a rejection");
    expect(r.error).toContain(expected);
  });
});

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

describe("runHeadless", () => {
  test("ready first, then a full run, split across chunk boundaries and CRLF", async () => {
    const s = session({
      // The line is cut mid-token and terminated with CRLF; the blank line in
      // between is protocol whitespace, not a malformed command.
      stdin: scriptedStdin([
        '{"type":"prompt","text":"hel',
        'lo","id":"c1"}\r\n',
        "\r\n",
        '{"type":"exit"}\n',
      ]),
      runAgent: chattyRun,
    });
    await s.done;

    expect(s.types()).toEqual([
      "ready",
      "run_started",
      "event",
      "reply",
      "event",
      "run_finished",
      "exiting",
    ]);
    const lines = s.lines();
    expect(lines[0]).toEqual({
      type: "ready",
      nodeId: "local",
      agentId: "pinky",
      tenantId: "t1",
      defaultModel: "fake/test-model",
    });
    expect(lines[1]).toEqual({ type: "run_started", threadId: "main", channelId: "jsonl:local" });
    expect(lines[2]).toEqual({
      type: "event",
      threadId: "main",
      channelId: "jsonl:local",
      event: threadEvent(1, "thinking"),
    });
    expect(lines[3]).toEqual({
      type: "reply",
      threadId: "main",
      channelId: "jsonl:local",
      text: "hi back",
    });
    expect(lines[5]).toEqual({
      type: "run_finished",
      threadId: "main",
      channelId: "jsonl:local",
      stopReason: "completed",
      turns: 2,
    });
    expect(lines[6]).toEqual({ type: "exiting" });

    // The prompt reached the log exactly once, under the client's own id.
    expect(s.events.calls).toHaveLength(1);
    expect(s.events.calls[0]!.ref).toEqual({
      tenantId: "t1",
      channelId: "jsonl:local",
      threadId: "main",
    });
    expect(s.events.calls[0]!.externalId).toBe("c1");
    expect(s.events.calls[0]!.data).toEqual([
      {
        type: "ingress",
        platform: "jsonl",
        author: { platform: "jsonl", userId: "local" },
        text: "hello",
        refs: [],
        externalId: "c1",
      },
    ]);
  });

  test("byte chunks split mid-UTF-8 still decode to one intact prompt", async () => {
    const bytes = new TextEncoder().encode('{"type":"prompt","text":"héllo 🎉"}\n');
    const emoji = bytes.indexOf(0xf0); // first byte of the 4-byte sequence
    const s = session({
      stdin: scriptedStdin([bytes.slice(0, emoji + 2), bytes.slice(emoji + 2)]),
      runAgent: chattyRun,
    });
    await s.done;

    const ingress = s.events.calls[0]!.data[0]!;
    if (ingress.type !== "ingress") throw new Error("expected an ingress event");
    expect(ingress.text).toBe("héllo 🎉");
    // No id given: the session mints a jsonl-prefixed dedup key.
    expect(s.events.calls[0]!.externalId).toMatch(/^jsonl:[0-9a-f-]{36}$/);
  });

  test("two prompts on one thread run serially, in arrival order", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "first", id: "a" }),
        prompt({ text: "second", id: "b" }),
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });

    await waitFor(() => runner.calls.length >= 1, "the first run to start");
    // Both prompts are already journaled, but only one run exists: the second
    // is parked behind the first on the lane's promise chain.
    await Bun.sleep(5);
    expect(runner.calls).toHaveLength(1);
    expect(s.events.calls.map((c) => c.externalId)).toEqual(["a", "b"]);
    expect(s.types()).toEqual(["ready", "run_started"]);
    expect(runner.calls[0]!.batch.map((i) => i.text)).toEqual(["first"]);

    runner.calls[0]!.finish({ turns: 1, stopReason: "completed" });
    await waitFor(() => runner.calls.length === 2, "the second run to start");
    expect(runner.calls[1]!.batch.map((i) => i.text)).toEqual(["second"]);
    runner.calls[1]!.finish({ turns: 3, stopReason: "max_turns" });
    await s.done;

    expect(s.types()).toEqual([
      "ready",
      "run_started",
      "run_finished",
      "run_started",
      "run_finished",
      "exiting",
    ]);
    expect(s.lines()[4]).toMatchObject({ stopReason: "max_turns", turns: 3 });
  });

  test("different threads run concurrently", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "a1", threadId: "alpha", id: "a" }),
        prompt({ text: "b1", threadId: "beta", id: "b" }),
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });

    // Both in flight at the same time — nothing serializes across threads.
    await waitFor(() => runner.calls.length === 2, "both runs to start");
    expect(runner.calls.map((c) => c.thread.threadId)).toEqual(["alpha", "beta"]);
    for (const call of runner.calls) call.finish({ turns: 1, stopReason: "completed" });
    await s.done;

    const started = s.lines().filter((l) => l.type === "run_started");
    expect(started.map((l) => l.threadId).sort()).toEqual(["alpha", "beta"]);
    expect(s.types().at(-1)).toBe("exiting");
  });

  test("abort cancels the run in flight on that thread", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "long job", id: "a" }),
        () => waitFor(() => runner.calls.length === 1, "the run to start"),
        '{"type":"abort","threadId":"main"}\n',
      ]),
      runAgent: runner.run,
    });

    await waitFor(() => runner.calls[0]?.hooks.signal.aborted === true, "the signal to fire");
    // The loop returns `aborted` rather than throwing; the run still finishes.
    runner.calls[0]!.finish({ turns: 1, stopReason: "aborted" });
    await s.done;

    expect(s.types()).toEqual(["ready", "run_started", "run_finished", "exiting"]);
    expect(s.lines()[2]).toMatchObject({ stopReason: "aborted", threadId: "main" });
  });

  test("abort with nothing in flight is an error, not a crash", async () => {
    const s = session({
      stdin: scriptedStdin(['{"type":"abort","threadId":"ghost"}\n', '{"type":"exit"}\n']),
      runAgent: chattyRun,
    });
    await s.done;

    expect(s.types()).toEqual(["ready", "error", "exiting"]);
    expect(s.lines()[1]).toMatchObject({ threadId: "ghost" });
    expect(String(s.lines()[1]!.message)).toContain("no run in flight");
  });

  test("a duplicate id is refused while that thread already has a run pending", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "once", id: "dup" }),
        () => waitFor(() => runner.calls.length === 1, "the first run"),
        prompt({ text: "once", id: "dup" }),
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });

    await waitFor(() => s.types().includes("error"), "the duplicate error");
    expect(runner.calls).toHaveLength(1); // the resend never became a run
    const dup = s.lines().find((l) => l.type === "error")!;
    expect(String(dup.message)).toContain("duplicate id dup");
    // A run IS pending for this thread, so replaying would double the reply.
    expect(String(dup.message)).toContain("run already pending");
    // Not a run-closing error: no `run` tag, and no run_started preceded it.
    expect("run" in dup).toBe(false);
    runner.calls[0]!.finish({ turns: 1, stopReason: "completed" });
    await s.done;

    expect(s.types()).toEqual(["ready", "run_started", "error", "run_finished", "exiting"]);
  });

  test("a failed ingest is reported and starts no run", async () => {
    const events = new FakeEvents();
    events.failOn = "boom";
    const runner = new ManualRunner();
    const s = session({
      events,
      stdin: scriptedStdin([prompt({ text: "hi", id: "boom" }), '{"type":"exit"}\n']),
      runAgent: runner.run,
    });
    await s.done;

    expect(runner.calls).toHaveLength(0);
    expect(s.types()).toEqual(["ready", "error", "exiting"]);
    expect(String(s.lines()[1]!.message)).toContain("ingest failed");
    // An ingest error closes no run, so it carries no `run` tag.
    expect("run" in s.lines()[1]!).toBe(false);
  });

  test("a malformed line is echoed back and the session keeps going", async () => {
    const s = session({
      stdin: scriptedStdin([
        "{not json\n",
        `{"type":"whatever"}\n`,
        prompt({ text: "still here", id: "ok" }),
        '{"type":"exit"}\n',
      ]),
      runAgent: chattyRun,
    });
    await s.done;

    expect(s.types()).toEqual([
      "ready",
      "error",
      "error",
      "run_started",
      "event",
      "reply",
      "event",
      "run_finished",
      "exiting",
    ]);
    expect(s.lines()[1]).toMatchObject({ line: "{not json" });
    expect(String(s.lines()[1]!.message)).toContain("invalid JSON");
    expect(String(s.lines()[2]!.message)).toContain("unknown command type");
  });

  test("a very long malformed line is truncated to 200 characters", async () => {
    const s = session({
      stdin: scriptedStdin([`${"x".repeat(500)}\n`, '{"type":"exit"}\n']),
      runAgent: chattyRun,
    });
    await s.done;
    expect(String(s.lines()[1]!.line)).toHaveLength(200);
  });

  test("exit waits for the in-flight run before writing exiting", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "slow", id: "a" }),
        () => waitFor(() => runner.calls.length === 1, "the run to start"),
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });

    await waitFor(() => runner.calls.length === 1, "the run to start");
    await Bun.sleep(5);
    // stdin is done, but the run is not: nothing has been closed off yet.
    expect(s.types()).toEqual(["ready", "run_started"]);
    expect(runner.calls[0]!.hooks.signal.aborted).toBe(false);

    runner.calls[0]!.finish({ turns: 1, stopReason: "completed" });
    await s.done;
    expect(s.types()).toEqual(["ready", "run_started", "run_finished", "exiting"]);
  });

  test("exit --abort cancels the runs queued behind the one in flight too", async () => {
    const runner = new ManualRunner();
    const s = session({
      // Three prompts on ONE thread: run 1 executes, runs 2 and 3 are parked on
      // the lane chain when the abort lands.
      stdin: scriptedStdin([
        prompt({ text: "one", id: "a" }),
        prompt({ text: "two", id: "b" }),
        prompt({ text: "three", id: "c" }),
        () => waitFor(() => runner.calls.length === 1, "the first run to start"),
        '{"type":"exit","abort":true}\n',
      ]),
      runAgent: runner.run,
    });

    await waitFor(() => runner.calls[0]?.hooks.signal.aborted === true, "the signal to fire");
    runner.calls[0]!.finish({ turns: 0, stopReason: "aborted" });
    await s.done;

    // The point of the test: the queued runs never reached the agent at all —
    // before the fix each one minted a fresh controller and ran to completion
    // while `exiting` waited for them.
    expect(runner.calls).toHaveLength(1);
    expect(runner.calls[0]!.batch.map((i) => i.text)).toEqual(["one"]);
    // …and every prompt still got a balanced started/finished pair.
    expect(s.types()).toEqual([
      "ready",
      "run_started",
      "run_finished",
      "run_started",
      "run_finished",
      "run_started",
      "run_finished",
      "exiting",
    ]);
    const finished = s.lines().filter((l) => l.type === "run_finished");
    expect(finished.map((l) => l.stopReason)).toEqual(["aborted", "aborted", "aborted"]);
    // A cancelled-while-parked run reports zero turns: it did no work.
    expect(finished.slice(1).map((l) => l.turns)).toEqual([0, 0]);
  });

  test("abort drops the queued runs on that thread and the lane stays usable", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "one", id: "a" }),
        prompt({ text: "two", id: "b" }),
        prompt({ text: "three", id: "c" }),
        () => waitFor(() => runner.calls.length === 1, "the first run to start"),
        '{"type":"abort","threadId":"main"}\n',
        () => waitFor(() => runner.calls[0]!.hooks.signal.aborted, "the abort to reach the run"),
        () => {
          runner.calls[0]!.finish({ turns: 1, stopReason: "aborted" });
        },
        () =>
          waitFor(
            () => s.types().filter((t) => t === "run_finished").length === 3,
            "all three runs to close",
          ),
        // The abort cancelled what was queued AT THAT MOMENT; a prompt sent
        // afterwards is new intent and must run normally.
        prompt({ text: "after", id: "d" }),
        () => waitFor(() => runner.calls.length === 2, "the post-abort run to start"),
        () => {
          runner.calls[1]!.finish({ turns: 1, stopReason: "completed" });
        },
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });
    await s.done;

    expect(runner.calls).toHaveLength(2);
    expect(runner.calls[1]!.batch.map((i) => i.text)).toEqual(["after"]);
    expect(runner.calls[1]!.abortedAtEntry).toBe(false);
    expect(
      s
        .lines()
        .filter((l) => l.type === "run_finished")
        .map((l) => l.stopReason),
    ).toEqual(["aborted", "aborted", "aborted", "completed"]);
    // No "nothing in flight" error: the abort had something to cancel.
    expect(s.types()).not.toContain("error");
  });

  test("the caller's signal ends the session even with stdin still open", async () => {
    const runner = new ManualRunner();
    const clientGone = new AbortController();
    const out: string[] = [];
    let release!: () => void;
    const forever = new Promise<void>((resolve) => {
      release = resolve;
    });
    // The EPIPE shape: the client stopped reading but never closed its end of
    // stdin, so the session can only be ended from the outside.
    const done = runHeadless({
      tenantId: "t1",
      agentId: "pinky",
      nodeId: "local",
      defaultModel: "fake/test-model",
      events: new FakeEvents(),
      runAgent: runner.run,
      stdin: scriptedStdin([
        prompt({ text: "one", id: "a" }),
        prompt({ text: "two", id: "b" }),
        () => forever,
      ]),
      write: (line) => {
        out.push(line);
      },
      signal: clientGone.signal,
    });

    await waitFor(() => runner.calls.length === 1, "the first run to start");
    clientGone.abort(new Error("EPIPE"));
    await waitFor(() => runner.calls[0]!.hooks.signal.aborted, "the in-flight run to be cancelled");
    runner.calls[0]!.finish({ turns: 0, stopReason: "aborted" });
    await done; // resolves despite stdin never reaching EOF
    release();

    expect(runner.calls).toHaveLength(1); // the queued run never started
    const types = out.map((l) => (JSON.parse(l) as { type: string }).type);
    expect(types).toEqual([
      "ready",
      "run_started",
      "run_finished",
      "run_started",
      "run_finished",
      "exiting",
    ]);
  });

  test("a write that throws does not park the lane", async () => {
    const runner = new ManualRunner();
    const out: string[] = [];
    const logged: string[] = [];
    let failed = false;
    const done = runHeadless({
      tenantId: "t1",
      agentId: "pinky",
      nodeId: "local",
      defaultModel: "fake/test-model",
      events: new FakeEvents(),
      runAgent: runner.run,
      stdin: scriptedStdin([
        prompt({ text: "first", id: "a" }),
        () => waitFor(() => logged.length === 1, "the dropped-line log"),
        prompt({ text: "second", id: "b" }),
        () => waitFor(() => runner.calls.length === 1, "the second prompt's run"),
        () => {
          runner.calls[0]!.finish({ turns: 1, stopReason: "completed" });
        },
      ]),
      write: (line) => {
        // The first run_started blows up; everything after it writes normally.
        if (!failed && line.includes('"run_started"')) {
          failed = true;
          throw new Error("stdout gone");
        }
        out.push(line);
      },
      log: (msg) => {
        logged.push(msg);
      },
    });
    await done;

    // The failed run is skipped (it could not be reported), but the lane keeps
    // serving: before the fix the rejection poisoned the chain and the second
    // prompt was ingested and never run — and drain() rejected with it.
    expect(runner.calls).toHaveLength(1);
    expect(runner.calls[0]!.batch.map((i) => i.text)).toEqual(["second"]);
    const types = out.map((l) => (JSON.parse(l) as { type: string }).type);
    expect(types).toEqual(["ready", "run_started", "run_finished", "exiting"]);
    expect(logged.join("\n")).toContain("dropped a protocol line");
  });

  test("a duplicate id with nothing running is replayed instead of refused", async () => {
    const runner = new ManualRunner();
    const s = session({
      stdin: scriptedStdin([
        prompt({ text: "once", id: "dup" }),
        () => waitFor(() => runner.calls.length === 1, "the first run"),
        () => {
          runner.calls[0]!.finish({ turns: 1, stopReason: "completed" });
        },
        () => waitFor(() => s.types().includes("run_finished"), "the first run to close"),
        // Same id, nothing in flight: the ingress is in the log but nothing is
        // answering it, which is what a crash between ingest and run leaves.
        prompt({ text: "once", id: "dup" }),
        () => waitFor(() => runner.calls.length === 2, "the replay run"),
        () => {
          runner.calls[1]!.finish({ turns: 1, stopReason: "completed" });
        },
        '{"type":"exit"}\n',
      ]),
      runAgent: runner.run,
    });
    await s.done;

    expect(s.types()).toEqual([
      "ready",
      "run_started",
      "run_finished",
      "run_started",
      "run_finished",
      "exiting",
    ]);
    expect(s.types()).not.toContain("error");
    expect(s.lines()[3]).toEqual({
      type: "run_started",
      threadId: "main",
      channelId: "jsonl:local",
      replay: true,
    });
    // Only the recovery run is flagged; a first-time prompt has no `replay`.
    expect("replay" in s.lines()[1]!).toBe(false);
    expect(s.events.calls).toHaveLength(2); // the second ingest returned null
  });

  test("a line over 1 MiB is dropped and the stream resynchronises", async () => {
    const huge = "x".repeat(1024 * 1024 + 1);
    const s = session({
      stdin: scriptedStdin([
        // No newline in this chunk: the framer would otherwise buffer forever.
        huge,
        `still-the-same-line\n${prompt({ text: "still here", id: "ok" })}`,
        '{"type":"exit"}\n',
      ]),
      runAgent: chattyRun,
    });
    await s.done;

    const errors = s.lines().filter((l) => l.type === "error");
    expect(errors).toEqual([{ type: "error", message: "line exceeds 1048576 bytes; dropped" }]);
    // One error for the whole oversized line, its tail discarded through the
    // next newline, and the command right after it still runs.
    expect(s.types()).toEqual([
      "ready",
      "error",
      "run_started",
      "event",
      "reply",
      "event",
      "run_finished",
      "exiting",
    ]);
  });

  test("stdin EOF is an exit", async () => {
    const s = session({
      stdin: scriptedStdin([prompt({ text: "bye", id: "a" })]), // no exit command
      runAgent: chattyRun,
    });
    await s.done;
    expect(s.types().at(-1)).toBe("exiting");
    expect(s.types()).toContain("run_finished");
  });

  test("a trailing line with no newline is still a command", async () => {
    const s = session({
      stdin: scriptedStdin(['{"type":"prompt","text":"no newline","id":"a"}']),
      runAgent: chattyRun,
    });
    await s.done;
    expect(s.events.calls).toHaveLength(1);
    expect(s.types()).toContain("run_finished");
  });

  test("a throwing runAgent yields an error, no run_finished, and the thread survives", async () => {
    const runner = new ManualRunner();
    const logged: string[] = [];
    const out: string[] = [];
    const events = new FakeEvents();
    const done = runHeadless({
      tenantId: "t1",
      agentId: "pinky",
      nodeId: "local",
      defaultModel: "fake/test-model",
      events,
      runAgent: runner.run,
      stdin: scriptedStdin([
        prompt({ text: "explodes", id: "a" }),
        () => waitFor(() => runner.calls.length === 1, "the first run"),
        () => {
          runner.calls[0]!.explode(new Error("provider on fire"));
        },
        () => waitFor(() => out.some((l) => l.includes('"error"')), "the error line"),
        prompt({ text: "next one", id: "b" }),
        () => waitFor(() => runner.calls.length === 2, "the second run"),
        () => {
          runner.calls[1]!.finish({ turns: 1, stopReason: "completed" });
        },
      ]),
      write: (line) => {
        out.push(line);
      },
      log: (msg) => {
        logged.push(msg);
      },
    });
    await done;

    const types = out.map((l) => (JSON.parse(l) as { type: string }).type);
    expect(types).toEqual([
      "ready",
      "run_started",
      "error",
      "run_started",
      "run_finished",
      "exiting",
    ]);
    const error = JSON.parse(out[2]!) as Record<string, unknown>;
    // `run: "failed"` is what lets a client close its own accounting for this
    // run without matching on the message text.
    expect(error).toMatchObject({ threadId: "main", channelId: "jsonl:local", run: "failed" });
    expect(String(error.message)).toContain("provider on fire");
    // Human detail goes to the log sink (stderr), never to the protocol stream.
    expect(logged.join("\n")).toContain("provider on fire");
  });
});
