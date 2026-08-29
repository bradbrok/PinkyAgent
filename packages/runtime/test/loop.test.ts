/**
 * Loop tests: exercise runAgentLoop against the real EventStore/projection
 * and the real ShedContextTool over an in-memory FakeDb (mirrors the FakeDb
 * shape in packages/core/test). Covers DESIGN.md §3 (projection from the
 * continuity boundary) and §4.1/§4.3/§4.5 (pressure ladder, restart cycle,
 * cut-point safety).
 */
import { describe, expect, test } from "bun:test";
import { EventStore, threadKey } from "@pinky/core";
import type { Db, SettingsSnapshot, ThreadEvent, ThreadEventData, ThreadRef, ToolCall } from "@pinky/core";
import { runAgentLoop } from "../src/loop";
import { ShedContextTool } from "../src/continuity";
import { FakeProvider } from "../src/providers/fake";
import type { AgentLoopOptions } from "../src/types";
import type { AgentRunResult, AssistantTurn, CompleteOptions, LlmMessage, Provider, Tool } from "../src/types";
import type { FakeScript } from "../src/providers/fake";
import type { RunAgentLoopOptions } from "../src/loop";

// ---------------------------------------------------------------------------
// FakeDb: implements the exact SQL core's EventStore issues (append, history,
// contextEvents, latestContinuitySeq).
// ---------------------------------------------------------------------------

const norm = (sql: string): string => sql.replace(/\s+/g, " ");

class FakeDb implements Db {
  readonly events: ThreadEvent[] = [];
  private seqByThread = new Map<string, number>();

  private eventsFor(key: string): ThreadEvent[] {
    return this.events.filter((e) => threadKey(e) === key).sort((a, b) => a.seq - b.seq);
  }

  /** Mirrors what postgres.js hands back for a jsonb column: a parsed value,
   *  not JSON text (see the JSONB CONTRACT in packages/core/src/pg.ts). */
  private static toRow(e: ThreadEvent): Record<string, unknown> {
    return {
      id: e.id,
      tenant_id: e.tenantId,
      channel_id: e.channelId,
      thread_id: e.threadId,
      seq: e.seq,
      ts: e.ts,
      data: e.data,
    };
  }

  query<T>(sql: string, params?: unknown[]): Promise<T[]> {
    const s = norm(sql);
    if (/insert into threads/.test(s)) {
      const key = (params as [string, string, string]).join(":");
      if (!this.seqByThread.has(key)) this.seqByThread.set(key, 0);
      return Promise.resolve([]);
    }
    if (/from threads .* for update/.test(s)) {
      const key = (params as [string, string, string]).join(":");
      const last = this.eventsFor(key).at(-1);
      return Promise.resolve((last ? [{ seq: last.seq }] : []) as T[]);
    }
    if (/insert into ingress_dedup/.test(s)) return Promise.resolve([]);
    if (/from events/.test(s) && /seq >= \$4/.test(s)) {
      // contextEvents: newest-first, capped.
      const [tenantId, channelId, threadId] = params as [string, string, string];
      const from = params![3] as number;
      const limit = params![4] as number;
      const rows = this.eventsFor(`${tenantId}:${channelId}:${threadId}`)
        .filter((e) => e.seq >= from)
        .sort((a, b) => b.seq - a.seq)
        .slice(0, limit);
      return Promise.resolve(rows.map(FakeDb.toRow) as T[]);
    }
    if (/from events/.test(s) && /seq > \$4/.test(s)) {
      const [tenantId, channelId, threadId] = params as [string, string, string];
      const after = params![3] as number;
      const limit = params![4] as number;
      const rows = this.eventsFor(`${tenantId}:${channelId}:${threadId}`)
        .filter((e) => e.seq > after)
        .slice(0, limit);
      return Promise.resolve(rows.map(FakeDb.toRow) as T[]);
    }
    return Promise.resolve([]);
  }

  queryOne<T>(sql: string, params?: unknown[]): Promise<T | null> {
    const s = norm(sql);
    if (/coalesce\(max\(seq\), 0\) \+ 1 as next/.test(s)) {
      const key = (params as [string, string, string]).join(":");
      return Promise.resolve({ next: (this.seqByThread.get(key) ?? 0) + 1 } as T);
    }
    if (/insert into events/.test(s)) {
      // `data` is bound as the PLAIN object, never pre-stringified JSON —
      // pg.ts's JSONB CONTRACT. The driver encodes it once on the way out.
      const [id, tenantId, channelId, threadId, seq, , data] = params as [
        string,
        string,
        string,
        string,
        number,
        string,
        ThreadEventData,
      ];
      const evt: ThreadEvent = {
        id,
        tenantId,
        channelId,
        threadId,
        seq,
        ts: new Date().toISOString(),
        data,
      };
      this.events.push(evt);
      this.seqByThread.set(`${tenantId}:${channelId}:${threadId}`, seq);
      return Promise.resolve({
        id,
        tenant_id: tenantId,
        channel_id: channelId,
        thread_id: threadId,
        seq,
        ts: evt.ts,
        data,
      } as T);
    }
    if (/type = 'continuity'/.test(s)) {
      const key = (params as [string, string, string]).join(":");
      const last = [...this.eventsFor(key)].reverse().find((e) => e.data.type === "continuity");
      return Promise.resolve((last ? { seq: last.seq } : null) as T | null);
    }
    if (/from events .* order by seq desc limit 1/.test(s)) {
      return Promise.resolve(null);
    }
    return Promise.resolve(null);
  }

  async tx<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    return fn(this);
  }

  async close(): Promise<void> {}
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const THREAD: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

function settings(context?: Partial<SettingsSnapshot["context"]>): SettingsSnapshot {
  return {
    tenantId: "t1",
    model: "fake/test-model",
    context: { advisoryFraction: 0.7, hardFraction: 0.9, approxWindowTokens: 180_000, ...context },
    replyGate: { classifierEnabled: false },
  };
}

function turn(partial: Partial<AssistantTurn> & Pick<AssistantTurn, "text">): AssistantTurn {
  return { toolCalls: [], stopReason: "stop", ...partial };
}

function call(id: string, name: string, args: Record<string, unknown> = {}): ToolCall {
  return { id, name, args };
}

/** A document that passes the §4.2 harness guard. */
const SHED_ARGS: Record<string, unknown> = {
  goal: "finish the continuity engine",
  plan: { done: ["read the design"], now: "rebuild context", next: ["verify"] },
  workingSet: { files: ["/src/loop.ts"] },
  decisions: [{ what: "shed last", why: "cut-point safety" }],
  openLoops: [],
  lessons: [],
  memoryHints: ["continuity"],
};

/**
 * The loop hands the provider its live `messages` array and keeps mutating it,
 * so assertions need a per-call snapshot of what the model actually saw.
 */
interface Prompt {
  model: string;
  system: string;
  tools: string[];
  messages: LlmMessage[];
}

class SnapshotProvider implements Provider {
  readonly name = "fake";
  readonly prompts: Prompt[] = [];
  private readonly inner: FakeProvider;

  constructor(script: FakeScript) {
    this.inner = new FakeProvider(script);
  }

  complete(opts: CompleteOptions): Promise<AssistantTurn> {
    this.prompts.push({
      model: opts.model,
      system: opts.system,
      tools: opts.tools.map((t) => t.name),
      messages: opts.messages.map((m) => ({ ...m })),
    });
    return this.inner.complete(opts);
  }
}

interface Harness {
  db: FakeDb;
  store: EventStore;
  run: (overrides?: HarnessOverrides) => Promise<AgentRunResult>;
  provider: SnapshotProvider;
  delivered: string[];
}

type HarnessOverrides = {
  [K in keyof RunAgentLoopOptions]?: RunAgentLoopOptions[K] | undefined;
};

function harness(script: FakeScript, tools: Tool[] = []): Harness {
  const db = new FakeDb();
  const store = new EventStore(db);
  const provider = new SnapshotProvider(script);
  const delivered: string[] = [];
  const run = (overrides: HarnessOverrides = {}) =>
    runAgentLoop({
      db,
      provider,
      tools,
      thread: THREAD,
      agentId: "pinky",
      systemPrompt: "sys",
      cwd: "/tmp",
      settings: settings(),
      deliver: async (text) => {
        delivered.push(text);
      },
      ...Object.fromEntries(Object.entries(overrides).filter(([, v]) => v !== undefined)),
    });
  return { db, store, run, provider, delivered };
}

const echoTool: Tool = {
  name: "echo",
  description: "echo args back",
  parameters: { type: "object" },
  execute: async (args) => ({ text: `echo:${JSON.stringify(args)}` }),
};

const ingress = (text: string): ThreadEventData => ({
  type: "ingress",
  platform: "cli",
  author: { platform: "cli", userId: "u1" },
  text,
  refs: [],
});

const notices = (msgs: { role: string; text: string }[]): string[] =>
  msgs.filter((m) => m.text.startsWith("[harness notice]")).map((m) => m.text);

const eventTypes = (db: FakeDb): string[] => db.events.map((e) => e.data.type);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("runAgentLoop", () => {
  test("text-only turn completes, delivers, and journals message + egress", async () => {
    const h = harness([turn({ text: "hello world" })]);
    const result = await h.run();
    expect(result).toEqual({ turns: 1, stopReason: "completed" });
    expect(h.delivered).toEqual(["hello world"]);

    expect(eventTypes(h.db)).toEqual(["message", "egress"]);
    const msg = h.db.events[0]!.data as Extract<ThreadEventData, { type: "message" }>;
    expect(msg.text).toBe("hello world");
    expect(msg.model).toBe("fake/test-model");
    const egress = h.db.events[1]!.data as Extract<ThreadEventData, { type: "egress" }>;
    expect(egress).toEqual({ type: "egress", target: { kind: "thread" }, text: "hello world" });
  });

  test("journals the provider's token usage on the assistant message", async () => {
    // DESIGN §13 cost model: $/task has to be answerable from the log alone,
    // and the cache counters are the half that makes restarts measurable.
    const h = harness([
      turn({
        text: "ok",
        usage: { input: 1200, output: 80, cacheRead: 900, cacheCreation: 300 },
      }),
    ]);
    await h.run();
    const msg = h.db.events[0]!.data as Extract<ThreadEventData, { type: "message" }>;
    expect(msg.usage).toEqual({ input: 1200, output: 80, cacheRead: 900, cacheCreation: 300 });
  });

  test("omits usage when the provider reported none (no empty key in the log)", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.run();
    const msg = h.db.events[0]!.data as Extract<ThreadEventData, { type: "message" }>;
    expect(msg.usage).toBeUndefined();
    expect(Object.keys(msg)).not.toContain("usage");
  });

  test("provider receives the bare model id (prefix stripped)", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.run();
    expect(h.provider.prompts[0]!.model).toBe("test-model");
  });

  test("tool-call turn executes the tool, journals tool_result, then completes", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("call_1", "echo", { n: 1 })], stopReason: "tool_calls" }),
        turn({ text: "done" }),
      ],
      [echoTool],
    );
    const result = await h.run();
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    expect(eventTypes(h.db)).toEqual(["message", "tool_result", "message", "egress"]);
    const tr = h.db.events[1]!.data as Extract<ThreadEventData, { type: "tool_result" }>;
    expect(tr).toMatchObject({ callId: "call_1", name: "echo", isError: false });
    expect(tr.text).toBe('echo:{"n":1}');

    // The tool result feeds back into the second provider call.
    const secondCall = h.provider.prompts[1]!;
    const toolMsgs = secondCall.messages.filter((m) => m.role === "tool");
    expect(toolMsgs).toHaveLength(1);
    expect(toolMsgs[0]).toMatchObject({ role: "tool", toolCallId: "call_1" });
  });

  test("unknown tool produces an isError tool_result and the loop continues", async () => {
    const h = harness([
      turn({ text: "", toolCalls: [call("call_x", "nope")], stopReason: "tool_calls" }),
      turn({ text: "recovered" }),
    ]);
    const result = await h.run();
    expect(result.stopReason).toBe("completed");
    const tr = h.db.events[1]!.data as Extract<ThreadEventData, { type: "tool_result" }>;
    expect(tr.isError).toBe(true);
    expect(tr.text).toContain("unknown tool");
  });

  test("stops at maxTurns while the model keeps calling tools", async () => {
    const h = harness(
      () => turn({ text: "", toolCalls: [call("c", "echo")], stopReason: "tool_calls" }),
      [echoTool],
    );
    const result = await h.run({ maxTurns: 3 });
    expect(result).toEqual({ turns: 3, stopReason: "max_turns" });
    expect(h.provider.prompts).toHaveLength(3);
  });

  test("finish 'length' is treated as completed", async () => {
    const h = harness([turn({ text: "partial", stopReason: "length" })]);
    const result = await h.run();
    expect(result).toEqual({ turns: 1, stopReason: "completed" });
    expect(h.delivered).toEqual(["partial"]);
  });

  test("pre-aborted signal stops before any provider call", async () => {
    const h = harness([turn({ text: "never" })]);
    const controller = new AbortController();
    controller.abort();
    const result = await h.run({ signal: controller.signal });
    expect(result).toEqual({ turns: 0, stopReason: "aborted" });
    expect(h.provider.prompts).toHaveLength(0);
    expect(h.db.events).toHaveLength(0);
  });

  test("works without a deliver callback", async () => {
    const h = harness([turn({ text: "quiet" })]);
    const result = await h.run({});
    expect(result.stopReason).toBe("completed");
    expect(eventTypes(h.db)).toEqual(["message", "egress"]);
  });
});

// ---------------------------------------------------------------------------
// DEFECT 1: context is the projection from the continuity boundary, not a
// fixed forward page of history.
// ---------------------------------------------------------------------------

describe("runAgentLoop context loading (DESIGN §3)", () => {
  test("builds the prompt from the latest continuity boundary, dropping older events", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("ancient business"));
    await h.store.append(THREAD, {
      type: "continuity",
      document: {
        goal: "carry on",
        plan: { done: [], now: "answer the fresh question", next: [] },
        workingSet: {},
        decisions: [],
        openLoops: [],
        lessons: [],
        memoryHints: [],
      },
      tokensBefore: 900,
    });
    await h.store.append(THREAD, ingress("fresh question"));

    await h.run();
    const msgs = h.provider.prompts[0]!.messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[0]!.text).toContain("# Pinky Continuity");
    expect(msgs[1]!.text).toContain("fresh question");
    expect(msgs.some((m) => m.text.includes("ancient business"))).toBe(false);
  });

  test("a thread longer than the history page still sees its newest events", async () => {
    // Regression: history() defaults to the first 500 events, which froze long
    // threads on their opening window forever.
    const h = harness([turn({ text: "ok" })]);
    for (let i = 0; i < 505; i++) await h.store.append(THREAD, ingress(`event-${i}`));

    await h.run();
    const msgs = h.provider.prompts[0]!.messages;
    expect(msgs).toHaveLength(505);
    expect(msgs.at(-1)!.text).toContain("event-504");
    expect(msgs[0]!.text).toContain("event-0");
  });
});

// ---------------------------------------------------------------------------
// DEFECT 2: the context-pressure ladder (DESIGN §4.1) and restart cycle (§4.3)
// ---------------------------------------------------------------------------

describe("runAgentLoop context pressure ladder (DESIGN §4.1)", () => {
  test("advisory pressure injects one user-role harness notice, not one per turn", async () => {
    // window 20 -> advisory at 14. Seed ~64 chars in one ingress event. No shed
    // tool is available, so the ladder stops at the advisory rung.
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("c1", "echo")], stopReason: "tool_calls" }),
        turn({ text: "", toolCalls: [call("c2", "echo")], stopReason: "tool_calls" }),
        turn({ text: "done" }),
      ],
      [echoTool],
    );
    await h.store.append(THREAD, ingress("x".repeat(64)));
    const before = h.db.events.length;
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }) });
    expect(result.stopReason).toBe("completed");

    for (const req of h.provider.prompts) {
      expect(notices(req.messages)).toHaveLength(1);
      expect(req.messages.every((m) => m.role !== "system")).toBe(true);
      expect(req.system).toBe("sys"); // stable cache prefix: never rewritten
    }
    const notice = notices(h.provider.prompts[2]!.messages)[0]!;
    expect(notice).toContain("context pressure");
    const noticeMsg = h.provider.prompts[2]!.messages.find((m) => m.text === notice)!;
    expect(noticeMsg.role).toBe("user");
    // Notices are never journaled.
    expect(eventTypes(h.db).slice(before)).toEqual([
      "message",
      "tool_result",
      "message",
      "tool_result",
      "message",
      "egress",
    ]);
  });

  test("without a shed_context tool, hard pressure degrades to normal operation", async () => {
    const h = harness([turn({ text: "fine" })], [echoTool]);
    await h.store.append(THREAD, ingress("x".repeat(80)));
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }) });
    expect(result.stopReason).toBe("completed");
    expect(h.provider.prompts[0]!.tools).toEqual(["echo"]);
  });

  test("hard pressure offers only shed_context and stops with 'shed' when no turns remain", async () => {
    const h = harness(
      () => turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("x".repeat(80))); // >= hardFraction * 20
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }), maxTurns: 1 });
    expect(result).toEqual({ turns: 1, stopReason: "shed" });

    const req = h.provider.prompts[0]!;
    expect(req.tools).toEqual(["shed_context"]);
    expect(req.system).toBe("sys"); // HARD_NOTE does not touch the system prompt
    const notice = notices(req.messages)[0]!;
    expect(notice).toContain("context limit reached");
    expect(req.messages.find((m) => m.text === notice)!.role).toBe("user");

    expect(eventTypes(h.db)).toEqual(["ingress", "message", "continuity", "tool_result"]);
    const cont = h.db.events[2]!.data as Extract<ThreadEventData, { type: "continuity" }>;
    expect(cont.document.goal).toBe("finish the continuity engine");
    expect(cont.tokensBefore).toBeGreaterThan(0); // loop's per-turn estimate
  });

  test("a forced turn refuses tools other than shed_context", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("e1", "echo")], stopReason: "tool_calls" }),
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("x".repeat(80)));
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }), maxTurns: 2 });
    expect(result).toEqual({ turns: 2, stopReason: "shed" });

    const first = h.db.events.find(
      (e) => e.data.type === "tool_result",
    )!.data as Extract<ThreadEventData, { type: "tool_result" }>;
    expect(first.isError).toBe(true);
    expect(first.text).toContain("only shed_context may be called");
    expect(h.db.events.some((e) => e.data.type === "tool_result" && e.data.text.startsWith("echo:"))).toBe(false);
    // Second forced turn carries the retry notice.
    expect(notices(h.provider.prompts[1]!.messages).at(-1)).toContain("final attempt");
  });

  test("an invalid document on the forced turn is retried once, then the run stops", async () => {
    const h = harness(
      () => turn({ text: "", toolCalls: [call("s", "shed_context", { goal: "g" })], stopReason: "tool_calls" }),
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("x".repeat(80)));
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }) });
    expect(result).toEqual({ turns: 2, stopReason: "shed_failed" });
    expect(h.db.events.some((e) => e.data.type === "continuity")).toBe(false);

    const rejected = h.db.events.filter(
      (e) => e.data.type === "tool_result",
    ) as (ThreadEvent & { data: Extract<ThreadEventData, { type: "tool_result" }> })[];
    expect(rejected).toHaveLength(2);
    expect(rejected[0]!.data.isError).toBe(true);
    expect(rejected[0]!.data.text).toContain("plan must be an object");

    const err = h.db.events.at(-1)!.data as Extract<ThreadEventData, { type: "error" }>;
    expect(err.type).toBe("error");
    expect(err.source).toBe("continuity");
    expect(err.count).toBe(2);
  });

  test("talking instead of shedding on a forced turn is bounded too", async () => {
    const h = harness(() => turn({ text: "I would rather not" }), [new ShedContextTool()]);
    await h.store.append(THREAD, ingress("x".repeat(80)));
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }) });
    expect(result).toEqual({ turns: 2, stopReason: "shed_failed" });
    expect(h.delivered).toHaveLength(2); // the text still reaches the user
    expect(h.provider.prompts).toHaveLength(2);
  });
});

describe("runAgentLoop restart cycle (DESIGN §4.3)", () => {
  test("a voluntary shed rebuilds the prompt from the boundary and the run continues", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed from the document" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("a long conversation about widgets"));

    const result = await h.run(); // no pressure: this is the preferred §4.1 rung
    expect(result).toEqual({ turns: 2, stopReason: "completed" });
    expect(h.delivered).toEqual(["resumed from the document"]);

    // Event order: assistant message -> continuity (during execution) -> tool_result.
    expect(eventTypes(h.db)).toEqual([
      "ingress",
      "message",
      "continuity",
      "tool_result",
      "message",
      "egress",
    ]);
    const cont = h.db.events[2]!.data as Extract<ThreadEventData, { type: "continuity" }>;
    expect(cont.tokensBefore).toBeGreaterThan(0);

    // The post-shed prompt is the continuity document alone: the pre-boundary
    // transcript is gone and the orphan tool_result was dropped.
    const after = h.provider.prompts[1]!.messages;
    expect(after).toHaveLength(1);
    expect(after[0]!.role).toBe("user");
    expect(after[0]!.text).toContain("# Pinky Continuity");
    expect(after[0]!.text).toContain("- now: rebuild context");
    expect(after.some((m) => m.role === "tool")).toBe(false);
    expect(after.some((m) => m.text.includes("widgets"))).toBe(false);
  });

  test("a shed batched with other tool calls executes last (cut-point safety)", async () => {
    const h = harness(
      [
        turn({
          text: "",
          // shed listed FIRST; the loop must still run it last.
          toolCalls: [call("s1", "shed_context", SHED_ARGS), call("e1", "echo", { n: 2 })],
          stopReason: "tool_calls",
        }),
        turn({ text: "done" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    const result = await h.run();
    expect(result).toEqual({ turns: 2, stopReason: "completed" });
    expect(eventTypes(h.db)).toEqual([
      "message",
      "tool_result", // echo
      "continuity", // boundary lands at the very end of the turn
      "tool_result", // shed
      "message",
      "egress",
    ]);
    const echoResult = h.db.events[1]!.data as Extract<ThreadEventData, { type: "tool_result" }>;
    expect(echoResult.name).toBe("echo");
    expect(h.provider.prompts[1]!.messages).toHaveLength(1);
  });

  test("the advisory notice is re-armed by a shed", async () => {
    // Big enough window that the rebuilt context sits under the advisory line,
    // and a seed big enough that the first window sits over it.
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "", toolCalls: [call("e1", "echo", { pad: "y".repeat(4000) })], stopReason: "tool_calls" }),
        turn({ text: "done" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("z".repeat(4400)));
    // advisory at 1000 tokens, hard at 1900: both windows below the hard rung.
    const opts = { settings: settings({ advisoryFraction: 0.5, hardFraction: 0.95, approxWindowTokens: 2000 }) };
    const result = await h.run(opts);
    expect(result.stopReason).toBe("completed");

    expect(notices(h.provider.prompts[0]!.messages)).toHaveLength(1); // first crossing
    expect(notices(h.provider.prompts[1]!.messages)).toHaveLength(0); // fresh window
    expect(notices(h.provider.prompts[2]!.messages)).toHaveLength(1); // re-armed, crossed again
  });
});
