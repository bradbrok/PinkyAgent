/**
 * Loop tests: exercise runAgentLoop against the real EventStore/projection
 * and the real ShedContextTool over an in-memory FakeDb (mirrors the FakeDb
 * shape in packages/core/test). Covers DESIGN.md §3 (projection from the
 * continuity boundary) and §4.1/§4.3/§4.5 (pressure ladder, restart cycle,
 * cut-point safety).
 */
import { describe, expect, test } from "bun:test";
import { DEFAULT_CONTEXT_EVENT_CAP, EventStore, buildContext, estimateTokens, threadKey } from "@pinky/core";
import type { Db, SettingsSnapshot, ThreadEvent, ThreadEventData, ThreadRef, ToolCall } from "@pinky/core";
import type { MemoryHit, MemoryStore, RecallScope, SearchInput } from "@pinky/core";
import { runAgentLoop } from "../src/loop";
import { ShedContextTool } from "../src/continuity";
import { FakeProvider } from "../src/providers/fake";
import type { AgentLoopOptions, DeferredTools, Embedder, MemoryContext } from "../src/types";
import type { AgentRunResult, AssistantTurn, CompleteOptions, LlmMessage, Provider, Tool, ToolChoice } from "../src/types";
import type { FakeScript } from "../src/providers/fake";
import type { RunAgentLoopOptions } from "../src/loop";

// ---------------------------------------------------------------------------
// FakeDb: implements the exact SQL core's EventStore issues (append, history,
// contextEvents, latestContinuitySeq).
// ---------------------------------------------------------------------------

const norm = (sql: string): string => sql.replace(/\s+/g, " ");

/**
 * Postgres `jsonb` key order: by (length, then bytes) — NOT insertion order.
 * Applied to what a read returns, which is where the reorder becomes visible.
 */
function jsonbOrder(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(jsonbOrder);
  if (value === null || typeof value !== "object") return value;
  const source = value as Record<string, unknown>;
  const keys = Object.keys(source).sort((a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0));
  const out: Record<string, unknown> = {};
  for (const key of keys) out[key] = jsonbOrder(source[key]);
  return out;
}

class FakeDb implements Db {
  readonly events: ThreadEvent[] = [];
  private seqByThread = new Map<string, number>();

  /**
   * Test seam: reorder every stored `data` object's keys the way jsonb does,
   * on READ. Off by default (most tests do not care); the cache-alignment
   * tests turn it on, because without it a fake store preserves insertion
   * order and the whole class of "wake N+1 renders different bytes" defects is
   * structurally invisible here.
   */
  reorderJson = false;

  private eventsFor(key: string): ThreadEvent[] {
    return this.events.filter((e) => threadKey(e) === key).sort((a, b) => a.seq - b.seq);
  }

  /**
   * Test seam: land events directly in the store, bypassing EventStore.append.
   *
   * The only way to observe a TRUNCATED context window is to hold more than
   * DEFAULT_CONTEXT_EVENT_CAP (5000) events, and 5000 real appends would be
   * 5000 trips through the tx path to prove nothing. Seq bookkeeping matches
   * append's, so the store keeps working normally afterwards.
   */
  seed(ref: ThreadRef, datas: ThreadEventData[]): void {
    const key = `${ref.tenantId}:${ref.channelId}:${ref.threadId}`;
    let seq = this.seqByThread.get(key) ?? 0;
    for (const data of datas) {
      seq += 1;
      this.events.push({ ...ref, id: `seed-${seq}`, seq, ts: new Date().toISOString(), data });
    }
    this.seqByThread.set(key, seq);
  }

  /** Mirrors what postgres.js hands back for a jsonb column: a parsed value,
   *  not JSON text (see the JSONB CONTRACT in packages/core/src/pg.ts). */
  private toRow(e: ThreadEvent): Record<string, unknown> {
    return {
      id: e.id,
      tenant_id: e.tenantId,
      channel_id: e.channelId,
      thread_id: e.threadId,
      seq: e.seq,
      ts: e.ts,
      data: this.reorderJson ? jsonbOrder(e.data) : e.data,
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
      return Promise.resolve(rows.map((r) => this.toRow(r)) as T[]);
    }
    if (/from events/.test(s) && /seq > \$4/.test(s)) {
      const [tenantId, channelId, threadId] = params as [string, string, string];
      const after = params![3] as number;
      const limit = params![4] as number;
      const rows = this.eventsFor(`${tenantId}:${channelId}:${threadId}`)
        .filter((e) => e.seq > after)
        .slice(0, limit);
      return Promise.resolve(rows.map((r) => this.toRow(r)) as T[]);
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

function settings(
  context?: Partial<SettingsSnapshot["context"]>,
  memory?: Partial<SettingsSnapshot["memory"]>,
): SettingsSnapshot {
  return {
    tenantId: "t1",
    model: "fake/test-model",
    context: { advisoryFraction: 0.7, hardFraction: 0.9, approxWindowTokens: 180_000, ...context },
    replyGate: { classifierEnabled: false },
    tools: { defaultMode: { builtin: "always", mcp: "deferred" }, alwaysOn: [], deferred: [], searchLimit: 8 },
    mcp: { servers: {} },
    memory: {
      embeddingModel: "none",
      autoRecall: true,
      recallLimit: 12,
      recallTokenBudget: 5_000,
      ...memory,
    },
    sleep: {
      enabled: false,
      intervalMs: 300_000,
      idleMs: 600_000,
      model: "",
      maxEventsPerPass: 200,
      maxThreadsPerSweep: 10,
      reflectMinMemories: 5,
      reflectBatch: 50,
    },
    selfConfig: { enabled: false, allowedKeys: [] },
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
  /** Provider-side tool mask; absent on every turn but a forced shed. */
  toolChoice: ToolChoice | undefined;
  /** Cache routing key — must be the same string on every turn of a thread. */
  cacheKey: string | undefined;
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
      toolChoice: opts.toolChoice,
      cacheKey: opts.cacheKey,
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
    // The notice is journaled, ONCE, immediately before the turn it provoked:
    // the prompt is a projection of the log (DESIGN §3), so a notice that
    // existed only in this run's array would be missing from the next wake's
    // prompt and the transcript would diverge from what the provider cached.
    expect(eventTypes(h.db).slice(before)).toEqual([
      "notice",
      "message",
      "tool_result",
      "message",
      "tool_result",
      "message",
      "egress",
    ]);
    const journaled = h.db.events[before]!.data as Extract<ThreadEventData, { type: "notice" }>;
    expect(journaled.text).toBe(notice);
  });

  test("without a shed_context tool, hard pressure degrades to normal operation", async () => {
    const h = harness([turn({ text: "fine" })], [echoTool]);
    await h.store.append(THREAD, ingress("x".repeat(80)));
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }) });
    expect(result.stopReason).toBe("completed");
    expect(h.provider.prompts[0]!.tools).toEqual(["echo"]);
  });

  test("hard pressure forces a shed and stops with 'shed' when no turns remain", async () => {
    const h = harness(
      () => turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("x".repeat(80))); // >= hardFraction * 20
    const result = await h.run({ settings: settings({ approxWindowTokens: 20 }), maxTurns: 1 });
    expect(result).toEqual({ turns: 1, stopReason: "shed" });

    const req = h.provider.prompts[0]!;
    // The tool DEFINITIONS are untouched — narrowing them would invalidate
    // every cache tier (tools render at prefix position 0). And the FIRST
    // forced attempt carries no `tool_choice` either: that invalidates the
    // messages tier, i.e. one uncached re-read of the biggest transcript this
    // thread will ever have. The notice plus the harness guard hold the
    // boundary; only the retry pays for the guarantee (DESIGN §4.5/§9).
    expect(req.tools).toEqual(["echo", "shed_context"]);
    expect(req.toolChoice).toBeUndefined();
    expect(req.system).toBe("sys"); // HARD_NOTE does not touch the system prompt
    const notice = notices(req.messages)[0]!;
    expect(notice).toContain("context limit reached");
    expect(req.messages.find((m) => m.text === notice)!.role).toBe("user");

    // `restart` closes the sequence: the rebuild happens even on the way out,
    // so the successor wake is not billed twice (DESIGN §13).
    expect(eventTypes(h.db)).toEqual([
      "ingress",
      "notice", // journaled ahead of the turn it forced
      "message",
      "continuity",
      "tool_result",
      "restart",
    ]);
    const cont = h.db.events[3]!.data as Extract<ThreadEventData, { type: "continuity" }>;
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
    // Why the guard is the load-bearing half: `echo` IS defined in the forced
    // request (the list is never narrowed) and the first attempt sends no mask
    // at all, so the guard is the only thing standing between the model and
    // another `echo`.
    expect(h.provider.prompts[0]!.tools).toEqual(["echo", "shed_context"]);
    expect(h.provider.prompts[0]!.toolChoice).toBeUndefined();
    // Second forced turn: the retry notice AND the mask, which is where a
    // messages-tier invalidation finally buys something.
    expect(notices(h.provider.prompts[1]!.messages).at(-1)).toContain("final attempt");
    expect(h.provider.prompts[1]!.tools).toEqual(["echo", "shed_context"]);
    expect(h.provider.prompts[1]!.toolChoice).toEqual({ type: "tool", name: "shed_context" });
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

    // Event order: assistant message -> continuity (during execution) ->
    // tool_result -> restart (the rebuild's cost, DESIGN §13).
    expect(eventTypes(h.db)).toEqual([
      "ingress",
      "message",
      "continuity",
      "tool_result",
      "restart",
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
      "restart", // what the rebuild cost
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

// ---------------------------------------------------------------------------
// Memory plane: auto-recall at context start and after each restart
// (DESIGN §5.4). The block is background context in a `user` message; the
// system prompt — the cached prefix (§4.5/§9) — is never touched.
// ---------------------------------------------------------------------------

const SCOPE: RecallScope = {
  agentId: "pinky",
  channelId: "c1",
  includeUser: false,
  includePrivate: false,
};

let hitSeq = 0;

function memoryHit(partial: Partial<MemoryHit> = {}): MemoryHit {
  hitSeq += 1;
  return {
    id: `mem-${hitSeq}`,
    tenantId: "t1",
    agentId: "pinky",
    visibility: "channel",
    userId: null,
    channelId: "c1",
    kind: "semantic",
    text: `memory ${hitSeq}`,
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

class RecallStub {
  readonly searches: SearchInput[] = [];
  constructor(
    private readonly opts: {
      hits?: MemoryHit[];
      /** Different hits per search, so two windows get distinguishable blocks. */
      hitsByCall?: MemoryHit[][];
      vectors?: boolean;
      searchError?: Error;
    } = {},
  ) {}

  async supportsVectors(): Promise<boolean> {
    return this.opts.vectors ?? false;
  }

  async search(input: SearchInput): Promise<MemoryHit[]> {
    this.searches.push(input);
    if (this.opts.searchError) throw this.opts.searchError;
    return this.opts.hitsByCall?.[this.searches.length - 1] ?? this.opts.hits ?? [];
  }
}

class BrokenEmbedder implements Embedder {
  readonly model = "fake/embed";
  readonly dimensions = 4;
  calls = 0;
  async embed(): Promise<number[][]> {
    this.calls += 1;
    throw new Error("embeddings provider down");
  }
}

function memoryContext(
  store: RecallStub,
  embedder?: Embedder,
  scope: RecallScope = SCOPE,
): MemoryContext {
  return {
    store: store as unknown as MemoryStore,
    scope,
    ...(embedder ? { embedder } : {}),
  };
}

/** A trusted local surface (default `pinky headless`): user + private rows. */
const WIDE_SCOPE: RecallScope = {
  ...SCOPE,
  userId: "u1",
  includeUser: true,
  includePrivate: true,
};

const memoryEvents = (db: FakeDb): Extract<ThreadEventData, { type: "memory" }>[] =>
  db.events
    .map((e) => e.data)
    .filter((d): d is Extract<ThreadEventData, { type: "memory" }> => d.type === "memory");

const RECALL_HEAD = "[harness notice] Recalled memories";

describe("runAgentLoop auto-recall (DESIGN §5.4)", () => {
  test("injects the <memories> block at index 0 as a user message", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("how do we handle provider retries?"));
    const recalled = memoryHit({ text: "retries use jittered backoff" });
    const store = new RecallStub({ hits: [recalled] });

    await h.run({ memory: memoryContext(store) });

    const msgs = h.provider.prompts[0]!.messages;
    expect(msgs[0]!.role).toBe("user");
    expect(msgs[0]!.text.startsWith(RECALL_HEAD)).toBe(true);
    expect(msgs[0]!.text).toContain("retries use jittered backoff");
    expect(msgs[1]!.text).toContain("how do we handle provider retries?");
    // The block is a conversation turn, never a system message or a prefix edit.
    expect(h.provider.prompts[0]!.system).toBe("sys");
    expect(msgs.every((m) => m.role !== "system")).toBe(true);

    // The query is seeded from the window, author prefix stripped.
    expect(store.searches).toHaveLength(1);
    expect(store.searches[0]!.query).toBe("how do we handle provider retries?");
    expect(store.searches[0]!.limit).toBe(12);
    expect(store.searches[0]!.scope).toEqual(SCOPE);

    // Audit-only except `block`, which is the injected text verbatim — that is
    // what the next wake's projection replays instead of searching again.
    expect(memoryEvents(h.db)).toEqual([
      {
        type: "memory",
        op: "recall",
        ids: [recalled.id],
        text: "how do we handle provider retries?",
        count: 1,
        block: msgs[0]!.text,
        scope: { includeUser: false, includePrivate: false },
      },
    ]);
  });

  test("no hits: nothing is injected, but the recall is still journaled", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("hello"));
    await h.run({ memory: memoryContext(new RecallStub({ hits: [] })) });

    const msgs = h.provider.prompts[0]!.messages;
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.text).toContain("hello");
    // `block: ""` — nothing to show the model, but the window is now marked as
    // recalled, which is what stops the NEXT wake from moving byte 0.
    expect(memoryEvents(h.db)).toEqual([
      {
        type: "memory",
        op: "recall",
        ids: [],
        text: "hello",
        count: 0,
        block: "",
        scope: { includeUser: false, includePrivate: false },
      },
    ]);
  });

  test("an empty memory plane on wake 1 does not move byte 0 on wake 2", async () => {
    // The reviewer's reproduction: wake 1 recalls into an empty plane, then
    // something is retained. If the gate read "is there a block?" instead of
    // "did auto-recall run?", wake 2 would search again, find the new row and
    // unshift it at index 0 — invalidating the provider's prefix cache for the
    // entire transcript on a thread that did nothing unusual (DESIGN §4.5).
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ hitsByCall: [[], [memoryHit({ text: "retained since" })]] });

    await h.run({ memory: memoryContext(store) });
    const first = h.provider.prompts[0]!.messages;

    await h.store.append(THREAD, ingress("second question"));
    await h.run({ memory: memoryContext(store) });
    const second = h.provider.prompts[1]!.messages;

    expect(store.searches).toHaveLength(1); // no second search
    expect(memoryEvents(h.db)).toHaveLength(1);
    expect(second[0]!.text).toBe(first[0]!.text); // byte 0 unmoved
    expect(second.slice(0, first.length)).toEqual(first); // a pure extension
    expect(second.some((m) => m.text.includes("retained since"))).toBe(false);
  });

  test("autoRecall=false never touches the store", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ hits: [memoryHit()] });
    await h.run({
      memory: memoryContext(store),
      settings: settings(undefined, { autoRecall: false }),
    });
    expect(store.searches).toHaveLength(0);
    expect(h.provider.prompts[0]!.messages).toHaveLength(1);
  });

  test("autoRecall=false strips a block the window already carries", async () => {
    // Turning memory off has to work on an EXISTING window too. The journaled
    // block is replayed by the projection, so honoring the setting means
    // un-hoisting it — a deliberate prefix break the operator asked for.
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ hits: [memoryHit({ text: "widgets cost $4" })] });

    await h.run({ memory: memoryContext(store) });
    expect(h.provider.prompts[0]!.messages[0]!.text).toContain("widgets cost $4");

    await h.run({
      memory: memoryContext(store),
      settings: settings(undefined, { autoRecall: false }),
    });
    const second = h.provider.prompts[1]!.messages;
    expect(second.some((m) => m.text.includes("widgets cost $4"))).toBe(false);
    expect(second[0]!.text).toContain("hello");
    expect(store.searches).toHaveLength(1); // stripped, not re-searched
    expect(memoryEvents(h.db)).toHaveLength(1);
  });

  test("a run with no memory context at all strips it too", async () => {
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ hits: [memoryHit({ text: "widgets cost $4" })] });

    await h.run({ memory: memoryContext(store) });
    await h.run(); // memory plane not wired on this surface
    expect(h.provider.prompts[1]!.messages.some((m) => m.text.includes("widgets"))).toBe(false);
  });

  test("a NARROWER scope strips the replayed block and recalls again (DESIGN §5.1)", async () => {
    // A default run opened this window with `includeUser`/`includePrivate`; a
    // `--shared` run picks the same thread up. Replaying that block would put
    // private rows into a shared context. Privacy wins over the prefix.
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({
      hitsByCall: [
        [memoryHit({ text: "a PRIVATE note" })],
        [memoryHit({ text: "only SHARED rows" })],
      ],
    });

    await h.run({ memory: memoryContext(store, undefined, WIDE_SCOPE) });
    await h.run({ memory: memoryContext(store, undefined, SCOPE) });

    expect(store.searches).toHaveLength(2);
    expect(store.searches[1]!.scope).toEqual(SCOPE);
    const second = h.provider.prompts[1]!.messages;
    expect(second[0]!.text).toContain("only SHARED rows");
    expect(second.some((m) => m.text.includes("a PRIVATE note"))).toBe(false);
    // Both passes are journaled, each with the width it ran under.
    const mem = memoryEvents(h.db);
    expect(mem).toHaveLength(2);
    expect(mem[0]!.scope).toEqual({ includeUser: true, includePrivate: true });
    expect(mem[1]!.scope).toEqual({ includeUser: false, includePrivate: false });
    // The projection keeps hoisting the FIRST, so the narrow surface pays this
    // strip-and-recall on every wake until the window turns over. Documented
    // trade: it only happens on a thread driven by two different surfaces.
    expect(buildContext(h.db.events)[0]!.text).toContain("a PRIVATE note");
  });

  test("a WIDER scope replays the narrow block as-is", async () => {
    // Nothing leaks in this direction: a trusted surface reading a block built
    // for a shared one just sees fewer rows. No re-search, no moved byte 0.
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ hits: [memoryHit({ text: "only SHARED rows" })] });

    await h.run({ memory: memoryContext(store, undefined, SCOPE) });
    await h.run({ memory: memoryContext(store, undefined, WIDE_SCOPE) });

    expect(store.searches).toHaveLength(1);
    expect(memoryEvents(h.db)).toHaveLength(1);
    expect(h.provider.prompts[1]!.messages[0]!.text).toContain("only SHARED rows");
  });

  test("without a memory context the prompt is the bare projection", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("hello"));
    await h.run();
    expect(h.provider.prompts[0]!.messages).toHaveLength(1);
    expect(memoryEvents(h.db)).toEqual([]);
  });

  test("a restart recalls again, seeded by the continuity memoryHints", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("widget pricing question"));
    const store = new RecallStub({ hits: [memoryHit({ text: "widgets cost $4" })] });

    const result = await h.run({ memory: memoryContext(store) });
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    // One recall per context: the initial window and the post-shed window.
    expect(store.searches.map((s) => s.query)).toEqual([
      "widget pricing question",
      "continuity", // SHED_ARGS.memoryHints — the only signal in a fresh window
    ]);

    const after = h.provider.prompts[1]!.messages;
    expect(after).toHaveLength(2);
    expect(after[0]!.role).toBe("user");
    expect(after[0]!.text.startsWith(RECALL_HEAD)).toBe(true);
    expect(after[1]!.text).toContain("# Pinky Continuity");
    expect(h.provider.prompts[1]!.system).toBe("sys");
    expect(memoryEvents(h.db)).toHaveLength(2);
  });

  test("an embedder failure degrades to FTS-only and journals an error", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("vector question"));
    const store = new RecallStub({ vectors: true, hits: [memoryHit({ text: "fts still works" })] });
    const embedder = new BrokenEmbedder();

    const result = await h.run({ memory: memoryContext(store, embedder) });
    expect(result.stopReason).toBe("completed");
    expect(embedder.calls).toBe(1);
    expect(store.searches[0]!.queryEmbedding).toBeUndefined();
    expect(h.provider.prompts[0]!.messages[0]!.text).toContain("fts still works");

    const err = h.db.events
      .map((e) => e.data)
      .find((d): d is Extract<ThreadEventData, { type: "error" }> => d.type === "error")!;
    expect(err).toMatchObject({ source: "memory", count: 1 });
    expect(err.message).toContain("embeddings provider down");
  });

  test("a store failure journals an error and the run continues unrecalled", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({ searchError: new Error("connection reset") });

    const result = await h.run({ memory: memoryContext(store) });
    expect(result).toEqual({ turns: 1, stopReason: "completed" });
    expect(h.provider.prompts[0]!.messages).toHaveLength(1); // no block
    expect(eventTypes(h.db)).toEqual(["ingress", "error", "message", "egress"]);
    const err = h.db.events[1]!.data as Extract<ThreadEventData, { type: "error" }>;
    expect(err.source).toBe("memory");
    expect(err.message).toContain("recall failed");
  });

  test("the token budget cuts the block down to the highest-scored hits", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("hello"));
    const store = new RecallStub({
      hits: [
        memoryHit({ text: `top ${"a".repeat(200)}`, score: 0.9 }),
        memoryHit({ text: `mid ${"b".repeat(200)}`, score: 0.8 }),
        memoryHit({ text: `low ${"c".repeat(200)}`, score: 0.1 }),
      ],
    });

    await h.run({
      memory: memoryContext(store),
      // ~104 tokens for header + one 200-char hit, as core's estimateTokens
      // counts it (chars/4 plus the per-message overhead); a second hit is ~60
      // more, so this budget admits exactly one.
      settings: settings(undefined, { recallTokenBudget: 110 }),
    });

    const block = h.provider.prompts[0]!.messages[0]!.text;
    expect(block).toContain("top a");
    expect(block).not.toContain("mid b");
    expect(block).not.toContain("low c");
    const ev = memoryEvents(h.db)[0]!;
    expect(ev.count).toBe(3); // candidates before the cut
    expect(ev.ids).toHaveLength(1); // what the model actually saw
  });
});

describe("runAgentLoop onEvent observer", () => {
  test("every appended event reaches onEvent, in log order, with the stored row", async () => {
    // Two turns and a tool call: message, tool_result, message, egress.
    const h = harness(
      [turn({ text: "", toolCalls: [call("c1", "echo", { a: 1 })] }), turn({ text: "done" })],
      [echoTool],
    );
    const seen: ThreadEvent[] = [];
    const result = await h.run({ onEvent: (e) => seen.push(e) });

    expect(result).toEqual({ turns: 2, stopReason: "completed" });
    // The observer sees exactly the log, in seq order — it is the append path,
    // not a second source of truth (headless JSONL streams straight from it).
    expect(eventTypes(h.db)).toEqual(seen.map((e) => e.data.type));
    expect(seen.map((e) => e.seq)).toEqual([1, 2, 3, 4]);
    expect(seen.map((e) => e.id)).toEqual(h.db.events.map((e) => e.id));
    expect(seen[0]).toEqual(h.db.events[0]!);
  });

  test("a throwing observer never breaks the run", async () => {
    const h = harness([turn({ text: "hello" })]);
    let calls = 0;
    const result = await h.run({
      onEvent: () => {
        calls++;
        throw new Error("observer exploded");
      },
    });

    expect(result).toEqual({ turns: 1, stopReason: "completed" });
    expect(calls).toBe(2); // message + egress
    expect(eventTypes(h.db)).toEqual(["message", "egress"]);
    expect(h.delivered).toEqual(["hello"]);
  });
});

describe("runAgentLoop tool context", () => {
  test("hands every tool the run's settings snapshot, read-only", async () => {
    // `settings_set` needs the snapshot to report `previous` and to read the
    // selfConfig allow-list (DESIGN.md P8, revised). It must be the object the
    // loop itself is running on — a stale or absent copy would let a tool
    // report a threshold nobody is using.
    const seen: (SettingsSnapshot | undefined)[] = [];
    const spy: Tool = {
      name: "spy",
      description: "records its context",
      parameters: { type: "object" },
      execute: async (_args, ctx) => {
        seen.push(ctx.settings);
        return { text: "ok" };
      },
    };
    const h = harness(
      [turn({ text: "", toolCalls: [call("c1", "spy")] }), turn({ text: "done" })],
      [spy],
    );
    const snapshot = settings({ advisoryFraction: 0.55 });
    const result = await h.run({ settings: snapshot });

    expect(result.stopReason).toBe("completed");
    expect(seen).toHaveLength(1);
    expect(seen[0]).toBe(snapshot);
    expect(seen[0]!.context.advisoryFraction).toBe(0.55);
  });

  test("passes the deferred-tool plane through, and omits the key without one", async () => {
    // tool_call reaches the catalog only through ctx.deferred (slice 9). The
    // absent case is the other half: the three meta-tools are registered on
    // every surface, so "no catalog here" has to arrive as a missing key they
    // can answer for — never as a half-built object.
    const seen: (DeferredTools | undefined)[] = [];
    const spy: Tool = {
      name: "spy",
      description: "records its context",
      parameters: { type: "object" },
      execute: async (_args, ctx) => {
        seen.push(ctx.deferred);
        return { text: "ok" };
      },
    };
    const deferred: DeferredTools = {
      catalog: { search: async () => [], describe: async () => null },
      call: async () => ({ text: "called" }),
    };

    const withPlane = harness(
      [turn({ text: "", toolCalls: [call("c1", "spy")] }), turn({ text: "done" })],
      [spy],
    );
    await withPlane.run({ deferred });
    expect(seen[0]).toBe(deferred);

    const without = harness(
      [turn({ text: "", toolCalls: [call("c1", "spy")] }), turn({ text: "done" })],
      [spy],
    );
    await without.run();
    expect(seen[1]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Restart economics (DESIGN §13 cost model, issue #5): every restart is
// journaled with what it cost, so "restarts vs a compaction baseline" is a
// query over the log rather than a study. Audit-only — the model never sees it.
// ---------------------------------------------------------------------------

const restarts = (db: FakeDb): Extract<ThreadEventData, { type: "restart" }>[] =>
  db.events
    .map((e) => e.data)
    .filter((d): d is Extract<ThreadEventData, { type: "restart" }> => d.type === "restart");

/** A boundary written by a previous run, exactly as the shed tool writes one. */
const continuityEvent = (tokensBefore: number): ThreadEventData => ({
  type: "continuity",
  document: {
    goal: "carry on",
    plan: { done: ["shed"], now: "resume from the document", next: [] },
    workingSet: {},
    decisions: [],
    openLoops: [],
    lessons: [],
    memoryHints: [],
  },
  tokensBefore,
});

describe("runAgentLoop restart economics (DESIGN §13)", () => {
  test("a shed journals what the rebuild cost", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed from the document" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("a long conversation about widgets".repeat(40)));

    const result = await h.run();
    expect(result.stopReason).toBe("completed");

    const cont = h.db.events.find((e) => e.data.type === "continuity")!;
    const before = (cont.data as Extract<ThreadEventData, { type: "continuity" }>).tokensBefore;
    // The fresh window as the provider actually received it, system prompt
    // included: the cached prefix is re-paid as a cache WRITE after a restart.
    const fresh = h.provider.prompts[1]!.messages;
    const expected = estimateTokens([{ role: "system", text: "sys" }, ...fresh]);

    expect(restarts(h.db)).toEqual([
      {
        type: "restart",
        boundarySeq: cont.seq,
        tokensBefore: before,
        tokensAfter: expected,
        recallTokens: 0, // no memory plane on this run
        messages: 1, // the continuity document alone
      },
    ]);
    // The whole point of the restart: the successor window is far smaller.
    expect(expected).toBeLessThan(before);
  });

  test("a plain fresh thread journals no restart", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, ingress("first message ever"));
    await h.run();
    expect(restarts(h.db)).toEqual([]);
  });

  test("a successor wake on an unbilled boundary backfills one, and only one", async () => {
    // The shedding run stopped with `shed` before it could take a turn (or
    // died on the way out): the rebuild is paid for HERE, on the next wake.
    const h = harness([turn({ text: "ok" }), turn({ text: "ok again" })]);
    await h.store.append(THREAD, ingress("ancient business"));
    const boundary = await h.store.append(THREAD, continuityEvent(9_000));
    await h.store.append(THREAD, ingress("fresh question"));

    await h.run();
    const first = restarts(h.db);
    expect(first).toHaveLength(1);
    expect(first[0]!.boundarySeq).toBe(boundary.seq);
    expect(first[0]!.tokensBefore).toBe(9_000); // mirrors the continuity event
    expect(first[0]!.messages).toBe(2); // document + the fresh question
    expect(first[0]!.tokensAfter).toBeGreaterThan(0);
    expect(first[0]!.tokensAfter).toBeLessThan(9_000);
    // The restart event lands inside the window it describes, so the next wake
    // on the same boundary finds it and does not bill it again.
    expect(h.db.events.some((e) => e.data.type === "restart" && e.seq > boundary.seq)).toBe(true);

    await h.run();
    expect(restarts(h.db)).toHaveLength(1);
  });

  test("the injected <memories> block is counted in recallTokens", async () => {
    const h = harness([turn({ text: "ok" })]);
    await h.store.append(THREAD, continuityEvent(7_000));
    const store = new RecallStub({ hits: [memoryHit({ text: "widgets cost $4" })] });

    await h.run({ memory: memoryContext(store) });

    const block = h.provider.prompts[0]!.messages[0]!;
    expect(block.text.startsWith(RECALL_HEAD)).toBe(true);
    const [restart] = restarts(h.db);
    expect(restart!.recallTokens).toBe(estimateTokens([{ role: "user", text: block.text }]));
    expect(restart!.recallTokens).toBeGreaterThan(0);
    // Recall is part of what the fresh window costs, not something beside it.
    expect(restart!.tokensAfter).toBeGreaterThan(restart!.recallTokens);
    expect(restart!.messages).toBe(2); // <memories> block + the document
  });
});

// ---------------------------------------------------------------------------
// Prompt-cache alignment (DESIGN §4.5 cache alignment, §9 "tool set masked not
// mutated mid-window"). A provider cache is a PREFIX match over
// tools -> system -> messages, so what the loop is allowed to vary per turn is
// the tail and nothing else: the tool definitions stay put and forcing happens
// through `tool_choice`, and the cache routing key is the thread, not the turn.
// ---------------------------------------------------------------------------

describe("runAgentLoop prompt-cache alignment (DESIGN §4.5/§9)", () => {
  test("an unforced turn sends every tool and no toolChoice", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("c1", "echo")], stopReason: "tool_calls" }),
        turn({ text: "done" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("hello"));

    const result = await h.run();
    expect(result.stopReason).toBe("completed");
    for (const req of h.provider.prompts) {
      expect(req.tools).toEqual(["echo", "shed_context"]);
      expect(req.toolChoice).toBeUndefined(); // absent = the provider default
    }
  });

  test("the first forced turn keeps the tool list AND the cache: no tool_choice", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    // Over the hard rung (1100 tokens) on the first window, comfortably under
    // it on the rebuilt one — so the forcing is on exactly one turn.
    await h.store.append(THREAD, ingress("z".repeat(4400)));
    const result = await h.run({
      settings: settings({ advisoryFraction: 0.5, hardFraction: 0.55, approxWindowTokens: 2000 }),
    });
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    const [forced, after] = h.provider.prompts as [Prompt, Prompt];
    expect(forced.tools).toEqual(["echo", "shed_context"]);
    // Neither tier is invalidated on the way into the most expensive window a
    // thread ever has: the appended notice EXTENDS the prefix, while
    // `tool_choice` would re-bill the whole transcript uncached.
    expect(forced.toolChoice).toBeUndefined();
    // Same definitions before and after; only the messages tier grew.
    expect(after.tools).toEqual(forced.tools);
    expect(after.toolChoice).toBeUndefined();
    expect(after.system).toBe(forced.system);
  });

  test("only the RETRY pays for tool_choice", async () => {
    // First attempt talks instead of shedding; the second gets the mask, which
    // is the one moment the guarantee is worth an uncached re-read.
    const h = harness(
      [
        turn({ text: "I would rather not" }),
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    // Over the hard rung on the first window, under it on the rebuilt one.
    await h.store.append(THREAD, ingress("z".repeat(4400)));
    const result = await h.run({
      settings: settings({ advisoryFraction: 0.5, hardFraction: 0.55, approxWindowTokens: 2000 }),
    });
    expect(result.stopReason).toBe("completed");

    const [first, retry] = h.provider.prompts as [Prompt, Prompt];
    expect(first.toolChoice).toBeUndefined();
    expect(retry.toolChoice).toEqual({ type: "tool", name: "shed_context" });
    // The tool LIST never moves, on either attempt.
    expect(first.tools).toEqual(["echo", "shed_context"]);
    expect(retry.tools).toEqual(["echo", "shed_context"]);
    expect(notices(retry.messages).at(-1)).toContain("final attempt");
  });

  test("every turn carries the same thread-derived cacheKey, restart included", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("e1", "echo")], stopReason: "tool_calls" }),
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "done" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("hello"));

    const result = await h.run();
    expect(result.stopReason).toBe("completed");
    // Identical on every call: a key that moved per turn would scatter one
    // thread across cache shards, which is the opposite of the point.
    expect(h.provider.prompts.map((p) => p.cacheKey)).toEqual([
      "t1/c1/th1",
      "t1/c1/th1",
      "t1/c1/th1",
    ]);
  });

  test("the cacheKey follows the thread, not the process", async () => {
    const other: ThreadRef = { tenantId: "t2", channelId: "c9", threadId: "th9" };
    const h = harness([turn({ text: "ok" })]);
    await h.run({ thread: other });
    expect(h.provider.prompts[0]!.cacheKey).toBe("t2/c9/th9");
  });

  test("a window the event cap truncated forces a shed on the next turn", async () => {
    // The cap keeps the NEWEST events, so a truncated window's START rolls
    // forward with every event appended — the prefix changes at the front on
    // every turn and can never hit a cache again. Treat it as hard pressure:
    // a shed installs a continuity boundary the cap no longer touches.
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    // One event past the cap, and all but the last are audit-only: the window
    // loads truncated while the PROMPT stays tiny, so truncation is provably
    // the rung that fired (the token fraction is nowhere near).
    h.db.seed(THREAD, [
      ...Array.from({ length: DEFAULT_CONTEXT_EVENT_CAP }, (_, i) => ({
        type: "error" as const,
        source: "seed",
        message: `noise ${i}`,
        count: 1,
      })),
      ingress("the newest question"),
    ]);

    const result = await h.run();
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    const forced = h.provider.prompts[0]!;
    const tokens = estimateTokens([{ role: "system", text: "sys" }, ...forced.messages]);
    expect(tokens).toBeLessThan(0.9 * 180_000); // the token rung never fired
    expect(forced.toolChoice).toBeUndefined(); // first attempt: cache stays warm
    expect(forced.tools).toEqual(["echo", "shed_context"]);
    expect(notices(forced.messages)[0]).toContain("context limit reached");
    // The capped window is still reported in the log (audit-only).
    expect(
      h.db.events.some((e) => e.data.type === "error" && e.data.source === "context"),
    ).toBe(true);

    // The rung re-arms off the rebuilt window: a fresh boundary is not capped.
    const after = h.provider.prompts[1]!;
    expect(after.toolChoice).toBeUndefined();
    expect(notices(after.messages)).toHaveLength(0);
    expect(h.db.events.some((e) => e.data.type === "continuity")).toBe(true);
  });

  test("a truncated window with no shed tool degrades to normal operation", async () => {
    const h = harness([turn({ text: "fine" })], [echoTool]);
    h.db.seed(THREAD, [
      ...Array.from({ length: DEFAULT_CONTEXT_EVENT_CAP }, (_, i) => ({
        type: "error" as const,
        source: "seed",
        message: `noise ${i}`,
        count: 1,
      })),
      ingress("still answerable"),
    ]);

    const result = await h.run();
    expect(result).toEqual({ turns: 1, stopReason: "completed" });
    expect(h.provider.prompts[0]!.toolChoice).toBeUndefined();
    expect(notices(h.provider.prompts[0]!.messages)).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Reconstructable requests (DESIGN §3 "prompt = projection", §4.5 cache
// alignment). A provider cache is a prefix match, so wake N+1's request has to
// be a byte-EXTENSION of wake N's. That only holds if everything the loop put
// in front of the model is in the log: the <memories> block and the pressure
// notices used to live in the run's in-memory array alone, so the next wake
// rebuilt a different conversation and missed the cache for the whole thread.
// ---------------------------------------------------------------------------

/**
 * A run that does BOTH things the projection used to lose: an auto-recall
 * block at index 0 and an advisory notice mid-conversation.
 *
 * window 20 -> advisory at 14, and the <memories> block alone clears that. No
 * shed tool is registered, so the hard rung falls through to the advisory one.
 */
function recallHarness(): { h: Harness; store: RecallStub; opts: HarnessOverrides } {
  const h = harness(
    [
      // Argument names in the model's own order, which is neither sorted nor
      // jsonb's (length, bytes) — and `aa`/`b` make those two disagree, so the
      // fixture catches a missing canonicalization at EITHER end.
      turn({
        text: "",
        toolCalls: [call("c1", "echo", { zulu: 1, a: 2, mm: 3, aa: 4, b: 5 })],
        stopReason: "tool_calls",
      }),
      turn({ text: "done" }),
      turn({ text: "and again" }),
    ],
    [echoTool],
  );
  // Read back the way Postgres hands jsonb back, not the way JS stored it.
  h.db.reorderJson = true;
  const store = new RecallStub({ hits: [memoryHit({ text: "retries use jittered backoff" })] });
  return {
    h,
    store,
    opts: { memory: memoryContext(store), settings: settings({ approxWindowTokens: 20 }) },
  };
}

describe("runAgentLoop reconstructable requests (DESIGN §3/§4.5)", () => {
  test("the log projects back to exactly the request the provider received", async () => {
    const { h, store, opts } = recallHarness();
    await h.store.append(THREAD, ingress("how do we handle provider retries?"));

    const result = await h.run(opts);
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    const sent = h.provider.prompts.at(-1)!.messages;
    expect(sent[0]!.text.startsWith(RECALL_HEAD)).toBe(true); // the block
    expect(sent.some((m) => m.text.includes("context pressure"))).toBe(true); // the notice
    expect(store.searches).toHaveLength(1);

    // The whole claim, in one assertion: replaying the log reproduces the last
    // request byte for byte — plus the reply that request produced.
    expect(buildContext(h.db.events)).toEqual([
      ...sent,
      { role: "assistant", text: "done" },
    ]);
  });

  test("the next wake extends that request instead of rewriting it", async () => {
    const { h, store, opts } = recallHarness();
    await h.store.append(THREAD, ingress("how do we handle provider retries?"));
    await h.run(opts);
    // What the first run finished holding: its last request plus the reply.
    const finished: LlmMessage[] = [
      ...h.provider.prompts.at(-1)!.messages,
      { role: "assistant", text: "done" },
    ];

    await h.store.append(THREAD, ingress("and what about timeouts?"));
    await h.run(opts);

    // (a) Recall is once per WINDOW, not once per wake (DESIGN §5.4). A second
    // live search could return different hits or a different order and move
    // byte 0 of the conversation.
    expect(store.searches).toHaveLength(1);
    expect(memoryEvents(h.db)).toHaveLength(1);

    // (b) Everything the provider already cached comes back unchanged, in the
    // same slots, with the new turn appended after it.
    const next = h.provider.prompts.at(-1)!.messages;
    expect(next.slice(0, finished.length)).toEqual(finished);
    expect(next[0]!.text).toBe(finished[0]!.text); // byte 0 is the same block
    expect(next[finished.length]!.text).toContain("and what about timeouts?");
    // + the ingress, and nothing else: the advisory notice is already in the
    // window from wake 1, so this wake replays it rather than appending a
    // second copy (DESIGN §4.1 — once per WINDOW).
    expect(next).toHaveLength(finished.length + 1);
    expect(next.filter((m) => m.text.includes("context pressure"))).toHaveLength(1);

    // (c) BYTES, not shapes. `args` is the one part of a tool call a provider
    // serializes wholesale (Anthropic `input`, OpenAI `arguments`), and jsonb
    // handed it back in a different key order than the model wrote it in — one
    // byte's difference in a `tool_use` block breaks the prefix match from
    // there to the end of the transcript. Both ends canonicalize, so wake 2
    // re-serializes wake 1's call identically.
    const argBytes = (msgs: LlmMessage[]): string[] =>
      msgs.flatMap((m) => (m.toolCalls ?? []).map((c) => JSON.stringify(c.args)));
    expect(argBytes(next.slice(0, finished.length))).toEqual(argBytes(finished));
    expect(argBytes(finished)).toEqual(['{"a":2,"aa":4,"b":5,"mm":3,"zulu":1}']);
  });

  test("three wakes above the advisory line produce exactly ONE notice", async () => {
    // The reviewer's reproduction. `advisoryArmed` used to start `true` every
    // run, but notices are journaled and replayed — so each wake appended
    // another identical one and the window walked toward the hard rung on
    // harness text alone.
    const h = harness(() => turn({ text: "still here" }), [echoTool]);
    await h.store.append(THREAD, ingress("x".repeat(64))); // window 20 -> over 14
    const opts = { settings: settings({ approxWindowTokens: 20 }) };

    for (let wake = 0; wake < 3; wake++) {
      await h.store.append(THREAD, ingress(`wake ${wake}`));
      await h.run(opts);
    }

    expect(h.provider.prompts).toHaveLength(3);
    for (const req of h.provider.prompts) expect(notices(req.messages)).toHaveLength(1);
    expect(h.db.events.filter((e) => e.data.type === "notice")).toHaveLength(1);
    // Same slot every time, which is the point: the notice is part of the
    // prefix the provider cached on wake 1.
    const slot = h.provider.prompts[0]!.messages.findIndex((m) => m.text.startsWith("[harness notice]"));
    for (const req of h.provider.prompts) {
      expect(req.messages.findIndex((m) => m.text.startsWith("[harness notice]"))).toBe(slot);
    }
  });

  test("a shed opens a fresh window with its own block, and later wakes reuse it", async () => {
    const h = harness(
      [
        turn({ text: "", toolCalls: [call("s1", "shed_context", SHED_ARGS)], stopReason: "tool_calls" }),
        turn({ text: "resumed" }),
        turn({ text: "still here" }),
      ],
      [echoTool, new ShedContextTool()],
    );
    await h.store.append(THREAD, ingress("widget pricing question"));
    const store = new RecallStub({
      hitsByCall: [
        [memoryHit({ text: "the OLD window block" })],
        [memoryHit({ text: "the FRESH window block" })],
      ],
    });

    const result = await h.run({ memory: memoryContext(store) });
    expect(result).toEqual({ turns: 2, stopReason: "completed" });

    // Recall ran once per window: the initial one and the post-shed one.
    const mem = memoryEvents(h.db);
    expect(mem).toHaveLength(2);
    expect(mem[0]!.block).toContain("the OLD window block");
    expect(mem[1]!.block).toContain("the FRESH window block");

    // The post-shed window opens with the FRESH block; the pre-boundary one is
    // gone, exactly as the pre-boundary transcript is.
    const projected = buildContext(h.db.events);
    expect(projected[0]!.text).toBe(mem[1]!.block as string);
    expect(projected.some((m) => m.text.includes("the OLD window block"))).toBe(false);

    // A later wake on that window reproduces it from the log — no third search.
    await h.run({ memory: memoryContext(store) });
    expect(store.searches).toHaveLength(2);
    expect(h.provider.prompts.at(-1)!.messages[0]!.text).toBe(mem[1]!.block as string);
  });
});
