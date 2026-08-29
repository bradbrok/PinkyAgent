/**
 * ShedContextTool tests (DESIGN.md §4.2): the harness guard that refuses
 * empty/low-signal continuity documents, and the `continuity` event it emits.
 */
import { describe, expect, test } from "bun:test";
import type { Db, ThreadEventData, ThreadRef } from "@pinky/core";
import { ShedContextTool, SHED_CONTEXT_TOOL_NAME, continuityDocSchema, validateContinuityDoc } from "../src/continuity";
import type { ToolContext } from "../src/types";

const THREAD: ThreadRef = { tenantId: "t1", channelId: "c1", threadId: "th1" };

const unusedDb: Db = {
  query: async () => [],
  queryOne: async () => null,
  tx: async (fn) => fn(unusedDb),
  close: async () => {},
};

function context(contextTokens?: number): { ctx: ToolContext; emitted: ThreadEventData[] } {
  const emitted: ThreadEventData[] = [];
  const ctx: ToolContext = {
    cwd: "/tmp",
    db: unusedDb,
    thread: THREAD,
    emit: async (data) => {
      emitted.push(data);
    },
    agentId: "pinky",
    ...(contextTokens === undefined ? {} : { contextTokens }),
  };
  return { ctx, emitted };
}

const goodArgs = (): Record<string, unknown> => ({
  goal: "Fix the context ladder so long threads stop freezing",
  plan: { done: ["read DESIGN §4"], now: "rework the loop", next: ["tests", "typecheck"] },
  workingSet: {
    files: ["/src/loop.ts"],
    urls: ["https://example.com/spec"],
    tools: ["mcp__srv__create_issue"],
  },
  decisions: [{ what: "shed runs last in a batched turn", why: "cut-point safety (§4.5)" }],
  openLoops: ["confirm the hard fraction with a real model"],
  lessons: ["a system-role note churns the cache prefix"],
  memoryHints: ["continuity ladder", "projection boundary"],
  mood: "steady",
});

describe("validateContinuityDoc", () => {
  test("normalizes a full document (trimming strings)", () => {
    const args = goodArgs();
    args["goal"] = "  spaced goal  ";
    const out = validateContinuityDoc(args);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.doc.goal).toBe("spaced goal");
    expect(out.doc.plan).toEqual({
      done: ["read DESIGN §4"],
      now: "rework the loop",
      next: ["tests", "typecheck"],
    });
    expect(out.doc.workingSet.files).toEqual(["/src/loop.ts"]);
    // Deferred tool names (slice 9): the successor's header does not list
    // them, so the document is the only place they can survive a restart.
    expect(out.doc.workingSet.tools).toEqual(["mcp__srv__create_issue"]);
    expect(out.doc.decisions[0]).toEqual({
      what: "shed runs last in a batched turn",
      why: "cut-point safety (§4.5)",
    });
    expect(out.doc.mood).toBe("steady");
  });

  test("workingSet.tools is absent, not empty, when the document omits it", () => {
    // Same rule as files/artifacts/urls: only keys the author wrote appear, so
    // an empty Working Set renders no section at all.
    const out = validateContinuityDoc({ goal: "g", plan: { now: "n" }, workingSet: { files: ["/a"] } });
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.doc.workingSet).toEqual({ files: ["/a"] });
  });

  test("defaults every optional list to empty", () => {
    const out = validateContinuityDoc({ goal: "g", plan: { now: "n" } });
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.doc).toEqual({
      goal: "g",
      plan: { done: [], now: "n", next: [] },
      workingSet: {},
      decisions: [],
      openLoops: [],
      lessons: [],
      memoryHints: [],
    });
  });

  test.each([
    [{}, "goal must be a non-empty string"],
    [{ goal: "   ", plan: { now: "n" } }, "goal must be a non-empty string"],
    [{ goal: "g" }, "plan must be an object"],
    [{ goal: "g", plan: { now: "" } }, "plan.now must be a non-empty string"],
    [{ goal: "g", plan: { now: "n", done: "not a list" } }, "plan.done must be an array of strings"],
    [{ goal: "g", plan: { now: "n" }, openLoops: "nope" }, "openLoops must be an array of strings"],
    [{ goal: "g", plan: { now: "n" }, lessons: [1] }, "lessons[0] must be a non-empty string"],
    [{ goal: "g", plan: { now: "n" }, memoryHints: [""] }, "memoryHints[0] must be a non-empty string"],
    [{ goal: "g", plan: { now: "n" }, decisions: [{ what: "x" }] }, "decisions[0].why must be a non-empty string"],
    [{ goal: "g", plan: { now: "n" }, decisions: ["x"] }, "decisions[0] must be an object"],
    [{ goal: "g", plan: { now: "n" }, decisions: "x" }, "decisions must be an array"],
    [{ goal: "g", plan: { now: "n" }, workingSet: [] }, "workingSet must be an object"],
    [{ goal: "g", plan: { now: "n" }, workingSet: { files: [7] } }, "workingSet.files[0] must be a non-empty string"],
    [{ goal: "g", plan: { now: "n" }, workingSet: { tools: "tool_call" } }, "workingSet.tools must be an array of strings"],
    [{ goal: "g", plan: { now: "n" }, workingSet: { tools: [""] } }, "workingSet.tools[0] must be a non-empty string"],
    [{ goal: "g", plan: { now: "n" }, mood: 3 }, "mood must be a string"],
  ])("rejects %o", (args, message) => {
    const out = validateContinuityDoc(args);
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.error).toContain(message);
  });

  test("rejects a non-object argument payload", () => {
    expect(validateContinuityDoc(null).ok).toBe(false);
    expect(validateContinuityDoc("{}").ok).toBe(false);
    expect(validateContinuityDoc([]).ok).toBe(false);
  });
});

describe("ShedContextTool", () => {
  test("exposes the ContinuityDoc schema under the shed_context name", () => {
    const tool = new ShedContextTool();
    expect(tool.name).toBe(SHED_CONTEXT_TOOL_NAME);
    expect(tool.name).toBe("shed_context");
    expect(tool.parameters).toBe(continuityDocSchema);
    const props = (tool.parameters as { properties: Record<string, unknown> }).properties;
    expect(Object.keys(props).sort()).toEqual([
      "decisions",
      "goal",
      "lessons",
      "memoryHints",
      "mood",
      "openLoops",
      "plan",
      "workingSet",
    ]);
    expect((tool.parameters as { required: string[] }).required).toEqual(["goal", "plan"]);
    // workingSet carries deferred tool names alongside files/artifacts/urls
    // (slice 9): they are NOT in the successor's header, so a document that
    // omitted them would make it rediscover the tool through tool_search.
    const workingSet = props["workingSet"] as { properties: Record<string, unknown> };
    expect(Object.keys(workingSet.properties).sort()).toEqual([
      "artifacts",
      "files",
      "tools",
      "urls",
    ]);
    expect((workingSet.properties["tools"] as { description: string }).description).toContain(
      "tool_describe",
    );
    expect(tool.description.split("\n")[0]!.length).toBeGreaterThan(20);
  });

  test("emits a continuity event carrying the loop's token estimate", async () => {
    const { ctx, emitted } = context(48_000);
    const result = await new ShedContextTool().execute(goodArgs(), ctx);
    expect(result.isError).toBeUndefined();
    expect(result.text).toContain("rework the loop");
    expect(emitted).toHaveLength(1);
    const evt = emitted[0] as Extract<ThreadEventData, { type: "continuity" }>;
    expect(evt.type).toBe("continuity");
    expect(evt.tokensBefore).toBe(48_000);
    expect(evt.document.goal).toContain("Fix the context ladder");
    expect(evt.document.plan.now).toBe("rework the loop");
  });

  test("tokensBefore falls back to 0 when the caller supplies no estimate", async () => {
    const { ctx, emitted } = context();
    await new ShedContextTool().execute(goodArgs(), ctx);
    const evt = emitted[0] as Extract<ThreadEventData, { type: "continuity" }>;
    expect(evt.tokensBefore).toBe(0);
  });

  test("an invalid document is rejected with a specific message and writes nothing", async () => {
    const { ctx, emitted } = context(1000);
    const result = await new ShedContextTool().execute({ goal: "g", plan: { now: "" } }, ctx);
    expect(result.isError).toBe(true);
    expect(result.text).toContain("plan.now must be a non-empty string");
    expect(result.text).toContain("call shed_context again");
    expect(emitted).toHaveLength(0);
  });
});
