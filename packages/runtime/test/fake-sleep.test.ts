/**
 * `fake/sleep` — the sleep-time worker's three forced calls, scripted (slice 6).
 *
 * Unlike every other fake behavior this one answers the REQUEST, not the
 * conversation: each pass is a one-shot `complete()` carrying one user message
 * and one forced tool, so the route keys on `opts.tools[0].name`. These tests
 * therefore drive it exactly as `packages/sleep` will — a realistic
 * CompleteOptions per call — and pin the branches smoke and the CLI e2e rely on:
 * the `remember:` scan (and its cap), the ADD/NOOP rule that makes a second
 * sweep idempotent, the insight text, and the fact that a malformed payload is
 * answered rather than thrown.
 */
import { describe, expect, test } from "bun:test";
import {
  FakeProvider,
  FAKE_BEHAVIORS,
  FAKE_SLEEP_REFLECT_PREFIX,
  FAKE_SLEEP_REMEMBER_RE,
  createFakeProvider,
} from "../src/providers/fake";
import { createProvider } from "../src/providers/index";
import type { AssistantTurn, CompleteOptions, LlmMessage, ToolSpec } from "../src/types";

const CANDIDATE = { kind: "semantic", importance: 7, visibility: "channel" } as const;

function toolSpec(name: string): ToolSpec {
  return { name, description: `${name} (test)`, parameters: { type: "object" } };
}

/** What the worker posts: one user message, one forced tool, nothing else. */
function opts(tool: string, text: string): CompleteOptions {
  return {
    model: "sleep-model",
    system: "SLEEP SYSTEM",
    messages: [{ role: "user", text }],
    tools: [toolSpec(tool)],
    toolChoice: { type: "tool", name: tool },
    maxTokens: 2048,
  };
}

/** The one tool call of a scripted turn, with the turn shape asserted first. */
function onlyCall(turn: AssistantTurn, name: string): { id: string; args: Record<string, unknown> } {
  expect(turn.stopReason).toBe("tool_calls");
  expect(turn.toolCalls.map((c) => c.name)).toEqual([name]);
  const call = turn.toolCalls[0]!;
  return { id: call.id, args: call.args };
}

describe("fake/sleep — extract", () => {
  test("one candidate per `remember:` line, trimmed, case-insensitive", async () => {
    const provider = createFakeProvider("sleep");
    // A rendered transcript (packages/sleep/src/transcript.ts), not bare text.
    const transcript = [
      "[1] user jsonl:local: remember: the smoke sleep canary is amber-falcon",
      "[2] assistant: noted",
      "  -> retain({})",
      "[3] user jsonl:local: REMEMBER:   trailing space is trimmed   ",
      "[4] tool read: nothing durable here",
    ].join("\n");

    const { id, args } = onlyCall(
      await provider.complete(opts("extract_memories", transcript)),
      "extract_memories",
    );
    // The id counts what the turn carries, so it changes with the payload.
    expect(id).toBe("fake-extract-2");
    expect(args).toEqual({
      candidates: [
        { text: "the smoke sleep canary is amber-falcon", ...CANDIDATE },
        { text: "trailing space is trimmed", ...CANDIDATE },
      ],
    });
  });

  test("a transcript with nothing to remember yields zero candidates", async () => {
    const provider = createFakeProvider("sleep");
    const { id, args } = onlyCall(
      await provider.complete(opts("extract_memories", "[1] user a:b: hello\n[2] assistant: hi")),
      "extract_memories",
    );
    // Still a tool call: the worker journals a receipt for an empty pass so the
    // cursor moves, and that only happens if the call itself was valid.
    expect(id).toBe("fake-extract-0");
    expect(args).toEqual({ candidates: [] });
  });

  test("caps at 12 candidates and re-reading the same transcript is identical", async () => {
    const provider = createFakeProvider("sleep");
    const transcript = Array.from({ length: 15 }, (_, i) => `[${i}] user a:b: remember: fact ${i}`).join("\n");

    const first = onlyCall(
      await provider.complete(opts("extract_memories", transcript)),
      "extract_memories",
    );
    expect(first.id).toBe("fake-extract-12");
    expect((first.args.candidates as unknown[]).length).toBe(12);
    expect((first.args.candidates as { text: string }[])[11]!.text).toBe("fact 11");

    // The regression this guards: the exported pattern is /g, so a scan that
    // stops at the cap would leave `lastIndex` mid-string and the NEXT pass over
    // the same events would extract a different set.
    const second = onlyCall(
      await provider.complete(opts("extract_memories", transcript)),
      "extract_memories",
    );
    expect(second.args).toEqual(first.args);
    expect(FAKE_SLEEP_REMEMBER_RE.lastIndex).toBe(0);
  });
});

describe("fake/sleep — decide", () => {
  const payload = (candidates: unknown[]): string => JSON.stringify({ candidates });

  test("NOOP when a neighbor's text matches exactly, ADD otherwise", async () => {
    const provider = createFakeProvider("sleep");
    const { id, args } = onlyCall(
      await provider.complete(
        opts(
          "decide_memory_updates",
          payload([
            {
              index: 0,
              text: "the canary is amber-falcon",
              ...CANDIDATE,
              neighbors: [
                { id: "m-other", text: "something else", kind: "semantic", importance: 5, recordedAt: "2026-08-01T00:00:00Z" },
                { id: "m-same", text: "the canary is amber-falcon", kind: "semantic", importance: 7, recordedAt: "2026-08-02T00:00:00Z" },
              ],
            },
            { index: 1, text: "a brand new fact", ...CANDIDATE, neighbors: [] },
          ]),
        ),
      ),
      "decide_memory_updates",
    );
    expect(id).toBe("fake-decide-2");
    // Exactly one decision per candidate index — the worker's validator rejects
    // anything else.
    expect(args).toEqual({
      decisions: [
        { candidate: 0, action: "NOOP" },
        { candidate: 1, action: "ADD" },
      ],
    });
  });

  test("a near-miss neighbor is still an ADD (equality, never similarity)", async () => {
    const provider = createFakeProvider("sleep");
    const { args } = onlyCall(
      await provider.complete(
        opts(
          "decide_memory_updates",
          payload([
            {
              index: 0,
              text: "the canary is amber-falcon",
              ...CANDIDATE,
              neighbors: [{ id: "m1", text: "the canary is amber-falcon.", kind: "semantic", importance: 7, recordedAt: "2026-08-02T00:00:00Z" }],
            },
          ]),
        ),
      ),
      "decide_memory_updates",
    );
    expect(args).toEqual({ decisions: [{ candidate: 0, action: "ADD" }] });
  });

  test("the payload's own index is what a decision names", async () => {
    const provider = createFakeProvider("sleep");
    const { args } = onlyCall(
      await provider.complete(
        opts("decide_memory_updates", payload([{ index: 7, text: "x", ...CANDIDATE, neighbors: [] }])),
      ),
      "decide_memory_updates",
    );
    expect(args).toEqual({ decisions: [{ candidate: 7, action: "ADD" }] });
  });

  test("a malformed payload is answered, not thrown", async () => {
    const provider = createFakeProvider("sleep");
    const turn = await provider.complete(opts("decide_memory_updates", "{not json"));
    // No tool call => the worker reports one clean `failed`; a throw would come
    // out of the middle of a sweep instead.
    expect(turn).toEqual({ text: "fake/sleep: unexpected call", toolCalls: [], stopReason: "stop" });
  });

  test("valid JSON without a candidates array is off-script too", async () => {
    const provider = createFakeProvider("sleep");
    const turn = await provider.complete(opts("decide_memory_updates", JSON.stringify({ memories: [] })));
    expect(turn.toolCalls).toEqual([]);
    expect(turn.text).toBe("fake/sleep: unexpected call");
  });
});

describe("fake/sleep — reflect", () => {
  const rows = [
    { id: "mem-1", text: "first fact", kind: "semantic", importance: 7, visibility: "channel", channelId: "c1", recordedAt: "2026-08-01T00:00:00Z" },
    { id: "mem-2", text: "second fact", kind: "semantic", importance: 5, visibility: "tenant", recordedAt: "2026-08-02T00:00:00Z" },
    { id: "mem-3", text: "third fact", kind: "episodic", importance: 4, visibility: "channel", channelId: "c2", recordedAt: "2026-08-03T00:00:00Z" },
  ];

  test("one insight over the batch, citing every id and superseding none", async () => {
    const provider = createFakeProvider("sleep");
    const { id, args } = onlyCall(
      await provider.complete(opts("reflect_memories", JSON.stringify({ memories: rows }))),
      "reflect_memories",
    );
    expect(id).toBe("fake-reflect-1");
    expect(args).toEqual({
      insights: [
        {
          text: `${FAKE_SLEEP_REFLECT_PREFIX} 3 memories: first fact`,
          importance: 5,
          sources: ["mem-1", "mem-2", "mem-3"],
          supersedes: [],
        },
      ],
    });
  });

  test("an empty batch returns zero insights", async () => {
    const provider = createFakeProvider("sleep");
    const { id, args } = onlyCall(
      await provider.complete(opts("reflect_memories", JSON.stringify({ memories: [] }))),
      "reflect_memories",
    );
    expect(id).toBe("fake-reflect-0");
    expect(args).toEqual({ insights: [] });
  });

  test("a malformed payload is answered, not thrown", async () => {
    const provider = createFakeProvider("sleep");
    const turn = await provider.complete(opts("reflect_memories", "]["));
    expect(turn).toEqual({ text: "fake/sleep: unexpected call", toolCalls: [], stopReason: "stop" });
  });
});

describe("fake/sleep — routing", () => {
  test("any other forced tool, or none, is an unexpected call", async () => {
    const provider = createFakeProvider("sleep");
    const other = await provider.complete(opts("recall", "hello"));
    expect(other).toEqual({ text: "fake/sleep: unexpected call", toolCalls: [], stopReason: "stop" });
    const bare = await provider.complete({ ...opts("extract_memories", "remember: x"), tools: [] });
    expect(bare.toolCalls).toEqual([]);
    expect(bare.text).toBe("fake/sleep: unexpected call");
  });

  test("createProvider routes fake/sleep with no key and lists it on a typo", () => {
    const env: Record<string, string | undefined> = {};
    expect(createProvider("fake/sleep", env).name).toBe("fake");
    expect(FAKE_BEHAVIORS).toContain("sleep");
    expect(() => createProvider("fake/sleepy", env)).toThrow(/sleep/);
  });
});

describe("FakeScript arity", () => {
  test("a two-argument script gets the CompleteOptions; a one-argument script still works", async () => {
    const seen: CompleteOptions[] = [];
    const two = new FakeProvider((messages, o) => {
      seen.push(o);
      return { text: `${messages.length}:${o.tools[0]?.name ?? "-"}`, toolCalls: [], stopReason: "stop" };
    });
    expect((await two.complete(opts("extract_memories", "hi"))).text).toBe("1:extract_memories");
    expect(seen[0]?.toolChoice).toEqual({ type: "tool", name: "extract_memories" });

    // The pre-slice-6 signature: every existing script in the repo is this
    // shape, and widening the type must not have changed what it receives.
    const one = new FakeProvider((messages: LlmMessage[]): AssistantTurn => ({
      text: `echo: ${messages[messages.length - 1]?.text ?? ""}`,
      toolCalls: [],
      stopReason: "stop",
    }));
    expect((await one.complete(opts("extract_memories", "hi"))).text).toBe("echo: hi");
  });
});

describe("fake/sleep — reflect stays inside the schema", () => {
  test("a very long source row is truncated, prefix intact", async () => {
    const provider = createFakeProvider("sleep");
    const { args } = onlyCall(
      await provider.complete(
        opts(
          "reflect_memories",
          JSON.stringify({ memories: [{ id: "mem-long", text: "x".repeat(4000) }] }),
        ),
      ),
      "reflect_memories",
    );
    const text = (args.insights as { text: string }[])[0]!.text;
    // maxLength 1500 in reflect_memories: quoting the row verbatim would fail
    // the worker's validator and report a `failed` pass over nothing.
    expect(text.length).toBe(1500);
    expect(text.startsWith(`${FAKE_SLEEP_REFLECT_PREFIX} 1 memories: `)).toBe(true);
  });
});
