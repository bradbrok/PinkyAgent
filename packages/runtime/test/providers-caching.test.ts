/**
 * Anthropic prompt-cache request shape (DESIGN.md §4.5 cache alignment, §9
 * "KV-cache thrash") and the OpenAI-compatible stream_options escape hatch.
 */
import { describe, expect, test } from "bun:test";
import { AnthropicProvider, buildSystemBlocks, type AnthropicUsage } from "../src/providers/anthropic";
import { OpenAIProvider } from "../src/providers/openai";
import { createProvider, SUPPORTED_PROVIDERS, transportOptionsFromEnv } from "../src/providers/index";
import { sseStreamFromText } from "../src/providers/sse";
import type { CompleteOptions } from "../src/types";

const OPTS: CompleteOptions = {
  model: "test-model",
  system: "STABLE PREFIX",
  messages: [{ role: "user", text: "hi" }],
  tools: [],
};

function sseResponse(body: string): Response {
  return new Response(sseStreamFromText(body), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/** Capture the single request body a provider posts. */
function capturing(body: string): {
  fetchFn: typeof fetch;
  read: () => Record<string, unknown>;
} {
  let captured: Record<string, unknown> | null = null;
  const fetchFn = (async (_url: string | URL | Request, init?: RequestInit) => {
    captured = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return sseResponse(body);
  }) as unknown as typeof fetch;
  return { fetchFn, read: () => captured! };
}

const ANTHROPIC_DONE =
  'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n';
const OPENAI_DONE = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

describe("buildSystemBlocks", () => {
  test("caches the stable prefix and leaves mid-conversation suffixes uncached", () => {
    expect(buildSystemBlocks("STABLE", ["turn note", "second note"])).toEqual([
      { type: "text", text: "STABLE", cache_control: { type: "ephemeral" } },
      { type: "text", text: "turn note" },
      { type: "text", text: "second note" },
    ]);
  });

  test("no stable prefix means no breakpoint at all", () => {
    expect(buildSystemBlocks("", ["only a suffix"])).toEqual([{ type: "text", text: "only a suffix" }]);
    expect(buildSystemBlocks("", [])).toEqual([]);
  });

  test("cacheSystem=false drops cache_control (proxies that reject it)", () => {
    expect(buildSystemBlocks("STABLE", ["note"], false)).toEqual([
      { type: "text", text: "STABLE" },
      { type: "text", text: "note" },
    ]);
  });
});

describe("AnthropicProvider request shape", () => {
  test("system is an array: cache_control on the first block, suffix blocks without it", async () => {
    const { fetchFn, read } = capturing(ANTHROPIC_DONE);
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn });
    await provider.complete({
      ...OPTS,
      messages: [
        { role: "user", text: "hi" },
        { role: "system", text: "mid-conversation operator note" },
        { role: "user", text: "again" },
      ],
    });
    const body = read();
    expect(body.system).toEqual([
      { type: "text", text: "STABLE PREFIX", cache_control: { type: "ephemeral" } },
      { type: "text", text: "mid-conversation operator note" },
    ]);
    // The suffix must never appear inside the cached prefix block.
    const blocks = body.system as { text: string; cache_control?: unknown }[];
    expect(blocks[0]!.text).not.toContain("mid-conversation");
    expect(blocks[1]!.cache_control).toBeUndefined();
    // Suffix text stays out of the messages array (it is system-role, not a
    // turn). The single message's last block carries the conversation
    // breakpoint — see "AnthropicProvider conversation caching" below.
    expect(body.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "hi" },
          { type: "text", text: "again", cache_control: { type: "ephemeral" } },
        ],
      },
    ]);
  });

  test("the cached prefix is byte-identical across turns (no thrash)", async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetchFn = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return sseResponse(ANTHROPIC_DONE);
    }) as unknown as typeof fetch;
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn });
    await provider.complete(OPTS);
    await provider.complete({
      ...OPTS,
      messages: [...OPTS.messages, { role: "system", text: "turn-specific note" }],
    });
    const first = (bodies[0]!.system as unknown[])[0];
    const second = (bodies[1]!.system as unknown[])[0];
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  test("records cache_read/cache_creation input tokens from message_start", async () => {
    const stream =
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":40}}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n';
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn: (async () => sseResponse(stream)) as unknown as typeof fetch,
    });
    const turn = await provider.complete(OPTS);
    const usage = turn.usage as AnthropicUsage | undefined;
    expect(usage).toEqual({ input: 12, output: 5, cacheRead: 900, cacheCreation: 40 });
  });

  test("omits the cache counters when the API does not report them", async () => {
    const stream =
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n';
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn: (async () => sseResponse(stream)) as unknown as typeof fetch,
    });
    expect((await provider.complete(OPTS)).usage).toEqual({ input: 12, output: 5 });
  });

  test("max_tokens: 64k streaming default, overridable per provider and per call", async () => {
    const a = capturing(ANTHROPIC_DONE);
    await new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn: a.fetchFn }).complete(
      OPTS,
    );
    expect(a.read().max_tokens).toBe(64_000);

    const b = capturing(ANTHROPIC_DONE);
    await new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn: b.fetchFn,
      maxTokens: 8192,
    }).complete(OPTS);
    expect(b.read().max_tokens).toBe(8192);

    const c = capturing(ANTHROPIC_DONE);
    await new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn: c.fetchFn }).complete({
      ...OPTS,
      maxTokens: 1024,
    });
    expect(c.read().max_tokens).toBe(1024);
  });
});

describe("OpenAIProvider stream_options", () => {
  test("includeUsage defaults to true", async () => {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await new OpenAIProvider({ apiKey: "k", baseUrl: "https://oai.test/v1", fetchFn }).complete(OPTS);
    expect(read().stream_options).toEqual({ include_usage: true });
  });

  test("includeUsage=false omits stream_options entirely", async () => {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      includeUsage: false,
    }).complete(OPTS);
    expect(read()).not.toHaveProperty("stream_options");
    expect(read().stream).toBe(true);
  });
});

describe("createProvider transport env", () => {
  test("reads PINKY_LLM_MAX_RETRIES / PINKY_LLM_TIMEOUT_MS", () => {
    expect(transportOptionsFromEnv({ PINKY_LLM_MAX_RETRIES: "5", PINKY_LLM_TIMEOUT_MS: "45000" })).toEqual({
      maxRetries: 5,
      timeoutMs: 45_000,
    });
    expect(transportOptionsFromEnv({})).toEqual({ maxRetries: undefined, timeoutMs: undefined });
    expect(transportOptionsFromEnv({ PINKY_LLM_MAX_RETRIES: "" })).toEqual({
      maxRetries: undefined,
      timeoutMs: undefined,
    });
  });

  test("malformed transport env fails loudly at provider construction", () => {
    expect(() => createProvider("anthropic/claude-opus-5", { PINKY_LLM_MAX_RETRIES: "lots" })).toThrow(
      /PINKY_LLM_MAX_RETRIES must be a non-negative integer/,
    );
    expect(() => createProvider("anthropic/claude-opus-5", { PINKY_LLM_TIMEOUT_MS: "-1" })).toThrow(
      /PINKY_LLM_TIMEOUT_MS must be a non-negative integer/,
    );
    expect(() => createProvider("openai/gpt-4o", { PINKY_LLM_INCLUDE_USAGE: "maybe" })).toThrow(
      /PINKY_LLM_INCLUDE_USAGE must be a boolean/,
    );
  });

  test("accepts the documented env values on every route", () => {
    const env = {
      ANTHROPIC_API_KEY: "a",
      OPENAI_API_KEY: "o",
      OPENROUTER_API_KEY: "r",
      PINKY_LLM_MAX_RETRIES: "0",
      PINKY_LLM_TIMEOUT_MS: "1000",
      PINKY_LLM_INCLUDE_USAGE: "false",
    };
    expect(createProvider("anthropic/claude-opus-5", env).name).toBe("anthropic");
    expect(createProvider("openai/gpt-4o", env).name).toBe("openai");
    expect(createProvider("openrouter/moonshotai/kimi-k2", env).name).toBe("openai");
  });
});

// The `fake/*` route is what makes a keyless end-to-end run possible (headless
// JSONL e2e, smoke): no API key is consulted and no request is ever made.
describe("createProvider fake route", () => {
  const NO_ENV: Record<string, string | undefined> = {};

  test("fake/echo echoes the last user message with no key configured", async () => {
    const provider = createProvider("fake/echo", NO_ENV);
    expect(provider.name).toBe("fake");
    const turn = await provider.complete({
      ...OPTS,
      messages: [
        { role: "user", text: "[harness notice] Recalled memories …" },
        { role: "assistant", text: "ok" },
        { role: "user", text: "[jsonl local]: hello there" },
      ],
    });
    expect(turn.text).toBe("echo: [jsonl local]: hello there");
    expect(turn.toolCalls).toEqual([]);
    expect(turn.stopReason).toBe("stop");
  });

  test("fake/retain-recall scripts retain -> recall -> text", async () => {
    const provider = createProvider("fake/retain-recall", NO_ENV);
    const first = await provider.complete(OPTS);
    expect(first.toolCalls.map((c) => c.name)).toEqual(["retain"]);
    expect(String((first.toolCalls[0]!.args as { text: string }).text)).toContain("zebra-quartz");
    const second = await provider.complete(OPTS);
    expect(second.toolCalls.map((c) => c.name)).toEqual(["recall"]);
    const third = await provider.complete(OPTS);
    expect(third.toolCalls).toEqual([]);
    expect(third.stopReason).toBe("stop");
  });

  test("an unknown fake behavior names the supported ones", () => {
    expect(() => createProvider("fake/nope", NO_ENV)).toThrow(/Supported: echo, retain-recall/);
  });

  test("fake is listed as a supported provider", () => {
    expect(SUPPORTED_PROVIDERS).toContain("fake");
  });
});

// ---------------------------------------------------------------------------
// OpenAI-compatible prompt caching (automatic prefix caching).
// Nothing is sent to opt in — only a byte-stable prefix and, on OpenAI, the
// `prompt_cache_key` routing hint. The work is on the way back: these routes
// count cached tokens INSIDE `prompt_tokens`, while `TokenUsage` is disjoint
// (input + cacheRead + cacheCreation = total prompt), which is the convention
// `pinky stats restarts` divides by.
// ---------------------------------------------------------------------------
describe("OpenAIProvider cache usage mapping", () => {
  /** Drive one stream whose final chunk carries `usage`, return the turn's usage. */
  async function usageFrom(usageJson: string) {
    const stream =
      'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n' +
      `data: {"choices":[],"usage":${usageJson}}\n\n` +
      OPENAI_DONE;
    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn: (async () => sseResponse(stream)) as unknown as typeof fetch,
    });
    return (await provider.complete(OPTS)).usage;
  }

  test("OpenAI/OpenRouter: cached_tokens becomes cacheRead and leaves input uncached", async () => {
    expect(
      await usageFrom(
        '{"prompt_tokens":1200,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":1000}}',
      ),
    ).toEqual({ input: 200, output: 20, cacheRead: 1000 });
  });

  test("DeepSeek: cacheRead is the hit count and input is exactly the miss count", async () => {
    // prompt_tokens = hit + miss on this route, so the subtraction must
    // reproduce prompt_cache_miss_tokens without ever reading that field.
    expect(
      await usageFrom(
        '{"prompt_tokens":1000,"completion_tokens":7,"prompt_cache_hit_tokens":768,"prompt_cache_miss_tokens":232}',
      ),
    ).toEqual({ input: 232, output: 7, cacheRead: 768 });
  });

  test("OpenRouter Anthropic passthrough: cache_write_tokens becomes cacheCreation", async () => {
    expect(
      await usageFrom(
        '{"prompt_tokens":1500,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":1000,"cache_write_tokens":300}}',
      ),
    ).toEqual({ input: 200, output: 9, cacheRead: 1000, cacheCreation: 300 });
  });

  test("a route reporting no cache counters leaves the KEYS ABSENT, not undefined", async () => {
    // "nothing was cached" and "nobody counted" are different answers to the
    // DESIGN §13 cost question — cacheWriteShare() in the CLI keys on null.
    const usage = (await usageFrom('{"prompt_tokens":300,"completion_tokens":4}'))!;
    expect(usage).toEqual({ input: 300, output: 4 });
    expect("cacheRead" in usage).toBe(false);
    expect("cacheCreation" in usage).toBe(false);
  });

  test("miscounting route cannot push input negative", async () => {
    expect(
      await usageFrom(
        '{"prompt_tokens":100,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":900}}',
      ),
    ).toEqual({ input: 0, output: 1, cacheRead: 900 });
  });
});

describe("OpenAIProvider prompt_cache_key", () => {
  const provider = (fetchFn: typeof fetch, promptCacheKey?: boolean) =>
    new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      ...(promptCacheKey === undefined ? {} : { promptCacheKey }),
    });

  test("sends the caller's cacheKey verbatim", async () => {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await provider(fetchFn).complete({ ...OPTS, cacheKey: "tenant/chan/thread" });
    expect(read().prompt_cache_key).toBe("tenant/chan/thread");
  });

  test("omitted when the caller supplies none", async () => {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await provider(fetchFn).complete(OPTS);
    expect(read()).not.toHaveProperty("prompt_cache_key");
  });

  test("promptCacheKey=false omits it even when supplied (endpoints that reject unknown fields)", async () => {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await provider(fetchFn, false).complete({ ...OPTS, cacheKey: "tenant/chan/thread" });
    expect(read()).not.toHaveProperty("prompt_cache_key");
  });
});

describe("OpenAIProvider tool_choice", () => {
  const TOOLS = [{ name: "recall", description: "d", parameters: { type: "object" } }];
  const withTools = { ...OPTS, tools: TOOLS };

  async function bodyFor(opts: CompleteOptions): Promise<Record<string, unknown>> {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    await new OpenAIProvider({ apiKey: "k", baseUrl: "https://oai.test/v1", fetchFn }).complete(opts);
    return read();
  }

  test('{type:"auto"} → "auto"', async () => {
    expect((await bodyFor({ ...withTools, toolChoice: { type: "auto" } })).tool_choice).toBe("auto");
  });

  test('{type:"none"} → "none"', async () => {
    expect((await bodyFor({ ...withTools, toolChoice: { type: "none" } })).tool_choice).toBe("none");
  });

  test('{type:"tool"} → the OpenAI function shape', async () => {
    const body = await bodyFor({ ...withTools, toolChoice: { type: "tool", name: "shed_context" } });
    expect(body.tool_choice).toEqual({ type: "function", function: { name: "shed_context" } });
    // The tool LIST is unchanged: masking must not invalidate the tools cache tier.
    expect((body.tools as unknown[]).length).toBe(1);
  });

  test("omitted when the caller sets none, and when there are no tools to choose from", async () => {
    expect(await bodyFor(withTools)).not.toHaveProperty("tool_choice");
    expect(await bodyFor({ ...OPTS, toolChoice: { type: "none" } })).not.toHaveProperty("tool_choice");
  });
});

// ---------------------------------------------------------------------------
// Conversation cache breakpoints (the rolling two-message window), the ttl
// knob, and tool_choice. Imports live here rather than at the head of the file
// because this block was appended alongside another author's; ESM hoists them.
// ---------------------------------------------------------------------------
import {
  applyMessageCacheBreakpoints,
  cacheControl,
  toAnthropicToolChoice,
  type AnthropicMessage,
} from "../src/providers/anthropic";
import { anthropicCacheTtlFromEnv } from "../src/providers/index";

/** Every cache_control anywhere in the request, however deeply nested. */
function markers(value: unknown, found: Record<string, unknown>[] = []): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    for (const item of value) markers(item, found);
    return found;
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (obj.cache_control) found.push(obj.cache_control as Record<string, unknown>);
    for (const child of Object.values(obj)) markers(child, found);
  }
  return found;
}

const TOOL = { name: "t", description: "d", parameters: { type: "object" } };

/** user "hi" / assistant text+tool_use / (tool_result + user) — 3 messages. */
const CONVERSATION: CompleteOptions = {
  ...OPTS,
  tools: [TOOL],
  messages: [
    { role: "user", text: "hi" },
    { role: "assistant", text: "sure", toolCalls: [{ id: "c1", name: "t", args: { a: 1 } }] },
    { role: "tool", toolCallId: "c1", text: "ok" },
    { role: "user", text: "next" },
  ],
};

describe("applyMessageCacheBreakpoints", () => {
  const msgs = (): AnthropicMessage[] => [
    { role: "user", content: [{ type: "text", text: "one" }] },
    { role: "assistant", content: [{ type: "text", text: "two" }] },
    { role: "user", content: [{ type: "text", text: "three" }, { type: "text", text: "four" }] },
  ];

  test("marks the last content block of the last two messages", () => {
    expect(applyMessageCacheBreakpoints(msgs())).toEqual([
      { role: "user", content: [{ type: "text", text: "one" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "two", cache_control: { type: "ephemeral" } }],
      },
      {
        role: "user",
        content: [
          { type: "text", text: "three" },
          { type: "text", text: "four", cache_control: { type: "ephemeral" } },
        ],
      },
    ]);
  });

  test("a single-message conversation gets exactly one marker", () => {
    const out = applyMessageCacheBreakpoints([
      { role: "user", content: [{ type: "text", text: "only" }] },
    ]);
    expect(markers(out)).toEqual([{ type: "ephemeral" }]);
  });

  test("no messages, or cache off, means no markers", () => {
    expect(applyMessageCacheBreakpoints([])).toEqual([]);
    expect(markers(applyMessageCacheBreakpoints(msgs(), false))).toEqual([]);
  });

  test("skips content-less messages and keeps looking back", () => {
    const out = applyMessageCacheBreakpoints([
      { role: "user", content: [{ type: "text", text: "one" }] },
      { role: "assistant", content: [{ type: "text", text: "two" }] },
      { role: "user", content: [] },
    ]);
    expect(markers(out)).toHaveLength(2);
    expect(out[0]!.content[0]!.cache_control).toEqual({ type: "ephemeral" });
    expect(out[1]!.content[0]!.cache_control).toEqual({ type: "ephemeral" });
  });

  test("is pure: the caller's messages and blocks are never stamped", () => {
    const input = msgs();
    const snapshot = JSON.stringify(input);
    applyMessageCacheBreakpoints(input, true, "1h");
    expect(JSON.stringify(input)).toBe(snapshot);
  });

  test("the ttl reaches every marker (mixed TTLs are not a thing)", () => {
    expect(markers(applyMessageCacheBreakpoints(msgs(), true, "1h"))).toEqual([
      { type: "ephemeral", ttl: "1h" },
      { type: "ephemeral", ttl: "1h" },
    ]);
    expect(cacheControl()).toEqual({ type: "ephemeral" });
    expect(cacheControl("5m")).toEqual({ type: "ephemeral" });
    expect(cacheControl("1h")).toEqual({ type: "ephemeral", ttl: "1h" });
  });
});

describe("toAnthropicToolChoice", () => {
  test("maps all three shapes and omits when absent", () => {
    expect(toAnthropicToolChoice(undefined)).toBeUndefined();
    expect(toAnthropicToolChoice({ type: "auto" })).toEqual({ type: "auto" });
    expect(toAnthropicToolChoice({ type: "none" })).toEqual({ type: "none" });
    expect(toAnthropicToolChoice({ type: "tool", name: "recall" })).toEqual({
      type: "tool",
      name: "recall",
    });
  });
});

describe("AnthropicProvider conversation caching", () => {
  const provider = (opts: Record<string, unknown> = {}, fetchFn?: typeof fetch): AnthropicProvider =>
    new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      ...(fetchFn ? { fetchFn } : {}),
      ...opts,
    });

  test("the request carries at most 3 markers: system + the last two messages", async () => {
    const { fetchFn, read } = capturing(ANTHROPIC_DONE);
    await provider({}, fetchFn).complete(CONVERSATION);
    const body = read();
    const all = markers(body);
    expect(all).toHaveLength(3);
    expect(all.length).toBeLessThanOrEqual(4); // the API's hard limit
    expect(markers(body.system)).toHaveLength(1);
    expect(markers(body.messages)).toHaveLength(2);
    // ...on the LAST block of each of the last two messages, nowhere else.
    const msgs = body.messages as AnthropicMessage[];
    expect(markers(msgs[0])).toHaveLength(0);
    expect(msgs[1]!.content.at(-1)!.cache_control).toEqual({ type: "ephemeral" });
    expect(msgs[2]!.content.at(-1)!.cache_control).toEqual({ type: "ephemeral" });
    expect(msgs[2]!.content[0]!.cache_control).toBeUndefined();
  });

  test("cache:false emits no marker anywhere (proxies that reject cache_control)", async () => {
    const off = capturing(ANTHROPIC_DONE);
    await provider({ cache: false }, off.fetchFn).complete(CONVERSATION);
    expect(markers(off.read())).toEqual([]);

    // `cacheSystem` is the legacy name and still gates the whole request.
    const alias = capturing(ANTHROPIC_DONE);
    await provider({ cacheSystem: false }, alias.fetchFn).complete(CONVERSATION);
    expect(markers(alias.read())).toEqual([]);
    expect(provider({ cacheSystem: false }).cache).toBe(false);
    expect(provider({ cache: true, cacheSystem: false }).cache).toBe(true);
    expect(provider().cache).toBe(true);
  });

  test("cacheTtl:1h stamps every marker; default is the bare 5m form", async () => {
    const hour = capturing(ANTHROPIC_DONE);
    await provider({ cacheTtl: "1h" }, hour.fetchFn).complete(CONVERSATION);
    const stamped = markers(hour.read());
    expect(stamped).toHaveLength(3);
    expect(stamped).toEqual([
      { type: "ephemeral", ttl: "1h" },
      { type: "ephemeral", ttl: "1h" },
      { type: "ephemeral", ttl: "1h" },
    ]);
    expect(provider().cacheTtl).toBe("5m");

    const short = capturing(ANTHROPIC_DONE);
    await provider({}, short.fetchFn).complete(CONVERSATION);
    for (const marker of markers(short.read())) expect(marker).not.toHaveProperty("ttl");
  });

  test("tool_choice: mapped when tools are present, omitted otherwise", async () => {
    for (const [choice, expected] of [
      [{ type: "auto" }, { type: "auto" }],
      [{ type: "none" }, { type: "none" }],
      [{ type: "tool", name: "t" }, { type: "tool", name: "t" }],
    ] as const) {
      const { fetchFn, read } = capturing(ANTHROPIC_DONE);
      await provider({}, fetchFn).complete({ ...CONVERSATION, toolChoice: choice });
      expect(read().tool_choice).toEqual(expected);
    }

    // Absent -> field omitted (the provider default stands).
    const bare = capturing(ANTHROPIC_DONE);
    await provider({}, bare.fetchFn).complete(CONVERSATION);
    expect(bare.read()).not.toHaveProperty("tool_choice");

    // No tools -> omitted even when set: the API rejects the pairing.
    const toolless = capturing(ANTHROPIC_DONE);
    await provider({}, toolless.fetchFn).complete({
      ...OPTS,
      tools: [],
      toolChoice: { type: "tool", name: "t" },
    });
    expect(toolless.read()).not.toHaveProperty("tool_choice");
    expect(toolless.read()).not.toHaveProperty("tools");
  });

  test("payload snapshot: the exact bytes a tool-using turn posts", async () => {
    const { fetchFn, read } = capturing(ANTHROPIC_DONE);
    await provider({}, fetchFn).complete({ ...CONVERSATION, toolChoice: { type: "tool", name: "t" } });
    expect(read()).toEqual({
      model: "test-model",
      max_tokens: 64_000,
      stream: true,
      system: [{ type: "text", text: "STABLE PREFIX", cache_control: { type: "ephemeral" } }],
      tools: [{ name: "t", description: "d", input_schema: { type: "object" } }],
      tool_choice: { type: "tool", name: "t" },
      messages: [
        { role: "user", content: [{ type: "text", text: "hi" }] },
        {
          role: "assistant",
          content: [
            { type: "text", text: "sure" },
            {
              type: "tool_use",
              id: "c1",
              name: "t",
              input: { a: 1 },
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "c1", content: "ok" },
            { type: "text", text: "next", cache_control: { type: "ephemeral" } },
          ],
        },
      ],
    });
  });

  test("tolerates the per-ttl cache_creation breakdown without inventing fields", async () => {
    const stream =
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7,"cache_read_input_tokens":800,"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":0}}}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n';
    const turn = await provider(
      {},
      (async () => sseResponse(stream)) as unknown as typeof fetch,
    ).complete(OPTS);
    expect(turn.usage).toEqual({ input: 7, output: 2, cacheRead: 800, cacheCreation: 30 });
  });
});

describe("PINKY_ANTHROPIC_CACHE_TTL", () => {
  test("reaches the provider; anything but 5m/1h throws at construction", () => {
    expect(anthropicCacheTtlFromEnv({})).toBeUndefined();
    expect(anthropicCacheTtlFromEnv({ PINKY_ANTHROPIC_CACHE_TTL: "" })).toBeUndefined();
    expect(anthropicCacheTtlFromEnv({ PINKY_ANTHROPIC_CACHE_TTL: " 1H " })).toBe("1h");

    const env = { ANTHROPIC_API_KEY: "a", PINKY_ANTHROPIC_CACHE_TTL: "1h" };
    expect((createProvider("anthropic/claude-opus-5", env) as AnthropicProvider).cacheTtl).toBe("1h");
    expect(
      (createProvider("anthropic/claude-opus-5", { ANTHROPIC_API_KEY: "a" }) as AnthropicProvider)
        .cacheTtl,
    ).toBe("5m");
    expect(() =>
      createProvider("anthropic/claude-opus-5", { PINKY_ANTHROPIC_CACHE_TTL: "2h" }),
    ).toThrow(/PINKY_ANTHROPIC_CACHE_TTL must be "5m" or "1h"/);
  });
});

describe("PINKY_LLM_PROMPT_CACHE_KEY", () => {
  const OPENROUTER_MODEL = "openrouter/moonshotai/kimi-k2";

  /** Drive a real route end to end: env -> createProvider -> posted body. */
  async function bodyFrom(
    model: string,
    env: Record<string, string | undefined>,
  ): Promise<Record<string, unknown>> {
    const { fetchFn, read } = capturing(OPENAI_DONE);
    const provider = createProvider(model, env) as OpenAIProvider;
    // The factory owns baseUrl/headers; only the socket is swapped out.
    (provider as unknown as { fetchFn: typeof fetch }).fetchFn = fetchFn;
    await provider.complete({ ...OPTS, cacheKey: "tenant/chan/thread" });
    return read();
  }

  const KEYS = { OPENAI_API_KEY: "o", OPENROUTER_API_KEY: "r" };
  const LOCAL_BASE_URL = "http://localhost:8000/v1";

  // `prompt_cache_key` is an OpenAI-native field, so the default follows the
  // one route that is known to accept it: real api.openai.com.
  test("defaults on for plain openai/, off for a custom base URL and for openrouter/", async () => {
    expect((await bodyFrom("openai/gpt-4o", KEYS)).prompt_cache_key).toBe("tenant/chan/thread");

    // A local vLLM / llama.cpp / LM Studio / Azure endpoint would 400 on the
    // unknown top-level field, so an OPENAI_BASE_URL override turns it off.
    expect(
      await bodyFrom("openai/gpt-4o", { ...KEYS, OPENAI_BASE_URL: LOCAL_BASE_URL }),
    ).not.toHaveProperty("prompt_cache_key");
    expect(await bodyFrom(OPENROUTER_MODEL, KEYS)).not.toHaveProperty("prompt_cache_key");
  });

  test("the env var, when set, wins on every route", async () => {
    const on = { ...KEYS, PINKY_LLM_PROMPT_CACHE_KEY: "true" };
    expect((await bodyFrom(OPENROUTER_MODEL, on)).prompt_cache_key).toBe("tenant/chan/thread");
    expect(
      (await bodyFrom("openai/gpt-4o", { ...on, OPENAI_BASE_URL: LOCAL_BASE_URL }))
        .prompt_cache_key,
    ).toBe("tenant/chan/thread");

    const off = { ...KEYS, PINKY_LLM_PROMPT_CACHE_KEY: "false" };
    expect(await bodyFrom("openai/gpt-4o", off)).not.toHaveProperty("prompt_cache_key");
    expect(await bodyFrom(OPENROUTER_MODEL, off)).not.toHaveProperty("prompt_cache_key");
  });

  test("anything but a boolean throws at construction", () => {
    expect(() => createProvider("openai/gpt-4o", { PINKY_LLM_PROMPT_CACHE_KEY: "sometimes" })).toThrow(
      /PINKY_LLM_PROMPT_CACHE_KEY must be a boolean/,
    );
  });
});
