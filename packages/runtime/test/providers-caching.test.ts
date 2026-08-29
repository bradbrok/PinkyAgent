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
    // Suffix text stays out of the messages array (it is system-role, not a turn).
    expect(body.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "hi" }, { type: "text", text: "again" }] },
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
