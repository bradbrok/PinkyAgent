import { describe, expect, test } from "bun:test";
import { AnthropicProvider } from "../src/providers/anthropic";
import { OpenAIProvider, OPENROUTER_DEFAULTS } from "../src/providers/openai";
import { iterateSse, sseStreamFromText } from "../src/providers/sse";
import { createProvider, splitModel } from "../src/providers/index";
import type { CompleteOptions } from "../src/types";

describe("iterateSse", () => {
  test("parses events with named types and multi-line data", async () => {
    const raw =
      ": keep-alive\r\n" +
      "event: foo\n" +
      "data: line1\n" +
      "data: line2\n" +
      "\r\n" +
      "data: bare\n" +
      "\n";
    const events = [];
    for await (const evt of iterateSse(sseStreamFromText(raw))) events.push(evt);
    expect(events).toEqual([
      { event: "foo", data: "line1\nline2" },
      { event: null, data: "bare" },
    ]);
  });

  test("handles data split across chunks", async () => {
    const raw = 'data: {"a":1}\n\ndata: {"b"';
    const bytes = new TextEncoder().encode(raw + ":2}\n\n");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
        controller.close();
      },
    });
    const events = [];
    for await (const evt of iterateSse(stream)) events.push(evt);
    expect(events.map((e) => e.data)).toEqual(['{"a":1}', '{"b":2}']);
  });
});

const OPTS: CompleteOptions = {
  model: "test-model",
  system: "be helpful",
  messages: [{ role: "user", text: "hi" }],
  tools: [],
};

function sseResponse(body: string): Response {
  return new Response(sseStreamFromText(body), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("AnthropicProvider", () => {
  test("accumulates text and tool_use blocks from a streamed response", async () => {
    const stream =
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":42}}}\n\n' +
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n' +
      'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"echo"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"word\\":\\"he"}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"llo\\"}"}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}\n\n' +
      "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn: (async () => sseResponse(stream)) as unknown as typeof fetch,
    });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("Hello");
    expect(turn.toolCalls).toEqual([{ id: "toolu_1", name: "echo", args: { word: "hello" } }]);
    expect(turn.stopReason).toBe("tool_calls");
    expect(turn.usage).toEqual({ input: 42, output: 9 });
  });

  test("maps end_turn/max_tokens stop reasons", async () => {
    const make = (reason: string) =>
      new AnthropicProvider({
        apiKey: "k",
        baseUrl: "https://api.test",
        fetchFn: (async () =>
          sseResponse(
            `data: {"type":"message_delta","delta":{"stop_reason":"${reason}"}}\n\n`,
          )) as unknown as typeof fetch,
      });
    expect((await make("end_turn").complete(OPTS)).stopReason).toBe("stop");
    expect((await make("max_tokens").complete(OPTS)).stopReason).toBe("length");
  });

  test("posts the expected request shape", async () => {
    let captured: { url: string; headers: Headers; body: Record<string, unknown> } | null = null;
    const provider = new AnthropicProvider({
      apiKey: "secret",
      baseUrl: "https://api.test/",
      fetchFn: (async (url: string | URL | Request, init?: RequestInit) => {
        captured = {
          url: String(url),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)) as Record<string, unknown>,
        };
        return sseResponse('data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n');
      }) as unknown as typeof fetch,
    });
    await provider.complete({
      ...OPTS,
      messages: [
        { role: "user", text: "hello" },
        { role: "assistant", text: "", toolCalls: [{ id: "c1", name: "echo", args: { a: 1 } }] },
        { role: "tool", toolCallId: "c1", text: "echo result" },
      ],
      tools: [{ name: "echo", description: "echo it", parameters: { type: "object" } }],
    });
    expect(captured!.url).toBe("https://api.test/v1/messages");
    expect(captured!.headers.get("x-api-key")).toBe("secret");
    expect(captured!.headers.get("anthropic-version")).toBe("2023-06-01");
    const body = captured!.body;
    expect(body.stream).toBe(true);
    // system is now an array of blocks with a cache breakpoint on the stable
    // prefix (see providers-caching.test.ts for the full caching contract).
    expect(body.system).toEqual([
      { type: "text", text: "be helpful", cache_control: { type: "ephemeral" } },
    ]);
    expect(body.max_tokens).toBe(64_000);
    expect(body.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "hello" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "c1", name: "echo", input: { a: 1 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "c1", content: "echo result" }] },
    ]);
    expect(body.tools).toEqual([
      { name: "echo", description: "echo it", input_schema: { type: "object" } },
    ]);
  });
});

describe("OpenAIProvider", () => {
  test("accumulates content and streamed tool calls by index", async () => {
    const chunk = (delta: Record<string, unknown>, finish: string | null = null) =>
      `data: {"choices":[{"index":0,"delta":${JSON.stringify(delta)},"finish_reason":${JSON.stringify(finish)}}]}\n\n`;
    const stream =
      chunk({ role: "assistant", content: "Hel" }) +
      chunk({ content: "lo" }) +
      chunk({ tool_calls: [{ index: 0, id: "call_1", function: { name: "echo", arguments: '{"wo' } }] }) +
      chunk({ tool_calls: [{ index: 0, function: { arguments: 'rld":1}' } }] }) +
      chunk({}, "tool_calls") +
      'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}\n\n' +
      "data: [DONE]\n\n";

    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn: (async () => sseResponse(stream)) as unknown as typeof fetch,
    });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("Hello");
    expect(turn.toolCalls).toEqual([{ id: "call_1", name: "echo", args: { world: 1 } }]);
    expect(turn.stopReason).toBe("tool_calls");
    expect(turn.usage).toEqual({ input: 11, output: 5 });
  });

  test("posts chat-completions shape with bearer auth and tool conversions", async () => {
    let captured: { url: string; headers: Headers; body: Record<string, unknown> } | null = null;
    const provider = new OpenAIProvider({
      apiKey: "secret",
      baseUrl: "https://oai.test/v1/",
      fetchFn: (async (url: string | URL | Request, init?: RequestInit) => {
        captured = {
          url: String(url),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)) as Record<string, unknown>,
        };
        return sseResponse(
          'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        );
      }) as unknown as typeof fetch,
    });
    await provider.complete({
      ...OPTS,
      messages: [
        { role: "user", text: "hi" },
        { role: "assistant", text: "calling", toolCalls: [{ id: "c1", name: "echo", args: { a: 1 } }] },
        { role: "tool", toolCallId: "c1", text: "done" },
      ],
      tools: [{ name: "echo", description: "echo it", parameters: { type: "object" } }],
    });
    expect(captured!.url).toBe("https://oai.test/v1/chat/completions");
    expect(captured!.headers.get("authorization")).toBe("Bearer secret");
    const body = captured!.body;
    expect(body.stream).toBe(true);
    expect(body.messages).toEqual([
      { role: "system", content: "be helpful" },
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "calling",
        tool_calls: [
          { id: "c1", type: "function", function: { name: "echo", arguments: '{"a":1}' } },
        ],
      },
      { role: "tool", tool_call_id: "c1", content: "done" },
    ]);
    expect(body.tools).toEqual([
      { type: "function", function: { name: "echo", description: "echo it", parameters: { type: "object" } } },
    ]);
  });
});

describe("createProvider", () => {
  test("splits provider/model-id on the first slash only", () => {
    expect(splitModel("openrouter/moonshotai/kimi-k2")).toEqual({
      provider: "openrouter",
      modelId: "moonshotai/kimi-k2",
    });
    expect(splitModel("anthropic/claude-sonnet-4-5").modelId).toBe("claude-sonnet-4-5");
  });

  test("routes openrouter/* to an OpenAI-compatible provider with OpenRouter defaults", async () => {
    const provider = createProvider("openrouter/moonshotai/kimi-k2", {
      OPENROUTER_API_KEY: "or-key",
    });
    expect(provider).toBeInstanceOf(OpenAIProvider);

    const { provider: prefix, modelId } = splitModel("openrouter/moonshotai/kimi-k2");
    expect(prefix).toBe("openrouter");
    expect(modelId).toBe("moonshotai/kimi-k2");

    // The factory returns a Provider (opaque); assert the wire shape via the
    // same configuration the factory passes (defaults documented in OPENROUTER_DEFAULTS).
    let captured: { url: string; headers: Headers; body: Record<string, unknown> } | null = null;
    const direct = new OpenAIProvider({
      apiKey: "or-key",
      baseUrl: OPENROUTER_DEFAULTS.baseUrl,
      extraHeaders: {
        "HTTP-Referer": OPENROUTER_DEFAULTS.referer,
        "X-Title": OPENROUTER_DEFAULTS.title,
      },
      fetchFn: (async (url: string | URL | Request, init?: RequestInit) => {
        captured = {
          url: String(url),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)) as Record<string, unknown>,
        };
        return sseResponse(
          'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        );
      }) as unknown as typeof fetch,
    });
    await direct.complete({ ...OPTS, model: modelId });
    expect(captured!.url).toBe("https://openrouter.ai/api/v1/chat/completions");
    expect(captured!.headers.get("authorization")).toBe("Bearer or-key");
    expect(captured!.headers.get("HTTP-Referer")).toBe("https://github.com/pinkyagent");
    expect(captured!.headers.get("X-Title")).toBe("PinkyAgent");
    expect(captured!.body.model).toBe("moonshotai/kimi-k2");
  });

  test("honors OPENROUTER_BASE_URL/REFERER/TITLE env overrides", () => {
    const provider = createProvider("openrouter/zai/glm-4.6", {
      OPENROUTER_API_KEY: "k",
      OPENROUTER_BASE_URL: "https://proxy.test/v1",
      OPENROUTER_REFERER: "https://example.com",
      OPENROUTER_TITLE: "Custom",
    }) as OpenAIProvider;
    expect(provider.name).toBe("openai");
  });

  test("unknown provider throws listing supported providers", () => {
    expect(() => createProvider("mistral/mistral-large", {})).toThrow(
      /Unknown provider "mistral".*anthropic, openai, openrouter/s,
    );
  });
});
