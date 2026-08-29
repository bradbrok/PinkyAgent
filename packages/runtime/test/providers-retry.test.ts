/**
 * Retry + timeout hardening for the streaming providers (DESIGN.md §8.1).
 * Everything runs against a fake fetch with an injected sleep — no network,
 * no real backoff timers.
 */
import { describe, expect, test } from "bun:test";
import { AnthropicProvider } from "../src/providers/anthropic";
import { OpenAIProvider } from "../src/providers/openai";
import {
  backoffDelayMs,
  HttpStatusError,
  isNetworkError,
  parseRetryAfter,
  RequestTimeoutError,
} from "../src/providers/retry";
import { sseStreamFromText } from "../src/providers/sse";
import type { CompleteOptions } from "../src/types";

const OPTS: CompleteOptions = {
  model: "test-model",
  system: "be helpful",
  messages: [{ role: "user", text: "hi" }],
  tools: [],
};

const ANTHROPIC_OK =
  'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n' +
  'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n' +
  'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n';

const OPENAI_OK =
  'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n' +
  'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

function sseResponse(body: string): Response {
  return new Response(sseStreamFromText(body), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function errorResponse(status: number, body = "boom", headers: Record<string, string> = {}): Response {
  return new Response(body, { status, headers });
}

/** Queue of per-call handlers; records how many times fetch was invoked. */
function scriptedFetch(
  handlers: ((url: string, init: RequestInit | undefined) => Promise<Response>)[],
): { fetchFn: typeof fetch; calls: { url: string; init: RequestInit | undefined }[] } {
  const calls: { url: string; init: RequestInit | undefined }[] = [];
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    const handler = handlers[calls.length];
    calls.push({ url: String(url), init });
    if (!handler) throw new Error(`fake fetch: unexpected call #${calls.length}`);
    return handler(String(url), init);
  }) as unknown as typeof fetch;
  return { fetchFn, calls };
}

/** Records every backoff sleep instead of waiting. */
function recordingSleep(): { sleep: (ms: number) => Promise<void>; slept: number[] } {
  const slept: number[] = [];
  return {
    sleep: (ms: number) => {
      slept.push(ms);
      return Promise.resolve();
    },
    slept,
  };
}

describe("runtime AbortSignal support (Bun 1.4)", () => {
  test("AbortSignal.any and AbortSignal.timeout exist", () => {
    expect(typeof (AbortSignal as unknown as { any?: unknown }).any).toBe("function");
    expect(typeof AbortSignal.timeout).toBe("function");
  });
});

describe("retry helpers", () => {
  test("parses retry-after as seconds and as an HTTP-date", () => {
    expect(parseRetryAfter("2")).toBe(2000);
    expect(parseRetryAfter("0.5")).toBe(500);
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter("not-a-date")).toBeUndefined();
    const now = Date.parse("2026-01-01T00:00:00Z");
    expect(parseRetryAfter("Thu, 01 Jan 2026 00:00:30 GMT", now)).toBe(30_000);
    // A date already in the past clamps to zero rather than going negative.
    expect(parseRetryAfter("Thu, 01 Jan 2026 00:00:00 GMT", now + 5_000)).toBe(0);
  });

  test("classifies fetch network failures", () => {
    expect(isNetworkError(new TypeError("fetch failed"))).toBe(true);
    expect(isNetworkError(Object.assign(new Error("read"), { code: "ECONNRESET" }))).toBe(true);
    expect(isNetworkError(new Error("bad json"))).toBe(false);
  });

  test("full jitter stays inside the exponential window; retry-after wins", () => {
    const cfg = { baseDelayMs: 500, maxDelayMs: 8_000, random: () => 1 };
    expect(backoffDelayMs(new Error("x"), 0, cfg)).toBe(500); // full window at random()=1
    expect(backoffDelayMs(new Error("x"), 3, cfg)).toBe(4_000);
    expect(backoffDelayMs(new Error("x"), 9, cfg)).toBe(8_000); // capped
    expect(backoffDelayMs(new Error("x"), 3, { ...cfg, random: () => 0 })).toBe(0);
    const rateLimited = new HttpStatusError("429", { status: 429, body: "", retryAfterMs: 1_500 });
    expect(backoffDelayMs(rateLimited, 0, cfg)).toBe(1_500);
  });
});

describe("AnthropicProvider retries", () => {
  test("429 then 200 succeeds and sleeps for retry-after", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => errorResponse(429, "rate limited", { "retry-after": "2" }),
      async () => sseResponse(ANTHROPIC_OK),
    ]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn, sleep });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("ok");
    expect(calls).toHaveLength(2);
    expect(slept).toEqual([2000]);
  });

  test("500 x3 then 200 succeeds (default maxRetries 3)", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => errorResponse(500, "oops"),
      async () => errorResponse(503, "oops"),
      async () => errorResponse(529, "overloaded"),
      async () => sseResponse(ANTHROPIC_OK),
    ]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      sleep,
      random: () => 0.5,
    });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("ok");
    expect(calls).toHaveLength(4);
    expect(slept).toEqual([250, 500, 1000]); // full jitter at 0.5 of 500/1000/2000
  });

  test("exhausting retries throws with status and truncated body", async () => {
    const body = "x".repeat(900);
    const { fetchFn, calls } = scriptedFetch(
      Array.from({ length: 3 }, () => async () => errorResponse(503, body)),
    );
    const { sleep } = recordingSleep();
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      sleep,
      maxRetries: 2,
    });
    const err = (await provider.complete(OPTS).catch((e: unknown) => e)) as HttpStatusError;
    expect(err).toBeInstanceOf(HttpStatusError);
    expect(err.status).toBe(503);
    expect(err.message).toBe(`Anthropic API error 503: ${"x".repeat(500)}`);
    expect(calls).toHaveLength(3);
  });

  test("400 is not retried", async () => {
    const { fetchFn, calls } = scriptedFetch([async () => errorResponse(400, "bad model")]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn, sleep });
    await expect(provider.complete(OPTS)).rejects.toThrow("Anthropic API error 400: bad model");
    expect(calls).toHaveLength(1);
    expect(slept).toEqual([]);
  });

  test("401/404 are not retried either", async () => {
    for (const status of [401, 404, 422]) {
      const { fetchFn, calls } = scriptedFetch([async () => errorResponse(status, "nope")]);
      const provider = new AnthropicProvider({
        apiKey: "k",
        baseUrl: "https://api.test",
        fetchFn,
        sleep: () => Promise.resolve(),
      });
      await expect(provider.complete(OPTS)).rejects.toThrow(`Anthropic API error ${status}`);
      expect(calls).toHaveLength(1);
    }
  });

  test("network TypeError from fetch is retried", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => {
        throw new TypeError("fetch failed");
      },
      async () => {
        throw Object.assign(new Error("read ECONNRESET"), { code: "ECONNRESET" });
      },
      async () => sseResponse(ANTHROPIC_OK),
    ]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      sleep,
      random: () => 0.5,
    });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("ok");
    expect(calls).toHaveLength(3);
    expect(slept).toEqual([250, 500]);
  });

  test("mid-stream failure after the first event is NOT retried", async () => {
    // The stream yields message_start, then the socket dies: retrying would
    // duplicate already-delivered output, so the error must surface.
    const failing = (): Response => {
      let sent = false;
      return new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            if (!sent) {
              sent = true;
              controller.enqueue(
                new TextEncoder().encode(
                  'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n' +
                    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                ),
              );
              return;
            }
            controller.error(Object.assign(new TypeError("fetch failed"), { code: "ECONNRESET" }));
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
    };
    const { fetchFn, calls } = scriptedFetch([async () => failing(), async () => sseResponse(ANTHROPIC_OK)]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn, sleep });
    await expect(provider.complete(OPTS)).rejects.toThrow(/fetch failed/);
    expect(calls).toHaveLength(1);
    expect(slept).toEqual([]);
  });

  test("a mid-stream `error` event surfaces instead of truncating silently", async () => {
    const stream =
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n' +
      'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n';
    const { fetchFn, calls } = scriptedFetch([async () => sseResponse(stream)]);
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      sleep: () => Promise.resolve(),
    });
    await expect(provider.complete(OPTS)).rejects.toThrow(
      "Anthropic stream error (overloaded_error): Overloaded",
    );
    expect(calls).toHaveLength(1);
  });

  test("abort during backoff rethrows AbortError promptly and stops retrying", async () => {
    const controller = new AbortController();
    const { fetchFn, calls } = scriptedFetch([
      async () => errorResponse(429, "slow down", { "retry-after": "30" }),
      async () => sseResponse(ANTHROPIC_OK),
    ]);
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      // A sleep that never resolves: only the abort can unblock the backoff.
      sleep: () => {
        queueMicrotask(() => controller.abort());
        return new Promise<void>(() => {});
      },
    });
    const started = Date.now();
    const err = (await provider
      .complete({ ...OPTS, signal: controller.signal })
      .catch((e: unknown) => e)) as Error;
    expect(err.name).toBe("AbortError");
    expect(Date.now() - started).toBeLessThan(1000);
    expect(calls).toHaveLength(1);
  });

  test("an already-aborted signal never reaches fetch", async () => {
    const controller = new AbortController();
    controller.abort();
    const { fetchFn, calls } = scriptedFetch([async () => sseResponse(ANTHROPIC_OK)]);
    const provider = new AnthropicProvider({ apiKey: "k", baseUrl: "https://api.test", fetchFn });
    const err = (await provider
      .complete({ ...OPTS, signal: controller.signal })
      .catch((e: unknown) => e)) as Error;
    expect(err.name).toBe("AbortError");
    expect(calls).toHaveLength(0);
  });

  test("timeout fires on a never-resolving fetch", async () => {
    const { fetchFn, calls } = scriptedFetch([() => new Promise<Response>(() => {})]);
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      timeoutMs: 30,
      maxRetries: 0,
    });
    const err = (await provider.complete(OPTS).catch((e: unknown) => e)) as RequestTimeoutError;
    expect(err).toBeInstanceOf(RequestTimeoutError);
    expect(err.message).toBe("Anthropic request timed out after 30ms");
    expect(calls).toHaveLength(1);
  });

  test("a timed-out attempt is retried when retries remain", async () => {
    const { fetchFn, calls } = scriptedFetch([
      () => new Promise<Response>(() => {}),
      async () => sseResponse(ANTHROPIC_OK),
    ]);
    const { sleep, slept } = recordingSleep();
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      timeoutMs: 25,
      maxRetries: 1,
      sleep,
      random: () => 0.5,
    });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("ok");
    expect(calls).toHaveLength(2);
    expect(slept).toEqual([250]);
  });

  test("each attempt gets the combined caller+deadline signal", async () => {
    const seen: (AbortSignal | null | undefined)[] = [];
    const controller = new AbortController();
    const { fetchFn } = scriptedFetch([
      async (_url, init) => {
        seen.push(init?.signal);
        return errorResponse(500, "again");
      },
      async (_url, init) => {
        seen.push(init?.signal);
        return sseResponse(ANTHROPIC_OK);
      },
    ]);
    const provider = new AnthropicProvider({
      apiKey: "k",
      baseUrl: "https://api.test",
      fetchFn,
      sleep: () => Promise.resolve(),
    });
    await provider.complete({ ...OPTS, signal: controller.signal });
    expect(seen).toHaveLength(2);
    expect(seen[0]).toBeInstanceOf(AbortSignal);
    expect(seen[0]).not.toBe(controller.signal); // combined, not the raw caller signal
    expect(seen[1]).not.toBe(seen[0]); // a fresh deadline per attempt
  });
});

describe("OpenAIProvider retries", () => {
  test("429 then 200 succeeds and sleeps for retry-after", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => errorResponse(429, "slow down", { "retry-after": "1" }),
      async () => sseResponse(OPENAI_OK),
    ]);
    const { sleep, slept } = recordingSleep();
    const provider = new OpenAIProvider({ apiKey: "k", baseUrl: "https://oai.test/v1", fetchFn, sleep });
    const turn = await provider.complete(OPTS);
    expect(turn.text).toBe("ok");
    expect(calls).toHaveLength(2);
    expect(slept).toEqual([1000]);
  });

  test("400 is not retried and keeps the status + body in the message", async () => {
    const { fetchFn, calls } = scriptedFetch([async () => errorResponse(400, "unknown model")]);
    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      sleep: () => Promise.resolve(),
    });
    await expect(provider.complete(OPTS)).rejects.toThrow("OpenAI API error 400: unknown model");
    expect(calls).toHaveLength(1);
  });

  test("timeout fires on a never-resolving fetch", async () => {
    const { fetchFn } = scriptedFetch([() => new Promise<Response>(() => {})]);
    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      timeoutMs: 30,
      maxRetries: 0,
    });
    await expect(provider.complete(OPTS)).rejects.toThrow("OpenAI request timed out after 30ms");
  });

  test("mid-stream failure after the first event is NOT retried", async () => {
    const failing = (): Response => {
      let sent = false;
      return new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            if (!sent) {
              sent = true;
              controller.enqueue(
                new TextEncoder().encode(
                  'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
                ),
              );
              return;
            }
            controller.error(new TypeError("fetch failed"));
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
    };
    const { fetchFn, calls } = scriptedFetch([async () => failing(), async () => sseResponse(OPENAI_OK)]);
    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      sleep: () => Promise.resolve(),
    });
    await expect(provider.complete(OPTS)).rejects.toThrow(/fetch failed/);
    expect(calls).toHaveLength(1);
  });

  test("a mid-stream error chunk surfaces", async () => {
    const stream =
      'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n' +
      'data: {"error":{"message":"upstream is down","code":502}}\n\n';
    const { fetchFn } = scriptedFetch([async () => sseResponse(stream)]);
    const provider = new OpenAIProvider({
      apiKey: "k",
      baseUrl: "https://oai.test/v1",
      fetchFn,
      sleep: () => Promise.resolve(),
    });
    await expect(provider.complete(OPTS)).rejects.toThrow("OpenAI stream error (502): upstream is down");
  });
});
