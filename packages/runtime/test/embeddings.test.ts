/**
 * Embedder contract + hardening (DESIGN.md §5.5, §8.1).
 * Everything runs against a fake fetch with an injected sleep — no network,
 * no real backoff timers, no API key.
 */
import { describe, expect, test } from "bun:test";
import {
  createEmbedder,
  DEFAULT_EMBEDDING_DIMENSIONS,
  DEFAULT_EMBEDDING_MAX_RETRIES,
  DEFAULT_EMBEDDING_TIMEOUT_MS,
  EMBEDDINGS_DISABLED_PREFIX,
  FakeEmbedder,
  isEmbeddingsDisabledError,
  OpenAIEmbedder,
} from "../src/embeddings";
import { HttpStatusError } from "../src/providers/retry";
import type { Embedder } from "../src/types";

/** Queue of per-call handlers; records how many times fetch was invoked. */
function scriptedFetch(handlers: ((url: string, init: RequestInit | undefined) => Promise<Response>)[]): {
  fetchFn: typeof fetch;
  calls: { url: string; init: RequestInit | undefined }[];
} {
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

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function embeddingsPayload(rows: { index: number; embedding: number[] }[]): unknown {
  return {
    object: "list",
    model: "text-embedding-3-small",
    data: rows.map((r) => ({ object: "embedding", index: r.index, embedding: r.embedding })),
    usage: { prompt_tokens: 4, total_tokens: 4 },
  };
}

function headersOf(init: RequestInit | undefined): Record<string, string> {
  return (init?.headers ?? {}) as Record<string, string>;
}

function bodyOf(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body)) as Record<string, unknown>;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += (a[i] ?? 0) * (b[i] ?? 0);
  return dot;
}

function norm(v: number[]): number {
  return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
}

/** The private RetryConfig, so the transport budget is assertable without a
 *  network call (the alternative is timing a real 15s timeout). */
function retryOf(e: Embedder | null): { maxRetries: number; timeoutMs: number } {
  return (e as unknown as { retry: { maxRetries: number; timeoutMs: number } }).retry;
}

function embedder(opts: Partial<ConstructorParameters<typeof OpenAIEmbedder>[0]> = {}): OpenAIEmbedder {
  return new OpenAIEmbedder({
    model: "openai/text-embedding-3-small",
    apiKey: "sk-test",
    baseUrl: "https://api.test/v1",
    ...opts,
  });
}

describe("OpenAIEmbedder request shape", () => {
  test("POSTs {baseUrl}/embeddings with bearer auth and the bare model id", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [0.1, 0.2] }])),
    ]);
    const e = embedder({ fetchFn });
    expect(e.model).toBe("openai/text-embedding-3-small");
    expect(e.dimensions).toBe(1536);

    const vectors = await e.embed(["hello"]);
    expect(vectors).toEqual([[0.1, 0.2]]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("https://api.test/v1/embeddings");
    expect(calls[0]!.init?.method).toBe("POST");
    expect(headersOf(calls[0]!.init).authorization).toBe("Bearer sk-test");
    expect(headersOf(calls[0]!.init)["content-type"]).toBe("application/json");
    // The configured id carries the routing prefix; the wire never sees it.
    expect(bodyOf(calls[0]!.init)).toEqual({
      model: "text-embedding-3-small",
      input: ["hello"],
      encoding_format: "float",
    });
  });

  test("trailing slashes in the base url are trimmed and extraHeaders are sent", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [1] }])),
    ]);
    const e = embedder({
      baseUrl: "https://api.test/v1///",
      extraHeaders: { "HTTP-Referer": "https://example.test", "X-Title": "PinkyAgent" },
      fetchFn,
    });
    await e.embed(["x"]);
    expect(calls[0]!.url).toBe("https://api.test/v1/embeddings");
    expect(headersOf(calls[0]!.init)["X-Title"]).toBe("PinkyAgent");
    expect(headersOf(calls[0]!.init)["HTTP-Referer"]).toBe("https://example.test");
  });

  test("dimensions is sent only when pinned, and is reported as Embedder.dimensions", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [0.5, 0.5, 0.5, 0.5] }])),
    ]);
    const e = embedder({ dimensions: 4, fetchFn });
    expect(e.dimensions).toBe(4);
    await e.embed(["x"]);
    expect(bodyOf(calls[0]!.init).dimensions).toBe(4);
  });

  test("an unknown model id falls back to the vector(1536) column width", () => {
    expect(embedder({ model: "openai/some-new-embedder" }).dimensions).toBe(
      DEFAULT_EMBEDDING_DIMENSIONS,
    );
    expect(embedder({ model: "openai/text-embedding-3-large" }).dimensions).toBe(3072);
  });

  test("empty input returns [] without a request", async () => {
    const { fetchFn, calls } = scriptedFetch([]);
    expect(await embedder({ fetchFn }).embed([])).toEqual([]);
    expect(calls).toHaveLength(0);
  });
});

describe("OpenAIEmbedder response handling", () => {
  test("vectors come back in input order however the API sorts them", async () => {
    const { fetchFn } = scriptedFetch([
      async () =>
        jsonResponse(
          embeddingsPayload([
            { index: 2, embedding: [3] },
            { index: 0, embedding: [1] },
            { index: 1, embedding: [2] },
          ]),
        ),
    ]);
    expect(await embedder({ fetchFn }).embed(["a", "b", "c"])).toEqual([[1], [2], [3]]);
  });

  test("a count mismatch throws instead of returning a short array", async () => {
    const { fetchFn } = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [1] }])),
    ]);
    await expect(embedder({ fetchFn }).embed(["a", "b"])).rejects.toThrow(/expected 2 embeddings, got 1/);
  });

  test("a non-numeric vector, a missing data array and a bad index all throw", async () => {
    const bad = scriptedFetch([
      async () => jsonResponse({ data: [{ index: 0, embedding: ["nope"] }] }),
    ]);
    await expect(embedder({ fetchFn: bad.fetchFn }).embed(["a"])).rejects.toThrow(/is not a number\[\]/);

    const noData = scriptedFetch([async () => jsonResponse({ object: "list" })]);
    await expect(embedder({ fetchFn: noData.fetchFn }).embed(["a"])).rejects.toThrow(/no "data" array/);

    const outOfRange = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 7, embedding: [1] }])),
    ]);
    await expect(embedder({ fetchFn: outOfRange.fetchFn }).embed(["a"])).rejects.toThrow(
      /index 7 out of range/,
    );
  });

  test("a pinned width is enforced on the response", async () => {
    const { fetchFn } = scriptedFetch([
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [1, 2, 3] }])),
    ]);
    await expect(embedder({ dimensions: 4, fetchFn }).embed(["a"])).rejects.toThrow(
      /has 3 dimensions, expected 4/,
    );
  });
});

describe("OpenAIEmbedder retries (DESIGN.md §8.1)", () => {
  test("429 then 200 succeeds and sleeps for retry-after", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => new Response("slow down", { status: 429, headers: { "retry-after": "2" } }),
      async () => jsonResponse(embeddingsPayload([{ index: 0, embedding: [0.25] }])),
    ]);
    const { sleep, slept } = recordingSleep();
    const vectors = await embedder({ fetchFn, sleep }).embed(["a"]);
    expect(vectors).toEqual([[0.25]]);
    expect(calls).toHaveLength(2);
    expect(slept).toEqual([2000]);
  });

  test("a 400 surfaces as HttpStatusError without a second attempt", async () => {
    const { fetchFn, calls } = scriptedFetch([
      async () => new Response('{"error":{"message":"bad input"}}', { status: 400 }),
    ]);
    const { sleep, slept } = recordingSleep();
    const err = await embedder({ fetchFn, sleep })
      .embed(["a"])
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(HttpStatusError);
    expect((err as HttpStatusError).status).toBe(400);
    expect((err as Error).message).toContain("OpenAI embeddings API error 400");
    expect(calls).toHaveLength(1);
    expect(slept).toEqual([]);
  });

  test("maxRetries caps the attempts on repeated 5xx", async () => {
    const fail = async (): Promise<Response> => new Response("boom", { status: 503 });
    const { fetchFn, calls } = scriptedFetch([fail, fail, fail]);
    const { sleep, slept } = recordingSleep();
    await expect(
      embedder({ fetchFn, sleep, maxRetries: 2, random: () => 1 }).embed(["a"]),
    ).rejects.toThrow(/API error 503/);
    expect(calls).toHaveLength(3);
    expect(slept).toEqual([500, 1000]);
  });

  test("the caller's abort signal wins immediately", async () => {
    const { fetchFn, calls } = scriptedFetch([]);
    const controller = new AbortController();
    controller.abort();
    await expect(embedder({ fetchFn }).embed(["a"], { signal: controller.signal })).rejects.toThrow();
    expect(calls).toHaveLength(0);
  });
});

describe("FakeEmbedder", () => {
  test("is deterministic across instances and records every batch", async () => {
    const a = new FakeEmbedder();
    const b = new FakeEmbedder();
    const [v1, v2] = await a.embed(["deploy failed on a missing env var", "deploy failed on a missing env var"]);
    const [v3] = await b.embed(["deploy failed on a missing env var"]);
    expect(v1).toEqual(v2!);
    expect(v1).toEqual(v3!);
    await a.embed(["second batch"]);
    expect(a.calls).toEqual([
      ["deploy failed on a missing env var", "deploy failed on a missing env var"],
      ["second batch"],
    ]);
  });

  test("honors `dimensions` (default 8) and always returns unit vectors", async () => {
    const small = new FakeEmbedder();
    const wide = new FakeEmbedder({ dimensions: 64, model: "fake/wide" });
    expect(small.dimensions).toBe(8);
    expect(wide.model).toBe("fake/wide");
    const vectors = await wide.embed(["hello world", "", "   ", "Brad prefers terse answers"]);
    for (const v of vectors) {
      expect(v).toHaveLength(64);
      expect(norm(v)).toBeCloseTo(1, 10);
    }
    expect((await small.embed(["hello world"]))[0]).toHaveLength(8);
  });

  test("shared vocabulary scores higher than disjoint vocabulary", async () => {
    const e = new FakeEmbedder({ dimensions: 32 });
    const [base, overlapping, disjoint] = await e.embed([
      "deploy failed on a missing env var",
      "the deploy failed again on a missing env var",
      "brad prefers terse answers about pricing",
    ]);
    expect(cosine(base!, overlapping!)).toBeGreaterThan(cosine(base!, disjoint!));
    expect(cosine(base!, base!)).toBeCloseTo(1, 10);
  });

  test("tokenization is case- and punctuation-insensitive", async () => {
    const e = new FakeEmbedder({ dimensions: 16 });
    const [lower, shouty] = await e.embed(["deploy failed", "  DEPLOY, failed!!  "]);
    expect(lower).toEqual(shouty!);
  });

  test("satisfies the Embedder interface", () => {
    const e: Embedder = new FakeEmbedder();
    expect(typeof e.embed).toBe("function");
  });
});

describe("createEmbedder", () => {
  test('"none" (and a blank setting) disables the vector voice', () => {
    expect(createEmbedder("none", {})).toBeNull();
    expect(createEmbedder("NONE", {})).toBeNull();
    expect(createEmbedder("", {})).toBeNull();
  });

  test("routes openai/* to api.openai.com by default and honors OPENAI_BASE_URL", () => {
    const e = createEmbedder("openai/text-embedding-3-small", { OPENAI_API_KEY: "sk-x" });
    expect(e).toBeInstanceOf(OpenAIEmbedder);
    expect(e!.model).toBe("openai/text-embedding-3-small");
    expect(e!.dimensions).toBe(1536);
    expect((e as OpenAIEmbedder).baseUrl).toBe("https://api.openai.com/v1");

    const local = createEmbedder("openai/bge-base", {
      OPENAI_API_KEY: "sk-x",
      OPENAI_BASE_URL: "http://localhost:8080/v1",
    }) as OpenAIEmbedder;
    expect(local.baseUrl).toBe("http://localhost:8080/v1");
  });

  test("routes openrouter/* with its base url and identifying headers", () => {
    const e = createEmbedder("openrouter/openai/text-embedding-3-small", {
      OPENROUTER_API_KEY: "or-x",
    }) as OpenAIEmbedder;
    expect(e.baseUrl).toBe("https://openrouter.ai/api/v1");
    expect(e.extraHeaders["X-Title"]).toBe("PinkyAgent");
    expect(e.extraHeaders["HTTP-Referer"]).toBeTruthy();
    expect(e.model).toBe("openrouter/openai/text-embedding-3-small");
  });

  test("a missing key throws the `embeddings disabled:` signal the wiring layer catches", () => {
    const err = (() => {
      try {
        createEmbedder("openai/text-embedding-3-small", {});
        return null;
      } catch (e) {
        return e as Error;
      }
    })();
    expect(err).toBeInstanceOf(Error);
    expect(err!.message.startsWith(EMBEDDINGS_DISABLED_PREFIX)).toBe(true);
    expect(err!.message).toBe(
      "embeddings disabled: OPENAI_API_KEY is not set (memory.embeddingModel = openai/text-embedding-3-small)",
    );
    expect(isEmbeddingsDisabledError(err)).toBe(true);
    expect(isEmbeddingsDisabledError(new Error("boom"))).toBe(false);

    expect(() => createEmbedder("openrouter/x", { OPENROUTER_API_KEY: "   " })).toThrow(
      /^embeddings disabled: OPENROUTER_API_KEY is not set/,
    );
  });

  test("an unknown provider or a bare model id is a loud misconfiguration", () => {
    const unknown = (() => {
      try {
        createEmbedder("cohere/embed-v3", {});
        return null;
      } catch (e) {
        return e as Error;
      }
    })();
    expect(unknown!.message).toContain('Unknown embeddings provider "cohere"');
    expect(unknown!.message).toContain("Supported providers: openai, openrouter, none");
    expect(isEmbeddingsDisabledError(unknown)).toBe(false);
    expect(() => createEmbedder("text-embedding-3-small", {})).toThrow(/provider\/model-id/);
  });

  test("embeddings carry their OWN transport budget, not the LLM's", () => {
    // A completion is a long stream someone is waiting on; an embedding is one
    // small request a recall is blocked behind, and it degrades to FTS. 120s x
    // 3 was six minutes of stalling an auto-recall for a vector we can do
    // without.
    const e = createEmbedder("openai/text-embedding-3-small", { OPENAI_API_KEY: "sk-x" });
    expect(retryOf(e)).toMatchObject({
      timeoutMs: DEFAULT_EMBEDDING_TIMEOUT_MS,
      maxRetries: DEFAULT_EMBEDDING_MAX_RETRIES,
    });
    expect(DEFAULT_EMBEDDING_TIMEOUT_MS).toBe(15_000);
    expect(DEFAULT_EMBEDDING_MAX_RETRIES).toBe(2);

    // The LLM knobs no longer reach here — including the one that would throw.
    const untouched = createEmbedder("openai/text-embedding-3-small", {
      OPENAI_API_KEY: "sk-x",
      PINKY_LLM_TIMEOUT_MS: "120000",
      PINKY_LLM_MAX_RETRIES: "-1",
    });
    expect(retryOf(untouched)).toMatchObject({
      timeoutMs: DEFAULT_EMBEDDING_TIMEOUT_MS,
      maxRetries: DEFAULT_EMBEDDING_MAX_RETRIES,
    });
  });

  test("PINKY_EMBED_* override the embedding budget, and are validated", () => {
    const e = createEmbedder("openrouter/openai/text-embedding-3-small", {
      OPENROUTER_API_KEY: "or-x",
      PINKY_EMBED_TIMEOUT_MS: "2500",
      PINKY_EMBED_MAX_RETRIES: "0",
    });
    expect(retryOf(e)).toMatchObject({ timeoutMs: 2500, maxRetries: 0 });

    expect(() =>
      createEmbedder("openai/text-embedding-3-small", {
        OPENAI_API_KEY: "sk-x",
        PINKY_EMBED_MAX_RETRIES: "-1",
      }),
    ).toThrow(/PINKY_EMBED_MAX_RETRIES/);
    expect(() =>
      createEmbedder("openai/text-embedding-3-small", {
        OPENAI_API_KEY: "sk-x",
        PINKY_EMBED_TIMEOUT_MS: "soon",
      }),
    ).toThrow(/PINKY_EMBED_TIMEOUT_MS/);
  });

  test("pins the output width to vector(1536), whatever the model's native size", async () => {
    // memory.embeddingModel is a SETTING, so "text-embedding-3-large" is a
    // legal thing for a human to write — and its native 3072 floats do not fit
    // memories.embedding. The v3 models are Matryoshka-trained, so asking for
    // 1536 truncates server-side instead of producing a vector the column
    // refuses (which used to cost the whole memory).
    const large = createEmbedder("openai/text-embedding-3-large", {
      OPENAI_API_KEY: "sk-x",
    }) as OpenAIEmbedder;
    expect(large.dimensions).toBe(DEFAULT_EMBEDDING_DIMENSIONS);

    const { fetchFn, calls } = scriptedFetch([
      async () =>
        jsonResponse(
          embeddingsPayload([{ index: 0, embedding: new Array(1536).fill(0.01) as number[] }]),
        ),
    ]);
    // Same object the factory builds, with fetch injected.
    const wired = new OpenAIEmbedder({
      model: "openai/text-embedding-3-large",
      apiKey: "sk-x",
      dimensions: large.dimensions,
      fetchFn,
    });
    await wired.embed(["x"]);
    expect(bodyOf(calls[0]!.init).dimensions).toBe(1536);
  });

  test("the pin is overridable — by option, and by PINKY_EMBED_DIMENSIONS", () => {
    expect(
      (createEmbedder("openai/text-embedding-3-large", { OPENAI_API_KEY: "sk-x" }, { dimensions: 3072 }) as OpenAIEmbedder)
        .dimensions,
    ).toBe(3072);
    expect(
      (
        createEmbedder("openai/text-embedding-3-large", {
          OPENAI_API_KEY: "sk-x",
          PINKY_EMBED_DIMENSIONS: "256",
        }) as OpenAIEmbedder
      ).dimensions,
    ).toBe(256);
    // 0 = "send no dimensions field": the escape hatch for an OpenAI-compatible
    // server (a local bge, say) that rejects the parameter outright.
    const native = createEmbedder("openai/text-embedding-3-large", {
      OPENAI_API_KEY: "sk-x",
      PINKY_EMBED_DIMENSIONS: "0",
    }) as OpenAIEmbedder;
    expect(native.dimensions).toBe(3072); // the model's own width, unpinned
  });
});
