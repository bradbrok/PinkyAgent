/**
 * Embedders: text -> vector, for the memory plane's vector recall voice
 * (DESIGN.md §5.5 — "text-embedding-3-small (1536d) behind an interface so a
 * local bge-base variant works for dev/edge").
 *
 * Same hardening story as the LLM providers (DESIGN.md §8.1, CLAUDE.md #7):
 * the request runs inside the shared `withRetry` (408/409/429/5xx + network
 * blips, jittered backoff, `retry-after`, per-attempt timeout) and non-2xx
 * surfaces as `HttpStatusError` via `assertOk`. Embeddings are a plain
 * request/response — there is no stream, so the whole call is retryable and
 * `markStreamProgress` is never used.
 *
 * The embedder is optional everywhere: `createEmbedder` returns null for
 * "none", and recall degrades to the FTS voice alone (DESIGN.md §5.4).
 */
import type { Embedder } from "./types";
import { OPENROUTER_DEFAULTS, splitModel } from "./providers/index";
import {
  assertOk,
  DEFAULT_MAX_RETRIES,
  DEFAULT_TIMEOUT_MS,
  realSleep,
  withRetry,
  type RetryConfig,
  type SleepFn,
} from "./providers/retry";

/** Matches the `memories.embedding vector(1536)` column (0002_embeddings.rerun.sql). */
export const DEFAULT_EMBEDDING_DIMENSIONS = 1536;

/**
 * Embeddings get their OWN transport budget, not the LLM's.
 *
 * A completion is a long stream a human is waiting on, so 120s x 3 retries is
 * proportionate. An embedding is one small request that a recall is blocked on
 * — on the auto-recall path the whole turn stalls behind it — and it degrades
 * gracefully (the FTS voice still answers). Waiting six minutes for a vector
 * is strictly worse than not having one.
 */
export const DEFAULT_EMBEDDING_TIMEOUT_MS = 15_000;
/** Retries after the first attempt; see DEFAULT_EMBEDDING_TIMEOUT_MS. */
export const DEFAULT_EMBEDDING_MAX_RETRIES = 2;

/** Native output width per known model; only used when `dimensions` is not pinned. */
export const KNOWN_EMBEDDING_DIMENSIONS: Readonly<Record<string, number>> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
};

/** Providers `createEmbedder` routes; "none" disables the vector voice. */
export const SUPPORTED_EMBEDDING_PROVIDERS = ["openai", "openrouter", "none"] as const;

/**
 * Prefix of the error `createEmbedder` throws when the route is configured but
 * its API key is blank. The wiring layer matches on it, warns once and runs
 * FTS-only; every other failure is a real misconfiguration and must be loud.
 */
export const EMBEDDINGS_DISABLED_PREFIX = "embeddings disabled:";

/** True for the "configured, but no API key" case (see EMBEDDINGS_DISABLED_PREFIX). */
export function isEmbeddingsDisabledError(err: unknown): boolean {
  return err instanceof Error && err.message.startsWith(EMBEDDINGS_DISABLED_PREFIX);
}

/** Strip the routing prefix: "openai/text-embedding-3-small" -> "text-embedding-3-small". */
function bareModelId(model: string): string {
  const slash = model.indexOf("/");
  return slash > 0 && slash < model.length - 1 ? model.slice(slash + 1) : model;
}

export interface OpenAIEmbedderOptions {
  /** Configured "provider/model-id" — what `Embedder.model` reports. */
  model: string;
  /** Bare model id sent on the wire; derived from `model` when omitted. */
  modelId?: string | undefined;
  /** Pin the output width (Matryoshka truncation). Sent as `dimensions` and
   *  enforced on the response; omitted => the model's native width. */
  dimensions?: number | undefined;
  /** Defaults to process.env.OPENAI_API_KEY. */
  apiKey?: string | undefined;
  /** Defaults to process.env.OPENAI_BASE_URL ?? https://api.openai.com/v1 */
  baseUrl?: string | undefined;
  /** Extra request headers (OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string> | undefined;
  /** Injectable for tests. */
  fetchFn?: typeof fetch | undefined;
  /** Retries after the first attempt for 408/409/429/5xx + network errors. Default 3. */
  maxRetries?: number | undefined;
  /** Per-attempt budget. Default 120_000ms. */
  timeoutMs?: number | undefined;
  /** Backoff sleep; injectable so tests never wait on real timers. */
  sleep?: SleepFn | undefined;
  /** Jitter source (test hook). Default Math.random. */
  random?: (() => number) | undefined;
  /** Label used in timeout/error messages. Default "OpenAI embeddings". */
  label?: string | undefined;
}

/** OpenAI-compatible `/embeddings` (api.openai.com, OpenRouter, any clone). */
export class OpenAIEmbedder implements Embedder {
  readonly model: string;
  readonly dimensions: number;
  readonly baseUrl: string;
  readonly extraHeaders: Record<string, string>;
  private readonly modelId: string;
  /** Only set when the caller pinned a width; drives the request field. */
  private readonly pinnedDimensions: number | undefined;
  private readonly apiKey: string;
  private readonly fetchFn: typeof fetch;
  private readonly label: string;
  private readonly retry: RetryConfig;

  constructor(opts: OpenAIEmbedderOptions) {
    this.model = opts.model;
    this.modelId = opts.modelId ?? bareModelId(opts.model);
    if (opts.dimensions !== undefined && (!Number.isInteger(opts.dimensions) || opts.dimensions <= 0)) {
      throw new Error(`dimensions must be a positive integer, got ${JSON.stringify(opts.dimensions)}`);
    }
    this.pinnedDimensions = opts.dimensions;
    this.dimensions =
      opts.dimensions ?? KNOWN_EMBEDDING_DIMENSIONS[this.modelId] ?? DEFAULT_EMBEDDING_DIMENSIONS;
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.baseUrl = (opts.baseUrl ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(
      /\/+$/,
      "",
    );
    this.extraHeaders = opts.extraHeaders ?? {};
    this.fetchFn = opts.fetchFn ?? fetch;
    this.label = opts.label ?? "OpenAI embeddings";
    this.retry = {
      label: this.label,
      maxRetries: opts.maxRetries ?? DEFAULT_MAX_RETRIES,
      timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      sleep: opts.sleep ?? realSleep,
      random: opts.random,
    };
  }

  async embed(texts: string[], opts: { signal?: AbortSignal } = {}): Promise<number[][]> {
    // Nothing to embed is not an API call: recall with no query text must not
    // cost a round trip (or a 400 for an empty `input`).
    if (texts.length === 0) return [];

    const body: Record<string, unknown> = {
      model: this.modelId,
      input: texts,
      encoding_format: "float",
    };
    if (this.pinnedDimensions !== undefined) body.dimensions = this.pinnedDimensions;
    const payload = JSON.stringify(body);

    return withRetry(this.retry, opts.signal, async ({ signal }) => {
      const res = await this.fetchFn(`${this.baseUrl}/embeddings`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
          ...this.extraHeaders,
        },
        body: payload,
        signal,
      });
      await assertOk(res, this.label);
      const parsed: unknown = await res.json();
      return this.parseVectors(parsed, texts.length);
    });
  }

  /** `data[].{index, embedding}` -> vectors in input order, fully validated. */
  private parseVectors(payload: unknown, expected: number): number[][] {
    const data = (payload as { data?: unknown } | null)?.data;
    if (!Array.isArray(data)) {
      throw new Error(`${this.label}: response has no "data" array`);
    }
    if (data.length !== expected) {
      throw new Error(`${this.label}: expected ${expected} embeddings, got ${data.length}`);
    }
    const out: (number[] | undefined)[] = new Array(expected).fill(undefined);
    for (let i = 0; i < data.length; i++) {
      const row = data[i] as { index?: unknown; embedding?: unknown } | null;
      const rawIndex = row?.index;
      // Position is the fallback for endpoints that omit `index`.
      const index = typeof rawIndex === "number" ? rawIndex : i;
      if (!Number.isInteger(index) || index < 0 || index >= expected) {
        throw new Error(`${this.label}: embedding index ${String(rawIndex)} out of range 0..${expected - 1}`);
      }
      if (out[index] !== undefined) {
        throw new Error(`${this.label}: duplicate embedding index ${index}`);
      }
      const vector = row?.embedding;
      if (
        !Array.isArray(vector) ||
        vector.length === 0 ||
        !vector.every((n) => typeof n === "number" && Number.isFinite(n))
      ) {
        throw new Error(`${this.label}: embedding at index ${index} is not a number[]`);
      }
      if (this.pinnedDimensions !== undefined && vector.length !== this.pinnedDimensions) {
        throw new Error(
          `${this.label}: embedding at index ${index} has ${vector.length} dimensions, expected ${this.pinnedDimensions}`,
        );
      }
      out[index] = vector as number[];
    }
    return out.map((vector, i) => {
      if (!vector) throw new Error(`${this.label}: no embedding returned for input ${i}`);
      return vector;
    });
  }
}

export interface FakeEmbedderOptions {
  /** Reported as `Embedder.model`. Default "fake/deterministic". */
  model?: string | undefined;
  /** Vector width. Default 8. */
  dimensions?: number | undefined;
}

/** FNV-1a 32-bit — stable across runs and processes (no Math.random, no Date). */
function hash32(token: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Deterministic bag-of-words hashing embedder for unit/integration tests and
 * `pinky smoke`: no network, no key, but shared vocabulary really does raise
 * cosine similarity, so a retain -> recall round trip is a meaningful check.
 *
 * Signed feature hashing (the sign comes from a different bit of the same
 * hash) keeps unrelated texts near-orthogonal even at 8 dimensions, instead of
 * the all-positive orthant a plain count vector would produce.
 */
export class FakeEmbedder implements Embedder {
  readonly model: string;
  readonly dimensions: number;
  /** Every batch this embedder was asked for, in order. */
  readonly calls: string[][] = [];

  constructor(opts: FakeEmbedderOptions = {}) {
    this.model = opts.model ?? "fake/deterministic";
    const dimensions = opts.dimensions ?? 8;
    if (!Number.isInteger(dimensions) || dimensions <= 0) {
      throw new Error(`dimensions must be a positive integer, got ${JSON.stringify(dimensions)}`);
    }
    this.dimensions = dimensions;
  }

  embed(texts: string[]): Promise<number[][]> {
    this.calls.push([...texts]);
    return Promise.resolve(texts.map((text) => this.vectorFor(text)));
  }

  /** Sync sibling of `embed` for tests that want a vector without awaiting. */
  vectorFor(text: string): number[] {
    const vector = new Array<number>(this.dimensions).fill(0);
    for (const token of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
      const h = hash32(token);
      const bucket = h % this.dimensions;
      const sign = (h >>> 31) & 1 ? -1 : 1;
      vector[bucket] = (vector[bucket] ?? 0) + sign;
    }
    let norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    if (norm === 0) {
      // Empty text (or a full sign cancellation): still return a unit vector so
      // callers never have to special-case a zero row.
      vector[0] = 1;
      norm = 1;
    }
    return vector.map((v) => v / norm);
  }
}

/** Bootstrap-only transport knob, same shape as providers/index.ts's envInt. */
function envInt(env: Record<string, string | undefined>, key: string): number | undefined {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${key} must be a non-negative integer, got ${JSON.stringify(raw)}`);
  }
  return value;
}

export interface CreateEmbedderOptions {
  /**
   * Output width to request and enforce. Defaults to
   * {@link DEFAULT_EMBEDDING_DIMENSIONS} (or `PINKY_EMBED_DIMENSIONS`); 0 sends
   * no `dimensions` field at all, for an OpenAI-compatible server that rejects
   * the parameter.
   */
  dimensions?: number | undefined;
}

/**
 * "provider/model-id" -> Embedder, or null for "none" (FTS-only recall).
 * Mirrors `createProvider`'s routing and OpenRouter defaults, but NOT its
 * transport budget: embeddings read `PINKY_EMBED_TIMEOUT_MS` /
 * `PINKY_EMBED_MAX_RETRIES` (see DEFAULT_EMBEDDING_TIMEOUT_MS).
 *
 * The output width is PINNED to 1536 by default. `memory.embeddingModel` is a
 * setting, so "openai/text-embedding-3-large" is a perfectly legal thing for a
 * human to write — and its native 3072 floats do not fit `vector(1536)`.
 * OpenAI's v3 models are Matryoshka-trained, so asking for 1536 truncates and
 * renormalizes server-side instead of silently producing a vector the column
 * (and MemoryStore.vectorDimensions) will refuse.
 *
 * Throws `embeddings disabled: <ENV_VAR> is not set (...)` when the route is
 * configured but its key is blank — the caller downgrades to FTS-only on that
 * one message and lets every other error escape.
 */
export function createEmbedder(
  model: string,
  env: Record<string, string | undefined> = process.env,
  opts: CreateEmbedderOptions = {},
): Embedder | null {
  const configured = model.trim();
  if (configured === "" || configured.toLowerCase() === "none") return null;

  const { provider, modelId } = splitModel(configured);
  const width =
    opts.dimensions ?? envInt(env, "PINKY_EMBED_DIMENSIONS") ?? DEFAULT_EMBEDDING_DIMENSIONS;
  if (!Number.isInteger(width) || width < 0) {
    throw new Error(`dimensions must be a non-negative integer, got ${JSON.stringify(width)}`);
  }
  const transport = {
    maxRetries: envInt(env, "PINKY_EMBED_MAX_RETRIES") ?? DEFAULT_EMBEDDING_MAX_RETRIES,
    timeoutMs: envInt(env, "PINKY_EMBED_TIMEOUT_MS") ?? DEFAULT_EMBEDDING_TIMEOUT_MS,
    // 0 => let the model use its native width (no `dimensions` on the wire).
    ...(width > 0 ? { dimensions: width } : {}),
  };

  const requireKey = (key: string | undefined, envVar: string): string => {
    const value = key?.trim() ?? "";
    if (value === "") {
      throw new Error(
        `${EMBEDDINGS_DISABLED_PREFIX} ${envVar} is not set (memory.embeddingModel = ${configured})`,
      );
    }
    return value;
  };

  switch (provider) {
    case "openai":
      return new OpenAIEmbedder({
        model: configured,
        modelId,
        apiKey: requireKey(env.OPENAI_API_KEY, "OPENAI_API_KEY"),
        baseUrl: env.OPENAI_BASE_URL,
        ...transport,
      });
    case "openrouter":
      return new OpenAIEmbedder({
        model: configured,
        modelId,
        apiKey: requireKey(env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
        baseUrl: env.OPENROUTER_BASE_URL ?? OPENROUTER_DEFAULTS.baseUrl,
        extraHeaders: {
          "HTTP-Referer": env.OPENROUTER_REFERER ?? OPENROUTER_DEFAULTS.referer,
          "X-Title": env.OPENROUTER_TITLE ?? OPENROUTER_DEFAULTS.title,
        },
        label: "OpenRouter embeddings",
        ...transport,
      });
    default:
      throw new Error(
        `Unknown embeddings provider ${JSON.stringify(provider)} in model ${JSON.stringify(configured)}. ` +
          `Supported providers: ${SUPPORTED_EMBEDDING_PROVIDERS.join(", ")}`,
      );
  }
}
