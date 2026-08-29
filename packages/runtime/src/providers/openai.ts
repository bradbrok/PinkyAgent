/**
 * OpenAI-compatible chat completions provider (streaming SSE).
 * Works against api.openai.com and any compatible endpoint (OPENAI_BASE_URL).
 *
 * Hardened per DESIGN.md §8.1: transient failures (408/409/429/5xx, network
 * blips, per-attempt timeouts) retry with jittered backoff inside `withRetry`;
 * anything after the first streamed event surfaces instead of restarting.
 *
 * Prompt caching on these routes (OpenAI, DeepSeek, OpenRouter) is AUTOMATIC
 * prefix caching: there is no `cache_control` to place and nothing to opt into.
 * The only things a client sends are (a) a byte-stable prefix — tools, then
 * system, then the transcript, which is what DESIGN.md §4.5/§9 are about — and
 * (b) on OpenAI, `prompt_cache_key`, a routing hint that keeps one thread
 * landing on the same cache shard. What comes BACK needs work: these APIs
 * count cached tokens INSIDE `prompt_tokens`, while `TokenUsage`
 * (core/events.ts) is disjoint, so the usage mapping below subtracts the
 * cached counts out of `input` (see `toTokenUsage`).
 */
import type { ToolCall } from "@pinky/core";
import type {
  AssistantTurn,
  CompleteOptions,
  LlmMessage,
  Provider,
  TokenUsage,
  ToolChoice,
  ToolSpec,
} from "../types";
import { iterateSse } from "./sse";
import {
  assertOk,
  DEFAULT_MAX_RETRIES,
  DEFAULT_TIMEOUT_MS,
  realSleep,
  withRetry,
  type RetryConfig,
  type SleepFn,
} from "./retry";

export interface OpenAIProviderOptions {
  /** Defaults to process.env.OPENAI_API_KEY. */
  apiKey?: string | undefined;
  /** Defaults to process.env.OPENAI_BASE_URL ?? https://api.openai.com/v1 */
  baseUrl?: string | undefined;
  /** Extra request headers (OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string> | undefined;
  /** Injectable for tests. */
  fetchFn?: typeof fetch | undefined;
  /** Send `stream_options: { include_usage: true }`. Default true. Some
   *  OpenAI-compatible endpoints reject the field outright — set false there
   *  (token usage is then simply absent from the turn). */
  includeUsage?: boolean | undefined;
  /** Send `prompt_cache_key: <opts.cacheKey>` when the caller supplies one.
   *  `createProvider` decides this per route, because the field is OpenAI-NATIVE:
   *  on for `openai/` against real api.openai.com, off for `openrouter/` and for
   *  any `openai/` with an OPENAI_BASE_URL override (vLLM, llama.cpp, LM Studio,
   *  Azure reject unknown body fields outright); PINKY_LLM_PROMPT_CACHE_KEY, when
   *  set, wins on every route. Constructed directly it defaults true. The key is
   *  only a cache-shard routing hint — dropping it costs hit rate, never
   *  correctness. */
  promptCacheKey?: boolean | undefined;
  /** Retries after the first attempt for 408/409/429/5xx + network errors. Default 3. */
  maxRetries?: number | undefined;
  /** Per-attempt budget covering connect *and* the streamed body. Default 120_000ms. */
  timeoutMs?: number | undefined;
  /** Backoff sleep; injectable so tests never wait on real timers. */
  sleep?: SleepFn | undefined;
  /** Jitter source (test hook). Default Math.random. */
  random?: (() => number) | undefined;
}
/** Defaults for the OpenRouter route (OpenAI-compatible). */
export const OPENROUTER_DEFAULTS = {
  baseUrl: "https://openrouter.ai/api/v1",
  referer: "https://github.com/pinkyagent",
  title: "PinkyAgent",
} as const;

type OpenAIMessage =
  | { role: "system" | "user"; content: string }
  | {
      role: "assistant";
      content: string | null;
      tool_calls?: {
        id: string;
        type: "function";
        function: { name: string; arguments: string };
      }[];
    }
  | { role: "tool"; tool_call_id: string; content: string };

/** Convert provider-agnostic messages to OpenAI chat shape. */
export function toOpenAIMessages(system: string, messages: LlmMessage[]): OpenAIMessage[] {
  const out: OpenAIMessage[] = [];
  if (system) out.push({ role: "system", content: system });
  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        out.push({ role: "system", content: msg.text });
        break;
      case "user":
        out.push({ role: "user", content: msg.text });
        break;
      case "assistant": {
        const toolCalls = msg.toolCalls ?? [];
        if (toolCalls.length > 0) {
          out.push({
            role: "assistant",
            content: msg.text || null,
            tool_calls: toolCalls.map((c) => ({
              id: c.id,
              type: "function" as const,
              function: { name: c.name, arguments: JSON.stringify(c.args) },
            })),
          });
        } else {
          out.push({ role: "assistant", content: msg.text || null });
        }
        break;
      }
      case "tool":
        out.push({ role: "tool", tool_call_id: msg.toolCallId ?? "", content: msg.text });
        break;
    }
  }
  return out;
}

function toOpenAITools(tools: ToolSpec[]): Record<string, unknown>[] {
  return tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }));
}

/**
 * Provider-side tool forcing. The tool LIST is sent unchanged either way:
 * shrinking `tools` invalidates every cache tier (tool definitions render at
 * position 0), while `tool_choice` invalidates only the messages tier — so the
 * loop masks with this (DESIGN.md §4.5/§9).
 */
export function toOpenAIToolChoice(choice: ToolChoice): unknown {
  switch (choice.type) {
    case "auto":
      return "auto";
    case "none":
      return "none";
    case "tool":
      return { type: "function", function: { name: choice.name } };
  }
}

/** The `usage` object as the OpenAI-compatible routes actually report it. */
interface OpenAIStreamUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  /** OpenAI / OpenRouter breakdown; OpenRouter adds `cache_write_tokens` when
   *  it is proxying a provider that bills cache writes (Anthropic). */
  prompt_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number } | null;
  /** DeepSeek reports the split at the top level instead. */
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
}

/**
 * Normalize a route's `usage` to the `TokenUsage` convention (core/events.ts):
 * `input` is the UNCACHED prompt remainder, DISJOINT from `cacheRead` and
 * `cacheCreation`, so total prompt = input + cacheRead + cacheCreation. That
 * is how Anthropic already reports it and what `pinky stats restarts` divides
 * by, but these APIs report the opposite: `prompt_tokens` INCLUDES the cached
 * tokens. Hence the subtraction (clamped at 0 against a route whose counters
 * disagree with its own total).
 *
 * Field names, by route:
 *   OpenAI / OpenRouter  `prompt_tokens_details.cached_tokens`
 *   OpenRouter passthrough of an Anthropic model
 *                        `prompt_tokens_details.cache_write_tokens`
 *   DeepSeek             `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`,
 *                        where prompt_tokens = hit + miss, so subtracting the
 *                        hit count reproduces the miss count exactly.
 *
 * A route reporting none of them leaves both fields ABSENT rather than 0 —
 * "nothing was cached" and "nobody counted" are different answers to the
 * DESIGN §13 cost question, and `exactOptionalPropertyTypes` forbids the
 * undefined-valued key that would blur them.
 */
export function toTokenUsage(u: OpenAIStreamUsage): TokenUsage {
  const details = u.prompt_tokens_details ?? undefined;
  const cacheRead = details?.cached_tokens ?? u.prompt_cache_hit_tokens;
  const cacheCreation = details?.cache_write_tokens;
  const prompt = u.prompt_tokens ?? 0;
  return {
    input: Math.max(0, prompt - (cacheRead ?? 0) - (cacheCreation ?? 0)),
    output: u.completion_tokens ?? 0,
    ...(cacheRead !== undefined ? { cacheRead } : {}),
    ...(cacheCreation !== undefined ? { cacheCreation } : {}),
  };
}

function mapFinishReason(raw: string | null | undefined): string {
  switch (raw) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
      return "tool_calls";
    default:
      return raw ?? "stop";
  }
}

export class OpenAIProvider implements Provider {
  readonly name = "openai";
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly extraHeaders: Record<string, string>;
  private readonly fetchFn: typeof fetch;
  private readonly includeUsage: boolean;
  private readonly promptCacheKey: boolean;
  private readonly retry: RetryConfig;

  constructor(opts: OpenAIProviderOptions = {}) {
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.baseUrl = (opts.baseUrl ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(
      /\/+$/,
      "",
    );
    this.extraHeaders = opts.extraHeaders ?? {};
    this.fetchFn = opts.fetchFn ?? fetch;
    this.includeUsage = opts.includeUsage ?? true;
    this.promptCacheKey = opts.promptCacheKey ?? true;
    this.retry = {
      label: "OpenAI",
      maxRetries: opts.maxRetries ?? DEFAULT_MAX_RETRIES,
      timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      sleep: opts.sleep ?? realSleep,
      random: opts.random,
    };
  }

  async complete(opts: CompleteOptions): Promise<AssistantTurn> {
    const body: Record<string, unknown> = {
      model: opts.model,
      messages: toOpenAIMessages(opts.system, opts.messages),
      stream: true,
    };
    if (this.includeUsage) body.stream_options = { include_usage: true };
    if (opts.tools.length > 0) {
      body.tools = toOpenAITools(opts.tools);
      // Only meaningful alongside a tool list; a bare `tool_choice` is a 400 on
      // some routes and a silent no-op on the rest.
      if (opts.toolChoice) body.tool_choice = toOpenAIToolChoice(opts.toolChoice);
    }
    // Routing hint only: it does not change the prompt, just which shard the
    // automatic prefix cache is looked up on.
    if (this.promptCacheKey && opts.cacheKey !== undefined) body.prompt_cache_key = opts.cacheKey;
    if (opts.maxTokens !== undefined) body.max_tokens = opts.maxTokens;
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    const payload = JSON.stringify(body);

    return withRetry(this.retry, opts.signal, async ({ signal, markStreamProgress }) => {
      const res = await this.fetchFn(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
          ...this.extraHeaders,
        },
        body: payload,
        signal,
      });
      await assertOk(res, "OpenAI");
      if (!res.body) throw new Error("OpenAI API returned no body");

      let text = "";
      let stopReason = "stop";
      let usage: TokenUsage | undefined;
      // streaming tool calls accumulate by their `index` field
      const calls = new Map<number, { id: string; name: string; argsJson: string }>();

      for await (const evt of iterateSse(res.body, signal)) {
        // Tokens have reached us: a later failure must surface, never restart.
        markStreamProgress();
        const data = evt.data.trim();
        if (!data || data === "[DONE]") continue;
        const parsed = JSON.parse(data) as Record<string, unknown>;

        const streamError = parsed.error;
        if (streamError && typeof streamError === "object") {
          const e = streamError as { message?: string; code?: unknown };
          throw new Error(
            `OpenAI stream error${e.code !== undefined ? ` (${String(e.code)})` : ""}: ${
              e.message ?? data.slice(0, 500)
            }`,
          );
        }

        const u = parsed.usage as OpenAIStreamUsage | null | undefined;
        if (u && (u.prompt_tokens !== undefined || u.completion_tokens !== undefined)) {
          usage = toTokenUsage(u);
        }

        const choices = parsed.choices as Record<string, unknown>[] | undefined;
        const choice = choices?.[0];
        if (!choice) continue;

        const delta = choice.delta as Record<string, unknown> | undefined;
        if (delta) {
          if (typeof delta.content === "string") text += delta.content;
          const toolDeltas = delta.tool_calls as Record<string, unknown>[] | undefined;
          for (const td of toolDeltas ?? []) {
            const index = td.index as number;
            let acc = calls.get(index);
            if (!acc) {
              acc = { id: "", name: "", argsJson: "" };
              calls.set(index, acc);
            }
            if (typeof td.id === "string" && td.id) acc.id = td.id;
            const fn = td.function as { name?: string; arguments?: string } | undefined;
            if (fn?.name) acc.name += fn.name;
            if (fn?.arguments) acc.argsJson += fn.arguments;
          }
        }

        const finish = choice.finish_reason as string | null | undefined;
        if (finish) stopReason = mapFinishReason(finish);
      }

      const toolCalls: ToolCall[] = [];
      for (const index of [...calls.keys()].sort((a, b) => a - b)) {
        const acc = calls.get(index)!;
        let args: Record<string, unknown> = {};
        try {
          args = acc.argsJson ? (JSON.parse(acc.argsJson) as Record<string, unknown>) : {};
        } catch {
          args = { _unparsed: acc.argsJson };
        }
        toolCalls.push({ id: acc.id, name: acc.name, args });
      }

      const turn: AssistantTurn = { text, toolCalls, stopReason };
      if (usage) turn.usage = usage;
      return turn;
    });
  }
}
