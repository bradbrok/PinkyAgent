/**
 * Provider factory: "provider/model-id" -> Provider instance.
 * The bare model id (only the routing prefix stripped) is what callers pass
 * as CompleteOptions.model — e.g. "openrouter/moonshotai/kimi-k2" yields the
 * model id "moonshotai/kimi-k2".
 *
 * Transport-class env knobs, all read here and nowhere else (they describe how
 * this deployment talks to a route, not what the agent does — hence env and not
 * the settings table):
 *   PINKY_LLM_MAX_RETRIES / PINKY_LLM_TIMEOUT_MS  retry budget, per-attempt deadline
 *   PINKY_LLM_INCLUDE_USAGE   stream_options.include_usage (OpenAI-compatible)
 *   PINKY_LLM_PROMPT_CACHE_KEY  send `prompt_cache_key` (an OpenAI-NATIVE field).
 *     Set, it wins on every route; unset it defaults true ONLY for `openai/`
 *     against real api.openai.com (no OPENAI_BASE_URL), false otherwise.
 *   PINKY_ANTHROPIC_CACHE_TTL   "5m" | "1h" on every cache_control breakpoint
 * Each parses loudly: a typo is a throw at construction, never a silent default.
 */
import type { Provider } from "../types";
import { AnthropicProvider, type CacheTtl } from "./anthropic";
import { createFakeProvider } from "./fake";
import { OpenAIProvider, OPENROUTER_DEFAULTS } from "./openai";

export {
  AnthropicProvider,
  DEFAULT_ANTHROPIC_MAX_TOKENS,
  applyMessageCacheBreakpoints,
  buildSystemBlocks,
  cacheControl,
  toAnthropicToolChoice,
} from "./anthropic";
export type {
  AnthropicMessage,
  AnthropicProviderOptions,
  AnthropicUsage,
  CacheControl,
  CacheTtl,
  ContentBlock,
  SystemTextBlock,
} from "./anthropic";
export { OpenAIProvider, OPENROUTER_DEFAULTS } from "./openai";
export type { OpenAIProviderOptions } from "./openai";
export {
  FakeProvider,
  createFakeProvider,
  FAKE_BEHAVIORS,
  FAKE_CANARY,
  FAKE_CANARY_QUERY,
  FAKE_DEFERRED_QUERY,
  FAKE_DEFERRED_MARKER,
} from "./fake";
export type { FakeBehavior, FakeScript } from "./fake";
export { iterateSse, sseStreamFromText } from "./sse";
export type { SseEvent } from "./sse";
export {
  DEFAULT_MAX_RETRIES,
  DEFAULT_TIMEOUT_MS,
  HttpStatusError,
  RequestTimeoutError,
  withRetry,
} from "./retry";
export type { RetryConfig, SleepFn } from "./retry";

export const SUPPORTED_PROVIDERS = ["anthropic", "openai", "openrouter", "fake"] as const;

/** Split "provider/model-id" on the first "/" — the model id may itself contain "/". */
export function splitModel(model: string): { provider: string; modelId: string } {
  const slash = model.indexOf("/");
  if (slash <= 0 || slash === model.length - 1) {
    throw new Error(
      `Model must be "provider/model-id", got ${JSON.stringify(model)}. ` +
        `Supported providers: ${SUPPORTED_PROVIDERS.join(", ")}`,
    );
  }
  return { provider: model.slice(0, slash), modelId: model.slice(slash + 1) };
}

/** Bootstrap-only env knobs (DESIGN: behavioral config lives in `settings`).
 *  These are transport hardening, not agent behavior, so env is the right home. */
function envInt(env: Record<string, string | undefined>, key: string): number | undefined {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${key} must be a non-negative integer, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function envBool(env: Record<string, string | undefined>, key: string): boolean | undefined {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  throw new Error(`${key} must be a boolean (true/false), got ${JSON.stringify(raw)}`);
}

/**
 * PINKY_ANTHROPIC_CACHE_TTL = "5m" (default) | "1h" — the lifetime stamped on
 * every `cache_control` breakpoint of an Anthropic request (system block +
 * the rolling two-message window; see providers/anthropic.ts). Transport-class,
 * not agent behavior: it trades a pricier cache write (~2x an input token
 * instead of ~1.25x) for a prefix that survives a wake hours later, which is a
 * property of how this *deployment* is driven, not of what the agent decides.
 * Unset/empty means the provider default. Anything else throws at construction.
 */
export function anthropicCacheTtlFromEnv(
  env: Record<string, string | undefined>,
): CacheTtl | undefined {
  const key = "PINKY_ANTHROPIC_CACHE_TTL";
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = raw.trim().toLowerCase();
  if (value === "5m" || value === "1h") return value;
  throw new Error(`${key} must be "5m" or "1h", got ${JSON.stringify(raw)}`);
}

/** Transport options shared by every route, read once per createProvider call. */
export function transportOptionsFromEnv(env: Record<string, string | undefined>): {
  maxRetries: number | undefined;
  timeoutMs: number | undefined;
} {
  return {
    maxRetries: envInt(env, "PINKY_LLM_MAX_RETRIES"),
    timeoutMs: envInt(env, "PINKY_LLM_TIMEOUT_MS"),
  };
}

/**
 * "provider/model-id" -> Provider.
 *
 * `fake/*` is a real route, not a test seam: `fake/echo`,
 * `fake/retain-recall` and `fake/deferred` need no API key and no network, so `pinky headless`,
 * `pinky smoke` and the integration suite can drive the whole stack on a bare
 * machine. It is for tests and smoke ONLY — it answers from a script, never a
 * model. See providers/fake.ts for the behavior list.
 */
export function createProvider(
  model: string,
  env: Record<string, string | undefined> = process.env,
): Provider {
  const { provider, modelId } = splitModel(model);
  const transport = transportOptionsFromEnv(env);
  // stream_options.include_usage is rejected by some OpenAI-compatible
  // endpoints; operators turn it off with PINKY_LLM_INCLUDE_USAGE=false.
  const includeUsage = envBool(env, "PINKY_LLM_INCLUDE_USAGE") ?? true;
  // `prompt_cache_key` is a cache-SHARD routing hint, not prompt content, so
  // dropping it costs hit rate and never correctness — but it is an OpenAI-NATIVE
  // body field, and a strict OpenAI-*compatible* server (vLLM, llama.cpp,
  // LM Studio, Azure) 400s an unknown top-level field on EVERY request. So it is
  // opt-out where it is known to work and opt-IN everywhere else: the env var,
  // when set, wins on every route; unset, the default is true only on `openai/`
  // with no OPENAI_BASE_URL override (i.e. real api.openai.com) and false on
  // `openrouter/` and on any `openai/` route pointed at a custom base URL.
  const promptCacheKeyEnv = envBool(env, "PINKY_LLM_PROMPT_CACHE_KEY");
  const customOpenAiBaseUrl = (env.OPENAI_BASE_URL ?? "").trim() !== "";
  // Parsed on every route, like the knobs above: a typo in a bootstrap env var
  // fails loudly at construction rather than silently on the one route it feeds.
  const cacheTtl = anthropicCacheTtlFromEnv(env);
  switch (provider) {
    case "anthropic":
      return new AnthropicProvider({ apiKey: env.ANTHROPIC_API_KEY, cacheTtl, ...transport });
    case "openai":
      return new OpenAIProvider({
        apiKey: env.OPENAI_API_KEY,
        baseUrl: env.OPENAI_BASE_URL,
        includeUsage,
        promptCacheKey: promptCacheKeyEnv ?? !customOpenAiBaseUrl,
        ...transport,
      });
    case "openrouter":
      return new OpenAIProvider({
        apiKey: env.OPENROUTER_API_KEY,
        baseUrl: env.OPENROUTER_BASE_URL ?? OPENROUTER_DEFAULTS.baseUrl,
        extraHeaders: {
          "HTTP-Referer": env.OPENROUTER_REFERER ?? OPENROUTER_DEFAULTS.referer,
          "X-Title": env.OPENROUTER_TITLE ?? OPENROUTER_DEFAULTS.title,
        },
        includeUsage,
        // OpenRouter is not OpenAI: the field is unknown to the router itself,
        // so it ships off unless an operator turns it on for a route they know.
        promptCacheKey: promptCacheKeyEnv ?? false,
        ...transport,
      });
    // Keyless, offline, scripted — tests and smoke only (see fake.ts).
    case "fake":
      return createFakeProvider(modelId);
    default:
      throw new Error(
        `Unknown provider ${JSON.stringify(provider)} in model ${JSON.stringify(model)}. ` +
          `Supported providers: ${SUPPORTED_PROVIDERS.join(", ")}`,
      );
  }
}
