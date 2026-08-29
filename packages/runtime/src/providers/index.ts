/**
 * Provider factory: "provider/model-id" -> Provider instance.
 * The bare model id (only the routing prefix stripped) is what callers pass
 * as CompleteOptions.model — e.g. "openrouter/moonshotai/kimi-k2" yields the
 * model id "moonshotai/kimi-k2".
 */
import type { Provider } from "../types";
import { AnthropicProvider } from "./anthropic";
import { createFakeProvider } from "./fake";
import { OpenAIProvider, OPENROUTER_DEFAULTS } from "./openai";

export { AnthropicProvider, DEFAULT_ANTHROPIC_MAX_TOKENS, buildSystemBlocks } from "./anthropic";
export type { AnthropicProviderOptions, AnthropicUsage, SystemTextBlock } from "./anthropic";
export { OpenAIProvider, OPENROUTER_DEFAULTS } from "./openai";
export type { OpenAIProviderOptions } from "./openai";
export {
  FakeProvider,
  createFakeProvider,
  FAKE_BEHAVIORS,
  FAKE_CANARY,
  FAKE_CANARY_QUERY,
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
 * `fake/*` is a real route, not a test seam: `fake/echo` and
 * `fake/retain-recall` need no API key and no network, so `pinky headless`,
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
  switch (provider) {
    case "anthropic":
      return new AnthropicProvider({ apiKey: env.ANTHROPIC_API_KEY, ...transport });
    case "openai":
      return new OpenAIProvider({
        apiKey: env.OPENAI_API_KEY,
        baseUrl: env.OPENAI_BASE_URL,
        includeUsage,
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
