/**
 * Anthropic Messages API provider (streaming SSE).
 * Docs: POST /v1/messages, headers x-api-key + anthropic-version.
 *
 * Hardened per DESIGN.md §8.1 (LLM call = activity boundary: a recorded result
 * or a loud failure) and §4.5/§9 (stable cached prefix, no KV-cache thrash):
 * transient failures retry with jittered backoff inside `withRetry` and every
 * attempt carries a deadline.
 *
 * ## Cache breakpoint layout (DESIGN §4.5/§9)
 *
 * Prompt caching is a PREFIX match over the rendered request, in render order
 * `tools` → `system` → `messages`. Only content *up to* a `cache_control`
 * marker is cached, so a marker on the system block alone caches tools+system
 * and nothing else — the conversation is then re-billed uncached on every
 * single turn, which is the exact defect this layout fixes. Up to three markers
 * go out, all sharing one ttl:
 *
 *   1. the stable system block (`buildSystemBlocks`) — covers tools + system;
 *   2. the LAST content block of the second-to-last message;
 *   3. the LAST content block of the last message.
 *
 * (2) and (3) are a rolling two-marker window (`applyMessageCacheBreakpoints`):
 * the older marker is where THIS request reads — it was the newest marker last
 * turn, so that entry exists — while the newer marker writes the entry the next
 * turn will read. The API allows 4 markers per request, so one slot stays free.
 *
 * Two markers rather than one because a breakpoint looks back at most ~20
 * content blocks for an existing entry to extend: a lone marker on the last
 * message would have to find last turn's entry across everything appended
 * since, and one fat turn (a batch of parallel tool_results) can outrun that
 * lookback and silently write a fresh entry instead of reading. The rolling
 * window keeps the read point within reach by construction.
 *
 * Mixing TTLs would require ordering longer-before-shorter, so `cacheTtl`
 * applies to EVERY marker or none. "1h" writes cost ~2x an input token (vs
 * ~1.25x for 5m) and exist for sporadically woken agents whose next wake lands
 * well past five minutes — see `PINKY_ANTHROPIC_CACHE_TTL` in providers/index.
 *
 * `cache: false` (alias: `cacheSystem: false`) emits NO marker anywhere, for
 * Anthropic-compatible proxies that reject `cache_control` outright.
 *
 * ## Why `tool_choice`, never a shorter tool list
 * Invalidation is tiered: changing tool DEFINITIONS invalidates every tier
 * (tools render at position 0, ahead of system and messages), whereas
 * `tool_choice` invalidates only the messages tier. So the loop masks tools
 * with `CompleteOptions.toolChoice` and keeps the rendered list byte-identical
 * (see types.ts ToolChoice). The field ships only when `tools` is non-empty —
 * the API rejects a tool_choice with no tools to choose from.
 */
import type { ToolCall } from "@pinky/core";
import type {
  AssistantTurn,
  CompleteOptions,
  LlmMessage,
  Provider,
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

/**
 * Streaming default. The Anthropic guidance for streamed requests is ~64k (HTTP
 * timeouts are not a concern once streaming, so leave the model room); it is a
 * ceiling, not a target. Models with a lower output cap (pre-4.x, 8192) reject
 * it — override with `maxTokens` here or per call via CompleteOptions.maxTokens.
 */
export const DEFAULT_ANTHROPIC_MAX_TOKENS = 64_000;

export interface AnthropicProviderOptions {
  /** Defaults to process.env.ANTHROPIC_API_KEY. */
  apiKey?: string | undefined;
  /** Defaults to https://api.anthropic.com */
  baseUrl?: string | undefined;
  /** Injectable for tests. */
  fetchFn?: typeof fetch | undefined;
  /** Retries after the first attempt for 408/409/429/5xx + network errors. Default 3. */
  maxRetries?: number | undefined;
  /** Per-attempt budget covering connect *and* the streamed body. Default 120_000ms. */
  timeoutMs?: number | undefined;
  /** Backoff sleep; injectable so tests never wait on real timers. */
  sleep?: SleepFn | undefined;
  /** Jitter source (test hook). Default Math.random. */
  random?: (() => number) | undefined;
  /** max_tokens when the caller does not set one. Default 64_000. */
  maxTokens?: number | undefined;
  /**
   * Emit `cache_control` breakpoints at all (system block + the rolling
   * two-message window). Default true. Turn it off for Anthropic-compatible
   * proxies that reject cache_control — it is all-or-nothing, never a
   * half-marked request.
   */
  cache?: boolean | undefined;
  /** Deprecated alias for `cache`; it never gated only the system block in
   *  practice, and now that markers ride on messages too the name lies.
   *  `cache` wins when both are set. */
  cacheSystem?: boolean | undefined;
  /** TTL stamped on EVERY breakpoint. Default "5m"; "1h" costs ~2x an input
   *  token to write and suits an agent woken hours apart. Mixed TTLs are not
   *  supported (the API wants longer-before-shorter ordering). */
  cacheTtl?: CacheTtl | undefined;
}

/** Cache lifetime for every breakpoint in a request. */
export type CacheTtl = "5m" | "1h";

/** `{type:"ephemeral"}` is the 5m form; the 1h form carries an explicit ttl. */
export interface CacheControl {
  type: "ephemeral";
  ttl?: "1h";
}

/** One `cache_control` value, shaped by the request-wide ttl. */
export function cacheControl(ttl: CacheTtl = "5m"): CacheControl {
  return ttl === "1h" ? { type: "ephemeral", ttl: "1h" } : { type: "ephemeral" };
}

type ContentBlockBody =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string };

/** Any message content block may carry a breakpoint (text, tool_use, tool_result). */
export type ContentBlock = ContentBlockBody & { cache_control?: CacheControl };

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: ContentBlock[];
}

/** A `system` array entry; only the stable prefix carries a cache breakpoint. */
export interface SystemTextBlock {
  type: "text";
  text: string;
  cache_control?: CacheControl;
}

/**
 * Usage as reported by Anthropic, including the cache counters. Structurally
 * identical to AssistantTurn["usage"] (core's TokenUsage), which declares
 * cacheRead/cacheCreation, so the counters survive into the journaled
 * `message` event rather than being dropped at the boundary.
 */
export interface AnthropicUsage {
  input: number;
  output: number;
  /** usage.cache_read_input_tokens — prefix served from cache (~0.1x cost). */
  cacheRead?: number;
  /** usage.cache_creation_input_tokens — prefix written to cache (~1.25x cost). */
  cacheCreation?: number;
}

/** Convert provider-agnostic messages to Anthropic's alternating shape. */
export function toAnthropicMessages(messages: LlmMessage[]): {
  systemSuffix: string[];
  messages: AnthropicMessage[];
} {
  const systemSuffix: string[] = [];
  const out: AnthropicMessage[] = [];

  const pushBlock = (role: "user" | "assistant", block: ContentBlock): void => {
    const last = out[out.length - 1];
    if (last && last.role === role) {
      last.content.push(block);
    } else {
      out.push({ role, content: [block] });
    }
  };

  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        // Anthropic takes system out-of-band; caller merges into the system param.
        if (msg.text) systemSuffix.push(msg.text);
        break;
      case "user":
        pushBlock("user", { type: "text", text: msg.text });
        break;
      case "assistant":
        if (msg.text) pushBlock("assistant", { type: "text", text: msg.text });
        for (const call of msg.toolCalls ?? []) {
          pushBlock("assistant", { type: "tool_use", id: call.id, name: call.name, input: call.args });
        }
        break;
      case "tool":
        // tool results ride in a user-role message
        pushBlock("user", { type: "tool_result", tool_use_id: msg.toolCallId ?? "", content: msg.text });
        break;
    }
  }
  return { systemSuffix, messages: out };
}

/**
 * Build the `system` array. Caching is a prefix match rendered tools -> system ->
 * messages, so the breakpoint goes on the *stable* prompt only: mid-conversation
 * system-role text (systemSuffix) follows as separate blocks with no
 * cache_control, which keeps it outside the cached prefix and stops it from
 * invalidating the cache on every turn (DESIGN §9 "KV-cache thrash").
 */
export function buildSystemBlocks(
  system: string,
  systemSuffix: string[],
  cache = true,
  ttl: CacheTtl = "5m",
): SystemTextBlock[] {
  const blocks: SystemTextBlock[] = [];
  if (system.length > 0) {
    const head: SystemTextBlock = { type: "text", text: system };
    if (cache) head.cache_control = cacheControl(ttl);
    blocks.push(head);
  }
  for (const text of systemSuffix) {
    if (text.length > 0) blocks.push({ type: "text", text });
  }
  return blocks;
}

/**
 * The conversation half of the layout: a marker on the LAST content block of
 * the last two messages (oh-my-pi's rolling window). Pure — it returns a new
 * array and clones only the blocks it stamps, so the caller's messages are
 * never mutated and an unmarked request is byte-identical to its input.
 *
 * Two markers is the steady state: the older is this request's read point (it
 * was written last turn, and sits within the ~20-block lookback), the newer is
 * what the next turn will read. A one-message conversation gets one marker;
 * `cache: false` gets none. Combined with the system marker that is at most 3
 * of the 4 the API allows. Messages with no content blocks are skipped — there
 * is nothing to stamp — and the window keeps looking further back.
 */
export function applyMessageCacheBreakpoints(
  messages: AnthropicMessage[],
  cache = true,
  ttl: CacheTtl = "5m",
): AnthropicMessage[] {
  if (!cache) return messages;
  const marks = new Set<number>();
  for (let i = messages.length - 1; i >= 0 && marks.size < 2; i -= 1) {
    if ((messages[i]?.content.length ?? 0) > 0) marks.add(i);
  }
  if (marks.size === 0) return messages;
  return messages.map((msg, i) => {
    if (!marks.has(i)) return msg;
    const content = [...msg.content];
    const last = content.length - 1;
    content[last] = { ...content[last]!, cache_control: cacheControl(ttl) };
    return { ...msg, content };
  });
}

/**
 * `CompleteOptions.toolChoice` → the Anthropic `tool_choice` body field (same
 * three shapes). Absent means "let the provider default"; the caller omits the
 * field entirely rather than sending `{type:"auto"}`, so a request that never
 * masks tools stays byte-identical to what it sent before this existed.
 */
export function toAnthropicToolChoice(
  choice: ToolChoice | undefined,
): Record<string, unknown> | undefined {
  if (!choice) return undefined;
  switch (choice.type) {
    case "auto":
      return { type: "auto" };
    case "none":
      return { type: "none" };
    case "tool":
      return { type: "tool", name: choice.name };
  }
}

function toAnthropicTools(tools: ToolSpec[]): Record<string, unknown>[] {
  return tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
}

function mapStopReason(raw: string | null | undefined): string {
  switch (raw) {
    case "end_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    default:
      return raw ?? "stop";
  }
}

export class AnthropicProvider implements Provider {
  readonly name = "anthropic";
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;
  private readonly maxTokens: number;
  /** Whether this provider emits cache_control at all (public so the factory's
   *  env wiring is observable without a round trip). */
  readonly cache: boolean;
  /** TTL stamped on every breakpoint — see PINKY_ANTHROPIC_CACHE_TTL. */
  readonly cacheTtl: CacheTtl;
  private readonly retry: RetryConfig;

  constructor(opts: AnthropicProviderOptions = {}) {
    this.apiKey = opts.apiKey ?? process.env.ANTHROPIC_API_KEY ?? "";
    this.baseUrl = (opts.baseUrl ?? "https://api.anthropic.com").replace(/\/+$/, "");
    this.fetchFn = opts.fetchFn ?? fetch;
    this.maxTokens = opts.maxTokens ?? DEFAULT_ANTHROPIC_MAX_TOKENS;
    // `cacheSystem` is the pre-rolling-window name, kept working as an alias.
    this.cache = opts.cache ?? opts.cacheSystem ?? true;
    this.cacheTtl = opts.cacheTtl ?? "5m";
    this.retry = {
      label: "Anthropic",
      maxRetries: opts.maxRetries ?? DEFAULT_MAX_RETRIES,
      timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      sleep: opts.sleep ?? realSleep,
      random: opts.random,
    };
  }

  async complete(opts: CompleteOptions): Promise<AssistantTurn> {
    const { systemSuffix, messages: converted } = toAnthropicMessages(opts.messages);
    const system = buildSystemBlocks(opts.system, systemSuffix, this.cache, this.cacheTtl);
    const messages = applyMessageCacheBreakpoints(converted, this.cache, this.cacheTtl);

    const body: Record<string, unknown> = {
      model: opts.model,
      max_tokens: opts.maxTokens ?? this.maxTokens,
      messages,
      stream: true,
    };
    if (system.length > 0) body.system = system;
    if (opts.tools.length > 0) {
      body.tools = toAnthropicTools(opts.tools);
      // Only with tools present: the API rejects a tool_choice that has
      // nothing to choose from, and masking is the loop's job either way.
      const toolChoice = toAnthropicToolChoice(opts.toolChoice);
      if (toolChoice) body.tool_choice = toolChoice;
    }
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    const payload = JSON.stringify(body);

    return withRetry(this.retry, opts.signal, async ({ signal, markStreamProgress }) => {
      const res = await this.fetchFn(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": this.apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: payload,
        signal,
      });
      await assertOk(res, "Anthropic");
      if (!res.body) throw new Error("Anthropic API returned no body");

      let text = "";
      let stopReason = "stop";
      let usage: AnthropicUsage | undefined;
      // content blocks accumulate by index; tool_use blocks buffer partial JSON
      const blocks = new Map<number, { kind: "text" } | { kind: "tool_use"; id: string; name: string; json: string }>();

      for await (const evt of iterateSse(res.body, signal)) {
        // Tokens have reached us: a later failure must surface, never restart.
        markStreamProgress();
        if (!evt.data) continue;
        const parsed = JSON.parse(evt.data) as Record<string, unknown>;
        const type = typeof parsed.type === "string" ? parsed.type : evt.event;
        switch (type) {
          case "message_start": {
            const msg = parsed.message as
              | {
                  usage?: {
                    input_tokens?: number;
                    cache_read_input_tokens?: number;
                    cache_creation_input_tokens?: number;
                    /**
                     * Per-ttl split of cache_creation_input_tokens, sent when
                     * the request mixes TTLs. Deliberately ignored: it is a
                     * breakdown of the same total, and TokenUsage carries one
                     * cacheCreation figure (this provider uses one ttl for the
                     * whole request anyway). Declared so it is plainly tolerated.
                     */
                    cache_creation?: {
                      ephemeral_5m_input_tokens?: number;
                      ephemeral_1h_input_tokens?: number;
                    };
                  };
                }
              | undefined;
            const u = msg?.usage;
            if (u) {
              usage = { input: u.input_tokens ?? 0, output: 0 };
              if (u.cache_read_input_tokens !== undefined) usage.cacheRead = u.cache_read_input_tokens;
              if (u.cache_creation_input_tokens !== undefined) {
                usage.cacheCreation = u.cache_creation_input_tokens;
              }
            }
            break;
          }
          case "content_block_start": {
            const index = parsed.index as number;
            const block = parsed.content_block as Record<string, unknown>;
            if (block.type === "tool_use") {
              blocks.set(index, {
                kind: "tool_use",
                id: block.id as string,
                name: block.name as string,
                json: "",
              });
            } else {
              blocks.set(index, { kind: "text" });
            }
            break;
          }
          case "content_block_delta": {
            const index = parsed.index as number;
            const delta = parsed.delta as Record<string, unknown>;
            const block = blocks.get(index);
            if (delta.type === "text_delta" && typeof delta.text === "string") {
              text += delta.text;
            } else if (delta.type === "input_json_delta" && typeof delta.partial_json === "string") {
              if (block && block.kind === "tool_use") block.json += delta.partial_json;
            }
            break;
          }
          case "message_delta": {
            const delta = parsed.delta as { stop_reason?: string } | undefined;
            if (delta?.stop_reason) stopReason = mapStopReason(delta.stop_reason);
            const u = parsed.usage as { output_tokens?: number } | undefined;
            if (u?.output_tokens !== undefined) {
              usage = { ...(usage ?? { input: 0, output: 0 }), output: u.output_tokens };
            }
            break;
          }
          case "error": {
            // Mid-stream API error (overloaded_error, ...). Loud, not truncated.
            const err = parsed.error as { type?: string; message?: string } | undefined;
            throw new Error(
              `Anthropic stream error${err?.type ? ` (${err.type})` : ""}: ${err?.message ?? evt.data.slice(0, 500)}`,
            );
          }
          default:
            break; // message_stop, content_block_stop, ping, ...
        }
      }

      const toolCalls: ToolCall[] = [];
      for (const index of [...blocks.keys()].sort((a, b) => a - b)) {
        const block = blocks.get(index)!;
        if (block.kind !== "tool_use") continue;
        let args: Record<string, unknown> = {};
        try {
          args = block.json ? (JSON.parse(block.json) as Record<string, unknown>) : {};
        } catch {
          args = { _unparsed: block.json };
        }
        toolCalls.push({ id: block.id, name: block.name, args });
      }

      const turn: AssistantTurn = { text, toolCalls, stopReason };
      if (usage) turn.usage = usage;
      return turn;
    });
  }
}
