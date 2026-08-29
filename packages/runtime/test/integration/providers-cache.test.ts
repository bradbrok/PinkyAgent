/**
 * The standing check that prompt caching actually WORKS against the live
 * Anthropic API — the one thing no unit test can prove. `providers-caching.ts`
 * asserts the request shape; this asserts the API agreed with it, which is the
 * only signal that catches a silent invalidator (a byte that moved in the
 * prefix, a marker the model tier ignores, a proxy stripping cache_control).
 *
 * Two identical requests, back to back: the first writes the prefix, the
 * second must READ it (`usage.cacheRead > 0`). Everything in the request is
 * deterministic — no timestamps, no run ids, no randomness — because a prefix
 * that differs by one byte between the two calls is exactly the defect this
 * test exists to catch, and a stable prefix also lets a re-run within the 5m
 * window read instead of paying for another write.
 *
 * Cost: two calls at `max_tokens: 16` on the cheapest model. Skipped unless
 * PINKY_INTEGRATION=1 *and* ANTHROPIC_API_KEY is set — a missing key is a clean
 * skip, never a red suite. Override the model with PINKY_CACHE_TEST_MODEL
 * (mind the per-model minimum cacheable prefix: 4096 tokens on Haiku 4.5, so
 * SYSTEM below is padded well past it — a shorter prefix simply never caches
 * and the failure looks like a bug in this provider).
 */
import { describe, expect, test } from "bun:test";
import { AnthropicProvider } from "../../src/providers/anthropic";
import type { CompleteOptions } from "../../src/types";

const ENABLED = process.env.PINKY_INTEGRATION === "1" && Boolean(process.env.ANTHROPIC_API_KEY);
const MODEL = process.env.PINKY_CACHE_TEST_MODEL ?? "claude-haiku-4-5";

/** ~60.5k chars (≈13-15k tokens) of deterministic filler: identical on every
 *  call, every run, and well clear of the 4096-token minimum cacheable
 *  prefix. Sized for margin, not economy — it costs ~$0.02 per run on
 *  Haiku 4.5, which is the price of the signal. */
function paddedSystem(): string {
  const lines = ["You are PinkyAgent's cache probe. Answer with a single word."];
  for (let i = 0; i < 400; i += 1) {
    lines.push(
      `Fact ${String(i).padStart(4, "0")}: the cached prefix is a byte-for-byte prefix match, ` +
        `rendered tools then system then messages, and only content up to a breakpoint is cached.`,
    );
  }
  return lines.join("\n");
}

const SYSTEM = paddedSystem();

describe.skipIf(!ENABLED)("Anthropic prompt caching (live)", () => {
  test(
    "a repeated prefix is served from cache on the second call",
    async () => {
      const provider = new AnthropicProvider({
        apiKey: process.env.ANTHROPIC_API_KEY!,
        maxRetries: 1,
      });
      const opts: CompleteOptions = {
        model: MODEL,
        system: SYSTEM,
        messages: [{ role: "user", text: "Reply with the single word: ok" }],
        tools: [],
        maxTokens: 16,
      };

      const first = await provider.complete(opts);
      const second = await provider.complete({ ...opts, messages: [...opts.messages] });

      // The first call either wrote the prefix or read one a recent run left
      // behind; either way the API must have engaged the cache at all.
      const firstCached = (first.usage?.cacheCreation ?? 0) + (first.usage?.cacheRead ?? 0);
      expect(firstCached).toBeGreaterThan(0);

      // The second is the actual assertion: the prefix came back from cache.
      expect(second.usage?.cacheRead ?? 0).toBeGreaterThan(0);
      // ...and TokenUsage's convention holds — `input` is the UNCACHED
      // remainder, so a cached prefix leaves far fewer billed input tokens
      // than the ~13-15k the system prompt renders to.
      expect(second.usage!.input).toBeLessThan(second.usage!.cacheRead!);
    },
    120_000,
  );
});
