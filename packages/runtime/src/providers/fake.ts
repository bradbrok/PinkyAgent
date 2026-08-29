/**
 * Scripted provider for tests and the CLI smoke command.
 * Pops turns from a script (or delegates to a function) and records every
 * CompleteOptions it receives for assertions.
 */
import type { AssistantTurn, CompleteOptions, LlmMessage, Provider } from "../types";

export type FakeScript = AssistantTurn[] | ((messages: LlmMessage[]) => AssistantTurn);

export class FakeProvider implements Provider {
  readonly name = "fake";
  private readonly script: FakeScript;
  /** Every CompleteOptions passed to complete(), in order. */
  readonly received: CompleteOptions[] = [];

  constructor(script: FakeScript) {
    this.script = Array.isArray(script) ? [...script] : script;
  }

  complete(opts: CompleteOptions): Promise<AssistantTurn> {
    this.received.push(opts);
    if (typeof this.script === "function") {
      return Promise.resolve(this.script(opts.messages));
    }
    const next = this.script.shift();
    if (!next) {
      return Promise.reject(
        new Error(`FakeProvider: script exhausted after ${this.received.length} call(s)`),
      );
    }
    return Promise.resolve(next);
  }
}

// ---------------------------------------------------------------------------
// The `fake/<behavior>` route (createProvider) — tests, smoke, headless e2e
// ---------------------------------------------------------------------------

/** Canary text `fake/retain-recall` stores, distinctive enough that the FTS
 *  voice finds it and nothing else. */
export const FAKE_CANARY = "The fake canary passphrase is zebra-quartz.";
/** Query `fake/retain-recall` recalls with. */
export const FAKE_CANARY_QUERY = "fake canary passphrase";

/** Behaviors reachable as `fake/<id>`; listed in the error for a typo. */
export const FAKE_BEHAVIORS = ["echo", "retain-recall"] as const;
export type FakeBehavior = (typeof FAKE_BEHAVIORS)[number];

/** Newest user-role message in the prompt — the ingress the loop just projected
 *  (an auto-recall `<memories>` block is also `user`, but it sits at index 0). */
function lastUserText(messages: LlmMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "user") return m.text;
  }
  return "";
}

/**
 * Keyless provider behaviors for `createProvider("fake/<behavior>")`.
 *
 * NOT for production: no network, no model, deterministic by construction.
 * It exists so an end-to-end run (headless JSONL, smoke, an integration test)
 * can exercise the whole stack — ingest, loop, tools, event log, delivery —
 * on a machine with no API key at all.
 *
 * - `fake/echo`          one turn, no tools: replies `echo: <last user text>`.
 * - `fake/retain-recall` three turns: `retain` the canary, `recall` it, then
 *   a closing message — the memory plane's round trip (DESIGN.md §5).
 */
export function createFakeProvider(behavior: string): FakeProvider {
  switch (behavior) {
    case "echo":
      return new FakeProvider((messages) => ({
        text: `echo: ${lastUserText(messages)}`,
        toolCalls: [],
        stopReason: "stop",
      }));
    case "retain-recall":
      return new FakeProvider([
        {
          text: "",
          toolCalls: [
            { id: "f1", name: "retain", args: { text: FAKE_CANARY, kind: "semantic", importance: 7 } },
          ],
          stopReason: "tool_calls",
        },
        {
          text: "",
          toolCalls: [{ id: "f2", name: "recall", args: { query: FAKE_CANARY_QUERY } }],
          stopReason: "tool_calls",
        },
        { text: "Retained and recalled.", toolCalls: [], stopReason: "stop" },
      ]);
    default:
      throw new Error(
        `Unknown fake behavior ${JSON.stringify(behavior)}. ` +
          `Supported: ${FAKE_BEHAVIORS.join(", ")}`,
      );
  }
}
