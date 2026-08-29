/**
 * Scripted provider for tests and the CLI smoke command.
 * Pops turns from a script (or delegates to a function) and records every
 * CompleteOptions it receives for assertions.
 */
import type { ToolCall } from "@pinky/core";
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

/** What `fake/deferred` searches the tool catalog for. Both fixtures publish an
 *  `echo` tool whose DESCRIPTION carries the word, so the FTS voice finds it
 *  without depending on how Postgres tokenizes `mcp__<server>__<raw>`. */
export const FAKE_DEFERRED_QUERY = "echo";
/** Closing marker of a completed `fake/deferred` script. A test asserting on it
 *  is asserting that all four turns ran — a lucky echo cannot produce it. */
export const FAKE_DEFERRED_MARKER = "[fake/deferred done]";

/** Behaviors reachable as `fake/<id>`; listed in the error for a typo. */
export const FAKE_BEHAVIORS = ["echo", "retain-recall", "deferred"] as const;
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

// ---------------------------------------------------------------------------
// fake/deferred — the meta-tool round trip, scripted
// ---------------------------------------------------------------------------

/**
 * The four turns of `fake/deferred`, as a function of the prompt.
 *
 * State lives in the MESSAGES, not in a counter: the loop hands the provider
 * the projected window every turn, so "which step is this" is answered by the
 * newest assistant message ({@link currentStep}: nothing -> search -> describe
 * -> call -> done). That is what makes the behavior survive a retry, a context
 * restart, a second prompt in the same thread, or a run that resumes on a
 * later wake — a shift()-ing script would answer turn 1 to a window that
 * already contains three tool results.
 *
 * 1. `tool_search` for {@link FAKE_DEFERRED_QUERY}.
 * 2. `tool_describe` the first `mcp__…` name in that numbered list.
 * 3. `tool_call` it, with arguments BUILT FROM THE SCHEMA the describe printed
 *    (see {@link fillArgs}) and the user's own text as every string value —
 *    so this works against any echo-shaped tool rather than one fixture's
 *    exact property names.
 * 4. A final message: the tool result verbatim, then {@link FAKE_DEFERRED_MARKER}.
 *
 * Every off-script branch ends the run with a sentence naming what went wrong,
 * so a red assertion reads as a diagnosis instead of a timeout.
 */
function deferredTurn(messages: LlmMessage[]): AssistantTurn {
  const call = currentStep(messages);
  const result = lastToolResultText(messages);

  if (!call) return callTurn("d1", "tool_search", { query: FAKE_DEFERRED_QUERY });

  switch (call.name) {
    case "tool_search": {
      const name = firstMcpToolName(result);
      if (!name) return say(`fake/deferred: no mcp__ tool matched "${FAKE_DEFERRED_QUERY}"`);
      return callTurn("d2", "tool_describe", { name });
    }
    case "tool_describe": {
      const name = typeof call.args.name === "string" ? call.args.name : "";
      if (!name) return say("fake/deferred: tool_describe was called without a name");
      return callTurn("d3", "tool_call", {
        name,
        args: fillArgs(parseSchemaFence(result), lastUserText(messages)),
      });
    }
    case "tool_call":
      return say(`${result}\n${FAKE_DEFERRED_MARKER}`);
    default:
      return say(`fake/deferred: unexpected tool ${call.name}`);
  }
}

/** A turn that calls exactly one tool. */
function callTurn(id: string, name: string, args: Record<string, unknown>): AssistantTurn {
  return { text: "", toolCalls: [{ id, name, args }], stopReason: "tool_calls" };
}

/** A turn that ends the run. */
function say(text: string): AssistantTurn {
  return { text, toolCalls: [], stopReason: "stop" };
}

/**
 * Which step this window is in: the tool call of the NEWEST assistant message,
 * or undefined to start over.
 *
 * The newest ASSISTANT message, not the newest tool call — the difference is
 * what makes a second prompt in the same thread work. A thread that already
 * ran the script (a long-lived headless session, or smoke's fixed thread ids,
 * whose log carries every previous run) projects the whole previous exchange
 * into this window; its last assistant message is a plain answer with no tool
 * calls, which is precisely the signal that a script FINISHED and the new user
 * turn starts a fresh one. Scanning for the newest tool call instead would
 * resume yesterday's run at step 4 and answer with yesterday's tool result.
 */
function currentStep(messages: LlmMessage[]): ToolCall | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "assistant") continue;
    const calls = m.toolCalls ?? [];
    return calls[calls.length - 1];
  }
  return undefined;
}

/** Newest tool result text — what the step above just produced. */
function lastToolResultText(messages: LlmMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "tool") return m.text;
  }
  return "";
}

/** First namespaced MCP name in a `tool_search` listing. The character class is
 *  the one mcpToolName() produces (`[A-Za-z0-9_-]` after the `mcp__` prefix). */
function firstMcpToolName(text: string): string | undefined {
  return /mcp__[A-Za-z0-9_-]+/.exec(text)?.[0];
}

/** The JSON Schema out of a `tool_describe` result's ```json fence. */
function parseSchemaFence(text: string): unknown {
  const fenced = /```json\n([\s\S]*?)\n```/.exec(text);
  if (!fenced?.[1]) return undefined;
  try {
    return JSON.parse(fenced[1]) as unknown;
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * The smallest argument object a schema accepts, with `text` in every string
 * slot. Only REQUIRED properties are filled (an optional one is noise the
 * server has to defend against), and nesting is followed so an object-valued
 * required property gets its own required children — which is exactly the
 * shape the modern MCP fixture publishes (`{ outer: { inner } }`) and also the
 * flat `{ text }` shape a simpler server would.
 *
 * Total and non-throwing: an absent or unparseable schema yields `{}`, which a
 * no-argument tool accepts and everything else answers with its schema (the
 * deferred registry's job, not this one's).
 */
function fillArgs(schema: unknown, text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!isRecord(schema)) return out;
  const properties = isRecord(schema.properties) ? schema.properties : {};
  const required = Array.isArray(schema.required) ? schema.required : [];
  for (const key of required) {
    if (typeof key !== "string") continue;
    const prop = properties[key];
    out[key] = isRecord(prop) ? fillValue(prop, text) : text;
  }
  return out;
}

/** One property's placeholder value, by declared JSON Schema type. */
function fillValue(schema: Record<string, unknown>, text: string): unknown {
  switch (schema.type) {
    case "object":
      return fillArgs(schema, text);
    case "array":
      return [];
    case "number":
    case "integer":
      return 1;
    case "boolean":
      return true;
    default:
      return text;
  }
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
 * - `fake/deferred`      four turns: `tool_search` -> `tool_describe` ->
 *   `tool_call` -> a final message — the deferred-tool round trip (slice 9),
 *   which is the only way to prove end to end that a tool NOT in the header
 *   was found, read and executed. See {@link deferredTurn}.
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
    case "deferred":
      return new FakeProvider(deferredTurn);
    default:
      throw new Error(
        `Unknown fake behavior ${JSON.stringify(behavior)}. ` +
          `Supported: ${FAKE_BEHAVIORS.join(", ")}`,
      );
  }
}
