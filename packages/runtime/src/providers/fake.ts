/**
 * Scripted provider for tests and the CLI smoke command.
 * Pops turns from a script (or delegates to a function) and records every
 * CompleteOptions it receives for assertions.
 */
import type { ToolCall } from "@pinky/core";
import type { AssistantTurn, CompleteOptions, LlmMessage, Provider } from "../types";

/**
 * A canned list of turns, or a function of the prompt.
 *
 * The function form takes the whole `CompleteOptions` as a second parameter
 * because some routes answer the REQUEST rather than the conversation:
 * `fake/sleep` keys on which tool the sleep worker forced, and that lives in
 * `opts.tools`, not in `messages`. Second parameter, never a replacement —
 * every existing one-argument script still type-checks and still runs.
 */
export type FakeScript =
  | AssistantTurn[]
  | ((messages: LlmMessage[], opts: CompleteOptions) => AssistantTurn);

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
      return Promise.resolve(this.script(opts.messages, opts));
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

/** Lines `fake/sleep` turns into extraction candidates: the text after
 *  `remember:` on any line of the rendered transcript. Multiline + insensitive
 *  because a transcript line is `[<seq>] user <platform>:<id>: remember: …`.
 *  Route-internal: exported for this module's unit test only, NOT re-exported
 *  from `@pinky/runtime` — a caller seeding a thread writes the literal word
 *  `remember:` in its text, it does not need the pattern. */
export const FAKE_SLEEP_REMEMBER_RE = /remember:\s*(.+)$/gim;
/** Opening words of the insight `fake/sleep` synthesizes. A memory row starting
 *  with this came from the reflect pass and nothing else, which is what lets a
 *  test assert the third LLM call ran rather than that *a* row exists. */
export const FAKE_SLEEP_REFLECT_PREFIX = "Reflection over";

/** Behaviors reachable as `fake/<id>`; listed in the error for a typo. */
export const FAKE_BEHAVIORS = ["echo", "retain-recall", "deferred", "sleep"] as const;
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

// ---------------------------------------------------------------------------
// fake/sleep — the sleep-time worker's three calls, scripted (slice 6)
// ---------------------------------------------------------------------------

/**
 * The tool names the worker forces (packages/sleep/src/schemas.ts:
 * EXTRACT_TOOL_NAME / DECIDE_TOOL_NAME / REFLECT_TOOL_NAME) and the candidate
 * cap (MAX_CANDIDATES). Copied, not imported: the dependency direction is
 * `core <- runtime <- sleep`, so runtime may not reach into @pinky/sleep. Keep
 * the two in step — a rename there makes every call here "unexpected", which
 * is at least a loud failure rather than a wrong memory.
 */
const SLEEP_EXTRACT_TOOL = "extract_memories";
const SLEEP_DECIDE_TOOL = "decide_memory_updates";
const SLEEP_REFLECT_TOOL = "reflect_memories";
const SLEEP_MAX_CANDIDATES = 12;
/** `maxLength` of an insight's `text` in the reflect schema. Quoting a long
 *  memory row verbatim would otherwise fail the worker's validator and turn a
 *  keyless smoke run into a `failed` pass over nothing. */
const SLEEP_MAX_INSIGHT_CHARS = 1500;

/**
 * The single sentence every off-script branch answers with. One fixed string
 * (not a per-branch diagnosis) because the worker's hand-rolled validator sees
 * a turn with NO tool call and reports `failed` cleanly; a route that threw
 * instead would surface as a stack trace from inside a sweep.
 */
const SLEEP_UNEXPECTED = "fake/sleep: unexpected call";

/**
 * `fake/sleep`, keyed on the tool the worker FORCED rather than on the
 * conversation: each of the three passes is a one-shot `complete()` with a
 * single user message and a single-entry `tools` array, so there is no window
 * to read a position out of (unlike {@link deferredTurn}).
 *
 * It exists so `bun run smoke` and the CLI e2e can drive the whole
 * extract -> decide -> reflect round trip — LLM calls, memory writes, receipts —
 * on a machine with no API key, and deterministically enough that a second
 * sweep over the same fact is provably a no-op.
 */
function sleepTurn(messages: LlmMessage[], opts: CompleteOptions): AssistantTurn {
  const text = lastUserText(messages);
  switch (opts.tools[0]?.name) {
    case SLEEP_EXTRACT_TOOL:
      return sleepExtractTurn(text);
    case SLEEP_DECIDE_TOOL:
      return sleepDecideTurn(text);
    case SLEEP_REFLECT_TOOL:
      return sleepReflectTurn(text);
    default:
      return say(SLEEP_UNEXPECTED);
  }
}

/** One candidate per `remember:` line of the transcript. */
function sleepExtractTurn(transcript: string): AssistantTurn {
  const candidates: Record<string, unknown>[] = [];
  // A fresh regex per call: FAKE_SLEEP_REMEMBER_RE is global, so `exec` leaves
  // `lastIndex` behind and a scan that stops at the cap would resume mid-string
  // on the next pass — the same transcript would extract different candidates
  // the second time it is read.
  const re = new RegExp(FAKE_SLEEP_REMEMBER_RE.source, FAKE_SLEEP_REMEMBER_RE.flags);
  for (const match of transcript.matchAll(re)) {
    const text = (match[1] ?? "").trim();
    // minLength 1 in the schema: a blank candidate would fail the whole pass.
    if (text === "") continue;
    candidates.push({ text, kind: "semantic", importance: 7, visibility: "channel" });
    if (candidates.length >= SLEEP_MAX_CANDIDATES) break;
  }
  return callTurn(`fake-extract-${candidates.length}`, SLEEP_EXTRACT_TOOL, { candidates });
}

/**
 * ADD unless the fact is already there verbatim, which is what makes a second
 * sweep over the same transcript idempotent in smoke: the first pass retains
 * the row, the second finds it as a neighbor with byte-identical text and says
 * NOOP. Exact equality, never a similarity heuristic — a fake must not have
 * judgement, only a rule a test can predict.
 */
function sleepDecideTurn(payload: string): AssistantTurn {
  const parsed = parseJsonRecord(payload);
  const candidates = parsed && Array.isArray(parsed.candidates) ? parsed.candidates : undefined;
  if (!candidates) return say(SLEEP_UNEXPECTED);

  const decisions = candidates.map((raw, i) => {
    const candidate = isRecord(raw) ? raw : {};
    // The payload's own index, so a decision still names its candidate if the
    // worker ever sends them out of array order (the validator demands each
    // index exactly once).
    const index = typeof candidate.index === "number" ? candidate.index : i;
    const text = typeof candidate.text === "string" ? candidate.text : "";
    const neighbors = Array.isArray(candidate.neighbors) ? candidate.neighbors : [];
    const known = neighbors.some((n) => isRecord(n) && n.text === text);
    return { candidate: index, action: known ? "NOOP" : "ADD" };
  });
  return callTurn(`fake-decide-${decisions.length}`, SLEEP_DECIDE_TOOL, { decisions });
}

/** ONE insight over the whole batch, citing every row and superseding none —
 *  invalidation is the interesting path and a fake should not take it by
 *  default. An empty batch returns zero insights (the worker still journals a
 *  receipt, so the watermark moves). */
function sleepReflectTurn(payload: string): AssistantTurn {
  const parsed = parseJsonRecord(payload);
  const memories = parsed && Array.isArray(parsed.memories) ? parsed.memories : undefined;
  if (!memories) return say(SLEEP_UNEXPECTED);

  const rows = memories.filter(isRecord);
  const first = rows[0];
  const insights =
    first === undefined
      ? []
      : [
          {
            text: `${FAKE_SLEEP_REFLECT_PREFIX} ${rows.length} memories: ${String(
              first.text ?? "",
            )}`.slice(0, SLEEP_MAX_INSIGHT_CHARS),
            importance: 5,
            sources: rows.map((r) => String(r.id ?? "")),
            supersedes: [],
          },
        ];
  return callTurn(`fake-reflect-${insights.length}`, SLEEP_REFLECT_TOOL, { insights });
}

/** The decide/reflect payloads arrive as `JSON.stringify(payload)` in the last
 *  user message. Total: malformed JSON is a value here, never a throw, so the
 *  worker reports one clean `failed` instead of an exception out of a sweep. */
function parseJsonRecord(text: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
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
 * - `fake/sleep`         answers the sleep-time worker's three forced calls
 *   (`extract_memories` / `decide_memory_updates` / `reflect_memories`, slice
 *   6) off the FORCED TOOL rather than the conversation, so a sweep writes real
 *   memory rows and receipts with no API key. See {@link sleepTurn}.
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
    case "sleep":
      return new FakeProvider(sleepTurn);
    default:
      throw new Error(
        `Unknown fake behavior ${JSON.stringify(behavior)}. ` +
          `Supported: ${FAKE_BEHAVIORS.join(", ")}`,
      );
  }
}
