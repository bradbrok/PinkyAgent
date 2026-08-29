/**
 * Agent loop: projection -> provider turns -> tool execution -> event log.
 * See DESIGN.md §3 (projection), §4.1 (context-pressure ladder), §4.3 (restart
 * cycle) and §4.5 (cut-point safety / cache alignment).
 */
import { EventStore, buildContext, canonicalizeArgs, estimateTokens, windowRecall } from "@pinky/core";
import type { RecallScope, ThreadEvent, ThreadEventData, WindowRecall } from "@pinky/core";
import { SHED_CONTEXT_TOOL_NAME } from "./continuity";
import { autoRecall, recallQueryFor } from "./memory-recall";
import type {
  AgentLoopOptions,
  AgentRunResult,
  AssistantTurn,
  LlmMessage,
  ToolChoice,
  ToolContext,
  ToolSpec,
} from "./types";
/** Loop-time options beyond the shared AgentLoopOptions contract. */
export interface RunAgentLoopOptions extends AgentLoopOptions {
  signal?: AbortSignal | undefined;
}

const DEFAULT_MAX_TURNS = 16;

/**
 * How many turns the hard boundary may burn demanding a continuity document
 * before the run is stopped (one attempt plus one retry that has seen the
 * validation error). Without a bound a model that refuses to shed would spin
 * over the context limit until maxTurns.
 */
const MAX_FORCED_SHED_TURNS = 2;

/**
 * How the forced turn is masked at the hard boundary — on the RETRY only.
 *
 * Sending `tools: [shed_context]` is never an option: tools render at position
 * 0 of the cached prefix, so narrowing the list invalidates every cache tier
 * (tools, system, and the whole conversation) at the most expensive window a
 * thread ever has. `tool_choice` is cheaper but not free: it invalidates the
 * MESSAGES tier, i.e. one uncached re-read of the entire transcript, at exactly
 * its largest. Appending the notice does NOT do that — an append extends the
 * prefix and stays cache-friendly — so the first forced attempt sends no
 * `toolChoice` at all: the HARD_NOTE plus the harness guard below (which
 * refuses every non-shed call) is what holds the boundary, warm. Only when that
 * attempt failed is the guarantee worth a full re-read, and the retry pays it.
 * DESIGN.md §9 "tool set masked not mutated mid-window" / §4.5 cache alignment.
 */
const FORCE_SHED_CHOICE: ToolChoice = { type: "tool", name: SHED_CONTEXT_TOOL_NAME };

/**
 * One rebuilt context window: what the model is about to see, plus the parts
 * that went into it. The loop carries the pieces rather than recomputing them
 * so the `restart` event reports the numbers actually used (DESIGN.md §13).
 */
interface LoadedContext {
  messages: LlmMessage[];
  /** The raw window, boundary event included — it seeds the recall query. */
  events: ThreadEvent[];
  /** Seq of the continuity event at the boundary; 0 when the thread has none. */
  boundarySeq: number;
  /**
   * The `<memories>` block this window carries: the one auto-recall just
   * produced, or the one already journaled in the window and replayed at index
   * 0 by the projection. Either way it is IN `messages` exactly once, so
   * `emitRestart` can bill it without double counting.
   */
  block: string | null;
  /**
   * True when the event-cap dropped the OLDEST events of this window. Not a
   * cosmetic flag: it is hard context pressure (see the ladder below).
   */
  truncated: boolean;
}

/** Has this window's boundary already been billed by a `restart` event? */
function hasRestart(loaded: LoadedContext): boolean {
  return loaded.events.some(
    (e) => e.data.type === "restart" && e.data.boundarySeq === loaded.boundarySeq,
  );
}

/**
 * The shedding turn's own estimate, read off the continuity event at this
 * window's boundary — the `tokensBefore` a `restart` event mirrors.
 *
 * Null means "not a window that opens on a restart": either the thread has
 * never shed, or the safety cap dropped the boundary event itself out of the
 * window, in which case this wake is not the successor of anything and must
 * not be billed as one.
 */
function boundaryTokensBefore(loaded: LoadedContext): number | null {
  if (loaded.boundarySeq <= 0) return null;
  const boundary = loaded.events.find((e) => e.seq === loaded.boundarySeq);
  return boundary?.data.type === "continuity" ? boundary.data.tokensBefore : null;
}

/**
 * Harness notices ride in a `user` message, never `role: "system"`: the
 * Anthropic provider hoists mid-conversation system messages into the
 * top-level system param, which churns the cached prefix (DESIGN.md §4.5/§9).
 * The prefix marks them as harness-authored, not human-authored.
 */
const NOTICE = "[harness notice]";

const ADVISORY_NOTE =
  `${NOTICE} context pressure: this window is filling up. Evacuate what matters to memory and call shed_context at your next natural boundary — a phase finished, a checkpoint reached, a sub-problem about to change.`;

/**
 * The hard rung's notice. "The only tool available" stays true from the model's
 * side on every forced turn: the harness guard further down turns any other
 * call into an error result, and the retry adds `tool_choice` on top. The tool
 * DEFINITIONS are left untouched throughout, precisely so the cached prefix
 * survives (FORCE_SHED_CHOICE). It also covers the truncated-window rung, where
 * the limit that was reached is the event cap rather than the token fraction.
 *
 * The wording is deliberately stable: it is prompt text, and editing it moves
 * bytes in every window that has ever carried it.
 */
const HARD_NOTE =
  `${NOTICE} context limit reached: call shed_context now to write your continuity document. It is the only tool available this turn.`;

const HARD_RETRY_NOTE =
  `${NOTICE} context limit reached and your last turn did not write a valid continuity document. This is the final attempt: call shed_context with a complete document (goal and plan.now are required) or the run stops here.`;

function isAbort(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

function toolSpecs(tools: AgentLoopOptions["tools"]): ToolSpec[] {
  return tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters }));
}

function turnToMessage(turn: AssistantTurn): LlmMessage {
  const msg: LlmMessage = { role: "assistant", text: turn.text };
  if (turn.toolCalls.length > 0) msg.toolCalls = turn.toolCalls;
  return msg;
}

/**
 * Freeze the key order of a turn's tool arguments before anything renders or
 * stores them (core's `canonicalizeArgs`).
 *
 * `data` is jsonb, and jsonb sorts an object's keys by (length, bytes). So the
 * in-run message renders the model's own order, the projection on the next wake
 * renders Postgres's, and the two diverge at the first tool call whose argument
 * names differ in length — a cold prefix from that block to the end of the
 * transcript, on EVERY wake (DESIGN.md §4.5). Canonicalizing here makes the
 * journaled event, the in-run message, and the projection of that event all
 * render the same bytes, whatever the store does with key order. Applied ONCE,
 * to the turn, so the message, the event and the executed calls cannot drift
 * apart; `buildContext` does the same on the way out.
 */
function canonicalizeTurn(turn: AssistantTurn): AssistantTurn {
  if (turn.toolCalls.length === 0) return turn;
  return {
    ...turn,
    toolCalls: turn.toolCalls.map((c) => ({ ...c, args: canonicalizeArgs(c.args) })),
  };
}

/**
 * Is the scope this run is recalling under NARROWER than the one a journaled
 * block was built with (DESIGN.md §5.1)? Only the two width flags can differ
 * between wakes on one thread — channelId/userId/agentId are fixed by the
 * thread and the agent.
 *
 * An undefined journaled scope (an event written before the field existed) is
 * "unknown", and unknown is not narrower: replay it rather than re-recalling
 * every wake forever on an old thread.
 */
/** Has this window already been given the advisory notice (in ANY earlier wake)? */
function windowHasAdvisory(events: ThreadEvent[]): boolean {
  return events.some((e) => e.data.type === "notice" && e.data.text === ADVISORY_NOTE);
}

function scopeNarrowed(journaled: WindowRecall["scope"], current: RecallScope): boolean {
  if (!journaled) return false;
  return (
    (journaled.includeUser && !current.includeUser) ||
    (journaled.includePrivate && !current.includePrivate)
  );
}

export async function runAgentLoop(opts: RunAgentLoopOptions): Promise<AgentRunResult> {
  const eventStore = new EventStore(opts.db);
  const maxTurns = opts.maxTurns ?? DEFAULT_MAX_TURNS;
  const { advisoryFraction, hardFraction, approxWindowTokens } = opts.settings.context;
  // Model id for the provider: strip the routing prefix ("anthropic/x" -> "x").
  const model = opts.settings.model;
  const slash = model.indexOf("/");
  const bareModel = slash >= 0 ? model.slice(slash + 1) : model;
  const allSpecs = toolSpecs(opts.tools);
  const shedTool = opts.tools.find((t) => t.name === SHED_CONTEXT_TOOL_NAME);
  /**
   * Cache routing key: the thread, and nothing that varies per turn. Providers
   * that shard their prompt cache by key (OpenAI `prompt_cache_key`) need every
   * turn of a thread to land on the same shard, so this is derived from the
   * thread identity alone — a per-turn value would scatter the prefix across
   * shards and cost exactly the hit rate it exists to buy. Others ignore it.
   */
  const cacheKey = `${opts.thread.tenantId}/${opts.thread.channelId}/${opts.thread.threadId}`;

  /**
   * The one append path. `opts.onEvent` observes what landed (the headless
   * JSONL mode streams it to stdout); it is fire-and-forget by contract, so a
   * throwing observer can never take down a run that is already journaled.
   */
  const emit = async (data: ThreadEventData): Promise<void> => {
    const event = await eventStore.append(opts.thread, data);
    if (!opts.onEvent) return;
    try {
      opts.onEvent(event);
    } catch {
      // Observer errors are the observer's problem.
    }
  };

  /**
   * Prompt = projection of the log from the latest continuity boundary
   * (DESIGN.md §3). Never a fixed forward page: that would pin a long thread
   * to its first N events forever.
   */
  const loadContext = async (): Promise<Omit<LoadedContext, "block">> => {
    const window = await eventStore.contextEvents(opts.thread);
    if (window.truncated) {
      // The cap keeps the NEWEST events; say so in the log rather than
      // silently serving a partial window.
      await emit({
        type: "error",
        source: "context",
        message: `context window capped: only the newest ${window.events.length} events since the continuity boundary (seq ${window.boundarySeq}) were loaded`,
        count: 1,
      });
    }
    return {
      messages: buildContext(window.events),
      events: window.events,
      boundarySeq: window.boundarySeq,
      truncated: window.truncated,
    };
  };

  /**
   * The projection plus the budgeted `<memories>` block (DESIGN.md §5.4:
   * "token-capped <memories> block at context start and after each restart").
   *
   * The block goes in at index 0 as a `user` message — it IS the context
   * start — and the system prompt is left alone, because that string is the
   * cached prefix (§4.5/§9). Recall failures are non-fatal by construction
   * (autoRecall journals them and returns null), and with no memory context
   * or `autoRecall` off this is exactly the old loadContext().
   *
   * Once per WINDOW, not once per wake. The gate is "has an auto-recall pass
   * already run in this window", read off the journaled event's `block` KEY —
   * NOT "is there a block to replay". A pass that found nothing journals
   * `block: ""`, which still closes the gate: otherwise the first wake on an
   * empty memory plane would leave it open, one `retain` would land, and the
   * next wake would unshift a block at index 0 — moving byte 0 and missing the
   * provider's prefix cache for the ENTIRE transcript (§4.5 cache alignment).
   * A window that already ran gets its block back FROM THE PROJECTION
   * (buildContext hoisted it to index 0 already), so `block` here is the
   * injected text either way and `emitRestart` bills the same `recallTokens`
   * without double counting.
   *
   * Two things still override a journaled block, both deliberate prefix breaks:
   *
   * 1. Memory turned OFF for this run (`opts.memory` gone, or
   *    `memory.autoRecall=false`). Replaying the block regardless would mean
   *    the setting no longer stops memories reaching the model on an existing
   *    window. The operator asked for no memories; the cost of honoring that is
   *    theirs to pay, once.
   * 2. A NARROWER scope than the block was built under (§5.1) — a `--shared`
   *    run picking up a window a default run opened with `includeUser` /
   *    `includePrivate`. Replaying it would put private rows into a shared
   *    context, so it is stripped and re-recalled under the current scope,
   *    which journals a second recall event. The projection keeps hoisting the
   *    FIRST, so a thread driven by two differently-scoped surfaces pays one
   *    prefix break per narrow wake. Privacy wins; the trade only exists for a
   *    thread someone runs both ways. The reverse (current scope WIDER) replays
   *    as-is: a wide reader seeing a narrow block leaks nothing.
   */
  const loadContextWithMemories = async (): Promise<LoadedContext> => {
    const loaded = await loadContext();
    const { messages, events } = loaded;
    const memory = opts.memory;
    // Optional-chained on purpose: a snapshot predating the memory settings
    // (older fixtures, a stale row) must degrade to "no recall", not throw.
    const cfg = opts.settings.memory;
    const active = memory && cfg?.autoRecall ? { memory, cfg } : null;
    const journaled = windowRecall(events);

    if (journaled) {
      const narrowed = active !== null && scopeNarrowed(journaled.scope, active.memory.scope);
      if (active && !narrowed) return { ...loaded, block: journaled.block || null };
      // Un-hoist what buildContext replayed at index 0 (nothing to do when the
      // journaled pass injected nothing), then fall through: disabled stops
      // here, narrowed re-recalls under the scope this run actually has.
      if (journaled.block !== "" && messages[0]?.text === journaled.block) messages.shift();
    }
    if (!active) return { ...loaded, block: null };

    const result = await autoRecall({
      memory: active.memory,
      query: recallQueryFor(messages, events),
      limit: active.cfg.recallLimit,
      tokenBudget: active.cfg.recallTokenBudget,
      emit,
      ...(opts.signal ? { signal: opts.signal } : {}),
    });
    // null = the store failed, so nothing was journaled and nothing claims the
    // window: the next wake retries. That is the one place where eventual
    // recall is worth more than a stable prefix.
    const block = result?.block ? result.block : null;
    if (block) messages.unshift({ role: "user", text: block });
    return { ...loaded, block };
  };

  /**
   * Journal what one restart cost (DESIGN.md §13: "restarts discard cache
   * warmth; measure $/task vs a compaction baseline early"). Audit-only, so
   * measuring the restart costs the model nothing.
   *
   * `tokensAfter` is core's estimate over exactly what the next provider call
   * receives — system prompt included, because the cached prefix is re-paid as
   * a cache WRITE on the first turn of a fresh window, and that write is the
   * cost this instrument exists to find. `recallTokens` is the injected
   * `<memories>` block measured the same way the recall budget measures it, so
   * the two numbers are comparable.
   */
  const emitRestart = async (loaded: LoadedContext, tokensBefore: number): Promise<void> => {
    await emit({
      type: "restart",
      boundarySeq: loaded.boundarySeq,
      tokensBefore,
      tokensAfter: estimateTokens([
        { role: "system", text: opts.systemPrompt },
        ...loaded.messages,
      ]),
      recallTokens: loaded.block ? estimateTokens([{ role: "user", text: loaded.block }]) : 0,
      messages: loaded.messages.length,
    });
  };

  const initial = await loadContextWithMemories();
  let messages: LlmMessage[] = initial.messages;

  // A run that OPENS on a continuity boundary is the successor wake a restart
  // paid for: the shedding run stopped with `shed` before it could take a turn
  // on the fresh window, or died between the two. Billed once per boundary —
  // the `restart` event lands inside the very window it describes, so every
  // later wake on the same boundary finds it and stays quiet.
  const openedAt = boundaryTokensBefore(initial);
  if (openedAt !== null && !hasRestart(initial)) await emitRestart(initial, openedAt);

  let turns = 0;
  /**
   * The advisory notice fires once per WINDOW, not once per wake — so it is
   * armed from the LOG, not from `true`.
   *
   * Notices are journaled and replayed (that is what keeps the transcript a
   * projection), so a per-run flag meant every wake above `advisoryFraction`
   * appended another identical copy: three wakes, three notices, the window
   * walking toward `hardFraction` on harness text alone. `initial.events` is
   * the window (contextEvents starts at the boundary), so "has this window
   * already been told?" is answerable exactly. A shed re-arms it below —
   * the fresh window has not been told anything.
   */
  let advisoryArmed = !windowHasAdvisory(initial.events);
  let forcedShedFailures = 0;
  /**
   * Did the window this run is working on load truncated? Pressure state like
   * `advisoryArmed`: a successful shed re-reads it from the fresh window.
   */
  let windowTruncated = initial.truncated;

  while (turns < maxTurns) {
    if (opts.signal?.aborted) return { turns, stopReason: "aborted" };

    // Context-pressure ladder (DESIGN.md §4.1). The system prompt is the cache
    // prefix and is never rewritten; pressure notices are conversation turns.
    const system = opts.systemPrompt;
    let forcingShed = false;
    const tokens = estimateTokens([{ role: "system", text: system }, ...messages]);
    /**
     * Hard rung: over the token fraction, OR working on a window the event cap
     * truncated. Truncation is real pressure, not a footnote — the cap keeps
     * the NEWEST events, so a truncated window's START rolls forward with
     * every event appended, and a prefix whose first bytes move each turn can
     * never hit a cache again (every turn is a full write). A shed replaces
     * the rolling start with a continuity boundary: a stable prefix, and a
     * window the cap no longer touches.
     */
    const overHard = tokens >= hardFraction * approxWindowTokens || windowTruncated;
    /** Second forced attempt: the one that pays for `tool_choice`. */
    let forcedRetry = false;
    if (overHard && shedTool) {
      // The tool list is NEVER narrowed here (that would invalidate every cache
      // tier). The first attempt rides on the notice plus the runtime guard
      // further down, which refuses non-shed calls; the retry adds
      // `FORCE_SHED_CHOICE` on top, at the price of one uncached re-read of the
      // transcript (see that const).
      //
      // These two notices are per forced TURN, not per window, so they are
      // bounded by MAX_FORCED_SHED_TURNS within a run — but a thread whose
      // model refuses to shed accumulates two more of them per wake, since each
      // wake re-enters the hard rung and the previous wake's notices are
      // replayed from the log. That is a stuck thread, and it stays stuck until
      // an operator raises the window, lowers the fractions, or intervenes.
      forcingShed = true;
      forcedRetry = forcedShedFailures > 0;
      await pushNotice(forcedRetry ? HARD_RETRY_NOTE : HARD_NOTE);
      advisoryArmed = false; // the hard notice supersedes the advisory one
    } else if (tokens >= advisoryFraction * approxWindowTokens && advisoryArmed) {
      await pushNotice(ADVISORY_NOTE);
      advisoryArmed = false;
    }

    let turn: AssistantTurn;
    try {
      turn = await opts.provider.complete({
        model: bareModel,
        system,
        messages,
        tools: allSpecs,
        cacheKey,
        ...(forcedRetry ? { toolChoice: FORCE_SHED_CHOICE } : {}),
        ...(opts.signal ? { signal: opts.signal } : {}),
      });
    } catch (err) {
      if (isAbort(err)) return { turns, stopReason: "aborted" };
      throw err;
    }
    turns += 1;
    // Key order frozen before anything renders or stores these args (jsonb
    // reorders them on the way back out) — see canonicalizeTurn.
    turn = canonicalizeTurn(turn);

    // Journal the assistant turn (activity boundary). Token usage rides along
    // when the provider reported it, so $/task is derivable from the log alone
    // (DESIGN.md §13: measure the cost of restarts vs a compaction baseline).
    await emit({
      type: "message",
      role: "assistant",
      text: turn.text,
      toolCalls: turn.toolCalls,
      model,
      ...(turn.usage ? { usage: turn.usage } : {}),
    });
    messages.push(turnToMessage(turn));

    if (turn.toolCalls.length === 0) {
      if (turn.text) {
        if (opts.deliver) await opts.deliver(turn.text);
        await emit({ type: "egress", target: { kind: "thread" }, text: turn.text });
      }
      if (forcingShed) {
        // Talking instead of shedding on a forced turn is a failed attempt.
        const stop = await noteForcedShedFailure();
        if (stop) return { turns, stopReason: "shed_failed" };
        continue;
      }
      // Provider hitting its output limit still terminates the run cleanly.
      return { turns, stopReason: "completed" };
    }

    // Cut-point safety (DESIGN.md §4.5): a successful shed moves the
    // projection boundary, so it runs LAST in the turn. Anything executed
    // after it would journal its tool_result past the boundary while its
    // assistant tool call sits before it — an orphan the projection must drop.
    // Array#sort is stable, so the other calls keep their order.
    const ordered = [...turn.toolCalls].sort(
      (a, b) =>
        Number(a.name === SHED_CONTEXT_TOOL_NAME) - Number(b.name === SHED_CONTEXT_TOOL_NAME),
    );

    let shed = false;
    for (const call of ordered) {
      const tool = opts.tools.find((t) => t.name === call.name);
      let resultText: string;
      let isError = false;
      if (forcingShed && call.name !== SHED_CONTEXT_TOOL_NAME) {
        // The forced turn asked for shed_context by name. The mask is
        // provider-side (`tool_choice`) and every tool is still DEFINED in the
        // request, so a provider that ignores or degrades the choice can hand
        // back some other call — this is the harness half of the guard, and it
        // is what actually holds the boundary.
        resultText = `context limit reached: only ${SHED_CONTEXT_TOOL_NAME} may be called right now`;
        isError = true;
      } else if (!tool) {
        resultText = `unknown tool: ${call.name}`;
        isError = true;
      } else {
        const ctx: ToolContext = {
          cwd: opts.cwd,
          db: opts.db,
          thread: opts.thread,
          emit,
          agentId: opts.agentId,
          contextTokens: tokens,
          // Read-only view of the run's config. `settings_set` writes through
          // the store, never through this object (DESIGN.md P8, revised).
          settings: opts.settings,
          ...(opts.messenger ? { messenger: opts.messenger } : {}),
          ...(opts.memory ? { memory: opts.memory } : {}),
          ...(opts.signal ? { signal: opts.signal } : {}),
        };
        try {
          const result = await tool.execute(call.args, ctx);
          resultText = result.text;
          isError = result.isError ?? false;
        } catch (err) {
          if (isAbort(err)) return { turns, stopReason: "aborted" };
          resultText = `tool threw: ${err instanceof Error ? err.message : String(err)}`;
          isError = true;
        }
      }
      await emit({
        type: "tool_result",
        callId: call.id,
        name: call.name,
        text: resultText,
        isError,
      });
      messages.push({ role: "tool", toolCallId: call.id, text: resultText });

      if (call.name === SHED_CONTEXT_TOOL_NAME && !isError) shed = true;
    }

    if (shed) {
      // Restart cycle (DESIGN.md §4.3): the continuity event is the new
      // boundary, so rebuild from a fresh projection and keep working in the
      // same run. The orphan tool_result for this very call is dropped by
      // buildContext, and the fresh window gets its own recall pass (§5.4)
      // seeded by the document's memoryHints.
      const rebuilt = await loadContextWithMemories();
      messages = rebuilt.messages;
      // Bill the restart before any turn spends the fresh window (§13); the
      // shedding turn's own estimate is what the continuity event recorded.
      await emitRestart(rebuilt, tokens);
      advisoryArmed = true;
      forcedShedFailures = 0;
      // Re-read the truncation rung from the window we are actually on now: a
      // fresh boundary is what un-truncates a thread, and pretending otherwise
      // would force a shed on every subsequent turn.
      windowTruncated = rebuilt.truncated;
      // No turns left to resume into: report the restart, not a turn overrun.
      if (turns >= maxTurns) return { turns, stopReason: "shed" };
      continue;
    }

    if (forcingShed) {
      const stop = await noteForcedShedFailure();
      if (stop) return { turns, stopReason: "shed_failed" };
    }
  }

  return { turns, stopReason: "max_turns" };

  /**
   * Add a harness notice to the conversation — journaled FIRST, then pushed.
   *
   * Order is the whole point: the `notice` event lands ahead of the `message`
   * event for the turn it provoked, so the successor wake's projection puts it
   * back in exactly the same slot and the transcript the provider cached still
   * matches (DESIGN.md §3 prompt = projection, §4.5 cache alignment). Two
   * forced attempts journal two notices; the projection renders both, in seq
   * order, because that is what the model saw.
   */
  async function pushNotice(text: string): Promise<void> {
    await emit({ type: "notice", text });
    messages.push({ role: "user", text });
  }

  /** Count a forced-shed miss; returns true when the run must stop. */
  async function noteForcedShedFailure(): Promise<boolean> {
    forcedShedFailures += 1;
    if (forcedShedFailures < MAX_FORCED_SHED_TURNS) return false;
    await emit({
      type: "error",
      source: "continuity",
      message: `no valid continuity document after ${forcedShedFailures} forced attempts at the hard context boundary; run stopped`,
      count: forcedShedFailures,
    });
    return true;
  }
}
