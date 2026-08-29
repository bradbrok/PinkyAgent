/**
 * Agent loop: projection -> provider turns -> tool execution -> event log.
 * See DESIGN.md §3 (projection), §4.1 (context-pressure ladder), §4.3 (restart
 * cycle) and §4.5 (cut-point safety / cache alignment).
 */
import { EventStore, buildContext, estimateTokens } from "@pinky/core";
import type { ThreadEventData } from "@pinky/core";
import { SHED_CONTEXT_TOOL_NAME } from "./continuity";
import type {
  AgentLoopOptions,
  AgentRunResult,
  AssistantTurn,
  LlmMessage,
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
 * Harness notices ride in a `user` message, never `role: "system"`: the
 * Anthropic provider hoists mid-conversation system messages into the
 * top-level system param, which churns the cached prefix (DESIGN.md §4.5/§9).
 * The prefix marks them as harness-authored, not human-authored.
 */
const NOTICE = "[harness notice]";

const ADVISORY_NOTE =
  `${NOTICE} context pressure: this window is filling up. Evacuate what matters to memory and call shed_context at your next natural boundary — a phase finished, a checkpoint reached, a sub-problem about to change.`;

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

  const emit = (data: ThreadEventData): Promise<void> =>
    eventStore.append(opts.thread, data).then(() => undefined);

  /**
   * Prompt = projection of the log from the latest continuity boundary
   * (DESIGN.md §3). Never a fixed forward page: that would pin a long thread
   * to its first N events forever.
   */
  const loadContext = async (): Promise<LlmMessage[]> => {
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
    return buildContext(window.events);
  };

  let messages: LlmMessage[] = await loadContext();

  let turns = 0;
  /** Advisory fires once per crossing; a shed re-arms it. */
  let advisoryArmed = true;
  let forcedShedFailures = 0;

  while (turns < maxTurns) {
    if (opts.signal?.aborted) return { turns, stopReason: "aborted" };

    // Context-pressure ladder (DESIGN.md §4.1). The system prompt is the cache
    // prefix and is never rewritten; pressure notices are conversation turns.
    const system = opts.systemPrompt;
    let specs = allSpecs;
    let forcingShed = false;
    const tokens = estimateTokens([{ role: "system", text: system }, ...messages]);
    if (tokens >= hardFraction * approxWindowTokens && shedTool) {
      forcingShed = true;
      specs = [
        { name: shedTool.name, description: shedTool.description, parameters: shedTool.parameters },
      ];
      messages.push({
        role: "user",
        text: forcedShedFailures === 0 ? HARD_NOTE : HARD_RETRY_NOTE,
      });
      advisoryArmed = false; // the hard notice supersedes the advisory one
    } else if (tokens >= advisoryFraction * approxWindowTokens && advisoryArmed) {
      messages.push({ role: "user", text: ADVISORY_NOTE });
      advisoryArmed = false;
    }

    let turn: AssistantTurn;
    try {
      turn = await opts.provider.complete({
        model: bareModel,
        system,
        messages,
        tools: specs,
        ...(opts.signal ? { signal: opts.signal } : {}),
      });
    } catch (err) {
      if (isAbort(err)) return { turns, stopReason: "aborted" };
      throw err;
    }
    turns += 1;

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
        // The forced turn offered only shed_context; honor that literally.
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
          ...(opts.messenger ? { messenger: opts.messenger } : {}),
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
      // buildContext.
      messages = await loadContext();
      advisoryArmed = true;
      forcedShedFailures = 0;
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
