/**
 * PinkyAgent thread event model (DESIGN.md §3).
 *
 * A thread is an append-only event log keyed by (tenantId, channelId, threadId).
 * The LLM prompt is a *projection* of the log — never the source of truth.
 */

export interface ThreadRef {
  tenantId: string;
  channelId: string;
  threadId: string;
}

export interface Principal {
  platform: string; // "slack" | "cli" | "a2a" | ...
  userId: string;
  displayName?: string;
}

export type EgressTarget =
  | { kind: "thread" }
  | { kind: "broadcast" }
  | { kind: "dm"; userId: string };

export type DecisionAction = "reply" | "broadcast" | "dm" | "react" | "defer" | "silent";

export interface ContinuityDoc {
  goal: string;
  plan: { done: string[]; now: string; next: string[] };
  workingSet: { files?: string[]; artifacts?: string[]; urls?: string[] };
  decisions: { what: string; why: string }[];
  openLoops: string[];
  lessons: string[];
  memoryHints: string[];
  mood?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/**
 * Per-turn token accounting. The cache counters are optional because only
 * some providers report them (Anthropic does; OpenAI-compatible routes
 * generally do not) — and they are the interesting half of the cost model,
 * since a cache read is ~0.1x and a cache write ~1.25x an ordinary input
 * token. Mirrored by AssistantTurn["usage"] in packages/runtime.
 */
export interface TokenUsage {
  input: number;
  output: number;
  cacheRead?: number;
  cacheCreation?: number;
}

export type ThreadEventData =
  | {
      type: "ingress";
      platform: string;
      author: Principal;
      text: string;
      refs: string[];
      /** External event id for dedup (e.g. Slack event_id). */
      externalId?: string;
    }
  | {
      type: "message";
      role: "assistant";
      text: string;
      toolCalls: ToolCall[];
      model: string;
      /**
       * Token counts as reported by the provider for THIS turn, journaled so
       * $/task is derivable from the log alone (DESIGN.md §13 cost model:
       * "restarts discard cache warmth; measure $/task vs compaction
       * baseline"). Absent when the provider reported none.
       */
      usage?: TokenUsage;
    }
  | { type: "tool_result"; callId: string; name: string; text: string; isError: boolean }
  | { type: "egress"; target: EgressTarget; text: string }
  | { type: "decision"; action: DecisionAction; reason: string }
  | { type: "continuity"; document: ContinuityDoc; tokensBefore: number }
  | { type: "subagent_spawn"; agent: string; task: string; outputRef?: string }
  | {
      type: "human_request";
      question: string;
      options?: unknown;
      status: "pending" | "answered";
      answer?: string;
    }
  | { type: "error"; source: string; message: string; count: number }
  | { type: "checkpoint"; ref: string }
  | {
      type: "a2a";
      /** Full sender address agentId@nodeId. */
      from: string;
      to: string;
      kind: "message" | "request" | "response";
      text: string;
      msgId: string;
    };

export interface ThreadEvent extends ThreadRef {
  id: string;
  seq: number;
  ts: string; // ISO
  data: ThreadEventData;
}

export function threadKey(ref: ThreadRef): string {
  return `${ref.tenantId}:${ref.channelId}:${ref.threadId}`;
}
