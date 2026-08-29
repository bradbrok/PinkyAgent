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
  | {
      /**
       * What one context restart cost, journaled at the moment the loop
       * rebuilt its window from a continuity boundary (DESIGN.md §13 cost
       * model: "restarts discard cache warmth; measure $/task vs a compaction
       * baseline early"). Audit-only — the projection never renders it, so
       * measuring a restart does not itself cost context.
       *
       * One of these per boundary: the loop writes it right after the rebuild
       * that follows a successful shed, and a successor wake that finds a
       * boundary with no `restart` recorded yet backfills one. Together with
       * the `usage` on the next `message` event (the successor's first turn,
       * which is where the cache-write bill lands) this is the whole §13 eval
       * query — see `pinky stats restarts`.
       */
      type: "restart";
      /** seq of the continuity event this window was rebuilt from. */
      boundarySeq: number;
      /** The loop's estimate at the turn that shed (mirrors continuity.tokensBefore). */
      tokensBefore: number;
      /** Estimate of the fresh window right after rebuild: system prompt + projection + <memories> block. */
      tokensAfter: number;
      /** Tokens of the injected <memories> block (0 when none). */
      recallTokens: number;
      /** Number of projected messages in the fresh window. */
      messages: number;
    }
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
    }
  | {
      type: "memory";
      /** Audit-only: the projection never renders it (DESIGN.md §5.3). */
      op: "recall" | "retain" | "update" | "invalidate";
      /** Ids touched (retain/update/invalidate: the row(s); recall: the hits, in rank order). */
      ids: string[];
      /** recall: the query text; retain/update: the stored text; invalidate: the reason. */
      text: string;
      /** recall: number of candidates before the token budget cut. */
      count?: number;
    }
  | {
      /**
       * A settings write the agent made through the `settings_set` tool
       * (DESIGN.md P8, revised: human-granted self-configuration). Audit-only:
       * the projection never renders it, so a self-config write costs no
       * context — but the log always answers "who changed this, from what, to
       * what, and when". Human writes go through the CLI and are not
       * journaled here.
       */
      type: "config";
      /** Settings scope written: "agent:<id>" or "channel:<id>". */
      scope: string;
      /** Dotted settings key, e.g. "context.advisoryFraction". */
      key: string;
      /** The value now stored (as passed to SettingsStore.set). */
      value: unknown;
      /** What the run's snapshot had at that key; undefined if unset. */
      previous: unknown;
      /** agentId that made the change. */
      by: string;
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
