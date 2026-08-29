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
  /** `tools`: deferred tool names in use — the successor `tool_describe`s them before acting (slice 9). */
  workingSet: { files?: string[]; artifacts?: string[]; urls?: string[]; tools?: string[] };
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
 * Sort object keys so a value renders the same bytes before and after a round
 * trip through the event log.
 *
 * Postgres `jsonb` does not preserve key order: it stores an object's keys
 * sorted by (length, then bytes). So `{zulu:1, a:2, mm:3}` goes in as the model
 * wrote it and comes back as `{a, mm, zulu}` — and a `tool_use` block whose
 * argument JSON differs by one byte from what the provider cached breaks the
 * prefix match from that block to the end of the transcript, on EVERY wake
 * (DESIGN.md §4.5 cache alignment). The fix is to stop caring what the store
 * does: canonicalize at both ends, so the in-run message, the journaled event,
 * and the projection of that event all render identical bytes.
 *
 * Recursive; arrays keep their order (order is data there), primitives are
 * returned as they are. UTF-16 code-unit order, i.e. plain `<` on strings — the
 * order does not have to match Postgres's, only be the SAME on both sides.
 */
export function canonicalizeArgs<T>(value: T): T {
  return canonicalize(value) as T;
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value === null || typeof value !== "object") return value;
  const source = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(source).sort()) out[key] = canonicalize(source[key]);
  return out;
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
      /**
       * A harness-authored turn: the context-pressure notices today
       * (DESIGN.md §4.1 ladder), any future harness aside tomorrow.
       *
       * MODEL-VISIBLE — the projection renders it as a `user` message in seq
       * order, never `role: "system"`: a mid-conversation system message is
       * hoisted into the cached system prefix by the Anthropic route and would
       * churn it every turn (§4.5/§9).
       *
       * It is journaled because the prompt is a projection of the log (§3). A
       * notice that lived only in the run's in-memory array would vanish from
       * the successor wake's prompt, and the conversation would diverge from
       * what the provider already cached at exactly that point. Journal the
       * notice immediately BEFORE pushing it, so it lands ahead of the
       * assistant `message` it provoked and re-renders in the same slot. No
       * dedupe: two forced attempts journal two notices, in order, which is
       * what the model saw.
       */
      type: "notice";
      text: string;
    }
  | {
      /**
       * A memory-plane write or read (DESIGN.md §5.3).
       *
       * Audit-only, with exactly one exception: the `block` on the recall that
       * OPENS a window. Everything else here — every retain, update,
       * invalidate, and every later recall in the same window — is never
       * rendered by the projection and costs no context.
       */
      type: "memory";
      op: "recall" | "retain" | "update" | "invalidate";
      /** Ids touched (retain/update/invalidate: the row(s); recall: the hits, in rank order). */
      ids: string[];
      /** recall: the query text; retain/update: the stored text; invalidate: the reason. */
      text: string;
      /** recall: number of candidates before the token budget cut. */
      count?: number;
      /**
       * AUTO-RECALL ONLY: the rendered `<memories>` text EXACTLY as it was
       * injected at index 0 of the prompt — `""` when the recall ran and
       * injected nothing (no hits, or nothing fit the budget).
       *
       * The KEY IS THE RECEIPT: `block` present ⇔ this event is the loop's
       * window-opening auto-recall pass. Present-and-empty is a real state and
       * must stay distinguishable from absent, because the loop's "have I
       * already recalled in this window?" gate reads the key, not the text. A
       * recall that journaled no key at all (the agent-initiated `recall` tool,
       * packages/tools/src/memory.ts) is invisible to that gate and can never
       * move byte 0 of a cached prompt.
       *
       * This is what makes the prompt a pure projection again (DESIGN.md §3).
       * Auto-recall is a live search: a new retention, a different query seed
       * or a different score order would otherwise re-render byte 0 of the
       * conversation on the next wake and miss the provider's prefix cache for
       * the whole thread (§4.5 cache alignment). Journaling the text means the
       * window keeps the block it opened with; the projection replays it and
       * the loop skips the search (DESIGN.md §5.4: once per WINDOW, not once
       * per wake). Only the FIRST recall-with-`block` in a window opens it; the
       * projection hoists that event's text when it is non-empty.
       */
      block?: string;
      /**
       * AUTO-RECALL ONLY (rides with `block`): how WIDE the scope was that
       * produced it (DESIGN.md §5.1). Only the two width flags matter —
       * channelId/userId/agentId are fixed by the thread and the agent, so they
       * cannot differ between two wakes on one thread.
       *
       * A journaled block is replayed into later wakes, and a later wake may be
       * driven by a NARROWER surface (`pinky headless --shared` after a default
       * run): replaying a block built with `includeUser`/`includePrivate` there
       * would put private rows into a shared context, which §5.1 forbids. The
       * loop compares this against the scope it is running under and re-recalls
       * when the current one is narrower. Absent on events written before this
       * field existed — treated as "unknown, replay as-is".
       */
      scope?: { includeUser: boolean; includePrivate: boolean };
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
