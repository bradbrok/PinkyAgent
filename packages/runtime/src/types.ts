/**
 * Runtime contracts: LLM providers, tools, and the A2A messenger.
 * These types are the cross-package boundary — change with care.
 */
import type {
  Db,
  SettingsSnapshot,
  ThreadEvent,
  ThreadRef,
  TokenUsage,
  ToolCall,
} from "@pinky/core";

export type { TokenUsage };

// ---------------------------------------------------------------------------
// LLM provider abstraction
// ---------------------------------------------------------------------------

export interface LlmMessage {
  role: "system" | "user" | "assistant" | "tool";
  /** Tool role messages carry the producing call id. */
  toolCallId?: string;
  /** Assistant messages may carry tool calls and no text. */
  toolCalls?: ToolCall[];
  text: string;
}

export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema object for the arguments. */
  parameters: Record<string, unknown>;
}

export interface CompleteOptions {
  model: string; // bare model id (provider prefix stripped)
  system: string;
  messages: LlmMessage[];
  tools: ToolSpec[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

export interface AssistantTurn {
  text: string;
  toolCalls: ToolCall[];
  /** Raw finish reason passthrough ("stop", "length", "tool_calls", ...). */
  stopReason: string;
  /**
   * Token counts for this turn. The cache counters are what make the DESIGN
   * §13 cost model answerable (a cache read is ~0.1x, a cache write ~1.25x an
   * input token), so they are part of the contract rather than a provider
   * extension; providers that do not report them simply leave them undefined.
   * The loop journals this onto the assistant `message` event.
   */
  usage?: TokenUsage;
}

export interface Provider {
  readonly name: string;
  complete(opts: CompleteOptions): Promise<AssistantTurn>;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface ToolContext {
  cwd: string;
  db: Db;
  thread: ThreadRef;
  /** Emit a thread event (tool-private observations should NOT emit; use sparingly). */
  emit: (data: ThreadEvent["data"]) => Promise<void>;
  /** Present when the runtime has A2A enabled. */
  messenger?: Messenger;
  /** This agent's stable id (for A2A addressing). */
  agentId?: string;
  /**
   * The loop's estimate of the prompt size (tokens) for the turn that issued
   * this call. Set by runAgentLoop; `shed_context` records it as the
   * `tokensBefore` of the continuity event (DESIGN.md §4).
   */
  contextTokens?: number;
  signal?: AbortSignal;
}

export interface ToolResult {
  text: string;
  isError?: boolean;
}

export interface Tool {
  readonly name: string;
  readonly description: string;
  readonly parameters: Record<string, unknown>;
  execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult>;
}

// ---------------------------------------------------------------------------
// A2A messenger (DESIGN.md §7: mailbox + wake-on-message, cross-machine)
// ---------------------------------------------------------------------------

export interface A2AEnvelope {
  id: string;
  /** agentId@nodeId */
  from: string;
  /** agentId@nodeId, or "broadcast" */
  to: string;
  kind: "message" | "request" | "response";
  text: string;
  threadHint?: string;
  sentAt: string; // ISO
}

export interface Messenger {
  readonly nodeId: string;
  /** Send a message; returns the assigned id. Cross-node delivery is fire-and-forget
   *  with durable local persistence first (at-least-once). */
  send(env: Omit<A2AEnvelope, "id" | "sentAt">): Promise<string>;
  /** Unread messages for an agent on this node; marks them read. */
  inbox(agentId: string, opts?: { limit?: number }): Promise<A2AEnvelope[]>;
  /** Subscribe to live delivery for an agent on this node. Returns unsubscribe. */
  onMessage(agentId: string, handler: (env: A2AEnvelope) => void): () => void;
  /**
   * Ingest an envelope that already carries an id/sentAt (the cross-node relay
   * side of at-least-once). MUST be idempotent on `id`: persist with the
   * ORIGINAL id, then CLAIM delivery for this node and wake subscribers only
   * if the claim succeeded. Returns true when this call was the delivery,
   * false when someone got there first (a sender retry — the caller still
   * answers 200).
   *
   * "Claim", not "did the insert write a row": two nodes may share one
   * database, where the sender has already persisted the very row being
   * delivered. Keying idempotency on row existence makes that first delivery
   * look like a replay and silently skips the wakeup.
   *
   * Required: the relay has no safe fallback. Re-minting a peer's envelope
   * through send() would give it a fresh id, so every sender retry would land
   * as a NEW message and wake the recipient again — at-least-once degraded to
   * at-least-once-per-retry.
   */
  receive(env: A2AEnvelope): Promise<boolean>;
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

export interface AgentLoopOptions {
  db: Db;
  provider: Provider;
  tools: Tool[];
  thread: ThreadRef;
  agentId: string;
  messenger?: Messenger;
  systemPrompt: string;
  cwd: string;
  maxTurns?: number;
  /** Human-owned settings snapshot (from the settings table). The loop reads
   *  model + context thresholds from here; agents cannot mutate it. */
  settings: SettingsSnapshot;
  /** Deliver assistant text to the outside world; return false to suppress. */
  deliver?: (text: string) => Promise<void>;
}

export interface AgentRunResult {
  turns: number;
  /**
   * - `completed`: the model ended its turn with text and no tool calls.
   * - `max_turns`: the turn budget ran out mid-work.
   * - `aborted`: the caller's signal fired.
   * - `shed`: the run ended immediately after a successful context restart
   *   (DESIGN.md §4.3) with no turns left to continue into. Work is not
   *   finished; it resumes from the continuity document on the next wake.
   * - `shed_failed`: context was over the hard boundary and the model did not
   *   produce a valid continuity document within the forced-shed budget, so
   *   the run was stopped rather than allowed to spin over the limit.
   */
  stopReason: "completed" | "max_turns" | "aborted" | "shed" | "shed_failed";
}
