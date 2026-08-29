/**
 * Runtime contracts: LLM providers, tools, and the A2A messenger.
 * These types are the cross-package boundary — change with care.
 */
import type {
  Db,
  MemoryStore,
  RecallScope,
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

/**
 * Provider-side tool forcing. The tool LIST stays in the request either way:
 * swapping the list invalidates every provider cache tier (tools render at
 * position 0), while `tool_choice` invalidates only the messages tier — so the
 * loop masks with this, never by sending a shorter `tools` (DESIGN.md §4.5/§9
 * "tool set masked not mutated mid-window").
 */
export type ToolChoice = { type: "auto" } | { type: "none" } | { type: "tool"; name: string };

export interface CompleteOptions {
  model: string; // bare model id (provider prefix stripped)
  system: string;
  messages: LlmMessage[];
  tools: ToolSpec[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
  /** Absent means the provider default ("auto"). */
  toolChoice?: ToolChoice;
  /**
   * Stable per-thread key for providers that ROUTE prompt caches by key
   * (OpenAI `prompt_cache_key`). Anthropic and DeepSeek key on the prefix
   * bytes alone and ignore it. Never part of the prompt text.
   */
  cacheKey?: string;
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
// Embeddings (DESIGN.md §5.5)
// ---------------------------------------------------------------------------

export interface Embedder {
  /** "provider/model-id" as configured, e.g. "openai/text-embedding-3-small". */
  readonly model: string;
  readonly dimensions: number; // 1536 for text-embedding-3-small
  embed(texts: string[], opts?: { signal?: AbortSignal }): Promise<number[][]>;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/**
 * The memory plane as the runtime sees it (DESIGN.md §5): a store, the scope
 * the caller is entitled to read/write (§5.1), and an optional embedder.
 *
 * Absent embedder => FTS-only: no vector voice on recall, no embedding written
 * on retain. That is a degraded mode, not an error — recall still works.
 */
export interface MemoryContext {
  store: MemoryStore;
  /** Absent => FTS-only (no vector voice on recall, no embedding on retain). */
  embedder?: Embedder;
  /** Who is asking and from where; the store turns this into a SQL predicate. */
  scope: RecallScope;
}

export interface ToolContext {
  cwd: string;
  db: Db;
  thread: ThreadRef;
  /** Emit a thread event (tool-private observations should NOT emit; use sparingly). */
  emit: (data: ThreadEvent["data"]) => Promise<void>;
  /** Present when the runtime has A2A enabled. */
  messenger?: Messenger;
  /** Present when the memory plane is enabled (DESIGN.md §5); the memory
   *  tools degrade to a clean error without it. */
  memory?: MemoryContext;
  /** This agent's stable id (for A2A addressing). */
  agentId?: string;
  /**
   * The settings snapshot this run started with — READ-ONLY, and a copy of
   * what the loop itself is using. Mutating it changes nothing: the loop read
   * its model and thresholds before the first turn, and the next run reloads
   * from the table.
   *
   * A tool that wants to *change* a setting writes through
   * `new SettingsStore(ctx.db).set(...)` under the `selfConfig` allow-list
   * (DESIGN.md P8, revised) — never by mutating this object, and never by
   * touching a file. Optional because a non-loop caller (a test, a one-shot
   * tool invocation) may have no snapshot to hand over.
   */
  settings?: SettingsSnapshot;
  /**
   * The loop's estimate of the prompt size (tokens) for the turn that issued
   * this call. Set by runAgentLoop; `shed_context` records it as the
   * `tokensBefore` of the continuity event (DESIGN.md §4).
   */
  contextTokens?: number;
  /**
   * The deferred-tool plane (slice 9): catalog search/describe plus dispatch.
   * Absent ⇒ the three meta-tools answer "no deferred tools on this surface".
   */
  deferred?: DeferredTools;
  signal?: AbortSignal;
}

export interface ToolResult {
  text: string;
  isError?: boolean;
}

// ---------------------------------------------------------------------------
// Deferred tools (slice 9). Tool schemas render at prefix position 0, so the
// header holds only always-on tools plus three fixed meta-tools; every other
// tool lives in the Postgres catalog and reaches the model as an ordinary tool
// result — appended, never a header rewrite (DESIGN.md §9 "masked not mutated").
// ---------------------------------------------------------------------------

export interface CatalogHit {
  name: string;
  /** Capped (~200 chars) in search results; full text via describe(). */
  description: string;
  source: "builtin" | "mcp";
  server?: string;
}

export interface CatalogEntry extends CatalogHit {
  /** JSON Schema for the tool's arguments (MCP inputSchema / Tool.parameters). */
  parameters: Record<string, unknown>;
}

export interface ToolCatalogView {
  search(query: string, limit: number): Promise<CatalogHit[]>;
  describe(name: string): Promise<CatalogEntry | null>;
}

export interface DeferredTools {
  catalog: ToolCatalogView;
  /**
   * Execute a DEFERRED tool by catalog name. Unknown name or arguments that
   * fail the schema come back as an isError result carrying the schema, so
   * the model can correct itself without a describe round-trip.
   */
  call(name: string, args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult>;
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
  /**
   * Recovery for the CONSUMPTION edge (issue #4). Re-fires the subscribers for
   * every message addressed to `agentId` on this node that carries no receipt
   * yet — `read_at is null`, regardless of delivered_at — oldest first, and
   * returns how many were fired.
   *
   * Two markers, two different claims:
   *   - `delivered_at` = the NODE accepted the message. Bookkeeping: it makes
   *     the relay idempotent and nothing more. A crash between the claim and
   *     the agent's turn leaves a row that is "delivered" forever and was
   *     never acted on — a wake lost with zero recovery.
   *   - `read_at` = an AGENT consumed it. THE RECEIPT, stamped by the consumer
   *     inside the same transaction as the work (see claimConsumption), so it
   *     exists if and only if the work was journaled.
   *   - recovery = re-fire everything unread.
   *
   * Safe to call as often as you like — at startup, on a timer, after a
   * reconnect — BECAUSE consumers claim the receipt transactionally: a second
   * fire finds the receipt already stamped and does nothing.
   */
  redeliverUnconsumed(agentId: string): Promise<number>;
  /**
   * Stamp the consumption receipt for one message, returning true only for the
   * caller that stamped it (so the winner does the work and everyone else
   * drops it).
   *
   * `tx` is the point of the method: the receipt belongs in the CONSUMER's
   * transaction, next to the events consuming it produced. Both commit or
   * neither does — a turn whose transaction rolls back leaves the message
   * unread and the next redeliverUnconsumed() fires it again. Never mark a
   * message consumed from the scheduling side.
   */
  claimConsumption(id: string, tx?: Db): Promise<boolean>;
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
  /** Enables auto-recall at context start / after each restart (DESIGN.md
   *  §5.4) and hands the memory tools their store. Absent => the loop runs
   *  exactly as it did before the memory plane existed. */
  memory?: MemoryContext;
  systemPrompt: string;
  cwd: string;
  /** Deferred-tool plane; copied into every ToolContext. Absent ⇒ none. */
  deferred?: DeferredTools;
  maxTurns?: number;
  /** Human-owned settings snapshot (from the settings table). The loop reads
   *  model + context thresholds from here; agents cannot mutate it. */
  settings: SettingsSnapshot;
  /** Deliver assistant text to the outside world; return false to suppress. */
  deliver?: (text: string) => Promise<void>;
  /**
   * Observer for every event the loop appends; errors are swallowed. Used by
   * the headless JSONL mode to stream the log live. Fire-and-forget: it must
   * not be a second write path, and nothing in the loop waits on it.
   */
  onEvent?: (event: ThreadEvent) => void;
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
