/**
 * The sleep-time worker's shared contracts (DESIGN.md §5.3 item 3, slice 6).
 *
 * The worker turns the event log into memory-plane rows while the agent is
 * idle. Two properties shape every type here:
 *
 * 1. **THE SCHEDULER HOLDS NO STATE.** Nothing in {@link SleepDeps} is a
 *    cursor, a watermark or a "last run at". A pass reads where it left off
 *    from its own newest RECEIPT in the log and writes the next one inside the
 *    same transaction as its memory writes (DESIGN.md §7/§8.1, and CLAUDE.md
 *    invariant #6: "a timer emits an event and the consumer journals the
 *    receipt — never mark something 'fired' from the scheduler side"). So a
 *    crash anywhere before commit leaves no receipt and no rows, and the next
 *    sweep simply redoes the pass.
 *
 * 2. **IT NEVER TOUCHES LIVE CONTEXT.** Every event a pass appends is
 *    audit-only, so nothing it writes RENDERS (DESIGN.md §3, §4.5 cache
 *    alignment). The one qualification: a window already over the event cap
 *    rolls on any append, this included — `contextEvents` keeps the newest
 *    DEFAULT_CONTEXT_EVENT_CAP events, so a truncated window's start moves
 *    whoever the writer is, and the loop already treats that as hard pressure.
 *    A row retained mid-window is first *seen* after the next continuity
 *    boundary, because auto-recall runs once per WINDOW and replays its
 *    journaled block (§5.4); that lag is the design, not a defect.
 */
import type {
  Db,
  EventStore,
  MemoryStore,
  SettingsSnapshot,
  ThreadEventData,
  ThreadRef,
} from "@pinky/core";
import type { Embedder, Provider } from "@pinky/runtime";

/** The `sleep` settings block (packages/core/src/config.ts). */
export type SleepSettings = SettingsSnapshot["sleep"];

/**
 * How WIDE the surface running the worker may read and write (DESIGN.md §5.1).
 *
 * The worker inherits the surface's width rather than picking its own: the CLI
 * (`pinky sleep run`) is a trusted local operator and passes both true, while
 * `pinky headless --shared` passes both false. A narrow surface must not mint
 * `user`-visible rows — it could not read them back, and a fact about one
 * person extracted on a shared pipe is exactly the leak §5.1 exists to stop.
 */
export interface SleepScope {
  includeUser: boolean;
  /**
   * RESERVED. Both passes hardcode `includePrivate: false` today — extraction
   * never compares a channel/tenant candidate against the agent's own scratch,
   * and reflection never reads it — so nothing consults this flag yet. It is
   * kept because it records the SURFACE'S GRANT (the CLI sets it, `--shared`
   * clears it), which is the thing a future pass that does read `private` rows
   * would need and could not reconstruct after the fact.
   */
  includePrivate: boolean;
}

/** Everything a pass needs. Assembled per surface; nothing here is stateful. */
export interface SleepDeps {
  /** withTenant-wrapped, so the `memories` RLS policy has its GUC. */
  db: Db;
  events: EventStore;
  /** The tenant store. A pass calls `memory.bind(tx)` INSIDE its transaction,
   *  so its rows and its receipt commit together or not at all. */
  memory: MemoryStore;
  /** Absent => FTS-only neighbor search and rows stored without an embedding.
   *  Degraded, never fatal — mirrors runtime/memory-recall.ts. */
  embedder?: Embedder;
  provider: Provider;
  /** "provider/model-id" actually in use — journaled on every receipt, so the
   *  log answers "which model wrote this memory". */
  model: string;
  agentId: string;
  tenantId: string;
  settings: SleepSettings;
  scope: SleepScope;
  /** STDERR. Never stdout — that is the headless JSONL protocol (CLAUDE.md #5). */
  log: (msg: string) => void;
  /** Injectable clock, so a receipt's `ms` is deterministic under test. */
  now?: () => Date;
  signal?: AbortSignal;
}

/** The `sleep`/`extract` event, pulled off the core union so the two cannot drift. */
export type ExtractReceipt = Extract<ThreadEventData, { type: "sleep"; phase: "extract" }>;
/** The `sleep`/`reflect` event, likewise. */
export type ReflectReceipt = Extract<ThreadEventData, { type: "sleep"; phase: "reflect" }>;

/**
 * What one extraction pass did.
 *
 * `skipped` is the ordinary outcome, not an anomaly: most sweeps find nothing
 * new, and a pass that loses the lock to a concurrent one is *supposed* to
 * write nothing. Only `failed` is a problem, and it is the only status that
 * journals an `error` event.
 */
export type ExtractPassResult =
  | { status: "done"; receipt: ExtractReceipt }
  | { status: "skipped"; reason: "no-new-events" | "lost-claim" }
  | { status: "failed"; error: string };

export type ReflectPassResult =
  | { status: "done"; receipt: ReflectReceipt }
  /** `not-idle`: the reflect thread itself was written to less than `idleMs`
   *  ago — the same gate discovery applies to a conversation, and the ONLY
   *  backoff a failing reflect pass has (its `error` event is the newest event
   *  on that thread, so it re-arms the gate). */
  | { status: "skipped"; reason: "below-threshold" | "lost-claim" | "not-idle" }
  | { status: "failed"; error: string };

/**
 * The worker's own thread — where reflect receipts live.
 *
 * Reflection is cross-thread, so its watermark belongs to no conversation; but
 * the event log is the state (P1), so the worker journals where everything
 * else does rather than inventing a second store. The `sleep:` channel prefix
 * is also what discovery excludes, so the worker can never extract from its
 * own bookkeeping.
 */
export function reflectThread(tenantId: string, agentId: string): ThreadRef {
  return { tenantId, channelId: `sleep:${agentId}`, threadId: "reflect" };
}

/**
 * Event types that count as extractable material.
 *
 * Everything else in the log is audit-only for the worker too — `decision`,
 * `egress`, `restart`, `config`, `memory`, and the worker's own `sleep`
 * receipts.
 *
 * `error` is DELIBERATELY NOT HERE, and it is the one exclusion with a bug
 * behind it rather than a taste: a FAILED pass journals an `error` event
 * (source "sleep") on the very thread it just failed on. If that event were
 * extractable the failure would feed itself — the thread stays due, the cursor
 * never advances past it, the transcript grows by one error line per sweep, and
 * every sweep re-pays two LLM calls to fail again. DESIGN.md §4.4's "negative
 * evidence" still reaches the worker, through `tool_result`s carrying
 * `isError` and through the `lessons` of a continuity document, both of which
 * are rendered.
 */
export const EXTRACT_EVENT_TYPES = [
  "ingress",
  "a2a",
  "message",
  "tool_result",
  "continuity",
] as const;

export type ExtractEventType = (typeof EXTRACT_EVENT_TYPES)[number];

/** Is this event material the worker extracts from? */
export function isExtractable(data: ThreadEventData): boolean {
  return (EXTRACT_EVENT_TYPES as readonly string[]).includes(data.type);
}
