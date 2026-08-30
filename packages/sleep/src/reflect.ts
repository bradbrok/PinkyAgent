/**
 * Reflection / consolidation — the sleep worker's second phase (DESIGN.md
 * §5.3 item 3, "Consolidation: cross-thread synthesis"; slice 6).
 *
 * This is the ONLY consolidation path in the system (DESIGN.md §9,
 * "sleep-worker-only consolidation"): the hot tools add rows, the continuity
 * engine adds rows, and nothing but this file ever synthesizes across them.
 *
 * Three properties hold it together:
 *
 * 1. **The event log is the state (DESIGN.md §3).** The scheduler keeps no
 *    cursor of its own: the watermark is read from the newest `sleep`/`reflect`
 *    receipt on the worker's own thread, and the receipt is written INSIDE the
 *    transaction that made the memory writes. A crash before commit leaves no
 *    receipt and no rows, so the next pass simply redoes the batch; two passes
 *    racing serialize on the thread lock and the loser writes nothing.
 * 2. **It never touches live context.** Every event it appends is audit-only,
 *    so a prompt's rendered bytes are identical before and after a pass
 *    (DESIGN.md §4.5 cache alignment).
 * 3. **It never reads `user` or `private` rows (DESIGN.md §5.1).** A
 *    `tenant`-visible insight synthesized from one person's facts would move
 *    them into the shared scope, permanently and invisibly. The scope is
 *    narrowed at the QUERY, not filtered afterwards.
 */
import { EventStore } from "@pinky/core";
import type {
  Db,
  MemoryRow,
  MemoryVisibility,
  RetainInput,
  ThreadEventData,
  ThreadRef,
  TokenUsage,
} from "@pinky/core";
import { REFLECT_SYSTEM } from "./prompts";
import { REFLECT_TOOL, REFLECT_TOOL_NAME, parseReflect } from "./schemas";
import { reflectThread } from "./types";
import type { ReflectPassResult, ReflectReceipt, SleepDeps } from "./types";
import { bareModelId, errText } from "./util";

/** Output cap for the reflect call. Three insights of ≤1500 chars fit easily. */
const REFLECT_MAX_TOKENS = 2048;

/** `meta.source` stamped on every row this phase writes; `pinky stats sleep` counts on it. */
export const REFLECT_SOURCE = "sleep:reflect";

/**
 * The tuple cursor into the memory plane. `recorded_at` alone is NOT unique —
 * rows retained in one transaction share it — so a timestamp-only watermark
 * would skip every row but the first of each batch.
 */
export interface ReflectWatermark {
  recordedAt: string;
  id: string;
}

/**
 * Newest reflect receipt's `through`, or null when the agent has never
 * reflected. Read with `data->>'phase'` rather than a second event type so the
 * two phases stay one `sleep` shape in the log (packages/core/src/events.ts).
 *
 * Called twice per pass — once to build the batch, once INSIDE the transaction
 * under the thread lock — which is what makes the lost-claim check meaningful.
 */
async function readWatermark(db: Db, thread: ThreadRef): Promise<ReflectWatermark | null> {
  const row = await db.queryOne<{ data: ThreadEventData | string }>(
    `select data from events
     where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
       and type = 'sleep' and data->>'phase' = 'reflect'
     order by seq desc limit 1`,
    [thread.tenantId, thread.channelId, thread.threadId],
  );
  if (!row) return null;
  // The string branch is tolerance for legacy doubly-encoded rows, exactly as
  // in EventStore.mapRow — those cannot match `data->>'phase'` anyway, so this
  // is belt-and-braces rather than a live path.
  const data = typeof row.data === "string" ? (JSON.parse(row.data) as ThreadEventData) : row.data;
  if (data.type !== "sleep" || data.phase !== "reflect") return null;
  return data.through;
}

function sameWatermark(a: ReflectWatermark | null, b: ReflectWatermark | null): boolean {
  if (a === null || b === null) return a === b;
  return a.recordedAt === b.recordedAt && a.id === b.id;
}

/**
 * Where a synthesized insight is allowed to live, decided by its sources
 * (DESIGN.md §5.1). `drop` means it is not allowed to live anywhere.
 */
export type InsightPlacement =
  | { visibility: "tenant" }
  | { visibility: "channel"; channelId: string }
  /** The distinct channels the sources came from — two or more, so it is refused. */
  | { drop: string[] };

/**
 * The rule, in one sentence: CHANNEL CONTENT MUST NEVER WIDEN TO THE TENANT.
 *
 * Count the distinct channels among the `channel`-visible sources:
 *  - none → `tenant`. Only tenant/global rows fed it, so nothing narrows.
 *  - exactly one → `channel` in that channel, even when tenant/global rows are
 *    mixed in: the insight is then NARROWER than some of its sources, which
 *    leaks nothing (the tenant rows themselves stay where they are).
 *  - two or more → DROPPED. There is no honest answer: `tenant` would publish
 *    one channel's content to every other, and picking one of the channels
 *    would file the other's content under a space that cannot see it. The
 *    reflect prompt tells the model to split such an insight per channel, so a
 *    drop is a prompt failure, not a normal outcome — hence the log line.
 */
export function insightVisibility(sources: MemoryRow[]): InsightPlacement {
  const channels: string[] = [];
  for (const s of sources) {
    if (s.visibility !== "channel" || s.channelId === null) continue;
    if (!channels.includes(s.channelId)) channels.push(s.channelId);
  }
  if (channels.length === 0) return { visibility: "tenant" };
  const only = channels[0];
  if (channels.length === 1 && only !== undefined) {
    return { visibility: "channel", channelId: only };
  }
  return { drop: channels };
}

/**
 * The rows an insight is allowed to retire: only those at EXACTLY its own
 * placement (same visibility, and same channel when that is `channel`).
 *
 * Invalidating a tenant-wide fact because a channel-scoped insight replaced it
 * would silently lose that fact in every other channel — the insight that
 * replaced it is not readable from there. A mismatched id is dropped from
 * `supersedes` and the insight itself is kept: the synthesis is still true,
 * only the retirement was overreach.
 */
function allowedSupersedes(
  ids: string[],
  byId: Map<string, MemoryRow>,
  placement: { visibility: MemoryVisibility; channelId?: string },
): { keep: string[]; refused: string[] } {
  const keep: string[] = [];
  const refused: string[] = [];
  for (const id of ids) {
    const row = byId.get(id);
    const ok =
      row !== undefined &&
      row.visibility === placement.visibility &&
      (placement.visibility !== "channel" || row.channelId === placement.channelId);
    (ok ? keep : refused).push(id);
  }
  return { keep, refused };
}

export interface ReflectPassOptions {
  /**
   * Bypass the idle gate below (`pinky sleep run --now`, smoke). Never set by
   * the headless timer: the gate is that timer's only backoff.
   */
  ignoreIdle?: boolean;
}

/**
 * Milliseconds since the newest event on `thread`, or null when it has none
 * (a thread nobody has ever written to is idle by definition).
 *
 * postgres.js hands `timestamptz` back as a Date, not a string, so the value is
 * coerced here rather than compared as text. An unparseable stamp also reads as
 * "idle": a gate that cannot read the clock must not wedge reflection forever.
 */
async function msSinceNewestEvent(db: Db, thread: ThreadRef, now: Date): Promise<number | null> {
  const row = await db.queryOne<{ ts: string | Date }>(
    `select ts from events
     where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
     order by seq desc limit 1`,
    [thread.tenantId, thread.channelId, thread.threadId],
  );
  if (!row) return null;
  const ms = row.ts instanceof Date ? row.ts.getTime() : Date.parse(String(row.ts));
  return Number.isNaN(ms) ? null : now.getTime() - ms;
}

/**
 * One reflection pass over the memories retained since the last one.
 *
 * Returns rather than throws: a sweep runs this after N extraction passes and
 * a dead provider must not take the sweep down with it. A failure journals ONE
 * `error` event on the reflect thread and is retried by the next sweep — the
 * batch is unchanged, so the retry is the same work, which is why no backoff
 * state is kept anywhere.
 */
export async function runReflectPass(
  deps: SleepDeps,
  opts: ReflectPassOptions = {},
): Promise<ReflectPassResult> {
  const clock = deps.now ?? ((): Date => new Date());
  const startedMs = clock().getTime();
  const thread = reflectThread(deps.tenantId, deps.agentId);

  try {
    // The idle gate, on the worker's OWN thread — the same rule discovery
    // applies to a conversation, and the only backoff this pass has. A failure
    // journals an `error` event here, so the newest event is fresh and the next
    // sweep is refused for `idleMs`; without it a dead provider costs two LLM
    // calls and one `error` row every tick, forever. A success re-arms it too,
    // which is intended: consolidation is expensive and nothing new can have
    // accumulated in seconds.
    if (!opts.ignoreIdle && deps.settings.idleMs > 0) {
      const age = await msSinceNewestEvent(deps.db, thread, clock());
      if (age !== null && age < deps.settings.idleMs) {
        return { status: "skipped", reason: "not-idle" };
      }
    }

    const after = await readWatermark(deps.db, thread);

    // DESIGN.md §5.1 is the whole privacy story of this phase, and it runs in
    // two directions at once:
    //  - WIDER than any conversation: `allChannels` is the worker-only read arm
    //    that makes `channel`-visibility rows of EVERY channel legible, because
    //    consolidation is cross-thread by definition and extraction writes
    //    `channel` by default. No run ever sets it — a conversation sees one
    //    channel.
    //  - NARROWER than the surface: `includeUser`/`includePrivate` stay false
    //    however trusted the caller is, and `visibilities` says the same thing
    //    again, so BOTH the scope predicate and the visibility filter would have
    //    to be widened before a personal row could reach the payload.
    const rows = await deps.memory.since({
      scope: { agentId: deps.agentId, allChannels: true, includeUser: false, includePrivate: false },
      after,
      limit: deps.settings.reflectBatch,
      visibilities: ["tenant", "channel", "global"],
      // Never consolidate the previous consolidation. Insights are retained at
      // tenant/channel visibility under this same agent and land AFTER the
      // watermark, so without this a pass reflects on its own output on the
      // next sweep — and with `reflectMinMemories <= MAX_INSIGHTS` (validation
      // allows 1; smoke uses 1) that recurs forever, each round further from
      // the events anything was learned from.
      excludeSources: [REFLECT_SOURCE],
    });

    if (rows.length < deps.settings.reflectMinMemories) {
      return { status: "skipped", reason: "below-threshold" };
    }
    const last = rows[rows.length - 1];
    if (!last) return { status: "skipped", reason: "below-threshold" };
    const through: ReflectWatermark = { recordedAt: last.recordedAt, id: last.id };

    const payload = JSON.stringify({
      memories: rows.map((r) => ({
        id: r.id,
        text: r.text,
        kind: r.kind,
        importance: r.importance,
        visibility: r.visibility,
        channelId: r.channelId,
        recordedAt: r.recordedAt,
      })),
    });

    const turn = await deps.provider.complete({
      // Lenient split (./util): the two passes must agree, and a settings typo
      // must cost one pass, not every sweep from here on.
      model: bareModelId(deps.model),
      system: REFLECT_SYSTEM,
      messages: [{ role: "user", text: payload }],
      tools: [REFLECT_TOOL],
      // Forced: the pass has exactly one shape of answer, and an unforced call
      // that replies in prose is a wasted round trip, not a degraded one.
      toolChoice: { type: "tool", name: REFLECT_TOOL_NAME },
      maxTokens: REFLECT_MAX_TOKENS,
      ...(deps.signal ? { signal: deps.signal } : {}),
    });

    const call = turn.toolCalls.find((c) => c.name === REFLECT_TOOL_NAME);
    if (!call) {
      return await fail(deps, thread, `model returned no ${REFLECT_TOOL_NAME} call`);
    }
    const parsed = parseReflect(call.args, rows.map((r) => r.id));
    if ("error" in parsed) {
      return await fail(deps, thread, `invalid ${REFLECT_TOOL_NAME} call: ${parsed.error}`);
    }
    const insights = parsed.insights;
    const usage: TokenUsage | undefined = turn.usage;

    // Embed OUTSIDE the transaction: it is a network call, and holding the
    // thread lock across it would block every concurrent pass on the provider's
    // latency. Failure is never fatal — an insight without a vector is still
    // found by the FTS voice (DESIGN.md §5.4, mirrors runtime/memory-recall.ts).
    const embeddings = await embedInsights(deps, insights.map((i) => i.text));

    const byId = new Map(rows.map((r) => [r.id, r]));

    const outcome = await deps.db.tx(async (tx): Promise<ReflectPassResult> => {
      await EventStore.lockThreadTx(tx, thread);
      // Re-read UNDER the lock: a concurrent sweep may have consumed this exact
      // batch while we were talking to the provider. Its receipt moved the
      // watermark, so ours is stale and this pass writes nothing at all —
      // the receipt, not a scheduler flag, is what makes the pass idempotent.
      const current = await readWatermark(tx, thread);
      if (!sameWatermark(current, after)) {
        return { status: "skipped", reason: "lost-claim" };
      }

      const store = deps.memory.bind(tx);
      const events: ThreadEventData[] = [];
      let added = 0;
      let invalidated = 0;
      let noop = 0;

      for (const [index, insight] of insights.entries()) {
        const sources = insight.sources
          .map((id) => byId.get(id))
          .filter((r): r is MemoryRow => r !== undefined);
        const placement = insightVisibility(sources);
        if ("drop" in placement) {
          // §5.1: an insight drawn from two channels has nowhere legal to live.
          // Counted as a noop rather than a failure — the pass is fine, this
          // one answer was not — and logged, because a run of these means the
          // reflect prompt's "group by channel" rule is not landing.
          noop += 1;
          deps.log(
            `[sleep] reflect: dropped an insight whose sources span channels ${placement.drop.join(", ")}`,
          );
          continue;
        }
        const placed =
          placement.visibility === "channel"
            ? { visibility: "channel" as const, channelId: placement.channelId }
            : { visibility: "tenant" as const };
        const embedding = embeddings?.[index];
        const input: RetainInput = {
          agentId: deps.agentId,
          ...placed,
          // Always semantic: procedural promotion is deliberately NOT automatic
          // (DESIGN.md §13 — "dangerous if automatic, start human-approved").
          kind: "semantic",
          text: insight.text,
          importance: insight.importance,
          ...(embedding && deps.embedder
            ? { embedding, embeddingModel: deps.embedder.model }
            : {}),
          meta: { source: REFLECT_SOURCE, sources: insight.sources },
        };
        const row = await store.retain(input);
        added += 1;
        events.push({ type: "memory", op: "retain", ids: [row.id], text: row.text });

        const { keep, refused } = allowedSupersedes(insight.supersedes ?? [], byId, placed);
        if (refused.length > 0) {
          deps.log(
            `[sleep] reflect: kept ${refused.length} row(s) the insight claimed to replace from outside its scope: ${refused.join(", ")}`,
          );
        }
        for (const supersededId of keep) {
          // §5.2: invalidate, never DELETE. The reason names the successor so
          // the chain is walkable forwards as well as through meta.supersedes.
          const reason = `${REFLECT_SOURCE} consolidated into ${row.id}`;
          const ok = await store.invalidate(supersededId, { reason });
          // Count what actually happened: invalidate() is idempotent and
          // returns false for a row some other pass already retired, and a
          // receipt that claimed it did the work would be a lie in the audit.
          if (!ok) continue;
          invalidated += 1;
          events.push({ type: "memory", op: "invalidate", ids: [supersededId], text: reason });
        }
      }

      const receipt: ReflectReceipt = {
        type: "sleep",
        phase: "reflect",
        after,
        through,
        scanned: rows.length,
        candidates: insights.length,
        added,
        updated: 0,
        invalidated,
        noop,
        model: deps.model,
        ...(usage ? { usage } : {}),
        ms: clock().getTime() - startedMs,
      };
      // Receipt LAST: it is the commit marker for everything before it in this
      // batch, and the next pass reads its `through` as the new watermark.
      await EventStore.appendTx(tx, thread, [...events, receipt]);
      return { status: "done", receipt };
    });

    if (outcome.status === "done") {
      const r = outcome.receipt;
      deps.log(
        `[sleep] reflect: scanned ${r.scanned}, +${r.added} insight(s), -${r.invalidated} superseded`,
      );
    }
    return outcome;
  } catch (err) {
    return await fail(deps, thread, errText(err));
  }
}

/**
 * The one failure path: log, journal ONE `error` event on the reflect thread
 * OUTSIDE any transaction (the failed one rolled back, taking its receipt with
 * it), and report.
 *
 * No backoff state: the error is now the thread's newest event, and the sweep's
 * idle gate is what throttles the retry (contract §3.2 step 8). A pass aborted
 * by shutdown journals nothing — the process is going away and the pool may
 * already be closing, so an `error` row about our own SIGTERM is noise at best
 * and a second failure at worst.
 */
async function fail(
  deps: SleepDeps,
  thread: ThreadRef,
  message: string,
): Promise<ReflectPassResult> {
  deps.log(`[sleep] reflect failed: ${message}`);
  if (!deps.signal?.aborted) {
    try {
      await deps.events.append(thread, { type: "error", source: "sleep", message, count: 1 });
    } catch (err) {
      // Nowhere better to report a log that cannot be written: whatever broke
      // will surface on the next append the process makes.
      deps.log(`[sleep] reflect: could not journal the failure: ${errText(err)}`);
    }
  }
  return { status: "failed", error: message };
}

/**
 * One embed call for the whole insight batch, or undefined when there is no
 * embedder / the call failed. Degraded, never fatal (DESIGN.md §5.5:
 * embeddings are optional everywhere).
 */
async function embedInsights(
  deps: SleepDeps,
  texts: string[],
): Promise<number[][] | undefined> {
  const embedder = deps.embedder;
  if (!embedder || texts.length === 0) return undefined;
  try {
    if (!(await deps.memory.supportsVectors())) return undefined;
    return await embedder.embed(texts, deps.signal ? { signal: deps.signal } : undefined);
  } catch (err) {
    deps.log(`[sleep] reflect: embedding failed, storing insights without vectors: ${errText(err)}`);
    return undefined;
  }
}
