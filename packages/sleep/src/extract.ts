/**
 * The extraction pass (DESIGN.md §5.3 item 3, Mem0's ADD/UPDATE/DELETE/NOOP
 * loop with DELETE implemented as invalidation).
 *
 * One pass = one thread, one range of its events, two LLM calls, and one
 * transaction. The ORDER below is the whole safety argument, so read it as a
 * sequence rather than as steps:
 *
 *   read cursor -> read events -> render -> extract -> neighbours -> decide
 *   -> [ lock thread | RE-READ cursor | write memories | append receipt ]
 *
 * Everything before the bracket is read-only and idempotent: a crash there
 * costs two LLM calls and nothing else. Everything inside the bracket is one
 * transaction, so the memory rows and the receipt that says they exist commit
 * together or not at all (CLAUDE.md invariant #6, the same receipt discipline
 * as the A2A consumption claim). The cursor is re-read UNDER the lock because
 * the reads happened minutes and two API calls ago: a concurrent pass may have
 * finished the same range in between, and the loser must write nothing rather
 * than duplicate every row.
 *
 * A pass that finds NOTHING still journals a receipt. Otherwise the cursor
 * would not move and every later sweep would re-extract the same events, at
 * full LLM price, forever.
 *
 * Nothing here is model-visible: `memory` writes and the `sleep` receipt are
 * audit-only in the projection (DESIGN.md §3), so nothing a pass writes
 * RENDERS. One qualification, so nobody reads this as a stronger promise than
 * it is: a window already over the event cap rolls on ANY append, this
 * included — `contextEvents` keeps the newest DEFAULT_CONTEXT_EVENT_CAP
 * events, and the loop already treats such a window as hard pressure (§4.5).
 */
import { EventStore } from "@pinky/core";
import type {
  Db,
  MemoryHit,
  MemoryRow,
  MemoryStore,
  MemoryVisibility,
  RecallScope,
  ThreadEventData,
  ThreadRef,
  TokenUsage,
} from "@pinky/core";
import { DECIDE_SYSTEM, EXTRACT_SYSTEM } from "./prompts";
import {
  DECIDE_TOOL,
  DECIDE_TOOL_NAME,
  EXTRACT_TOOL,
  EXTRACT_TOOL_NAME,
  parseDecide,
  parseExtract,
} from "./schemas";
import type { Candidate, Decision } from "./schemas";
import { renderTranscript } from "./transcript";
import { bareModelId, errText, toNum } from "./util";
import { isExtractable } from "./types";
import type { ExtractPassResult, ExtractReceipt, SleepDeps, SleepScope } from "./types";

/** Similar memories shown per candidate (DESIGN.md §5.3: "retrieve top-10 similar"). */
export const NEIGHBOR_LIMIT = 10;

/** Output budget per worker call. Both tools answer with a short JSON object;
 *  the ceiling exists to bound a runaway generation, not to shape the answer. */
export const SLEEP_MAX_TOKENS = 2048;

/** Provenance stamped on everything an extraction pass writes. */
export const EXTRACT_META_SOURCE = "sleep:extract";

/**
 * Event types the MEMORY PLANE itself produces — the worker's receipts and
 * every `memory` write (the worker's, and the agent's own recall/retain tools).
 *
 * Used for exactly one decision: whether a full page of non-extractable events
 * is worth a catch-up receipt (see {@link runPass}). A page made only of these
 * is a page the memory plane wrote, and journaling a receipt for it produces
 * another one — the treadmill.
 */
const SELF_WRITTEN_TYPES: ReadonlySet<string> = new Set(["sleep", "memory"]);

/**
 * Where this thread's last extraction pass stopped: the `toSeq` of its newest
 * `sleep`/`extract` receipt, or 0 when it has never had one.
 *
 * The scheduler holds no cursor of its own — this IS the cursor (CLAUDE.md #6).
 * Takes a `Db` rather than the store because it is also called INSIDE the
 * pass's transaction, against the tx handle, for the lost-claim re-check.
 */
export async function readExtractCursor(db: Db, thread: ThreadRef): Promise<number> {
  const row = await db.queryOne<{ to_seq: number | string | null }>(
    `select (data->>'toSeq')::bigint as to_seq from events
     where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
       and type = 'sleep' and data->>'phase' = 'extract'
     order by seq desc limit 1`,
    [thread.tenantId, thread.channelId, thread.threadId],
  );
  return toNum(row?.to_seq, 0);
}

/**
 * Sum the pass's LLM calls into one `TokenUsage`.
 *
 * A counter nobody reported stays ABSENT, never 0: "nothing cached" and
 * "nobody counted" are different facts, and `pinky stats` prices them
 * differently (CLAUDE.md #7). So the whole result is undefined when neither
 * call reported usage, and the cache counters appear only if some call had them.
 */
function sumUsage(parts: (TokenUsage | undefined)[]): TokenUsage | undefined {
  const present = parts.filter((p): p is TokenUsage => p !== undefined);
  if (present.length === 0) return undefined;
  let input = 0;
  let output = 0;
  let cacheRead: number | undefined;
  let cacheCreation: number | undefined;
  for (const part of present) {
    input += part.input;
    output += part.output;
    if (part.cacheRead !== undefined) cacheRead = (cacheRead ?? 0) + part.cacheRead;
    if (part.cacheCreation !== undefined) {
      cacheCreation = (cacheCreation ?? 0) + part.cacheCreation;
    }
  }
  return {
    input,
    output,
    ...(cacheRead !== undefined ? { cacheRead } : {}),
    ...(cacheCreation !== undefined ? { cacheCreation } : {}),
  };
}

/**
 * Apply the §5.1 visibility rules to what the model proposed.
 *
 * A `user` row survives only when the model named a person who ACTUALLY SPEAKS
 * in this transcript and the surface is wide enough to read `user` rows back.
 * Anything else is downgraded to `channel`: an invented userId would write a
 * row no scope predicate ever matches (invisible forever), and a `user` row
 * minted on a shared surface is the leak §5.1 exists to prevent. `private` is
 * not in the schema at all — the worker never writes the agent's own scratch.
 *
 * A non-`user` row drops `userId` entirely: it is a column no predicate
 * consults for that visibility, so keeping it would be provenance that looks
 * like scope.
 */
export function resolveVisibility(
  candidate: Candidate,
  authors: string[],
  scope: SleepScope,
): Candidate {
  const keepUser =
    candidate.visibility === "user" &&
    candidate.userId !== undefined &&
    scope.includeUser &&
    authors.includes(candidate.userId);
  if (keepUser) return candidate;
  return {
    text: candidate.text,
    kind: candidate.kind,
    importance: candidate.importance,
    visibility: candidate.visibility === "user" ? "channel" : candidate.visibility,
  };
}

/**
 * Embed `texts` into `into`, keyed by text. Never throws.
 *
 * Mirrors runtime/memory-recall.ts's ladder: no embedder, or no pgvector
 * column, or a failing embedding API, all degrade to FTS-only — one log line
 * and the pass continues. A blipping embedding provider must not cost the
 * agent its memories, and an unembedded row is still found by the lexical
 * voice (DESIGN.md §5.4).
 */
async function embedInto(
  deps: SleepDeps,
  texts: string[],
  into: Map<string, number[]>,
): Promise<void> {
  const embedder = deps.embedder;
  if (!embedder || texts.length === 0) return;
  const wanted = [...new Set(texts.filter((t) => !into.has(t)))];
  if (wanted.length === 0) return;
  try {
    if (!(await deps.memory.supportsVectors())) return;
    const vectors = await embedder.embed(
      wanted,
      deps.signal ? { signal: deps.signal } : undefined,
    );
    wanted.forEach((text, i) => {
      const vector = vectors[i];
      if (vector && vector.length > 0) into.set(text, vector);
    });
  } catch (err) {
    deps.log(`sleep: embedding failed, continuing FTS-only: ${errText(err)}`);
  }
}

/** Where a memory row LIVES: the tuple every scope predicate matches on (§5.1). */
export interface Placement {
  visibility: MemoryVisibility;
  channelId?: string | null;
  userId?: string | null;
}

/**
 * Does `row` live in exactly the same place as `placement`?
 *
 * This is the guard on UPDATE and DELETE, and it exists because NEIGHBOURS ARE
 * WIDER THAN THE CANDIDATE. `scopePredicate` always includes the `tenant` and
 * `global` arms, so a `channel` candidate is routinely shown `tenant` rows —
 * and an UPDATE of one would rewrite it AT CHANNEL SCOPE, making a
 * tenant-visible fact invisible everywhere but this conversation. The other
 * direction is worse: a `tenant` candidate updating a `channel` row
 * republishes one conversation's content tenant-wide. Neither is a memory
 * edit; both are a scope change wearing one (§5.1).
 *
 * So: same visibility, plus the id that visibility is matched on — `channelId`
 * for `channel`, `userId` for `user`. `tenant`/`global` carry no subject id, so
 * visibility alone settles them. Mirrors `allowedSupersedes` in reflect.ts.
 */
export function samePlacement(row: Placement, placement: Placement): boolean {
  if (row.visibility !== placement.visibility) return false;
  if (placement.visibility === "channel") {
    return (row.channelId ?? null) === (placement.channelId ?? null);
  }
  if (placement.visibility === "user") return (row.userId ?? null) === (placement.userId ?? null);
  return true;
}

/** A placement in one phrase, for the log line a refusal writes. */
export function placementLabel(p: Placement): string {
  if (p.visibility === "channel") return `channel ${p.channelId ?? "(none)"}`;
  if (p.visibility === "user") return `user ${p.userId ?? "(none)"}`;
  return p.visibility;
}

/** The payload row for one candidate's neighbour (what the decide call reads). */
function neighborPayload(hit: MemoryHit | MemoryRow): Record<string, unknown> {
  return {
    id: hit.id,
    text: hit.text,
    kind: hit.kind,
    importance: hit.importance,
    recordedAt: hit.recordedAt,
  };
}

/**
 * The CLAIM: lock the thread, re-read the cursor, run `apply`, append what it
 * produced with the receipt LAST — all in one transaction.
 *
 * Every path that journals a receipt goes through here, so the lock discipline
 * exists in one place (the same reason core's EventStore funnels every append
 * through `appendLockedTx`). The re-read is the whole point: the caller's reads
 * are minutes and up to two API calls old, so a concurrent pass may have
 * finished the same range in between — the loser writes nothing rather than
 * duplicating its rows.
 */
async function claim(
  deps: SleepDeps,
  thread: ThreadRef,
  fromSeq: number,
  apply: (tx: Db) => Promise<{ events: ThreadEventData[]; receipt: ExtractReceipt }>,
): Promise<ExtractPassResult> {
  return await deps.db.tx(async (tx): Promise<ExtractPassResult> => {
    await EventStore.lockThreadTx(tx, thread);
    const cursorNow = await readExtractCursor(tx, thread);
    if (cursorNow >= fromSeq) return { status: "skipped", reason: "lost-claim" };
    const { events, receipt } = await apply(tx);
    // The receipt goes LAST, after the audit events for the writes it is the
    // receipt FOR — so a reader of the log sees the work, then the claim.
    await EventStore.appendTx(tx, thread, [...events, receipt]);
    return { status: "done", receipt };
  });
}

/**
 * One extraction pass over `thread`. Never throws: every failure comes back as
 * `{ status: "failed" }` with the message that was journaled.
 */
export async function runExtractPass(
  deps: SleepDeps,
  thread: ThreadRef,
): Promise<ExtractPassResult> {
  const clock = deps.now ?? ((): Date => new Date());
  const startedMs = clock().getTime();
  try {
    return await runPass(deps, thread, clock, startedMs);
  } catch (err) {
    const message = errText(err);
    deps.log(`sleep: extract ${thread.channelId}/${thread.threadId} failed: ${message}`);
    // An abort is a shutdown, not a defect: journaling an `error` event on the
    // way out would leave every stopped sweep looking like a broken thread —
    // and the append would likely fail anyway against a closing pool.
    if (!deps.signal?.aborted) {
      try {
        await deps.events.append(thread, {
          type: "error",
          source: "sleep",
          message,
          count: 1,
        });
      } catch (journalErr) {
        // Nowhere better to report this: whatever broke the append will
        // surface again on the next sweep.
        deps.log(`sleep: could not journal the failure: ${errText(journalErr)}`);
      }
    }
    return { status: "failed", error: message };
  }
}

async function runPass(
  deps: SleepDeps,
  thread: ThreadRef,
  clock: () => Date,
  startedMs: number,
): Promise<ExtractPassResult> {
  // 1. Cursor.
  const cursor = await readExtractCursor(deps.db, thread);

  // 2. The range. `history` is a forward page from the cursor, so a busy
  //    thread is consumed maxEventsPerPass at a time across successive sweeps.
  const events = await deps.events.history(thread, {
    afterSeq: cursor,
    limit: deps.settings.maxEventsPerPass,
  });
  const fromSeq = cursor + 1;
  // The range is what was CONSUMED, not what was rendered: audit-only events
  // inside it are covered by the receipt and never re-read.
  const lastRead = events[events.length - 1]?.seq ?? cursor;
  const kept = events.filter((e) => isExtractable(e.data));

  // A FULL PAGE OF AUDIT-ONLY EVENTS still has to move the cursor — but only
  // when somebody OTHER than the memory plane filled it.
  //
  // `history` is a forward page of `maxEventsPerPass` rows, and it can come
  // back holding nothing extractable. Skipping there parks the cursor at the
  // head of that page while discovery's `exists` clause still sees real
  // material BEYOND it, so the thread is re-picked every sweep and never
  // extracted. A zero-count receipt covering the page is the fix; no LLM call
  // is made, so it costs one transaction.
  //
  // TWO guards keep that from becoming a receipt TREADMILL, and both are
  // load-bearing:
  //
  //  1. The page must be FULL. A short page proves there is nothing beyond it
  //     (anything newer would have been in it), so skipping is correct and free.
  //
  //  2. The page must contain at least one event the memory plane did not
  //     write. This is the one that bites: a normal pass's OWN output is up to
  //     12 `memory` events plus its `sleep` receipt, so with
  //     `maxEventsPerPass <= 13` the next page is made entirely of that
  //     output — and a catch-up receipt there appends yet another `sleep`
  //     event, which fills the next page, forever. At `maxEventsPerPass: 1` it
  //     never converges at all. Sweeps are immune (discovery's `exists` clause
  //     needs an EXTRACTABLE event past the cursor, and none of these are one),
  //     but a surface that pins a thread and calls straight in —
  //     `pinky sleep run --thread`, the smoke leg — would write one receipt per
  //     invocation.
  //
  // Residual, stated rather than hidden: a full page of PURELY memory-plane
  // events that genuinely hides material behind it is skipped, so the cursor
  // parks until the page stops being full. That needs `maxEventsPerPass <= 13`
  // — i.e. a page smaller than one pass's own output, which is a
  // misconfiguration (the default is 200). A quiet stall a human can see in
  // `pinky stats sleep` beats unbounded audit-event growth they cannot.
  const foreignPage = events.some((e) => !SELF_WRITTEN_TYPES.has(e.data.type));
  if (kept.length === 0) {
    if (events.length < deps.settings.maxEventsPerPass || !foreignPage) {
      return { status: "skipped", reason: "no-new-events" };
    }
    return await claim(deps, thread, fromSeq, () =>
      Promise.resolve({
        events: [],
        receipt: {
          type: "sleep",
          phase: "extract",
          fromSeq,
          toSeq: lastRead,
          scanned: 0,
          candidates: 0,
          added: 0,
          updated: 0,
          invalidated: 0,
          noop: 0,
          model: deps.model,
          // `usage` absent, never 0: nobody was asked, so nobody counted.
          ms: clock().getTime() - startedMs,
        },
      }),
    );
  }

  // 3. Transcript. When the char budget bound, the cursor stops where the
  //    transcript did — the rest of the range is the next pass's material.
  const transcript = renderTranscript(kept);
  const budgetBound = transcript.scanned < kept.length;
  const toSeq = budgetBound ? transcript.toSeq : lastRead;

  // 4. Extract.
  const extractTurn = await deps.provider.complete({
    model: bareModelId(deps.model),
    system: EXTRACT_SYSTEM,
    messages: [{ role: "user", text: transcript.text }],
    tools: [EXTRACT_TOOL],
    toolChoice: { type: "tool", name: EXTRACT_TOOL_NAME },
    maxTokens: SLEEP_MAX_TOKENS,
    ...(deps.signal ? { signal: deps.signal } : {}),
  });
  const extractCall = extractTurn.toolCalls.find((c) => c.name === EXTRACT_TOOL_NAME);
  if (!extractCall) {
    throw new Error(
      `${EXTRACT_TOOL_NAME} was forced but the model returned no call to it (stopReason ${extractTurn.stopReason})`,
    );
  }
  const parsed = parseExtract(extractCall.args);
  if ("error" in parsed) throw new Error(parsed.error);
  const candidates = parsed.candidates.map((c) =>
    resolveVisibility(c, transcript.authors, deps.scope),
  );

  // 5. Neighbours — one embedding call for every candidate, then one search
  //    each. The scope is the candidate's own: a `user` candidate compares
  //    against that person's rows, everything else against the channel's.
  const embeddings = new Map<string, number[]>();
  await embedInto(deps, candidates.map((c) => c.text), embeddings);

  const neighborsByCandidate: MemoryHit[][] = [];
  for (const candidate of candidates) {
    const scope: RecallScope = {
      agentId: deps.agentId,
      channelId: thread.channelId,
      ...(candidate.userId !== undefined ? { userId: candidate.userId } : {}),
      includeUser: deps.scope.includeUser && candidate.userId !== undefined,
      // Never: the agent's private scratch is not material for a memory the
      // worker is about to write at channel or tenant visibility (§5.1).
      includePrivate: false,
    };
    const embedding = embeddings.get(candidate.text);
    neighborsByCandidate.push(
      await deps.memory.search({
        scope,
        query: candidate.text,
        limit: NEIGHBOR_LIMIT,
        ...(embedding ? { queryEmbedding: embedding } : {}),
      }),
    );
  }

  // 6. Decide — skipped entirely with no candidates. There is nothing to
  //    reconcile, and the pass still needs its receipt (step 7).
  let decisions: Decision[] = [];
  let decideUsage: TokenUsage | undefined;
  if (candidates.length > 0) {
    const payload = {
      candidates: candidates.map((c, index) => ({
        index,
        text: c.text,
        kind: c.kind,
        importance: c.importance,
        visibility: c.visibility,
        neighbors: (neighborsByCandidate[index] ?? []).map(neighborPayload),
      })),
    };
    const decideTurn = await deps.provider.complete({
      model: bareModelId(deps.model),
      system: DECIDE_SYSTEM,
      messages: [{ role: "user", text: JSON.stringify(payload) }],
      tools: [DECIDE_TOOL],
      toolChoice: { type: "tool", name: DECIDE_TOOL_NAME },
      maxTokens: SLEEP_MAX_TOKENS,
      ...(deps.signal ? { signal: deps.signal } : {}),
    });
    decideUsage = decideTurn.usage;
    const decideCall = decideTurn.toolCalls.find((c) => c.name === DECIDE_TOOL_NAME);
    if (!decideCall) {
      throw new Error(
        `${DECIDE_TOOL_NAME} was forced but the model returned no call to it (stopReason ${decideTurn.stopReason})`,
      );
    }
    const parsedDecisions = parseDecide(
      decideCall.args,
      candidates,
      neighborsByCandidate.map((hits) => hits.map((h) => h.id)),
    );
    if ("error" in parsedDecisions) throw new Error(parsedDecisions.error);
    decisions = parsedDecisions.decisions;

    // An UPDATE whose merged wording differs needs its own vector; identical
    // wording reuses the candidate's, which is already in the map.
    await embedInto(
      deps,
      decisions
        .filter((d) => d.action === "UPDATE")
        .map((d) => d.text ?? "")
        .filter((t) => t !== ""),
      embeddings,
    );
  }

  const usage = sumUsage([extractTurn.usage, decideUsage]);
  const meta: Record<string, unknown> = {
    source: EXTRACT_META_SOURCE,
    channelId: thread.channelId,
    threadId: thread.threadId,
    fromSeq,
    toSeq,
  };

  /** Embedding fields for `text`, or nothing at all (never `undefined` keys —
   *  exactOptionalPropertyTypes). */
  const embeddingFor = (text: string): { embedding?: number[]; embeddingModel?: string } => {
    const embedding = embeddings.get(text);
    if (!embedding || !deps.embedder) return {};
    return { embedding, embeddingModel: deps.embedder.model };
  };

  // 7. Apply + receipt, ONE transaction (see {@link claim}).
  //
  // Neighbour rows are keyed by id here, OUTSIDE the transaction, because the
  // placement guard below needs the row as the model saw it — the id alone
  // says nothing about where it lives.
  const neighborById = new Map<string, MemoryHit>();
  for (const hits of neighborsByCandidate) for (const hit of hits) neighborById.set(hit.id, hit);

  return await claim(deps, thread, fromSeq, async (tx) => {
    // bind(tx): the same store, running on the caller's transaction handle, so
    // every memory write below commits with the receipt or not at all.
    const store: MemoryStore = deps.memory.bind(tx);
    const memoryEvents: ThreadEventData[] = [];
    let added = 0;
    let updated = 0;
    let invalidated = 0;
    let noop = 0;

    for (const decision of decisions) {
      const candidate = candidates[decision.candidate];
      if (!candidate) {
        throw new Error(`decision names candidate ${decision.candidate}, which does not exist`);
      }
      // `channelId` on every row, whatever the visibility: it is provenance
      // (where this was learned). Only `channel` rows are MATCHED on it by
      // scopePredicate, and a `channel` row without it would be unreadable.
      const common = {
        agentId: deps.agentId,
        visibility: candidate.visibility,
        channelId: thread.channelId,
        ...(candidate.userId !== undefined ? { userId: candidate.userId } : {}),
        kind: candidate.kind,
        importance: candidate.importance,
        meta,
      };

      // THE PLACEMENT GUARD (§5.1). Both destructive actions share it, before
      // the switch, because both take a `target` from a neighbour list that is
      // WIDER than the candidate (see {@link samePlacement}). A target that
      // lives somewhere else is not edited, not retired, and not silently
      // dropped either — it becomes a NOOP with a log line, so the receipt's
      // counts still add up to the candidates the model was asked about.
      let targetRow: MemoryHit | undefined;
      if (decision.action === "UPDATE" || decision.action === "DELETE") {
        const target = decision.target;
        if (!target) throw new Error(`${decision.action} decision reached apply with no target`);
        const row = neighborById.get(target);
        const placement: Placement = {
          visibility: candidate.visibility,
          channelId: thread.channelId,
          ...(candidate.userId !== undefined ? { userId: candidate.userId } : {}),
        };
        if (!row || !samePlacement(row, placement)) {
          deps.log(
            `sleep: refusing to ${decision.action} ${target} — it is ` +
              `${row ? placementLabel(row) : "not in this candidate's neighbours"} but the candidate ` +
              `is ${placementLabel(placement)}; counted as NOOP`,
          );
          noop += 1;
          continue;
        }
        targetRow = row;
      }

      switch (decision.action) {
        case "ADD": {
          const row = await store.retain({
            ...common,
            text: candidate.text,
            ...embeddingFor(candidate.text),
          });
          memoryEvents.push({ type: "memory", op: "retain", ids: [row.id], text: row.text });
          added += 1;
          break;
        }
        case "UPDATE": {
          const target = decision.target;
          const text = decision.text ?? candidate.text;
          if (!target || !targetRow) throw new Error("UPDATE decision reached apply with no target");
          // Placement comes from the TARGET, not the candidate. Belt and
          // braces — the guard just proved they are equal — but it states the
          // rule in the code that does the writing: an update is "the same
          // fact, better detail", so it never moves house (§5.2).
          const row = await store.update(target, {
            ...common,
            visibility: targetRow.visibility,
            ...(targetRow.channelId !== null ? { channelId: targetRow.channelId } : {}),
            ...(targetRow.userId !== null ? { userId: targetRow.userId } : {}),
            text,
            ...embeddingFor(text),
          });
          memoryEvents.push({
            type: "memory",
            op: "update",
            // Both rows are touched: the superseded one, then its replacement
            // (the same shape the agent-facing memory_edit tool journals).
            ids: [target, row.id],
            text: row.text,
          });
          updated += 1;
          break;
        }
        case "DELETE": {
          const target = decision.target;
          if (!target) throw new Error("DELETE decision reached apply with no target");
          const reason = `${EXTRACT_META_SOURCE} contradicted by: ${candidate.text}`;
          // Invalidation, never a DELETE (§5.2). A false return means the row
          // was already retired between the neighbour search and now — count
          // what happened, not what was decided, so the receipt stays a receipt.
          const ok = await store.invalidate(target, { reason });
          if (ok) {
            memoryEvents.push({ type: "memory", op: "invalidate", ids: [target], text: reason });
            invalidated += 1;
          } else {
            deps.log(`sleep: ${target} was already invalidated, skipping`);
          }
          break;
        }
        case "NOOP":
          noop += 1;
          break;
      }
    }

    const receipt: ExtractReceipt = {
      type: "sleep",
      phase: "extract",
      fromSeq,
      toSeq,
      scanned: transcript.scanned,
      candidates: candidates.length,
      added,
      updated,
      invalidated,
      noop,
      model: deps.model,
      ...(usage ? { usage } : {}),
      ms: clock().getTime() - startedMs,
    };
    return { events: memoryEvents, receipt };
  });
}
