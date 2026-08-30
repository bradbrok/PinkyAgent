/**
 * Prompt projection (DESIGN.md §3) — the only thing the model ever sees.
 *
 * Events are append-only; the continuity event is the boundary. Anything
 * before the latest continuity is dropped from context. Everything after it
 * renders in a stable, serialized form so prompts are cache-friendly.
 *
 * Model-visible: `continuity`, `ingress`, `a2a`, `notice`, `message`,
 * `tool_result`, and the `block` on the ONE `memory` recall event that opens a
 * window. Everything else (`decision`, `egress`, `error`, `restart`, `config`,
 * `sleep`, every other `memory` event) is audit-only and never costs context.
 */
import { canonicalizeArgs } from "./events";
import type { ContinuityDoc, ThreadEvent, ToolCall } from "./events";

export interface ProjectedMessage {
  role: "system" | "user" | "assistant" | "tool";
  /** Present on tool messages. */
  toolCallId?: string;
  /** Present on assistant messages that invoked tools. */
  toolCalls?: ToolCall[];
  text: string;
}

/** Render a ContinuityDoc in stable markdown-ish form. */
export function serializeContinuity(doc: ContinuityDoc): string {
  const lines: string[] = [];
  lines.push("# Pinky Continuity");
  lines.push(`**Goal:** ${doc.goal}`);
  lines.push("## Plan");
  lines.push(`- now: ${doc.plan.now || "(none)"}`);
  for (const item of doc.plan.done) lines.push(`- done: ${item}`);
  for (const item of doc.plan.next) lines.push(`- next: ${item}`);
  if (
    doc.workingSet.files?.length ||
    doc.workingSet.artifacts?.length ||
    doc.workingSet.urls?.length ||
    doc.workingSet.tools?.length
  ) {
    lines.push("## Working Set");
    for (const f of doc.workingSet.files ?? []) lines.push(`- file: ${f}`);
    for (const a of doc.workingSet.artifacts ?? []) lines.push(`- artifact: ${a}`);
    for (const u of doc.workingSet.urls ?? []) lines.push(`- url: ${u}`);
    // Deferred tools (slice 9): these names are NOT in the successor's header,
    // so carrying them across the boundary is what saves it a tool_search to
    // rediscover the tool this window was already using.
    for (const t of doc.workingSet.tools ?? []) lines.push(`- tool: ${t}`);
  }
  if (doc.decisions.length) {
    lines.push("## Decisions");
    for (const d of doc.decisions) lines.push(`- ${d.what} (because: ${d.why})`);
  }
  if (doc.openLoops.length) {
    lines.push("## Open Loops");
    for (const l of doc.openLoops) lines.push(`- ${l}`);
  }
  if (doc.lessons.length) {
    lines.push("## Lessons");
    for (const l of doc.lessons) lines.push(`- ${l}`);
  }
  if (doc.memoryHints.length) {
    lines.push("## Memory Hints");
    for (const h of doc.memoryHints) lines.push(`- ${h}`);
  }
  if (doc.mood) lines.push(`**Mood:** ${doc.mood}`);
  return lines.join("\n");
}

/** The newest `continuity` event in `events`, else null. */
export function latestContinuity(events: ThreadEvent[]): { seq: number; doc: ContinuityDoc } | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e?.data.type === "continuity") {
      return { seq: e.seq, doc: e.data.document };
    }
  }
  return null;
}

/**
 * The auto-recall pass that OPENED this window, as journaled — or null when the
 * window has never had one.
 *
 * `ran` exists to make the null the only signal: a recall that found nothing
 * journals `block: ""`, and "ran and injected nothing" must not read as "never
 * ran". The loop gates on the OBJECT, not on the text; if it gated on a
 * non-empty block, a wake on an empty memory plane would leave the gate open
 * and the NEXT wake would inject a block at index 0 — moving byte 0 and
 * invalidating the whole cached prefix, which is exactly the failure this
 * journaling exists to prevent (DESIGN.md §4.5 cache alignment, §5.4 once per
 * WINDOW rather than once per wake).
 */
export interface WindowRecall {
  /** Always true — the presence of this object IS "auto-recall already ran". */
  ran: true;
  /** The injected text; `""` when the pass injected nothing. */
  block: string;
  /** The scope width that produced it; absent on events written before §5.1 scoping. */
  scope?: { includeUser: boolean; includePrivate: boolean };
}

/** {@link WindowRecall} for the window `events` ends in (boundary rule of buildContext). */
export function windowRecall(events: ThreadEvent[]): WindowRecall | null {
  const boundarySeq = latestContinuity(events)?.seq ?? 0;
  return firstAutoRecall(events.filter((e) => e.seq >= boundarySeq));
}

/**
 * The first auto-recall event inside an already-windowed list: the first
 * `memory`/`recall` carrying the `block` KEY. An event without the key is an
 * agent-initiated `recall` tool call — audit-only, and deliberately unable to
 * claim a window (packages/core/src/events.ts, `block`).
 */
function firstAutoRecall(visible: ThreadEvent[]): WindowRecall | null {
  for (const e of visible) {
    const d = e.data;
    if (d.type !== "memory" || d.op !== "recall") continue;
    if (typeof d.block !== "string") continue;
    return { ran: true, block: d.block, ...(d.scope ? { scope: d.scope } : {}) };
  }
  return null;
}

/**
 * Project the model context: the continuity boundary payload plus every
 * model-visible event after it (DESIGN.md §3). Pre-boundary events are never
 * sent to the model; they stay in the log for audit, replay, and memory
 * extraction. All other event types are silently skipped.
 *
 * Two robustness rules keep the projection accepted by providers, which reject
 * unpaired tool blocks (DESIGN.md §4.5, cut-point safety):
 *
 * 1. A `tool_result` renders only when its `callId` belongs to a tool call on
 *    an assistant message that was itself rendered in this window. This is the
 *    normal state right after a restart: the loop journals the assistant
 *    message (pre-boundary, dropped), the `shed_context` tool emits the
 *    `continuity` event, and the shed's own `tool_result` lands post-boundary
 *    as an orphan.
 * 2. An assistant tool call with no `tool_result` in the window is dropped too
 *    (a run aborted mid-tool leaves one behind); an assistant message left with
 *    no text and no calls is skipped entirely.
 *
 * Together these also guarantee the projection never starts with a tool
 * message; a defensive trim enforces that regardless of input.
 *
 * The recalled-memories block is hoisted to index 0, ahead of the continuity
 * document — literally where the loop injected it (DESIGN.md §5.4, "at context
 * start and after each restart"). It is taken from the FIRST `memory` recall
 * event in the window carrying the `block` KEY (the loop's own auto-recall
 * pass); any later one is audit-only, so a mid-window recall the agent asked
 * for cannot rewrite byte 0 of a prompt the provider has already cached. When
 * that opening pass injected nothing its `block` is `""` and nothing is
 * hoisted — the window opens with no memories and keeps it that way.
 */
export function buildContext(events: ThreadEvent[]): ProjectedMessage[] {
  const boundary = latestContinuity(events);
  const boundarySeq = boundary?.seq ?? 0;
  const visible = events.filter((e) => e.seq >= boundarySeq);

  // Pass 1: which tool calls actually have a result inside this window?
  const answered = new Set<string>();
  for (const e of visible) {
    if (e.data.type === "tool_result") answered.add(e.data.callId);
  }

  // Pass 2: render, tracking the call ids that made it into the prompt.
  const rendered = new Set<string>();
  const msgs: ProjectedMessage[] = [];
  for (const e of visible) {
    const d = e.data;
    switch (d.type) {
      case "continuity": {
        msgs.push({ role: "user", text: serializeContinuity(d.document) });
        break;
      }
      case "ingress": {
        const author = `[${d.author.platform} ${d.author.displayName ?? d.author.userId}]`;
        msgs.push({ role: "user", text: `${author}: ${d.text}` });
        break;
      }
      case "a2a": {
        // A peer's message is ingress from another agent, so it renders as a
        // `user` turn like any other arrival (DESIGN.md §7 wake-on-message):
        // a run woken by A2A must SEE what woke it, and the journaled event is
        // the only record — the mailbox row is not part of the projection.
        // The address is spelled out so the model can reply to it by name.
        msgs.push({ role: "user", text: `[a2a ${d.kind} from ${d.from}]: ${d.text}` });
        break;
      }
      case "notice": {
        // A harness-authored turn (the §4.1 pressure ladder). It renders in
        // seq order, exactly where the loop pushed it, and as `user` — never
        // `system`, which would move the cached prefix (§4.5/§9).
        msgs.push({ role: "user", text: d.text });
        break;
      }
      case "message": {
        // Canonical key order, because `data` is jsonb and jsonb SORTS an
        // object's keys by (length, bytes) on the way in. Without this the
        // `tool_use` arguments the loop sent in-run ({zulu, a, mm}) come back
        // out reordered ({a, mm, zulu}), so wake N+1's request diverges from
        // wake N's at the first tool call whose argument names differ in
        // length — one cold prefix per wake (DESIGN.md §4.5). The loop
        // canonicalizes at the other end too, so both renders agree.
        const calls = d.toolCalls
          .filter((c) => answered.has(c.id))
          .map((c) => ({ ...c, args: canonicalizeArgs(c.args) }));
        if (!d.text && calls.length === 0) break; // nothing left to render
        const msg: ProjectedMessage = { role: "assistant", text: d.text };
        if (calls.length) {
          msg.toolCalls = calls;
          for (const c of calls) rendered.add(c.id);
        }
        msgs.push(msg);
        break;
      }
      case "tool_result": {
        if (!rendered.has(d.callId)) break; // orphan: its call is not in the window
        msgs.push({ role: "tool", toolCallId: d.callId, text: d.text });
        break;
      }
      // Every other event type is deliberately skipped from the prompt.
      default:
        break;
    }
  }

  // A prompt must never open with a tool message. Trimmed BEFORE the block is
  // hoisted, so the rule is about the conversation, not about the block.
  while (msgs[0]?.role === "tool") msgs.shift();

  // The recalled-memories block IS the context start (DESIGN.md §5.4): it goes
  // in front of the continuity document, which is where the loop put it. An
  // empty block is a recall that ran and injected nothing — journaled so the
  // loop's gate can see it, but there is no message to replay.
  const opened = firstAutoRecall(visible);
  if (opened && opened.block !== "") msgs.unshift({ role: "user", text: opened.block });
  return msgs;
}

/** Cheap estimate (chars/4) for pressure heuristics. */
export function estimateTokens(msgs: ProjectedMessage[]): number {
  let chars = 0;
  for (const m of msgs) chars += m.text.length + 20; // overhead per message
  return Math.ceil(chars / 4);
}
