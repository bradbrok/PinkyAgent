/**
 * Prompt projection (DESIGN.md §3) — the only thing the model ever sees.
 *
 * Events are append-only; the continuity event is the boundary. Anything
 * before the latest continuity is dropped from context. Everything after it
 * renders in a stable, serialized form so prompts are cache-friendly.
 */
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
  if (doc.workingSet.files?.length || doc.workingSet.artifacts?.length || doc.workingSet.urls?.length) {
    lines.push("## Working Set");
    for (const f of doc.workingSet.files ?? []) lines.push(`- file: ${f}`);
    for (const a of doc.workingSet.artifacts ?? []) lines.push(`- artifact: ${a}`);
    for (const u of doc.workingSet.urls ?? []) lines.push(`- url: ${u}`);
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
      case "message": {
        const calls = d.toolCalls.filter((c) => answered.has(c.id));
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

  // A prompt must never open with a tool message.
  while (msgs[0]?.role === "tool") msgs.shift();
  return msgs;
}

/** Cheap estimate (chars/4) for pressure heuristics. */
export function estimateTokens(msgs: ProjectedMessage[]): number {
  let chars = 0;
  for (const m of msgs) chars += m.text.length + 20; // overhead per message
  return Math.ceil(chars / 4);
}
