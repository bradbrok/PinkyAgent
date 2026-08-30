/**
 * The event log rendered for extraction (DESIGN.md §5.3 item 3).
 *
 * Pure and boring on purpose: this is the ONE place the worker turns events
 * into text, so what the extraction model reads is a property of the log and
 * of nothing else. It is not the projection (packages/core/src/projection.ts)
 * and must not become one — the projection is the model's live context and is
 * a cached prefix (§4.5), while this is a throwaway prompt built from a
 * *range* of events, renders a failed tool result as `(error)` rather than
 * dropping it (DESIGN.md §4.4 keeps failures as negative evidence), and
 * prefixes every line with its seq so a candidate memory can be traced back.
 *
 * The one rule with teeth is the char budget: when the range does not fit, the
 * transcript STOPS at the last event that did and reports its seq. It never
 * drops older lines to make room for newer ones — the caller's cursor advances
 * to `toSeq`, so a dropped line would be material the worker skipped forever.
 */
import type { ThreadEvent, ThreadEventData } from "@pinky/core";

/**
 * Default char budget for one pass's transcript (~6k tokens at 4 chars/token).
 * Big enough for a few hundred ordinary events, small enough that a pass over
 * a busy thread costs less than the turn that produced it.
 */
export const DEFAULT_TRANSCRIPT_CHARS = 24_000;

/** Tool-call argument JSON per call. Arguments are evidence, not payload. */
export const MAX_ARGS_CHARS = 200;

/** Tool result text per result. Same cap the MCP renderer uses, one order down. */
export const MAX_TOOL_RESULT_CHARS = 300;

export interface TranscriptOptions {
  /** Default {@link DEFAULT_TRANSCRIPT_CHARS}. */
  maxChars?: number;
}

export interface Transcript {
  /** The rendered lines, newline-joined. Empty only for an empty input. */
  text: string;
  /** Distinct `ingress` userIds IN ORDER — the only ids a `user`-visible
   *  candidate may claim (extract.ts's downgrade rule). */
  authors: string[];
  /** Events actually rendered. Fewer than the input means the budget bound. */
  scanned: number;
  /** Seq of the LAST event rendered; 0 for an empty transcript. */
  toSeq: number;
}

/** `text`, cut to `max` chars with a marker, so the cut is visible to the model. */
function cap(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

/**
 * One event as transcript line(s), or null when it is not extractable material.
 *
 * Total over the event union rather than trusting the caller's filter: a new
 * event type added to core lands here as `null` (invisible to the worker)
 * instead of as an `undefined` in a template string.
 */
function renderEvent(seq: number, data: ThreadEventData): string | null {
  const head = `[${seq}]`;
  switch (data.type) {
    case "ingress":
      return `${head} user ${data.author.platform}:${data.author.userId}: ${data.text}`;
    case "a2a":
      return `${head} peer ${data.from}: ${data.text}`;
    case "message": {
      // Text may be empty on a pure tool-call turn; no trailing space then.
      const lines = [`${head} assistant:${data.text ? ` ${data.text}` : ""}`];
      for (const call of data.toolCalls) {
        lines.push(`  -> ${call.name}(${cap(JSON.stringify(call.args ?? {}), MAX_ARGS_CHARS)})`);
      }
      return lines.join("\n");
    }
    case "tool_result":
      return `${head} tool ${data.name}${data.isError ? " (error)" : ""}: ${cap(data.text, MAX_TOOL_RESULT_CHARS)}`;
    case "continuity": {
      // Defensive like runtime/memory-recall.ts: a document comes back out of
      // the log, where an older schema or a hand-fixed row may have left a
      // field missing. One bad row must not make every later pass over this
      // thread throw — the cursor would never advance again.
      const doc = data.document as
        | { goal?: string; plan?: { now?: string }; lessons?: unknown }
        | undefined;
      const goal = doc?.goal ?? "";
      const now = doc?.plan?.now ?? "";
      const lessons = Array.isArray(doc?.lessons)
        ? doc.lessons.filter((l): l is string => typeof l === "string").join("; ")
        : "";
      return `${head} continuity: goal=${goal} | now=${now} | lessons=${lessons}`;
    }
    // No `error` case: standalone `error` events are NOT extractable material
    // (types.ts EXTRACT_EVENT_TYPES says why — a failed pass journals one, and
    // extracting it would make the failure feed itself). Failures reach the
    // model through the `(error)` tool results above and continuity `lessons`.
    default:
      return null;
  }
}

/**
 * Render events (ascending seq) into one extraction prompt.
 *
 * The FIRST renderable event is always included, even when it alone blows the
 * budget. Otherwise a single oversized event — one giant paste — would produce
 * an empty transcript with `toSeq: 0`, the cursor could never move past it,
 * and every sweep from then on would re-read the same event forever.
 */
export function renderTranscript(
  events: ThreadEvent[],
  opts: TranscriptOptions = {},
): Transcript {
  const maxChars = Math.max(1, Math.floor(opts.maxChars ?? DEFAULT_TRANSCRIPT_CHARS));
  const lines: string[] = [];
  const authors: string[] = [];
  const seenAuthors = new Set<string>();
  let used = 0;
  let scanned = 0;
  let toSeq = 0;

  for (const event of events) {
    const rendered = renderEvent(event.seq, event.data);
    if (rendered === null) continue;
    // +1 for the newline this line will be joined with.
    const cost = rendered.length + (lines.length > 0 ? 1 : 0);
    if (scanned > 0 && used + cost > maxChars) break;
    lines.push(rendered);
    used += cost;
    scanned += 1;
    toSeq = event.seq;
    if (event.data.type === "ingress") {
      const userId = event.data.author.userId;
      if (userId && !seenAuthors.has(userId)) {
        seenAuthors.add(userId);
        authors.push(userId);
      }
    }
  }

  return { text: lines.join("\n"), authors, scanned, toSeq };
}
