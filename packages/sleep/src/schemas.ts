/**
 * The three tool schemas the sleep worker forces, and the hand-rolled
 * validators for what comes back (DESIGN.md §5.3 item 3).
 *
 * WHY VALIDATE AT ALL, given the call is forced against a schema? Because a
 * tool call is the model's output, not the API's: providers do not enforce
 * `maxItems`, `minimum` or an enum, and the two rules that actually protect
 * the memory plane are not expressible in JSON Schema at all — every candidate
 * gets exactly one decision, and an UPDATE/DELETE target must be one of THAT
 * candidate's own neighbours. Without the second, one hallucinated id
 * invalidates an unrelated row, and §5.2's "invalidate, never delete" becomes
 * a promise about the mechanism rather than about the data.
 *
 * Hand-rolled, in the style of runtime/deferred.ts `validateArgs`: the repo has
 * two runtime dependencies and a JSON-Schema library is not going to be the
 * third. Every rejection message NAMES the field, because it is what the
 * failure lands in the `error` event as, and "invalid arguments" tells whoever
 * reads that event nothing.
 *
 * The `parameters` objects are written with their keys in canonical (code-unit)
 * order. Not load-bearing here — these never round-trip through jsonb — but it
 * is the house convention (core/events.ts `canonicalizeArgs`) and a schema that
 * later moves into the tool catalog would need it.
 */
import type { ToolSpec } from "@pinky/runtime";

export const EXTRACT_TOOL_NAME = "extract_memories";
export const DECIDE_TOOL_NAME = "decide_memory_updates";
export const REFLECT_TOOL_NAME = "reflect_memories";

/** Candidates one extraction pass may propose. */
export const MAX_CANDIDATES = 12;
/** Insights one reflection pass may synthesize (Generative Agents' reflection
 *  is a trickle, not a firehose — three is already a lot to justify). */
export const MAX_INSIGHTS = 3;

/** Chars in a candidate memory / an UPDATE's merged text. */
export const MAX_MEMORY_CHARS = 1000;
/** Chars in a synthesized insight (it spans several rows, so a little longer). */
export const MAX_INSIGHT_CHARS = 1500;
/** Chars in a decision's `reason` (audit colour, never stored as memory). */
export const MAX_REASON_CHARS = 300;

export type CandidateKind = "semantic" | "episodic";
export type CandidateVisibility = "channel" | "tenant" | "user";

/** One proposed memory, as the model wrote it (before the §5.1 downgrade). */
export interface Candidate {
  text: string;
  kind: CandidateKind;
  importance: number;
  visibility: CandidateVisibility;
  /** Only meaningful with `visibility: "user"`; extract.ts drops it otherwise. */
  userId?: string;
}

/**
 * Mem0's four actions. Named `MemoryAction` rather than `DecisionAction`
 * because core already exports that name for the reply-gate decision events
 * (§6) and a consumer importing both barrels would collide.
 */
export type MemoryAction = "ADD" | "UPDATE" | "DELETE" | "NOOP";

export interface Decision {
  /** Index into the candidate list this decision answers. */
  candidate: number;
  action: MemoryAction;
  /** Neighbour row id; required (and checked) for UPDATE and DELETE. */
  target?: string;
  /** UPDATE's merged text. {@link parseDecide} fills in the candidate's own
   *  text when the model omitted it, so the apply step never has to. */
  text?: string;
  reason?: string;
}

export interface Insight {
  text: string;
  importance: number;
  /** Ids of the batch rows that support the insight; at least one. */
  sources: string[];
  /** Subset of `sources` the insight fully replaces (invalidated on apply). */
  supersedes?: string[];
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

export const EXTRACT_TOOL: ToolSpec = {
  name: EXTRACT_TOOL_NAME,
  description:
    "Record the durable memories worth keeping from this transcript. Return an empty list when nothing in it is worth remembering.",
  parameters: {
    additionalProperties: false,
    properties: {
      candidates: {
        description: `Up to ${MAX_CANDIDATES} standalone memories. Fewer is better; an empty list is a valid answer.`,
        items: {
          additionalProperties: false,
          properties: {
            importance: {
              description:
                "1-10. 1 = trivia, 5 = useful background, 8+ = a decision, a commitment, or a lesson that would be expensive to relearn.",
              maximum: 10,
              minimum: 1,
              type: "integer",
            },
            kind: {
              description:
                "'semantic' for a standing fact ('Brad prefers terse answers'); 'episodic' for a dated event and its outcome.",
              enum: ["semantic", "episodic"],
              type: "string",
            },
            text: {
              description:
                "One self-contained statement, understandable with no other context. Name the subject; never write 'he', 'that', or 'the above'.",
              maxLength: MAX_MEMORY_CHARS,
              minLength: 1,
              type: "string",
            },
            userId: {
              description:
                "Required with visibility 'user': the id of the person this is about, exactly as it appears in the transcript.",
              type: "string",
            },
            visibility: {
              description:
                "'channel' = about this conversation; 'tenant' = about the organisation as a whole; 'user' = about one person, and only for a person who speaks in this transcript.",
              enum: ["channel", "tenant", "user"],
              type: "string",
            },
          },
          required: ["text", "kind", "importance", "visibility"],
          type: "object",
        },
        maxItems: MAX_CANDIDATES,
        type: "array",
      },
    },
    required: ["candidates"],
    type: "object",
  },
};

export const DECIDE_TOOL: ToolSpec = {
  name: DECIDE_TOOL_NAME,
  description:
    "Decide what each candidate memory does to what is already stored: one decision per candidate, no more and no fewer.",
  parameters: {
    additionalProperties: false,
    properties: {
      decisions: {
        description: "Exactly one entry per candidate index, in any order.",
        items: {
          additionalProperties: false,
          properties: {
            action: {
              description:
                "ADD = new information. UPDATE = the same fact as `target`, with better or newer detail (put the merged wording in `text`). DELETE = the candidate contradicts `target`, so `target` is now false. NOOP = already known, nothing to change.",
              enum: ["ADD", "UPDATE", "DELETE", "NOOP"],
              type: "string",
            },
            candidate: {
              description: "The candidate's `index` from the payload.",
              minimum: 0,
              type: "integer",
            },
            reason: {
              description: "One short sentence: why this action. Recorded, never shown to the agent.",
              maxLength: MAX_REASON_CHARS,
              type: "string",
            },
            target: {
              description:
                "Required for UPDATE and DELETE: the `id` of one of THIS candidate's neighbours. Never an id from another candidate's list.",
              type: "string",
            },
            text: {
              description:
                "UPDATE only: the merged statement that replaces the target. Defaults to the candidate's own text.",
              maxLength: MAX_MEMORY_CHARS,
              minLength: 1,
              type: "string",
            },
          },
          required: ["candidate", "action"],
          type: "object",
        },
        type: "array",
      },
    },
    required: ["decisions"],
    type: "object",
  },
};

export const REFLECT_TOOL: ToolSpec = {
  name: REFLECT_TOOL_NAME,
  description:
    "Synthesize cross-cutting insights from a batch of recently stored memories. Return an empty list when nothing connects them.",
  parameters: {
    additionalProperties: false,
    properties: {
      insights: {
        description: `At most ${MAX_INSIGHTS}. Each must be supported by two or more of the memories in the batch.`,
        items: {
          additionalProperties: false,
          properties: {
            importance: {
              description: "1-10, on the same scale the source memories use.",
              maximum: 10,
              minimum: 1,
              type: "integer",
            },
            sources: {
              description: "Ids of the batch memories this insight is drawn from.",
              items: { type: "string" },
              minItems: 1,
              type: "array",
            },
            supersedes: {
              description:
                "Ids from `sources` this insight FULLY replaces — they will be invalidated. Leave empty unless the insight says everything they said.",
              items: { type: "string" },
              type: "array",
            },
            text: {
              description:
                "One self-contained statement of the pattern, not a list of the sources.",
              maxLength: MAX_INSIGHT_CHARS,
              minLength: 1,
              type: "string",
            },
          },
          required: ["text", "importance", "sources"],
          type: "object",
        },
        maxItems: MAX_INSIGHTS,
        type: "array",
      },
    },
    required: ["insights"],
    type: "object",
  },
};

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

/** A rejection. Disjoint from every success shape, so `"error" in r` narrows. */
export interface ParseError {
  error: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The first key of `value` that is not in `allowed`, or null. */
function extraKey(value: Record<string, unknown>, allowed: readonly string[]): string | null {
  for (const key of Object.keys(value)) {
    if (value[key] === undefined) continue;
    if (!allowed.includes(key)) return key;
  }
  return null;
}

function badString(
  value: unknown,
  field: string,
  max: number,
): { value: string } | ParseError {
  if (typeof value !== "string") return { error: `${field} must be a string` };
  const trimmed = value.trim();
  if (trimmed === "") return { error: `${field} must be a non-empty string` };
  if (trimmed.length > max) {
    return { error: `${field} is ${trimmed.length} characters, at most ${max} are allowed` };
  }
  return { value: trimmed };
}

function badImportance(value: unknown, field: string): { value: number } | ParseError {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > 10) {
    return { error: `${field} must be an integer 1..10, got ${JSON.stringify(value)}` };
  }
  return { value };
}

function badEnum<T extends string>(
  value: unknown,
  field: string,
  allowed: readonly T[],
): { value: T } | ParseError {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    return {
      error: `${field} must be one of ${allowed.join(", ")}, got ${JSON.stringify(value)}`,
    };
  }
  return { value: value as T };
}

/** Ids as a list of non-empty strings, or the field that broke. */
function stringList(value: unknown, field: string): { value: string[] } | ParseError {
  if (!Array.isArray(value)) return { error: `${field} must be an array` };
  const out: string[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "string" || item.trim() === "") {
      return { error: `${field}[${i}] must be a non-empty string` };
    }
    out.push(item);
  }
  return { value: out };
}

const CANDIDATE_KEYS = ["importance", "kind", "text", "userId", "visibility"] as const;
const DECISION_KEYS = ["action", "candidate", "reason", "target", "text"] as const;
const INSIGHT_KEYS = ["importance", "sources", "supersedes", "text"] as const;
const KINDS: readonly CandidateKind[] = ["semantic", "episodic"];
const VISIBILITIES: readonly CandidateVisibility[] = ["channel", "tenant", "user"];
const ACTIONS: readonly MemoryAction[] = ["ADD", "UPDATE", "DELETE", "NOOP"];

/**
 * Validate an `extract_memories` call. An EMPTY candidate list is a success:
 * "this transcript held nothing worth keeping" is the common answer, and the
 * pass still journals its receipt so the cursor moves past those events.
 */
export function parseExtract(args: unknown): { candidates: Candidate[] } | ParseError {
  if (!isRecord(args)) return { error: `${EXTRACT_TOOL_NAME}: arguments must be an object` };
  const unknownKey = extraKey(args, ["candidates"]);
  if (unknownKey !== null) {
    return { error: `${EXTRACT_TOOL_NAME}: unexpected property "${unknownKey}"` };
  }
  const raw = args["candidates"];
  if (!Array.isArray(raw)) return { error: `${EXTRACT_TOOL_NAME}: "candidates" must be an array` };
  if (raw.length > MAX_CANDIDATES) {
    return {
      error: `${EXTRACT_TOOL_NAME}: "candidates" has ${raw.length} items, at most ${MAX_CANDIDATES} are allowed`,
    };
  }

  const candidates: Candidate[] = [];
  for (let i = 0; i < raw.length; i++) {
    const field = `candidates[${i}]`;
    const item: unknown = raw[i];
    if (!isRecord(item)) return { error: `${field} must be an object` };
    const bad = extraKey(item, CANDIDATE_KEYS);
    if (bad !== null) return { error: `${field}: unexpected property "${bad}"` };

    const text = badString(item["text"], `${field}.text`, MAX_MEMORY_CHARS);
    if ("error" in text) return text;
    const kind = badEnum(item["kind"], `${field}.kind`, KINDS);
    if ("error" in kind) return kind;
    const importance = badImportance(item["importance"], `${field}.importance`);
    if ("error" in importance) return importance;
    const visibility = badEnum(item["visibility"], `${field}.visibility`, VISIBILITIES);
    if ("error" in visibility) return visibility;

    let userId: string | undefined;
    if (item["userId"] !== undefined) {
      const parsed = badString(item["userId"], `${field}.userId`, 256);
      if ("error" in parsed) return parsed;
      userId = parsed.value;
    }

    candidates.push({
      text: text.value,
      kind: kind.value,
      importance: importance.value,
      visibility: visibility.value,
      ...(userId !== undefined ? { userId } : {}),
    });
  }
  return { candidates };
}

/**
 * Validate a `decide_memory_updates` call against the candidates it answers.
 *
 * `neighborIdsByCandidate[i]` is the id list candidate `i` was actually shown.
 * Checking `target` against THAT list rather than against every id in the
 * payload is the whole point: a model that mixes up two candidates' neighbours
 * would otherwise invalidate a row it never compared anything to (§5.2).
 *
 * Decisions come back sorted by candidate index, so the apply step runs in
 * candidate order without re-sorting.
 */
export function parseDecide(
  args: unknown,
  candidates: Candidate[],
  neighborIdsByCandidate: string[][],
): { decisions: Decision[] } | ParseError {
  if (!isRecord(args)) return { error: `${DECIDE_TOOL_NAME}: arguments must be an object` };
  const unknownKey = extraKey(args, ["decisions"]);
  if (unknownKey !== null) {
    return { error: `${DECIDE_TOOL_NAME}: unexpected property "${unknownKey}"` };
  }
  const raw = args["decisions"];
  if (!Array.isArray(raw)) return { error: `${DECIDE_TOOL_NAME}: "decisions" must be an array` };

  const byCandidate = new Map<number, Decision>();
  for (let i = 0; i < raw.length; i++) {
    const field = `decisions[${i}]`;
    const item: unknown = raw[i];
    if (!isRecord(item)) return { error: `${field} must be an object` };
    const bad = extraKey(item, DECISION_KEYS);
    if (bad !== null) return { error: `${field}: unexpected property "${bad}"` };

    const index = item["candidate"];
    if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= candidates.length) {
      return {
        error: `${field}.candidate must be an integer 0..${candidates.length - 1}, got ${JSON.stringify(index)}`,
      };
    }
    if (byCandidate.has(index)) {
      return { error: `${field}.candidate ${index} already has a decision` };
    }
    const action = badEnum(item["action"], `${field}.action`, ACTIONS);
    if ("error" in action) return action;

    const candidate = candidates[index];
    // Unreachable (the range check above), but the compiler does not know that
    // and noUncheckedIndexedAccess is on.
    if (!candidate) return { error: `${field}.candidate ${index} is out of range` };

    const decision: Decision = { candidate: index, action: action.value };

    if (action.value === "UPDATE" || action.value === "DELETE") {
      const target = item["target"];
      const allowed = neighborIdsByCandidate[index] ?? [];
      if (typeof target !== "string" || !allowed.includes(target)) {
        return {
          error:
            `${field}.target must be one of candidate ${index}'s neighbour ids ` +
            `(${allowed.length > 0 ? allowed.join(", ") : "it has none, so only ADD or NOOP are possible"}), ` +
            `got ${JSON.stringify(target)}`,
        };
      }
      decision.target = target;
    }

    if (item["text"] !== undefined) {
      const text = badString(item["text"], `${field}.text`, MAX_MEMORY_CHARS);
      if ("error" in text) return text;
      decision.text = text.value;
    }
    // An UPDATE with no text means "same statement, keep it current" — take the
    // candidate's own wording rather than failing the whole pass on an omission
    // the schema calls optional.
    if (action.value === "UPDATE" && decision.text === undefined) {
      decision.text = candidate.text;
    }

    if (item["reason"] !== undefined) {
      const reason = badString(item["reason"], `${field}.reason`, MAX_REASON_CHARS);
      if ("error" in reason) return reason;
      decision.reason = reason.value;
    }

    byCandidate.set(index, decision);
  }

  // Exactly once, not at least once: a missing decision is a candidate silently
  // dropped, and the pass would journal counts that do not add up to what it
  // was asked about.
  const decisions: Decision[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const decision = byCandidate.get(i);
    if (!decision) return { error: `decisions: candidate ${i} has no decision` };
    decisions.push(decision);
  }
  return { decisions };
}

/**
 * Validate a `reflect_memories` call against the batch it was shown.
 *
 * `supersedes ⊆ sources` is enforced, not just `supersedes ⊆ batch`: an
 * insight may only retire rows it claims to be built from. Otherwise a model
 * could invalidate a row it never cited, which is a destructive edit with no
 * stated evidence — exactly what DESIGN.md §9 blames for Mem0's quality
 * regression.
 */
export function parseReflect(args: unknown, batchIds: string[]): { insights: Insight[] } | ParseError {
  if (!isRecord(args)) return { error: `${REFLECT_TOOL_NAME}: arguments must be an object` };
  const unknownKey = extraKey(args, ["insights"]);
  if (unknownKey !== null) {
    return { error: `${REFLECT_TOOL_NAME}: unexpected property "${unknownKey}"` };
  }
  const raw = args["insights"];
  if (!Array.isArray(raw)) return { error: `${REFLECT_TOOL_NAME}: "insights" must be an array` };
  if (raw.length > MAX_INSIGHTS) {
    return {
      error: `${REFLECT_TOOL_NAME}: "insights" has ${raw.length} items, at most ${MAX_INSIGHTS} are allowed`,
    };
  }

  const insights: Insight[] = [];
  for (let i = 0; i < raw.length; i++) {
    const field = `insights[${i}]`;
    const item: unknown = raw[i];
    if (!isRecord(item)) return { error: `${field} must be an object` };
    const bad = extraKey(item, INSIGHT_KEYS);
    if (bad !== null) return { error: `${field}: unexpected property "${bad}"` };

    const text = badString(item["text"], `${field}.text`, MAX_INSIGHT_CHARS);
    if ("error" in text) return text;
    const importance = badImportance(item["importance"], `${field}.importance`);
    if ("error" in importance) return importance;

    const sources = stringList(item["sources"], `${field}.sources`);
    if ("error" in sources) return sources;
    if (sources.value.length === 0) {
      return { error: `${field}.sources must name at least one memory from the batch` };
    }
    for (const id of sources.value) {
      if (!batchIds.includes(id)) {
        return { error: `${field}.sources contains ${JSON.stringify(id)}, which is not in this batch` };
      }
    }

    let supersedes: string[] | undefined;
    if (item["supersedes"] !== undefined) {
      const parsed = stringList(item["supersedes"], `${field}.supersedes`);
      if ("error" in parsed) return parsed;
      for (const id of parsed.value) {
        if (!sources.value.includes(id)) {
          return {
            error: `${field}.supersedes contains ${JSON.stringify(id)}, which is not one of its own sources`,
          };
        }
      }
      supersedes = parsed.value;
    }

    insights.push({
      text: text.value,
      importance: importance.value,
      sources: sources.value,
      ...(supersedes !== undefined ? { supersedes } : {}),
    });
  }
  return { insights };
}
