/**
 * The continuity write path (DESIGN.md §4).
 *
 * `shed_context` is the agent-initiated rung of the trigger ladder (§4.1) and
 * the mechanism the hard boundary forces. It validates an agent-authored
 * ContinuityDoc (§4.2 — "written as a tool call, not free text, so the harness
 * can reject empty/low-signal documents") and appends it to the log as the
 * `continuity` event. That event is the new projection boundary: everything
 * before it stops being sent to the model (§3), so the loop rebuilds its
 * message list from a fresh projection right after a successful call (§4.3).
 */
import type { ContinuityDoc } from "@pinky/core";
import type { Tool, ToolContext, ToolResult } from "./types";

export const SHED_CONTEXT_TOOL_NAME = "shed_context";

const stringArray = (description: string): Record<string, unknown> => ({
  type: "array",
  items: { type: "string" },
  description,
});

/** JSON Schema for {@link ContinuityDoc} (DESIGN.md §4.2). */
export const continuityDocSchema: Record<string, unknown> = {
  type: "object",
  properties: {
    goal: {
      type: "string",
      description: "The current objective in one paragraph. Required, must be substantive.",
    },
    plan: {
      type: "object",
      description: "The plan, recited rather than summarized.",
      properties: {
        done: stringArray("Steps already finished."),
        now: { type: "string", description: "The single step in flight. Required." },
        next: stringArray("Steps still to do, in order."),
      },
      required: ["now"],
    },
    workingSet: {
      type: "object",
      description: "What the successor must load before acting.",
      properties: {
        files: stringArray("Absolute file paths."),
        artifacts: stringArray("Artifact refs (blob:sha256:...)."),
        urls: stringArray("URLs."),
        tools: stringArray(
          "Deferred tool names in use; the successor should tool_describe them before acting.",
        ),
      },
    },
    decisions: {
      type: "array",
      description: "Implicit decisions made explicit, so the successor does not relitigate them.",
      items: {
        type: "object",
        properties: {
          what: { type: "string", description: "The decision." },
          why: { type: "string", description: "The reason it was made." },
        },
        required: ["what", "why"],
      },
    },
    openLoops: stringArray("Unanswered questions and pending human requests."),
    lessons: stringArray("Mistakes turned into negative evidence, so failures are not repeated."),
    memoryHints: stringArray("Queries the successor should run against memory on wake."),
    mood: { type: "string", description: "Optional relational/affective note for persona continuity." },
  },
  required: ["goal", "plan"],
};

export type ContinuityValidation =
  | { ok: true; doc: ContinuityDoc }
  | { ok: false; error: string };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Trimmed non-empty string, else null. */
function text(v: unknown): string | null {
  return typeof v === "string" && v.trim().length > 0 ? v.trim() : null;
}

type Fail = { error: string };

function strings(value: unknown, field: string): string[] | Fail {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) return { error: `${field} must be an array of strings` };
  const out: string[] = [];
  for (let i = 0; i < value.length; i++) {
    const s = text(value[i]);
    if (s === null) return { error: `${field}[${i}] must be a non-empty string` };
    out.push(s);
  }
  return out;
}

function isFail(v: unknown): v is Fail {
  return isRecord(v) && typeof v["error"] === "string";
}

/**
 * Validate an agent-authored continuity document (§4.2 harness guard).
 *
 * Hard requirements: a non-empty `goal` and a non-empty `plan.now` — a document
 * without them cannot orient a successor, which is exactly the low-signal case
 * the design says to refuse. Every list field is optional (missing means empty)
 * but, when present, must be an array of non-empty strings; `decisions` entries
 * must carry both `what` and `why`.
 */
export function validateContinuityDoc(args: unknown): ContinuityValidation {
  if (!isRecord(args)) return { ok: false, error: "arguments must be a JSON object" };

  const goal = text(args["goal"]);
  if (goal === null) return { ok: false, error: "goal must be a non-empty string" };

  const planRaw = args["plan"];
  if (!isRecord(planRaw)) {
    return { ok: false, error: "plan must be an object with { done, now, next }" };
  }
  const now = text(planRaw["now"]);
  if (now === null) {
    return { ok: false, error: "plan.now must be a non-empty string naming the step in flight" };
  }
  const done = strings(planRaw["done"], "plan.done");
  if (isFail(done)) return { ok: false, error: done.error };
  const next = strings(planRaw["next"], "plan.next");
  if (isFail(next)) return { ok: false, error: next.error };

  const workingSetRaw = args["workingSet"];
  if (workingSetRaw !== undefined && workingSetRaw !== null && !isRecord(workingSetRaw)) {
    return {
      ok: false,
      error: "workingSet must be an object with optional files/artifacts/urls/tools arrays",
    };
  }
  const ws = isRecord(workingSetRaw) ? workingSetRaw : {};
  const workingSet: ContinuityDoc["workingSet"] = {};
  // `tools` is the deferred-tool half of the working set (slice 9): the names
  // are not in the successor's header, so without them it would have to
  // rediscover through tool_search what this window already found.
  for (const key of ["files", "artifacts", "urls", "tools"] as const) {
    if (ws[key] === undefined || ws[key] === null) continue;
    const list = strings(ws[key], `workingSet.${key}`);
    if (isFail(list)) return { ok: false, error: list.error };
    workingSet[key] = list;
  }

  const decisionsRaw = args["decisions"];
  const decisions: ContinuityDoc["decisions"] = [];
  if (decisionsRaw !== undefined && decisionsRaw !== null) {
    if (!Array.isArray(decisionsRaw)) {
      return { ok: false, error: "decisions must be an array of { what, why } objects" };
    }
    for (let i = 0; i < decisionsRaw.length; i++) {
      const entry = decisionsRaw[i];
      if (!isRecord(entry)) {
        return { ok: false, error: `decisions[${i}] must be an object with { what, why }` };
      }
      const what = text(entry["what"]);
      if (what === null) return { ok: false, error: `decisions[${i}].what must be a non-empty string` };
      const why = text(entry["why"]);
      if (why === null) return { ok: false, error: `decisions[${i}].why must be a non-empty string` };
      decisions.push({ what, why });
    }
  }

  const openLoops = strings(args["openLoops"], "openLoops");
  if (isFail(openLoops)) return { ok: false, error: openLoops.error };
  const lessons = strings(args["lessons"], "lessons");
  if (isFail(lessons)) return { ok: false, error: lessons.error };
  const memoryHints = strings(args["memoryHints"], "memoryHints");
  if (isFail(memoryHints)) return { ok: false, error: memoryHints.error };

  const moodRaw = args["mood"];
  if (moodRaw !== undefined && moodRaw !== null && typeof moodRaw !== "string") {
    return { ok: false, error: "mood must be a string" };
  }
  const mood = text(moodRaw);

  const doc: ContinuityDoc = {
    goal,
    plan: { done, now, next },
    workingSet,
    decisions,
    openLoops,
    lessons,
    memoryHints,
    ...(mood !== null ? { mood } : {}),
  };
  return { ok: true, doc };
}

/**
 * Write the continuity document and restart the context window (§4.1/§4.3).
 *
 * Emitting the `continuity` event is the whole side effect: it becomes the new
 * projection boundary, and the loop rebuilds the prompt from it and keeps
 * going within the same run.
 */
export class ShedContextTool implements Tool {
  readonly name = SHED_CONTEXT_TOOL_NAME;
  // First line doubles as the one-line summary in the system prompt's tool
  // list (see system-prompt.ts), so it has to stand alone.
  readonly description = [
    "Write your continuity document and restart your context window: the transcript before it stops being visible and you continue from the document alone.",
    "Call it at a natural boundary — a task phase finished, a plan checkpoint reached, about to switch sub-problems — or when a harness notice reports context pressure. Everything load-bearing must be inside the document, because nothing outside it survives: the goal, the plan (done/now/next), the files and artifacts to reload, the deferred tools you were using, decisions with their reasons, open loops, and lessons learned from anything that went wrong.",
  ].join("\n");
  readonly parameters = continuityDocSchema;

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    const parsed = validateContinuityDoc(args);
    if (!parsed.ok) {
      return {
        text: `continuity document rejected: ${parsed.error}. Nothing was written; call shed_context again with a corrected document.`,
        isError: true,
      };
    }
    await ctx.emit({
      type: "continuity",
      document: parsed.doc,
      tokensBefore: ctx.contextTokens ?? 0,
    });
    return {
      text: `continuity written. Context restarted at this boundary: only the continuity document is visible from here. Resume with plan.now (${parsed.doc.plan.now}), loading workingSet refs as needed.`,
    };
  }
}
