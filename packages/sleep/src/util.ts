/**
 * Small shared coercions for the sleep worker (slice 6).
 *
 * All four exist because a value crosses a boundary that changes its type or
 * its trustworthiness, and every pass was solving the same four problems
 * privately. One copy means the two passes cannot disagree — which they had
 * already started to do about model ids.
 */

/** An unknown throw as a message. Never `[object Object]`. */
export function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Coerce a Postgres `bigint` at the boundary.
 *
 * postgres.js returns int8 as a STRING (it does not fit a double in general),
 * and `cursor + 1` on a string CONCATENATES — "200" becomes "2001", a cursor
 * that jumps a thousand events ahead and silently skips everything in between.
 * Comparisons go lexicographic the same way ("9" > "10"). Same rule, same
 * reason as `toSeq` in packages/core/src/event-store.ts.
 */
export function toNum(value: number | string | null | undefined, fallback: number): number {
  if (value === null || value === undefined) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Coerce a Postgres `timestamptz` at the boundary.
 *
 * The driver hands it back as a Date, while every surface that prints or
 * compares one wants the ISO string — and `String(new Date())` is
 * "Fri Aug 29 2026 …", so the CLI would print one format and the log another.
 */
export function toIso(value: string | Date): string {
  return value instanceof Date ? value.toISOString() : String(value);
}

/**
 * Bare model id for the provider ("anthropic/x" -> "x"), exactly as
 * runtime/loop.ts derives it.
 *
 * Deliberately LENIENT about a missing prefix rather than throwing like
 * `splitModel`: a model id is settings data, and the worker runs unattended on
 * a timer, so a typo must degrade to one failed pass — not a crash loop that
 * re-throws every `intervalMs` forever.
 */
export function bareModelId(model: string): string {
  const slash = model.indexOf("/");
  return slash >= 0 ? model.slice(slash + 1) : model;
}
