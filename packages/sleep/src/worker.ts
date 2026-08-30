/**
 * The sweep and its timer (DESIGN.md §5.3 item 3, §8.1 wake sources; slice 6).
 *
 * A sweep is: discover due threads, run one extraction pass per thread, then
 * one reflection pass. It is deliberately the dumbest possible scheduler —
 * every bit of durable state lives in the event log as a receipt (CLAUDE.md
 * invariant #6), so this file may be killed at any instant with no cleanup and
 * no bookkeeping to reconcile.
 *
 * Two things it must never do:
 *
 *  - **Run passes concurrently.** Each pass takes a thread row lock and talks
 *    to an LLM inside it. Fanning out would put N provider round trips inside N
 *    held locks and let one slow model block the agent's own appends, so the
 *    passes run SEQUENTIALLY and a sweep is bounded by `maxThreadsPerSweep`.
 *  - **Let one thread stop the sweep.** A throw from a pass becomes that
 *    thread's `failed` result and nothing more; the log already carries the
 *    `error` event, and the next thread is unaffected.
 */
import type { ThreadRef } from "@pinky/core";
import { discoverDueThreads } from "./discovery";
import { runExtractPass } from "./extract";
import { runReflectPass, type ReflectPassOptions } from "./reflect";
import type { ExtractPassResult, ReflectPassResult, SleepDeps } from "./types";
import { errText } from "./util";

export interface SweepOptions {
  /** Pin the thread set (smoke, `pinky sleep run --thread`); discovery is skipped. */
  threads?: ThreadRef[];
  /** Ignore the idle gate (`pinky sleep run --now`). */
  ignoreIdle?: boolean;
  /** Restrict discovery to one channel. Ignored when `threads` is pinned. */
  channelId?: string;
  /** Default true. */
  reflect?: boolean;
}

export interface SweepReport {
  threads: { thread: ThreadRef; result: ExtractPassResult }[];
  /** null when reflection was not attempted (`reflect: false`, shutdown, or a halt). */
  reflect: ReflectPassResult | null;
  /**
   * Set when the sweep stopped early because two CONSECUTIVE threads failed
   * with the identical error — the value is that error. Threads after it were
   * never attempted, so they are absent from `threads`.
   */
  halted?: string;
}

/**
 * The two passes, injectable.
 *
 * @internal TEST SEAM ONLY — not part of the slice-6 contract surface. It is a
 * third, optional positional parameter so every real caller writes
 * `sweep(deps)` / `sweep(deps, opts)` exactly as the contract says. The
 * alternative (bun's `mock.module`) is process-global and would hand the
 * mocked `./extract` to every other test file in the run.
 */
export interface SweepPasses {
  extract: (deps: SleepDeps, thread: ThreadRef) => Promise<ExtractPassResult>;
  reflect: (deps: SleepDeps, opts?: ReflectPassOptions) => Promise<ReflectPassResult>;
}

const DEFAULT_PASSES: SweepPasses = { extract: runExtractPass, reflect: runReflectPass };

/**
 * One sweep. Never throws for a per-thread failure; only a broken DISCOVERY
 * query (i.e. the database itself) propagates, because there is no per-thread
 * result to attach that to and a caller polling a dead database should hear
 * about it.
 */
export async function sweep(
  deps: SleepDeps,
  opts: SweepOptions = {},
  passes: SweepPasses = DEFAULT_PASSES,
): Promise<SweepReport> {
  const threads = opts.threads ?? (await discover(deps, opts));
  const report: SweepReport = { threads: [], reflect: null };
  /** The previous thread's failure, or null when it succeeded/skipped. */
  let previousError: string | null = null;

  for (const thread of threads) {
    if (deps.signal?.aborted) return report;
    let result: ExtractPassResult;
    try {
      result = await passes.extract(deps, thread);
    } catch (err) {
      // A pass is supposed to return `failed` rather than throw; if one ever
      // does throw, it must still not cost the other threads their sweep.
      result = { status: "failed", error: errText(err) };
      deps.log(`[sleep] extract threw on ${thread.channelId}/${thread.threadId}: ${result.error}`);
    }
    report.threads.push({ thread, result });

    if (result.status !== "failed") {
      previousError = null;
      continue;
    }
    // Two consecutive threads failing with the SAME message is not N unlucky
    // threads, it is one broken dependency — a missing API key answers 401 for
    // every thread, and the sweep would otherwise pay N LLM calls and journal N
    // `error` events for it, every `intervalMs`, forever. Stop the sweep: the
    // remaining threads are left untouched (and absent from `threads`), and
    // reflection — which would fail the same way — is not attempted.
    if (previousError === result.error) {
      report.halted = result.error;
      deps.log(`[sleep] sweep halted after repeated failure: ${result.error}`);
      return report;
    }
    previousError = result.error;
  }

  // Reflection is cross-thread, so it runs once per sweep and AFTER extraction:
  // the rows this sweep just wrote are exactly the material it consolidates.
  if (opts.reflect === false || deps.signal?.aborted) return report;
  try {
    // `--now` bypasses the reflect thread's own idle gate as well as
    // discovery's: an operator asking for a sweep right now means both.
    report.reflect = await passes.reflect(
      deps,
      opts.ignoreIdle !== undefined ? { ignoreIdle: opts.ignoreIdle } : {},
    );
  } catch (err) {
    report.reflect = { status: "failed", error: errText(err) };
    deps.log(`[sleep] reflect threw: ${report.reflect.error}`);
  }
  return report;
}

async function discover(deps: SleepDeps, opts: SweepOptions): Promise<ThreadRef[]> {
  const due = await discoverDueThreads(deps.db, {
    tenantId: deps.tenantId,
    // `--now` collapses the gate rather than removing the clause, so the query
    // shape (and its plan) is the same one production runs.
    idleMs: opts.ignoreIdle ? 0 : deps.settings.idleMs,
    limit: deps.settings.maxThreadsPerSweep,
    ...(opts.channelId !== undefined ? { channelId: opts.channelId } : {}),
    ...(deps.now ? { now: deps.now() } : {}),
  });
  return due.map((d) => d.thread);
}

/** One-line summary of a sweep, for the timer's stderr log. */
export function summarizeSweep(report: SweepReport): string {
  const counts = new Map<string, number>();
  for (const t of report.threads) counts.set(t.result.status, (counts.get(t.result.status) ?? 0) + 1);
  const threadPart =
    report.threads.length === 0
      ? "no due threads"
      : `${report.threads.length} thread(s): ` +
        [...counts.entries()].map(([status, n]) => `${n} ${status}`).join(", ");
  const reflectPart =
    report.reflect === null
      ? "reflect skipped"
      : `reflect ${report.reflect.status}${
          report.reflect.status === "skipped" ? ` (${report.reflect.reason})` : ""
        }`;
  const haltPart = report.halted === undefined ? "" : `; halted: ${report.halted}`;
  return `[sleep] sweep: ${threadPart}; ${reflectPart}${haltPart}`;
}

/**
 * The headless timer (DESIGN.md §8.1: "wake sources are all the same shape …
 * cron heartbeat, timer"). Mirrors `startA2ASweep` in packages/cli/src/index.ts:
 * once now, then every `intervalMs`, unref'd so it never keeps the process
 * alive by itself, every line to stderr via `deps.log` (stdout is the JSONL
 * protocol). Returns a stop function.
 *
 * A tick that arrives while a sweep is still running is SKIPPED, not queued: a
 * sweep is bounded by `maxThreadsPerSweep` and each pass is an LLM round trip,
 * so a slow provider would otherwise pile up overlapping sweeps that all fight
 * for the same thread locks and all but one report `lost-claim`. Nothing is
 * lost by skipping — due threads are recomputed from the log on the next tick.
 */
export function startSleepSweep(
  deps: SleepDeps,
  opts: { intervalMs: number },
  passes: SweepPasses = DEFAULT_PASSES,
): () => Promise<void> {
  /** The current run, or null when idle. Doubles as the "skip this tick" flag. */
  let inFlight: Promise<void> | null = null;

  const runSweep = async (): Promise<void> => {
    try {
      const report = await sweep(deps, {}, passes);
      // Quiet when there is nothing to say: this runs every few minutes for the
      // life of the process, and an idle agent should not fill stderr.
      if (report.threads.length > 0 || report.reflect?.status === "done") {
        deps.log(summarizeSweep(report));
      }
    } catch (err) {
      deps.log(`[sleep] sweep failed: ${errText(err)}`);
    }
  };

  const tick = (): void => {
    if (inFlight) {
      deps.log("[sleep] sweep still in flight, skipping this tick");
      return;
    }
    const run = runSweep().finally(() => {
      if (inFlight === run) inFlight = null;
    });
    inFlight = run;
  };

  tick();
  const timer = setInterval(tick, opts.intervalMs);
  timer.unref();
  // Returns a PROMISE that settles when the in-flight sweep does, so a shutdown
  // can `abort(); await stop()` and know no pass is still mid-transaction when
  // the pool closes. `() => Promise<void>` is assignable wherever `() => void`
  // was expected, so a caller that ignores it keeps working.
  return async (): Promise<void> => {
    clearInterval(timer);
    await inFlight;
  };
}
