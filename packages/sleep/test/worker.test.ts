/**
 * The sweep and its timer (slice 6, contract §3.5).
 *
 * The passes are injected here (the third, `@internal` parameter of
 * `sweep`/`startSleepSweep`): what this file is about is the SCHEDULING —
 * sequential passes, one bad thread not taking the sweep down, the idle gate,
 * and a timer that cannot pile sweeps on top of each other. Bun's
 * `mock.module` would have done it too, at the price of replacing `./extract`
 * for every other test file in the run — the module registry is process-global.
 */
import { describe, expect, it } from "bun:test";
import type { ThreadRef } from "@pinky/core";
import type { ReflectPassOptions } from "../src/reflect";
import type { ExtractPassResult, ReflectPassResult, SleepDeps } from "../src/types";
import { startSleepSweep, summarizeSweep, sweep, type SweepPasses } from "../src/worker";
import { FakeDb, dueRow, installDueThreads, makeDeps } from "./helpers";

const DUE_SQL = /from threads t cross join lateral/;

const SKIPPED: ExtractPassResult = { status: "skipped", reason: "no-new-events" };
const DONE_EXTRACT = { status: "done" } as unknown as ExtractPassResult;
const REFLECT_SKIPPED: ReflectPassResult = { status: "skipped", reason: "below-threshold" };

interface Recorder {
  passes: SweepPasses;
  /** start:/end: per extraction, then "reflect" — an interleave means concurrency. */
  trace: string[];
  extractCalls: ThreadRef[];
  /** The options each reflect pass was called with, in order. */
  reflectOpts: (ReflectPassOptions | undefined)[];
  readonly reflectCalls: number;
}

function recorder(
  opts: {
    extract?: (thread: ThreadRef) => Promise<ExtractPassResult>;
    reflect?: () => Promise<ReflectPassResult>;
  } = {},
): Recorder {
  const trace: string[] = [];
  const extractCalls: ThreadRef[] = [];
  const reflectOpts: (ReflectPassOptions | undefined)[] = [];
  const state = { reflectCalls: 0 };
  const passes: SweepPasses = {
    async extract(_deps: SleepDeps, thread: ThreadRef): Promise<ExtractPassResult> {
      extractCalls.push(thread);
      trace.push(`start:${thread.channelId}`);
      // Yield: an implementation that fired the passes concurrently would
      // interleave the trace instead of producing clean start/end pairs.
      await Promise.resolve();
      const out = opts.extract ? await opts.extract(thread) : SKIPPED;
      trace.push(`end:${thread.channelId}`);
      return out;
    },
    async reflect(_deps: SleepDeps, o?: ReflectPassOptions): Promise<ReflectPassResult> {
      reflectOpts.push(o);
      state.reflectCalls += 1;
      trace.push("reflect");
      return opts.reflect ? await opts.reflect() : REFLECT_SKIPPED;
    },
  };
  return {
    passes,
    trace,
    extractCalls,
    reflectOpts,
    get reflectCalls() {
      return state.reflectCalls;
    },
  };
}

function dueDb(rows: Record<string, unknown>[]): FakeDb {
  const db = new FakeDb();
  installDueThreads(db, rows);
  return db;
}

describe("sweep — running the passes", () => {
  it("runs one extraction per due thread, in order, one at a time", async () => {
    const db = dueDb([dueRow("cli:a"), dueRow("cli:b")]);
    const { deps } = makeDeps({ db });
    const rec = recorder();

    const report = await sweep(deps, {}, rec.passes);

    // Sequential: each pass holds a thread row lock across an LLM round trip,
    // so fanning out would park N provider latencies inside N held locks and
    // block the agent's own appends.
    expect(rec.trace).toEqual(["start:cli:a", "end:cli:a", "start:cli:b", "end:cli:b", "reflect"]);
    expect(report.threads.map((t) => t.thread.channelId)).toEqual(["cli:a", "cli:b"]);
    expect(report.threads.every((t) => t.result.status === "skipped")).toBe(true);
    expect(report.reflect).toEqual(REFLECT_SKIPPED);
  });

  it("turns a thrown pass into that thread's failed result and keeps going", async () => {
    const db = dueDb([dueRow("cli:a"), dueRow("cli:b")]);
    const { deps, logs } = makeDeps({ db });
    const rec = recorder({
      extract: (thread) => {
        if (thread.channelId === "cli:a") throw new Error("boom");
        return Promise.resolve(DONE_EXTRACT);
      },
    });

    const report = await sweep(deps, {}, rec.passes);

    expect(report.threads[0]?.result).toEqual({ status: "failed", error: "boom" });
    expect(report.threads[1]?.result.status).toBe("done");
    expect(rec.reflectCalls).toBe(1);
    expect(logs.some((l) => l.includes("extract threw on cli:a/main"))).toBe(true);
  });

  it("reflects once, after every extraction, and only when asked", async () => {
    const db = dueDb([dueRow("cli:a")]);
    const { deps } = makeDeps({ db });

    const on = recorder();
    const withReflect = await sweep(deps, {}, on.passes);
    expect(on.reflectCalls).toBe(1);
    // AFTER extraction: the rows this sweep just wrote are its material.
    expect(on.trace[on.trace.length - 1]).toBe("reflect");
    expect(withReflect.reflect).not.toBeNull();

    const off = recorder();
    const without = await sweep(deps, { reflect: false }, off.passes);
    // null means "not attempted", which the CLI prints differently from a
    // reflect pass that ran and skipped.
    expect(without.reflect).toBeNull();
    expect(off.reflectCalls).toBe(0);
  });

  it("stops before the next thread when the process is shutting down", async () => {
    const db = dueDb([dueRow("cli:a"), dueRow("cli:b")]);
    const controller = new AbortController();
    const { deps } = makeDeps({ db, signal: controller.signal });
    const rec = recorder({
      extract: () => {
        controller.abort();
        return Promise.resolve(SKIPPED);
      },
    });

    const report = await sweep(deps, {}, rec.passes);

    expect(rec.extractCalls.map((t) => t.channelId)).toEqual(["cli:a"]);
    expect(report.threads.length).toBe(1);
    // Reflection is skipped too. A half-run sweep is normal: the next one
    // recomputes what is due from the log, and nothing was lost.
    expect(report.reflect).toBeNull();
    expect(rec.reflectCalls).toBe(0);
  });
});

describe("sweep — halting on a broken dependency", () => {
  const failWith = (error: string) => ({ status: "failed" as const, error });

  it("stops after two consecutive identical failures and skips reflection", async () => {
    const db = dueDb([dueRow("cli:a"), dueRow("cli:b"), dueRow("cli:c"), dueRow("cli:d")]);
    const { deps, logs } = makeDeps({ db });
    // What a missing API key looks like: the same 401 for every thread.
    const rec = recorder({ extract: () => Promise.resolve(failWith("401 no api key")) });

    const report = await sweep(deps, {}, rec.passes);

    expect(report.halted).toBe("401 no api key");
    // Two attempted, two never touched — the point of the rule is the LLM calls
    // and `error` events the sweep does not make.
    expect(rec.extractCalls.map((t) => t.channelId)).toEqual(["cli:a", "cli:b"]);
    expect(report.threads.map((t) => t.thread.channelId)).toEqual(["cli:a", "cli:b"]);
    expect(report.reflect).toBeNull();
    expect(rec.reflectCalls).toBe(0);
    expect(logs.filter((l) => l.includes("sweep halted after repeated failure"))).toEqual([
      "[sleep] sweep halted after repeated failure: 401 no api key",
    ]);
  });

  it("keeps going when the failures differ, or when a success separates them", async () => {
    const db = dueDb([dueRow("cli:a"), dueRow("cli:b"), dueRow("cli:c")]);
    const { deps } = makeDeps({ db });
    const distinct = recorder({
      extract: (thread) => Promise.resolve(failWith(`broke on ${thread.channelId}`)),
    });

    const report = await sweep(deps, {}, distinct.passes);

    // Three different errors are three thread-specific problems, not one
    // broken dependency.
    expect(report.halted).toBeUndefined();
    expect(report.threads.length).toBe(3);
    expect(distinct.reflectCalls).toBe(1);

    const db2 = dueDb([dueRow("cli:a"), dueRow("cli:b"), dueRow("cli:c")]);
    const { deps: deps2 } = makeDeps({ db: db2 });
    const separated = recorder({
      extract: (thread) =>
        Promise.resolve(thread.channelId === "cli:b" ? SKIPPED : failWith("same")),
    });

    const second = await sweep(deps2, {}, separated.passes);

    // a fails, b succeeds, c fails: the run is broken again, not still broken.
    expect(second.halted).toBeUndefined();
    expect(second.threads.length).toBe(3);
  });

  it("names the halt in the summary line", () => {
    expect(
      summarizeSweep({
        threads: [
          {
            thread: { tenantId: "t1", channelId: "cli:a", threadId: "main" },
            result: { status: "failed", error: "401" },
          },
        ],
        reflect: null,
        halted: "401",
      }),
    ).toContain("halted: 401");
  });
});

describe("sweep — thread selection", () => {
  it("skips discovery entirely when the thread set is pinned", async () => {
    // No discovery route installed: FakeDb throws on unrouted SQL, so a query
    // here fails the test instead of passing unnoticed.
    const db = new FakeDb();
    const { deps } = makeDeps({ db });
    const rec = recorder();
    const pinned: ThreadRef[] = [{ tenantId: "t1", channelId: "cli:smoke", threadId: "sleep" }];

    const report = await sweep(deps, { threads: pinned }, rec.passes);

    expect(db.all(DUE_SQL).length).toBe(0);
    expect(rec.extractCalls).toEqual(pinned);
    expect(report.threads.map((t) => t.thread)).toEqual(pinned);
  });

  it("passes the settings idle gate and thread cap through to discovery", async () => {
    const db = dueDb([]);
    const { deps } = makeDeps({ db, settings: { idleMs: 600_000, maxThreadsPerSweep: 7 } });

    await sweep(deps, {}, recorder().passes);

    const params = db.find(DUE_SQL)?.params ?? [];
    expect(params[2]).toBe(600_000);
    expect(params[params.length - 1]).toBe(7);
  });

  it("collapses the idle gate for --now and forwards a channel filter", async () => {
    const db = dueDb([]);
    const { deps } = makeDeps({ db, settings: { idleMs: 600_000 } });

    await sweep(deps, { ignoreIdle: true, channelId: "cli:main" }, recorder().passes);

    const call = db.find(DUE_SQL);
    // Collapsed to 0 rather than dropped: the query shape (and its plan) stays
    // the one production runs.
    expect(call?.params?.[2]).toBe(0);
    expect(call?.params?.[4]).toBe("cli:main");
    expect((call?.sql ?? "").replace(/\s+/g, " ")).toContain("and t.channel_id = $5");
  });

  it("forwards --now to the reflect pass's own idle gate", async () => {
    const db = dueDb([]);
    const { deps } = makeDeps({ db });

    const bypass = recorder();
    await sweep(deps, { ignoreIdle: true }, bypass.passes);
    // An operator asking for a sweep RIGHT NOW means both gates: discovery's
    // and the reflect thread's.
    expect(bypass.reflectOpts).toEqual([{ ignoreIdle: true }]);

    const gated = recorder();
    await sweep(deps, {}, gated.passes);
    expect(gated.reflectOpts).toEqual([{}]);
  });

  it("uses the injected clock, so a sweep's idle window is reproducible", async () => {
    const db = dueDb([]);
    const at = new Date("2026-08-29T12:00:00.000Z");
    const { deps } = makeDeps({ db, now: () => at });

    await sweep(deps, {}, recorder().passes);

    expect(db.find(DUE_SQL)?.params?.[1]).toBe(at.toISOString());
  });
});

describe("startSleepSweep — the timer", () => {
  interface TimerControl {
    fire: () => void;
    unrefCalled: () => boolean;
    cleared: () => boolean;
  }

  /** Drive the interval by hand: capture the callback, record unref/clear. */
  async function withFakeTimers(fn: (ctl: TimerControl) => Promise<void>): Promise<void> {
    const g = globalThis as unknown as { setInterval: unknown; clearInterval: unknown };
    const realSet = g.setInterval;
    const realClear = g.clearInterval;
    let captured: (() => void) | null = null;
    let unrefCalled = false;
    let cleared = false;
    g.setInterval = (cb: () => void): unknown => {
      captured = cb;
      return {
        unref: (): void => {
          unrefCalled = true;
        },
      };
    };
    g.clearInterval = (): void => {
      cleared = true;
    };
    try {
      await fn({
        fire: () => captured?.(),
        unrefCalled: () => unrefCalled,
        cleared: () => cleared,
      });
    } finally {
      g.setInterval = realSet;
      g.clearInterval = realClear;
    }
  }

  /** Let the in-flight sweep's microtasks (and any awaited promise) run. */
  const settle = (): Promise<void> =>
    new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

  it("sweeps once immediately, unrefs the timer, and stops on demand", async () => {
    const db = dueDb([dueRow("cli:a")]);
    const { deps, logs } = makeDeps({ db });
    const rec = recorder();

    await withFakeTimers(async (ctl) => {
      const stop = startSleepSweep(deps, { intervalMs: 10_000 }, rec.passes);
      await settle();

      expect(rec.extractCalls.length).toBe(1);
      // Unref'd exactly like startA2ASweep: the worker must never be the reason
      // a process stays alive.
      expect(ctl.unrefCalled()).toBe(true);
      expect(logs.some((l) => l.startsWith("[sleep] sweep:"))).toBe(true);
      await stop();
      expect(ctl.cleared()).toBe(true);
    });
  });

  it("skips a tick that arrives while a sweep is still running", async () => {
    const db = dueDb([dueRow("cli:a")]);
    const { deps, logs } = makeDeps({ db });
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const rec = recorder({
      extract: async () => {
        await gate;
        return SKIPPED;
      },
    });

    await withFakeTimers(async (ctl) => {
      startSleepSweep(deps, { intervalMs: 10_000 }, rec.passes);
      await settle();
      expect(rec.extractCalls.length).toBe(1); // the first sweep is in flight

      ctl.fire();
      await settle();
      // Skipped, not queued: overlapping sweeps take the same thread locks and
      // all but one report lost-claim, at the price of a full set of LLM calls.
      expect(rec.extractCalls.length).toBe(1);
      expect(logs.some((l) => l.includes("still in flight"))).toBe(true);

      release();
      await settle();

      ctl.fire();
      await settle();
      expect(rec.extractCalls.length).toBe(2);
    });
  });

  it("stop() resolves only once the in-flight sweep has settled", async () => {
    const db = dueDb([dueRow("cli:a")]);
    const { deps } = makeDeps({ db });
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const rec = recorder({
      extract: async () => {
        await gate;
        return SKIPPED;
      },
    });

    await withFakeTimers(async () => {
      const stop = startSleepSweep(deps, { intervalMs: 10_000 }, rec.passes);
      await settle();

      let stopped = false;
      const stopping = stop().then(() => {
        stopped = true;
      });
      await settle();
      // A shutdown that returned here would close the pool under a pass that is
      // mid-transaction.
      expect(stopped).toBe(false);

      release();
      await stopping;
      expect(stopped).toBe(true);
    });
  });

  it("logs a failing sweep instead of throwing out of the timer", async () => {
    // A dead database: discovery is the one thing a sweep cannot recover from,
    // and an unhandled rejection inside a setInterval callback takes the
    // process down — this timer runs inside `pinky headless`.
    const db = new FakeDb();
    db.failOn(DUE_SQL, "connection terminated");
    const { deps, logs } = makeDeps({ db });

    await withFakeTimers(async () => {
      startSleepSweep(deps, { intervalMs: 10_000 }, recorder().passes);
      await settle();
      expect(logs.some((l) => l.startsWith("[sleep] sweep failed:"))).toBe(true);
    });
  });

  it("says nothing on an idle tick", async () => {
    const db = dueDb([]);
    const { deps, logs } = makeDeps({ db });

    await withFakeTimers(async () => {
      startSleepSweep(deps, { intervalMs: 10_000 }, recorder().passes);
      await settle();
      // This runs every few minutes for the life of the process; an idle agent
      // must not fill stderr with "nothing happened".
      expect(logs).toEqual([]);
    });
  });
});

describe("summarizeSweep", () => {
  it("says nothing happened plainly", () => {
    expect(summarizeSweep({ threads: [], reflect: null })).toBe(
      "[sleep] sweep: no due threads; reflect skipped",
    );
  });

  it("counts thread outcomes and names the reflect skip reason", () => {
    const line = summarizeSweep({
      threads: [
        { thread: { tenantId: "t1", channelId: "cli:a", threadId: "main" }, result: SKIPPED },
        {
          thread: { tenantId: "t1", channelId: "cli:b", threadId: "main" },
          result: { status: "failed", error: "x" },
        },
      ],
      reflect: REFLECT_SKIPPED,
    });
    expect(line).toContain("2 thread(s)");
    expect(line).toContain("1 skipped");
    expect(line).toContain("1 failed");
    expect(line).toContain("reflect skipped (below-threshold)");
  });
});
