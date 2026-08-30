/**
 * The sleep-time worker through the CLI, as real child processes (slice 6,
 * DESIGN.md §5.3 item 3).
 *
 * What only this file can prove:
 *
 *  - `pinky sleep run` end to end — discovery/pinning, the two forced LLM
 *    calls, the memory writes and the RECEIPT that commits with them, all
 *    against real Postgres. The unit suite runs on a FakeDb, so the lock, the
 *    jsonb accessors and the bigint coercions are unexercised there.
 *  - `pinky stats sleep`, whose whole body is two SQL statements — including
 *    the reflect branch, whose range column reads a different jsonb path
 *    (`data->'through'->>'recordedAt'`) than the extract one.
 *  - That a second run over the same thread is a SKIP, which is the property
 *    that makes a stateless scheduler safe to re-fire (CLAUDE.md invariant #6).
 *  - That `pinky headless` can run the sweep timer WITHOUT putting a single
 *    non-protocol byte on stdout. gateway/test/headless.test.ts injects
 *    `write`, so a stray console.log in a sweep is structurally invisible
 *    there and corrupts the stream here.
 *
 * Keyless by construction: `sleep.model` is `fake/sleep` (the scripted route
 * in runtime/providers/fake.ts), which turns every `remember:` line of a
 * transcript into one candidate, and the prompt channel's model is
 * `fake/echo`. No API key is consulted anywhere.
 *
 * WHY THIS FILE OVERRIDES `tenantId`. The worker cannot be aimed: `pinky sleep
 * run --thread` pins one thread, but the `pinky headless` timer sweeps
 * whatever the TENANT has — oldest-idle first, `maxThreadsPerSweep` at a time
 * — and would otherwise park extract cursors on the operator's own
 * `cli:local` / `cli:smoke/*` threads and reflect over their real memory
 * plane. So `agent:pinky` gets a throwaway `tenantId` for the duration (the
 * child processes resolve it from exactly that overlay, which is why
 * `pinky stats sleep` boots with `agent:pinky` in scope), and every event,
 * receipt, catalog row and memory the children write lands in a tenant that
 * is deleted whole at the end. It also makes the counts here EXACT rather
 * than ">= 1" against whatever a dev database happens to hold.
 *
 * The two settings rows that could hurt an operator are ordered around that:
 * `sleep.model` is written LAST (so `fake/sleep` never exists without the
 * tenant override in force) and removed FIRST, and `tenantId` is removed LAST
 * — every cleanup step runs even if an earlier one throws.
 *
 * Skipped unless PINKY_INTEGRATION=1:
 *
 *   bun run db:up && bun run migrate
 *   PINKY_INTEGRATION=1 bun test packages/cli/test/integration/sleep.test.ts
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import {
  createDb,
  loadEnvConfig,
  EventStore,
  SettingsStore,
  type Db,
  type ThreadEventData,
  type ThreadRef,
} from "@pinky/core";
import { FAKE_SLEEP_REFLECT_PREFIX } from "@pinky/runtime";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const ENV = loadEnvConfig();
/** Path to the CLI entry, so the test does not depend on the runner's cwd. */
const CLI = new URL("../../src/index.ts", import.meta.url).pathname;

/** Unique per run: a rerun (or a crash) never collides with earlier rows. */
const RUN = `sleep-test-${crypto.randomUUID().slice(0, 8)}`;
/** The throwaway tenant every child process resolves (see the file header). */
const TENANT = `it-sleep-cli-${RUN}`;
/** NOT a `sleep:` channel — discovery excludes those (that is where the
 *  worker journals its own receipts). */
const CHANNEL = `slp:${RUN}`;
const THREAD = "t1";
/** Where the reflect receipts land: the worker's own thread, agent `pinky`. */
const REFLECT_CHANNEL = "sleep:pinky";
const REFLECT_THREAD = "reflect";
/** The headless leg's own channel: its model row is per channel. */
const HEADLESS_CHANNEL = `slpjsonl:${RUN}`;
const HEADLESS_SCOPE = `channel:${HEADLESS_CHANNEL}`;
/** The thread the headless sweep must discover on its own (no --thread here). */
const SWEEP_THREAD = "sleepy";

/**
 * `pinky sleep run`, `pinky stats sleep` and `pinky headless` all boot with
 * `agent:pinky` in scope, so this is where the worker's configuration — and
 * the tenant override — goes. Every key is removed in afterAll.
 */
const AGENT_SCOPE = "agent:pinky";
/** Removed in THIS order: the scripted route first, the tenant fence last. */
const AGENT_KEYS = [
  "sleep.model",
  "sleep.enabled",
  "sleep.idleMs",
  "sleep.intervalMs",
  "sleep.maxThreadsPerSweep",
  "sleep.reflectMinMemories",
  "tenantId",
];

/** Extracted verbatim by `fake/sleep` from the `remember:` line. */
const CANARY = `${RUN} canary indigo-heron`;
const SWEEP_CANARY = `${RUN} sweep canary teal-otter`;

/** A hung child must fail the test, not block the suite. */
const DEADLINE_MS = 60_000;
const TEST_TIMEOUT_MS = 120_000;
/** How long the headless session stays open waiting for its startup sweep. */
const SWEEP_WAIT_MS = 30_000;

const ingress = (text: string): ThreadEventData => ({
  type: "ingress",
  platform: "cli",
  author: { platform: "cli", userId: "u1" },
  text,
  refs: [],
});

const assistant = (text: string): ThreadEventData => ({
  type: "message",
  role: "assistant",
  text,
  toolCalls: [],
  model: "fake/sleep",
});

interface Result {
  out: string;
  err: string;
  exitCode: number;
}

/** Run the CLI to completion and collect both streams. */
async function cli(args: string[]): Promise<Result> {
  const proc = Bun.spawn(["bun", "run", CLI, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PINKY_INTEGRATION: undefined },
  });
  const timer = setTimeout(() => proc.kill(), DEADLINE_MS);
  try {
    const [out, err] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    return { out, err, exitCode: await proc.exited };
  } finally {
    clearTimeout(timer);
  }
}

/** The one printed line mentioning `<channel>/<thread>`, or "". */
const rowFor = (out: string, channel: string, thread: string): string =>
  out.split("\n").find((l) => l.includes(`${channel}/${thread}`)) ?? "";

/** The `reflect` line of a `pinky sleep run` report (not a thread row). */
const reflectLine = (out: string): string =>
  out.split("\n").find((l) => l.startsWith("reflect ")) ?? "";

interface MemoryRowLite {
  text: string;
  meta: Record<string, unknown>;
}

suite("pinky sleep (live process, live db)", () => {
  let db: Db;

  /** The newest `sleep`/<phase> receipt on a thread, as stored. */
  async function receipt(
    channelId: string,
    threadId: string,
    phase: "extract" | "reflect",
  ): Promise<Record<string, unknown> | null> {
    const row = await db.queryOne<{ data: Record<string, unknown> }>(
      `select data from events
        where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
          and type = 'sleep' and data->>'phase' = $4
        order by seq desc limit 1`,
      [TENANT, channelId, threadId, phase],
    );
    return row?.data ?? null;
  }

  /** Every current memory row in the throwaway tenant, oldest first. */
  async function memories(): Promise<MemoryRowLite[]> {
    return await db.query<MemoryRowLite>(
      `select text, meta from memories
        where tenant_id = $1 and valid_to is null
        order by recorded_at asc, id asc`,
      [TENANT],
    );
  }

  const written = (rows: MemoryRowLite[], source: string): MemoryRowLite[] =>
    rows.filter((r) => r.meta.source === source);

  beforeAll(async () => {
    // Admin url: cleanup touches events/threads/settings, and `memories` has
    // RLS that a superuser bypasses — one privileged handle covers both jobs.
    db = createDb(ENV.databaseAdminUrl);
    const settings = new SettingsStore(db);

    // FIRST: the fence. Everything the children do from here lands in a tenant
    // nobody else owns.
    await settings.set(AGENT_SCOPE, "tenantId", TENANT);
    await settings.set(AGENT_SCOPE, "sleep.enabled", true);
    // 0 = no idle gate, so the headless timer's first tick finds the thread
    // seeded below instead of waiting ten minutes for it to go quiet.
    await settings.set(AGENT_SCOPE, "sleep.idleMs", 0);
    await settings.set(AGENT_SCOPE, "sleep.intervalMs", 10_000);
    // The tenant holds a handful of threads and discovery is oldest-first, so
    // a cap of 1000 means "all of them" — the seeded thread cannot be crowded
    // out of a sweep by the ones these tests create as they go.
    await settings.set(AGENT_SCOPE, "sleep.maxThreadsPerSweep", 1000);
    // One new memory is enough to reflect: the throwaway tenant contains
    // nothing but what these tests wrote, so consolidation is safe to exercise
    // for real — and it is the only coverage `stats sleep`'s reflect branch has.
    await settings.set(AGENT_SCOPE, "sleep.reflectMinMemories", 1);
    await settings.set(HEADLESS_SCOPE, "model", "fake/echo");
    // LAST: the scripted route never exists without the fence above it.
    await settings.set(AGENT_SCOPE, "sleep.model", "fake/sleep");

    const store = new EventStore(db);
    const thread = (channelId: string, threadId: string): ThreadRef => ({
      tenantId: TENANT,
      channelId,
      threadId,
    });
    // The material the pass extracts: `fake/sleep` takes everything after
    // `remember:` on a line, so the row it writes is exactly CANARY.
    await store.appendBatch(thread(CHANNEL, THREAD), [
      ingress(`remember: ${CANARY}`),
      assistant("Noted."),
    ]);
    // A second thread nothing pins, for the headless timer to DISCOVER.
    await store.appendBatch(thread(HEADLESS_CHANNEL, SWEEP_THREAD), [
      ingress(`remember: ${SWEEP_CANARY}`),
      assistant("Noted."),
    ]);
  });

  afterAll(async () => {
    if (!db) return;
    const settings = new SettingsStore(db);
    // Every step runs even if an earlier one throws: a half-finished cleanup
    // that left `sleep.model = fake/sleep` (or the tenant override) on the
    // operator's agent scope is the one outcome this file must never have.
    const steps: [string, () => Promise<unknown>][] = [
      // The dangerous rows first, in the reverse order they were written.
      ...AGENT_KEYS.slice(0, -1).map(
        (key) => [`unset ${key}`, () => settings.unset(AGENT_SCOPE, key)] as [string, () => Promise<unknown>],
      ),
      ["settings channel", () => db.query(`delete from settings where scope = $1`, [HEADLESS_SCOPE])],
      ["memories", () => db.query(`delete from memories where tenant_id = $1`, [TENANT])],
      ["events", () => db.query(`delete from events where tenant_id = $1`, [TENANT])],
      ["threads", () => db.query(`delete from threads where tenant_id = $1`, [TENANT])],
      ["dedup", () => db.query(`delete from ingress_dedup where tenant_id = $1`, [TENANT])],
      // `pinky headless` publishes its built-ins into the catalog per surface.
      ["catalog", () => db.query(`delete from tool_catalog where tenant_id = $1`, [TENANT])],
      // The fence comes down last, so everything above ran with it in place.
      ["unset tenantId", () => settings.unset(AGENT_SCOPE, "tenantId")],
    ];
    for (const [label, step] of steps) {
      try {
        await step();
      } catch (err) {
        console.error(`[sleep.test] cleanup step '${label}' failed: ${String(err)}`);
      }
    }
    await db.close();
  });

  it(
    "extracts a pinned thread, consolidates it, and journals both receipts",
    async () => {
      const { out, exitCode } = await cli([
        "sleep",
        "run",
        "--now",
        "--channel",
        CHANNEL,
        "--thread",
        THREAD,
      ]);
      expect(exitCode).toBe(0);

      const row = rowFor(out, CHANNEL, THREAD);
      expect(row).toContain("done");
      // `+1 ~0 -0 =0`: one ADD, nothing updated, invalidated or skipped.
      expect(row).toContain("+1 ~0 -0 =0");
      expect(out).toContain("model fake/sleep");
      // Reflection runs in the same sweep and AFTER extraction, so the row the
      // extract pass just wrote is exactly what it consolidates.
      expect(reflectLine(out)).toContain("done");
      expect(reflectLine(out)).toContain("+1 ~0 -0 =0");

      // The receipts: a pass's only durable record, committed in the same
      // transaction as the rows below or not at all.
      const extract = await receipt(CHANNEL, THREAD, "extract");
      expect(extract).not.toBeNull();
      expect(extract!.phase).toBe("extract");
      expect(Number(extract!.added)).toBe(1);
      expect(Number(extract!.scanned)).toBeGreaterThanOrEqual(2);
      expect(Number(extract!.fromSeq)).toBe(1);
      expect(Number(extract!.toSeq)).toBeGreaterThanOrEqual(2);
      expect(extract!.model).toBe("fake/sleep");

      const reflect = await receipt(REFLECT_CHANNEL, REFLECT_THREAD, "reflect");
      expect(reflect).not.toBeNull();
      expect(Number(reflect!.added)).toBe(1);
      // The watermark tuple: `after` is null on the first pass ever, and
      // `through` is the last row of the batch it read.
      expect(reflect!.after).toBeNull();
      expect((reflect!.through as { recordedAt?: string }).recordedAt).toBeTruthy();

      const rows = await memories();
      const extracted = written(rows, "sleep:extract");
      expect(extracted).toHaveLength(1);
      expect(extracted[0]!.text).toBe(CANARY);
      // Provenance: which phase wrote it, and off which slice of which thread.
      expect(extracted[0]!.meta.channelId).toBe(CHANNEL);
      expect(extracted[0]!.meta.threadId).toBe(THREAD);

      const insights = written(rows, "sleep:reflect");
      expect(insights).toHaveLength(1);
      expect(insights[0]!.text.startsWith(FAKE_SLEEP_REFLECT_PREFIX)).toBe(true);
      // `fake/sleep` supersedes nothing, so the source row is still current.
      expect(rows).toHaveLength(2);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "prints both receipts in `pinky stats sleep`",
    async () => {
      // Unfiltered: the tenant holds nothing but what the test above wrote, so
      // this is the whole table and every count is exact.
      const { out, exitCode } = await cli(["stats", "sleep"]);
      expect(exitCode).toBe(0);

      const extract = rowFor(out, CHANNEL, THREAD);
      expect(extract).toContain("extract");
      expect(extract).toContain("+1 ~0 -0 =0");
      // `1..N`, the inclusive seq span the cursor advanced across.
      expect(extract).toMatch(/\s1\.\.\d+\s/);

      // The reflect branch reads a different jsonb path for its range column.
      const reflect = rowFor(out, REFLECT_CHANNEL, REFLECT_THREAD);
      expect(reflect).toContain("reflect");
      expect(reflect).toMatch(/->\s\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
      expect(reflect).toContain("+1 ~0 -0 =0");

      expect(out).toContain("extract  1 pass(es)");
      expect(out).toContain("reflect  1 pass(es)");
      // Exact, not ">= 1": the extract row plus the insight, and nothing else
      // lives in this tenant.
      expect(out).toContain("memories written by the worker: 2 (current rows, all agents, all time)");

      // --channel scopes to one channel, which excludes the worker's own thread.
      const scoped = await cli(["stats", "sleep", "--channel", CHANNEL]);
      expect(scoped.exitCode).toBe(0);
      expect(rowFor(scoped.out, CHANNEL, THREAD)).toContain("extract");
      expect(rowFor(scoped.out, REFLECT_CHANNEL, REFLECT_THREAD)).toBe("");
      expect(scoped.out).toContain("reflect  0 pass(es)");

      const elsewhere = await cli(["stats", "sleep", "--channel", `${CHANNEL}-nothing-here`]);
      expect(elsewhere.exitCode).toBe(0);
      expect(elsewhere.out).toContain("no sleep receipts");
      expect(elsewhere.out).toContain("extract  0 pass(es)");
      expect(elsewhere.out).toContain("reflect  0 pass(es)");
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "skips a thread whose events it has already consumed",
    async () => {
      const { out, exitCode } = await cli([
        "sleep",
        "run",
        "--now",
        "--channel",
        CHANNEL,
        "--thread",
        THREAD,
      ]);
      // A skip is the ordinary outcome, not a failure: the cursor is past
      // everything extractable, and what the first pass appended (`memory`,
      // `sleep`) is material the worker deliberately does not read back.
      expect(exitCode).toBe(0);
      expect(rowFor(out, CHANNEL, THREAD)).toContain("skipped (no-new-events)");

      // Idempotent all the way down: no second extracted row, no second
      // extract receipt. (Reflection legitimately runs again — its own
      // watermark advanced past the insight the first sweep wrote.)
      expect(written(await memories(), "sleep:extract")).toHaveLength(1);
      const n = await db.query<{ n: string }>(
        `select count(*) as n from events
          where (tenant_id, channel_id, thread_id) = ($1, $2, $3) and type = 'sleep'`,
        [TENANT, CHANNEL, THREAD],
      );
      expect(Number(n[0]!.n)).toBe(1);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "runs the sweep inside `pinky headless` without putting a byte on stdout",
    async () => {
      // The session is held OPEN until the startup sweep has committed its
      // receipt, so this is a real assertion about the timer rather than a
      // race with process exit.
      const proc = Bun.spawn(["bun", "run", CLI, "headless"], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, PINKY_INTEGRATION: undefined },
      });
      const timer = setTimeout(() => proc.kill(), DEADLINE_MS);
      let swept: Record<string, unknown> | null = null;
      try {
        proc.stdin.write(
          `${JSON.stringify({
            type: "prompt",
            text: "hello there",
            channelId: HEADLESS_CHANNEL,
            threadId: "chat",
            id: `${RUN}:p1`,
          })}\n`,
        );
        await proc.stdin.flush();

        const deadline = Date.now() + SWEEP_WAIT_MS;
        for (;;) {
          swept = await receipt(HEADLESS_CHANNEL, SWEEP_THREAD, "extract");
          if (swept || Date.now() >= deadline) break;
          await Bun.sleep(250);
        }

        proc.stdin.write(`${JSON.stringify({ type: "exit" })}\n`);
        await proc.stdin.flush();
        await proc.stdin.end();
      } finally {
        clearTimeout(timer);
      }

      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
      const exitCode = await proc.exited;

      // The point of the whole leg: nothing but the protocol reached stdout,
      // with a sweep running the entire time.
      const lines: Record<string, unknown>[] = [];
      const garbage: string[] = [];
      for (const raw of stdout.split("\n")) {
        if (raw.trim() === "") continue;
        try {
          lines.push(JSON.parse(raw) as Record<string, unknown>);
        } catch {
          garbage.push(raw);
        }
      }
      expect(garbage).toEqual([]);
      expect(exitCode).toBe(0);

      const types = lines.map((l) => String(l.type));
      expect(types[0]).toBe("ready");
      expect(types).toContain("reply");
      expect(types.at(-1)).toBe("exiting");
      expect(types).not.toContain("error");
      // The child really did resolve the throwaway tenant from `agent:pinky`.
      expect(lines[0]!.tenantId).toBe(TENANT);

      // Everything the worker says goes to stderr, starting with the line
      // that says the timer is armed at all.
      expect(stderr).toContain("[sleep] sweep every 10000ms with fake/sleep");

      // And it really swept: a thread nothing pinned, found by discovery,
      // extracted into a row that outlives the process.
      expect(swept).not.toBeNull();
      expect(Number(swept!.added)).toBe(1);
      const rows = written(await memories(), "sleep:extract");
      expect(rows.some((r) => r.text === SWEEP_CANARY)).toBe(true);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "shuts down cleanly on SIGTERM with the sweep timer armed",
    async () => {
      // The timer holds an AbortSignal that `release()` trips BEFORE it awaits
      // `stopSleep()` — which clears the interval and waits for the sweep in
      // flight — and only then closes the pool. Whether a sweep happens to be
      // running when the signal lands is a race; what is deterministic, and
      // what this asserts, is that the abort-then-await cannot wedge or crash
      // the shutdown: 143 is `installShutdown`'s "stopped by SIGTERM", not a
      // crash, and stdout stays protocol-only throughout.
      const proc = Bun.spawn(["bun", "run", CLI, "headless"], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, PINKY_INTEGRATION: undefined },
      });
      const timer = setTimeout(() => proc.kill("SIGKILL"), DEADLINE_MS);
      try {
        // Wait for `ready`, so the signal lands on a booted process (the sleep
        // timer is armed before runHeadless writes it).
        const reader = proc.stdout.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (!buf.includes("\n")) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
        }
        expect(JSON.parse(buf.split("\n")[0] ?? "{}").type).toBe("ready");

        proc.kill("SIGTERM");
        // Keep reading through the SAME reader: `new Response(proc.stdout)`
        // would find the stream locked by the one that read `ready`.
        const stderrText = new Response(proc.stderr).text();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
        }
        const stderr = await stderrText;
        const exitCode = await proc.exited;

        expect(exitCode).toBe(143);
        expect(stderr).toContain("[shutdown] SIGTERM");
        expect(stderr).toContain("[sleep] sweep every");
        // Still protocol-only, shutdown included.
        for (const raw of buf.split("\n")) {
          if (raw.trim() === "") continue;
          expect(() => JSON.parse(raw)).not.toThrow();
        }
      } finally {
        clearTimeout(timer);
        try {
          await proc.stdin.end();
        } catch {
          // stdin is already closed with the process.
        }
      }
    },
    TEST_TIMEOUT_MS,
  );
});
