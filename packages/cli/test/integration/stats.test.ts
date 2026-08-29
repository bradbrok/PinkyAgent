/**
 * `pinky stats restarts` against a live database (DESIGN.md §13 cost model).
 *
 * The command is one SQL statement — a lateral join from each `restart` event
 * to the FIRST `message` after it in the same thread, plus jsonb accessors and
 * a `bigint` cast — and none of that runs in the unit suite. This is the only
 * place the query is executed at all, so it is where a wrong join, a missing
 * `::int`, or a jsonb path typo shows up.
 *
 * Spawned as a real child process for the same reason headless.test.ts is: the
 * CLI entry point runs its command switch at import time, so importing it here
 * would execute one.
 *
 * Skipped unless PINKY_INTEGRATION=1:
 *
 *   bun run db:up && bun run migrate
 *   PINKY_INTEGRATION=1 bun test packages/cli/test/integration/stats.test.ts
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

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const ENV = loadEnvConfig();
const CLI = new URL("../../src/index.ts", import.meta.url).pathname;

/** Unique per run: a rerun (or a crash) never collides with earlier rows. */
const RUN = `stats-test-${crypto.randomUUID().slice(0, 8)}`;
const CHANNEL = `stats:${RUN}`;

const TEST_TIMEOUT_MS = 60_000;

const ingress = (text: string): ThreadEventData => ({
  type: "ingress",
  platform: "cli",
  author: { platform: "cli", userId: "u1" },
  text,
  refs: [],
});

const continuity = (tokensBefore: number): ThreadEventData => ({
  type: "continuity",
  document: {
    goal: "measure what a restart costs",
    plan: { done: [], now: "resume from the document", next: [] },
    workingSet: {},
    decisions: [],
    openLoops: [],
    lessons: [],
    memoryHints: [],
  },
  tokensBefore,
});

/** Run the CLI and return its stdout. */
async function stats(args: string[]): Promise<{ out: string; exitCode: number; err: string }> {
  const proc = Bun.spawn(["bun", "run", CLI, "stats", ...args], {
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PINKY_INTEGRATION: undefined },
  });
  const [out, err] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  return { out, err, exitCode: await proc.exited };
}

/** The one printed line mentioning `thread`, or "" when the row is missing. */
const rowFor = (out: string, thread: string): string =>
  out.split("\n").find((l) => l.includes(`${CHANNEL}/${thread}`)) ?? "";

suite("pinky stats restarts (live db)", () => {
  let db: Db;
  let tenantId: string;

  beforeAll(async () => {
    db = createDb(ENV.databaseAdminUrl);
    // The command filters on the tenant its settings snapshot resolves to, so
    // the fixture has to be written under that same id.
    tenantId = (await new SettingsStore(db).load()).tenantId;
    const store = new EventStore(db);
    const thread = (threadId: string): ThreadRef => ({ tenantId, channelId: CHANNEL, threadId });

    // t1: a restart followed by the successor's first turn (cache-write heavy,
    // which is the whole point of the measurement) and a LATER turn that must
    // not be the one reported.
    await store.appendBatch(thread("t1"), [
      ingress("a long conversation"),
      continuity(120_000),
      {
        type: "restart",
        boundarySeq: 2,
        tokensBefore: 120_000,
        tokensAfter: 4_200,
        recallTokens: 800,
        messages: 3,
      },
      {
        type: "message",
        role: "assistant",
        text: "resumed",
        toolCalls: [],
        model: "anthropic/claude",
        usage: { input: 300, output: 120, cacheRead: 200, cacheCreation: 3_500 },
      },
      {
        type: "message",
        role: "assistant",
        text: "warm now",
        toolCalls: [],
        model: "anthropic/claude",
        usage: { input: 10, output: 5, cacheRead: 4_000, cacheCreation: 0 },
      },
    ]);

    // t2: a restart nothing has answered yet — the `n/a` column, which is also
    // every OpenAI-compatible route (they report no cache counters).
    await store.appendBatch(thread("t2"), [
      continuity(60_000),
      {
        type: "restart",
        boundarySeq: 1,
        tokensBefore: 60_000,
        tokensAfter: 1_000,
        recallTokens: 0,
        messages: 1,
      },
    ]);
  });

  afterAll(async () => {
    if (!db) return;
    await db.query(`delete from events where channel_id = $1`, [CHANNEL]);
    await db.query(`delete from threads where channel_id = $1`, [CHANNEL]);
    await db.close();
  });

  it(
    "joins each restart to the successor's first turn and summarizes the cost",
    async () => {
      const { out, exitCode } = await stats(["restarts", "--channel", CHANNEL]);
      expect(exitCode).toBe(0);

      const t1 = rowFor(out, "t1");
      expect(t1).toContain("120000 ->");
      expect(t1).toContain("4200");
      expect(t1).toContain("-115800 (-96%)");
      expect(t1).toContain("800"); // recallTokens
      // The FIRST message after the restart, not the warm turn behind it.
      expect(t1).toContain("in 300 read 200 write 3500 out 120");
      expect(t1).not.toContain("4000");

      const t2 = rowFor(out, "t2");
      expect(t2).toContain("60000 ->");
      expect(t2).toContain("n/a"); // no successor turn: no usage to report

      // mean tokensAfter (4200 + 1000) / 2; cache-write share 3500/4000 over
      // the one turn that reported usage; rebuild cost the sum of tokensAfter.
      expect(out).toContain("restarts 2");
      expect(out).toContain("mean tokensAfter 2600");
      expect(out).toContain("mean cache-write share 88% (1/2 turns reported cache usage)");
      expect(out).toContain("est. rebuild cost 5200 tokens");
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "honors --limit and scopes to the channel",
    async () => {
      const limited = await stats(["restarts", "--channel", CHANNEL, "--limit", "1"]);
      expect(limited.exitCode).toBe(0);
      expect(limited.out).toContain("restarts 1");

      const elsewhere = await stats(["restarts", "--channel", `${CHANNEL}-nothing-here`]);
      expect(elsewhere.exitCode).toBe(0);
      expect(elsewhere.out).toContain("no restart events");
      expect(elsewhere.out).toContain("restarts 0");
      expect(elsewhere.out).toContain("est. rebuild cost 0 tokens");
    },
    TEST_TIMEOUT_MS,
  );
});
