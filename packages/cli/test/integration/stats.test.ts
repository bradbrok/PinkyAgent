/**
 * `pinky stats restarts` and `pinky stats cache` against a live database
 * (DESIGN.md §13 cost model).
 *
 * Each command is one SQL statement — `restarts` is a lateral join from each
 * `restart` event to the FIRST `message` after it in the same thread; `cache`
 * is a newest-N scan of the `message` events that carry `usage` — plus jsonb
 * accessors and a `bigint` cast, and none of that runs in the unit suite. This
 * is the only place either query is executed at all, so it is where a wrong
 * join, a missing `::int`, or a jsonb path typo shows up.
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
  type TokenUsage,
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
/** `stats cache` reads `message` events, so it needs its own channel: the
 *  restart fixture's turns would otherwise land in its window too. */
const CACHE_CHANNEL = `cache:${RUN}`;
/**
 * The plain OpenAI / DeepSeek usage SHAPE, in its own channel so its summary
 * lines can be asserted whole: every turn reports `cacheRead` and NO
 * `cacheCreation` key at all. Everything keyed off the write counter is
 * unmeasurable here and must say so; everything keyed off the read counter —
 * the mean hit rate and the warm -> cold marker — must still work.
 */
const OPENAI_CHANNEL = `openai:${RUN}`;

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

// ---------------------------------------------------------------------------
// `pinky stats cache` — the prompt-cache hit rate (DESIGN.md §13 cost model)
// ---------------------------------------------------------------------------

/**
 * An assistant turn as the loop journals it. `usage` absent is the honest
 * "this provider counted nothing" case, which the command must exclude from
 * the mean rather than average in as a zero.
 */
const turn = (model: string, usage?: TokenUsage): ThreadEventData => ({
  type: "message",
  role: "assistant",
  text: "ok",
  toolCalls: [],
  model,
  ...(usage ? { usage } : {}),
});

const MODEL = "anthropic/claude-fable-5";
/** A route whose usage carries `cached_tokens` and nothing about writes. */
const OPENAI_MODEL = "openai/gpt-5";

/** One printed turn line, parsed back into the numbers that produced it. */
interface ParsedTurn {
  thread: string;
  seq: number;
  prompt: number;
  read: string;
  write: string;
  uncached: string;
  hit: string;
  cold: boolean;
}

const TURN_LINE =
  /^(\S+)\s+(\d+)\s+(\S+)\s+prompt\s+(\d+) = read\s+(\S+) \+ write\s+(\S+) \+ uncached\s+(\S+)\s+hit\s+(n\/a|\d+%)(\s+⊘ cold)?\s*$/;

/** Assert on numbers, not on column widths: parse every turn row back out. */
function parseTurns(out: string, channel: string): ParsedTurn[] {
  const rows: ParsedTurn[] = [];
  for (const line of out.split("\n")) {
    const m = TURN_LINE.exec(line);
    if (!m || !m[1]!.startsWith(`${channel}/`)) continue;
    rows.push({
      thread: m[1]!.slice(channel.length + 1),
      seq: Number(m[2]),
      prompt: Number(m[4]),
      read: m[5]!,
      write: m[6]!,
      uncached: m[7]!,
      hit: m[8]!,
      cold: m[9] !== undefined,
    });
  }
  return rows;
}

suite("pinky stats cache (live db)", () => {
  let db: Db;
  let tenantId: string;

  beforeAll(async () => {
    db = createDb(ENV.databaseAdminUrl);
    tenantId = (await new SettingsStore(db).load()).tenantId;
    const store = new EventStore(db);
    const thread = (threadId: string): ThreadRef => ({
      tenantId,
      channelId: CACHE_CHANNEL,
      threadId,
    });

    // Threads are written oldest-first so `order by ts desc` (one now() per
    // transaction) puts `c` at the head of the window — that is what --limit
    // trims to. Ids sort a < b < c on BOTH server collations and in JS.

    // a: three warm turns, then the invalidation, then an uncounted turn.
    await store.appendBatch(thread("a"), [
      ingress("a long conversation"),
      turn(MODEL, { input: 100, output: 20, cacheRead: 4_000, cacheCreation: 0 }),
      turn(MODEL, { input: 200, output: 20, cacheRead: 4_000, cacheCreation: 0 }),
      turn(MODEL, { input: 300, output: 20, cacheRead: 4_000, cacheCreation: 0 }),
      // warm -> cold: the prefix moved, so the whole thing is re-written.
      turn(MODEL, { input: 500, output: 20, cacheRead: 0, cacheCreation: 4_500 }),
      // Usage, but no cache counters (most OpenAI-compatible routes report
      // only prompt/completion) — that is "nobody counted", not a 0% hit.
      turn("openrouter/moonshotai/kimi-k2", { input: 6_000, output: 20 }),
      // No `usage` key at all: the query must not return this one, and it must
      // not become a zero anywhere in the summary.
      turn(MODEL),
    ]);

    // b ends WARM and c starts cold-looking: if the transition analysis kept
    // one global "previous turn" instead of one per thread, c's first turn
    // would be miscounted as an invalidation.
    await store.appendBatch(thread("b"), [
      turn(MODEL, { input: 50, output: 10, cacheRead: 8_000, cacheCreation: 0 }),
    ]);
    await store.appendBatch(thread("c"), [
      turn(MODEL, { input: 400, output: 15, cacheRead: 0, cacheCreation: 3_200 }),
    ]);

    // d: the OpenAI/DeepSeek shape — a read counter, never a write one. Warm,
    // warm, then the prefix moves: the last turn reads nothing off a prompt
    // far over the smallest cacheable size, which is an invalidation whether
    // or not the route can tell us what the re-write cost.
    await store.appendBatch(
      { tenantId, channelId: OPENAI_CHANNEL, threadId: "d" },
      [
        ingress("a long conversation"),
        turn(OPENAI_MODEL, { input: 100, output: 20, cacheRead: 5_000 }),
        turn(OPENAI_MODEL, { input: 150, output: 20, cacheRead: 5_000 }),
        turn(OPENAI_MODEL, { input: 5_400, output: 20, cacheRead: 0 }),
      ],
    );
  });

  afterAll(async () => {
    if (!db) return;
    for (const channel of [CACHE_CHANNEL, OPENAI_CHANNEL]) {
      await db.query(`delete from events where channel_id = $1`, [channel]);
      await db.query(`delete from threads where channel_id = $1`, [channel]);
    }
    await db.close();
  });

  it(
    "reports per-turn hit shares, marks the warm -> cold transition, and excludes uncounted turns from the mean",
    async () => {
      const { out, exitCode } = await stats(["cache", "--channel", CACHE_CHANNEL]);
      expect(exitCode).toBe(0);

      const rows = parseTurns(out, CACHE_CHANNEL);
      // Thread-major, seq-ascending — sorted on the JS side of the comparison.
      expect(rows.map((r) => `${r.thread}/${r.seq}`)).toEqual([
        "a/2",
        "a/3",
        "a/4",
        "a/5",
        "a/6",
        "b/1",
        "c/1",
      ]);

      // prompt = input + cacheRead + cacheCreation; hit = cacheRead / prompt.
      expect(rows.map((r) => r.prompt)).toEqual([4_100, 4_200, 4_300, 5_000, 6_000, 8_050, 3_600]);
      expect(rows.map((r) => r.hit)).toEqual([
        "98%", // 4000/4100
        "95%", // 4000/4200
        "93%", // 4000/4300
        "0%", // the invalidated turn read nothing
        "n/a", // the provider counted nothing — NOT a 0% hit
        "99%", // 8000/8050
        "0%",
      ]);

      // "-" is "nobody counted", which is why that turn has no hit share.
      const uncounted = rows[4]!;
      expect(uncounted.read).toBe("-");
      expect(uncounted.write).toBe("-");
      expect(uncounted.uncached).toBe("6000");

      // Exactly one invalidation, and it is a/5 — c/1 is the FIRST turn of its
      // thread, so it has no predecessor to have gone cold from.
      expect(rows.filter((r) => r.cold).map((r) => `${r.thread}/${r.seq}`)).toEqual(["a/5"]);

      // ONE denominator: the 6 turns that carry counters are the mean's
      // divisor AND the set the token totals below are summed over.
      expect(out).toContain("turns 7  with cache counters 6");
      // (0.9756 + 0.9524 + 0.9302 + 0 + 0.9938 + 0) / 6 = 64%. Averaging the
      // uncounted turn in as a zero would print 55%.
      expect(out).toContain("mean hit 64%");
      expect(out).not.toContain("mean hit 55%");
      expect(out).toContain("cold transitions 1");
      // Measured turns only — a/6's 6000 uncached tokens are NOT in the totals,
      // because nothing about a/6's prompt was measured against the cache.
      expect(out).toContain(
        "tokens over the 6 measured turns  read 20000  write 7700  uncached 1550  prompt 29250",
      );
      // a/5 (4500/5000) and c/1 (3200/3600) are both >= 80% write, over the 6
      // turns that reported a write counter (a/6 reported none).
      expect(out).toContain(
        "prefix rewritten (write >= 80% of prompt) 2/6 turns reporting writes (33%)",
      );
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "honors --thread, --limit, and an empty scope",
    async () => {
      const one = await stats(["cache", "--channel", CACHE_CHANNEL, "--thread", "a"]);
      expect(one.exitCode).toBe(0);
      const rows = parseTurns(one.out, CACHE_CHANNEL);
      expect(rows.map((r) => r.thread)).toEqual(["a", "a", "a", "a", "a"]);
      expect(one.out).toContain("turns 5  with cache counters 4");
      // (0.9756 + 0.9524 + 0.9302 + 0) / 4
      expect(one.out).toContain("mean hit 71%");
      expect(one.out).toContain("cold transitions 1");
      expect(one.out).toContain(
        "tokens over the 4 measured turns  read 12000  write 4500  uncached 1100  prompt 17600",
      );
      expect(one.out).toContain(
        "prefix rewritten (write >= 80% of prompt) 1/4 turns reporting writes (25%)",
      );

      const limited = await stats(["cache", "--channel", CACHE_CHANNEL, "--limit", "2"]);
      expect(limited.exitCode).toBe(0);
      expect(parseTurns(limited.out, CACHE_CHANNEL)).toHaveLength(2);
      expect(limited.out).toContain("turns 2");

      // The sample EDGE. Newest-4 is c/1, b/1, a/6, a/5 (one `now()` per
      // batch, then seq desc inside a batch), so a/5's predecessor a/4 is
      // outside the window: the invalidation cannot be seen from this sample
      // and must not be claimed — a thread's first SAMPLED turn is never cold,
      // even when it looks exactly like one (read 0, a whole prompt written).
      const edge = await stats(["cache", "--channel", CACHE_CHANNEL, "--limit", "4"]);
      expect(edge.exitCode).toBe(0);
      const edgeRows = parseTurns(edge.out, CACHE_CHANNEL);
      expect(edgeRows.map((r) => `${r.thread}/${r.seq}`)).toEqual(["a/5", "a/6", "b/1", "c/1"]);
      expect(edgeRows.some((r) => r.cold)).toBe(false);
      expect(edge.out).toContain("cold transitions 0");

      const elsewhere = await stats(["cache", "--channel", `${CACHE_CHANNEL}-nothing-here`]);
      expect(elsewhere.exitCode).toBe(0);
      expect(elsewhere.out).toContain("no assistant turns with usage");
      expect(elsewhere.out).toContain("turns 0  with cache counters 0");
      expect(elsewhere.out).toContain("mean hit n/a");
      expect(elsewhere.out).toContain("tokens over the 0 measured turns  read 0");
      expect(elsewhere.out).toContain(
        "prefix rewritten (write >= 80% of prompt) n/a (no turn reported a cache-write counter)",
      );
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "detects warm -> cold on a route that reports reads only, and says n/a for the rewrite share",
    async () => {
      const { out, exitCode } = await stats(["cache", "--channel", OPENAI_CHANNEL]);
      expect(exitCode).toBe(0);

      const rows = parseTurns(out, OPENAI_CHANNEL);
      expect(rows.map((r) => `${r.thread}/${r.seq}`)).toEqual(["d/2", "d/3", "d/4"]);
      // No write counter anywhere: the column is "nobody counted", not 0.
      expect(rows.map((r) => r.write)).toEqual(["-", "-", "-"]);
      // A read counter of 0 IS a measurement, so the hit share is 0%, not n/a.
      expect(rows.map((r) => r.hit)).toEqual(["98%", "97%", "0%"]);
      expect(rows.map((r) => r.prompt)).toEqual([5_100, 5_150, 5_400]);

      // The regression this test exists for: the marker used to require
      // `cacheCreation > 0`, which this route never sends — so the cliff at
      // d/4 was invisible and the summary printed a reassuring 0.
      expect(rows.filter((r) => r.cold).map((r) => `${r.thread}/${r.seq}`)).toEqual(["d/4"]);
      expect(out).toContain("cold transitions 1");
      expect(out).not.toContain("cold transitions 0");

      // (0.9804 + 0.9709 + 0) / 3 = 65%, over all three turns: each one
      // reported a cache counter.
      expect(out).toContain("turns 3  with cache counters 3");
      expect(out).toContain("mean hit 65%");
      expect(out).toContain(
        "tokens over the 3 measured turns  read 10000  write 0  uncached 5650  prompt 15650",
      );
      // Nothing reported a write, so the share has no denominator to be a
      // fraction of — printing "0/3 (0%)" would read as "no prefix rewrites".
      expect(out).toContain(
        "prefix rewritten (write >= 80% of prompt) n/a (no turn reported a cache-write counter)",
      );
      expect(out).not.toContain("prefix rewritten (write >= 80% of prompt) 0/");
    },
    TEST_TIMEOUT_MS,
  );
});
