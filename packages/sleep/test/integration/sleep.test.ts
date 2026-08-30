/**
 * The sleep-time worker against a live database (DESIGN.md §5.3 item 3, slice 6).
 *
 * The unit suite drives every pass through a FakeDb that records SQL text, so
 * it proves the SHAPE of the statements and nothing about whether Postgres
 * accepts them, nor about what happens when two of them run at once. Everything
 * that only exists on the server is therefore untested until it runs here:
 *
 *   - discovery's lateral-over-`threads` query: the idle gate's
 *     `now - (idleMs * interval '1 millisecond')` arithmetic, the
 *     `data->>'phase' = 'extract'` cursor probe, `type = any($n::text[])`, the
 *     `channel_id not like 'sleep:%'` exclusion, and `seq`/`to_seq` coming back
 *     as int8 STRINGS that have to be coerced (the same bug class as
 *     event-store's `toSeq`);
 *   - `memory.bind(tx)` really writing inside the caller's transaction, which
 *     is the whole safety argument: the rows and the receipt that says they
 *     exist commit together or not at all;
 *   - `EventStore.lockThreadTx` really serializing two passes on one thread, so
 *     the loser's cursor re-read sees the winner's receipt;
 *   - `MemoryStore.since()`'s truncated tuple watermark, whose failure mode
 *     (a batch that is re-read forever) is invisible against a fake clock;
 *   - the visibility rules, which are the §5.1 privacy boundary and are decided
 *     by rows the query returned rather than by anything in the code path.
 *
 * Skipped unless PINKY_INTEGRATION=1. Connections come from loadEnvConfig(),
 * never a literal port (local 5544/5545, CI 5432). Reads and writes go through
 * the unprivileged `pinky_app` role — derived from DATABASE_URL the way
 * core/test/integration/memory.test.ts does it — because a superuser bypasses
 * RLS and the worker writes memories; the admin handle is used only for
 * migrate, cleanup, and looking at columns the store never selects (`user_id`,
 * `valid_to`, raw `meta`).
 *
 * NOTHING HERE MAY DEPEND ON PGVECTOR. The worker is built with no embedder on
 * purpose: both CI images run this file, and the FTS-only branch is the one a
 * stock local checkout (postgres:16-alpine) executes anyway.
 *
 * Every row this file writes carries a run-unique tenant id starting
 * `it-sleep-`, so cleanup is a scoped DELETE and two concurrent runs cannot see
 * each other.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import {
  DEFAULT_SETTINGS,
  EventStore,
  MemoryStore,
  createDb,
  loadEnvConfig,
  migrate,
  withTenant,
} from "@pinky/core";
import type { Db, MemoryRow, ThreadEvent, ThreadEventData, ThreadRef } from "@pinky/core";
import { FAKE_SLEEP_REFLECT_PREFIX, FakeProvider, createFakeProvider } from "@pinky/runtime";
import type { AssistantTurn, CompleteOptions, LlmMessage, Provider } from "@pinky/runtime";
import { discoverDueThreads } from "../../src/discovery";
import { EXTRACT_META_SOURCE, runExtractPass } from "../../src/extract";
import { REFLECT_SOURCE, runReflectPass } from "../../src/reflect";
import { DECIDE_TOOL_NAME, EXTRACT_TOOL_NAME, REFLECT_TOOL_NAME } from "../../src/schemas";
import { reflectThread } from "../../src/types";
import type {
  ExtractReceipt,
  ReflectReceipt,
  SleepDeps,
  SleepScope,
  SleepSettings,
} from "../../src/types";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

/** DATABASE_URL with pinky_app's dev credentials substituted (0003_rls.sql,
 *  docker-compose.yml). Host, port and database are kept, so a stock checkout
 *  whose DATABASE_URL is still the superuser tests the RLS path anyway. */
function asAppRole(url: string): string {
  const u = new URL(url);
  u.username = "pinky_app";
  u.password = "pinky";
  return u.toString();
}

const ENV = loadEnvConfig();
const ADMIN_URL = ENV.databaseAdminUrl;
const APP_URL = process.env.PINKY_TEST_APP_URL ?? asAppRole(ENV.databaseUrl);
const SCHEMA_DIR = new URL("../../../core/schema", import.meta.url).pathname;

const PREFIX = "it-sleep-";
const RUN = crypto.randomUUID().slice(0, 8);
const TENANT = `${PREFIX}${RUN}`;

/** Channels are per-group so one group's threads cannot appear in another's
 *  discovery result and turn an exact-set assertion into a flake. */
const chan = (name: string): string => `it:${name}-${RUN}`;
/** Agent ids likewise: every memory read goes through `agent_id = $n`, so a
 *  distinct agent per group is what lets a test assert on an exact row set. */
const agent = (name: string): string => `sleep-${name}-${RUN}`;

const ref = (channelId: string, threadId: string): ThreadRef => ({
  tenantId: TENANT,
  channelId,
  threadId,
});

function ingress(text: string, userId = "brad"): ThreadEventData {
  return { type: "ingress", platform: "cli", author: { platform: "cli", userId }, text, refs: [] };
}

/** An `extract` receipt as a previous pass would have left it. Only `toSeq`
 *  and `phase` are read back (by discovery and by readExtractCursor); the rest
 *  is filled in so the event is a real one rather than a shape the union
 *  happens to allow. */
function extractReceipt(fromSeq: number, toSeq: number): ThreadEventData {
  return {
    type: "sleep",
    phase: "extract",
    fromSeq,
    toSeq,
    scanned: 1,
    candidates: 0,
    added: 0,
    updated: 0,
    invalidated: 0,
    noop: 0,
    model: "fake/sleep",
    ms: 1,
  };
}

/** The audit events the worker must NOT consider extractable material. */
const RESTART_EVENT: ThreadEventData = {
  type: "restart",
  boundarySeq: 1,
  tokensBefore: 10,
  tokensAfter: 5,
  recallTokens: 0,
  messages: 1,
};
const MEMORY_EVENT: ThreadEventData = {
  type: "memory",
  op: "retain",
  ids: ["00000000-0000-0000-0000-000000000000"],
  text: "audit only",
};
/** `error` left EXTRACT_EVENT_TYPES in the fix wave, so it belongs in the
 *  audit-only set here. It is also the retry path: a failed pass journals one,
 *  which must push the thread out of the idle window WITHOUT by itself making
 *  it due — the material the pass choked on is still there to make it due. */
const ERROR_EVENT: ThreadEventData = {
  type: "error",
  source: "sleep",
  message: "a previous pass failed",
  count: 1,
};

// ---------------------------------------------------------------------------
// Provider helpers
// ---------------------------------------------------------------------------

/** Newest user-role message — the payload every worker call puts its input in. */
function lastUserText(messages: LlmMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "user") return m.text;
  }
  return "";
}

function call(id: string, name: string, args: Record<string, unknown>): AssistantTurn {
  return { text: "", toolCalls: [{ id, name, args }], stopReason: "tool_calls" };
}

const UNEXPECTED: AssistantTurn = { text: "unexpected call", toolCalls: [], stopReason: "stop" };

/**
 * A provider that proposes exactly `candidates` and then ADDs every one of
 * them.
 *
 * `fake/sleep` can only ever emit `visibility: "channel"` with no userId, so
 * the §5.1 downgrade rules — which are the interesting part of extraction —
 * are unreachable through it. Hand-written scripts are the only way in.
 */
function proposing(candidates: Record<string, unknown>[]): FakeProvider {
  return new FakeProvider((messages: LlmMessage[], opts: CompleteOptions) => {
    switch (opts.tools[0]?.name) {
      case EXTRACT_TOOL_NAME:
        return call("x", EXTRACT_TOOL_NAME, { candidates });
      case DECIDE_TOOL_NAME: {
        const payload = JSON.parse(lastUserText(messages)) as {
          candidates: { index: number }[];
        };
        return call("d", DECIDE_TOOL_NAME, {
          decisions: payload.candidates.map((c) => ({ candidate: c.index, action: "ADD" })),
        });
      }
      default:
        return UNEXPECTED;
    }
  });
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

let admin: Db;
/** The raw app-role pool. Closed once in afterAll; `appTenant` wraps it. */
let app: Db;
/** withTenant-wrapped: what the worker is handed, so RLS has its GUC. */
let appTenant: Db;
let events: EventStore;
let memory: MemoryStore;

interface DepsOptions {
  provider: Provider;
  agentId: string;
  settings?: Partial<SleepSettings>;
  scope?: Partial<SleepScope>;
  /** Collector for `deps.log`; the worker's only output channel. */
  logs?: string[];
}

/** Real deps over the live pool. No embedder on purpose — see the file header. */
function makeDeps(o: DepsOptions): SleepDeps {
  const logs = o.logs;
  return {
    db: appTenant,
    events,
    memory,
    provider: o.provider,
    model: "fake/sleep",
    agentId: o.agentId,
    tenantId: TENANT,
    settings: { ...DEFAULT_SETTINGS.sleep, ...o.settings },
    scope: { includeUser: true, includePrivate: true, ...o.scope },
    log: (msg: string): void => {
      if (logs) logs.push(msg);
    },
  };
}

// ---------------------------------------------------------------------------
// Inspection helpers (admin handle: RLS-bypassing, and it can read columns the
// store never selects)
// ---------------------------------------------------------------------------

interface RawMemory {
  id: string;
  visibility: string;
  user_id: string | null;
  channel_id: string | null;
  kind: string;
  text: string;
  importance: number;
  valid_to: Date | null;
  meta: Record<string, unknown>;
}

/** Every row this agent owns, sorted ON THE JS SIDE (the pgvector image is
 *  glibc en_US and alpine is C, so `order by text` disagrees with `.sort()`
 *  for some run ids — a real CI flake, see CLAUDE.md testing conventions). */
async function rowsFor(agentId: string): Promise<RawMemory[]> {
  const rows = await admin.query<RawMemory>(
    `select id, visibility, user_id, channel_id, kind, text, importance, valid_to, meta
       from memories where tenant_id = $1 and agent_id = $2`,
    [TENANT, agentId],
  );
  return [...rows].sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0));
}

const current = (rows: RawMemory[]): RawMemory[] => rows.filter((r) => r.valid_to === null);

function extractReceiptsOf(evts: ThreadEvent[]): ExtractReceipt[] {
  const out: ExtractReceipt[] = [];
  for (const e of evts) {
    if (e.data.type === "sleep" && e.data.phase === "extract") out.push(e.data);
  }
  return out;
}

function reflectReceiptsOf(evts: ThreadEvent[]): ReflectReceipt[] {
  const out: ReflectReceipt[] = [];
  for (const e of evts) {
    if (e.data.type === "sleep" && e.data.phase === "reflect") out.push(e.data);
  }
  return out;
}

const typesOf = (evts: ThreadEvent[]): string[] => evts.map((e) => e.data.type);

// ---------------------------------------------------------------------------

suite("sleep worker (live postgres)", () => {
  const purge = async (): Promise<void> => {
    // Superuser: bypasses RLS, so one statement clears every tenant this file
    // may have created (including a crashed previous run).
    await admin.query(`delete from memories where tenant_id like $1`, [`${PREFIX}%`]);
    await admin.query(`delete from events where tenant_id like $1`, [`${PREFIX}%`]);
    await admin.query(`delete from threads where tenant_id like $1`, [`${PREFIX}%`]);
    await admin.query(`delete from ingress_dedup where tenant_id like $1`, [`${PREFIX}%`]);
  };

  beforeAll(async () => {
    admin = createDb(ADMIN_URL, { max: 2 });
    await migrate(admin, SCHEMA_DIR);
    await purge();
    // max >= the widest concurrent fan-out below (two racing passes, each of
    // which also runs standalone statements in their own transactions), or the
    // race just queues in the client pool and proves nothing about the lock.
    app = createDb(APP_URL, { max: 10 });
    appTenant = withTenant(app, TENANT);
    events = new EventStore(appTenant);
    memory = new MemoryStore(appTenant, TENANT);
  }, 120_000);

  afterAll(async () => {
    // Ordered so a failure in one close cannot skip the cleanup: purge first,
    // then the pools.
    if (admin) await purge().catch(() => undefined);
    if (app) await app.close().catch(() => undefined);
    if (admin) await admin.close().catch(() => undefined);
  }, 60_000);

  it("runs as pinky_app, so the memories RLS policy actually applies", async () => {
    const who = await app.queryOne<{ role: string; superuser: boolean; bypassrls: boolean }>(
      `select current_user as role, rolsuper as superuser, rolbypassrls as bypassrls
         from pg_roles where rolname = current_user`,
    );
    expect(who?.role).toBe("pinky_app");
    expect(who?.superuser).toBe(false);
    expect(who?.bypassrls).toBe(false);
  });

  // =========================================================================
  describe("discoverDueThreads: the due query", () => {
    const CH = chan("disc");
    const CH_AUDIT = chan("disc-audit");
    const CH_CURSOR = chan("disc-cursor");
    const CH_SLEEP = `sleep:${agent("disc")}`;

    const dueA = ref(CH, "due-a");
    const dueB = ref(CH, "due-b");
    const audited = ref(CH_AUDIT, "audit");
    const resumed = ref(CH_CURSOR, "resumed");
    const workerOwn = ref(CH_SLEEP, "reflect");

    /** A clock far enough past every seeded event that the idle gate is open
     *  whatever the server's clock offset is. */
    const wayLater = (): Date => new Date(Date.now() + 86_400_000);

    beforeAll(async () => {
      // Appended in order and one transaction each, so `last.ts` is strictly
      // increasing across threads — which is what the ordering assertion reads.
      await events.append(dueA, ingress("first thread, oldest event"));
      await new Promise((r) => setTimeout(r, 10));
      await events.append(dueB, ingress("second thread, newer event"));

      await events.appendBatch(resumed, [
        ingress("already extracted"),
        extractReceipt(1, 1),
        ingress("new material after the cursor"),
      ]);

      await events.append(workerOwn, ingress("the worker's own bookkeeping"));
    }, 60_000);

    /** The query under test, with this file's tenant and an open idle gate
     *  unless the case says otherwise. */
    const due = (over: {
      idleMs?: number;
      limit?: number;
      channelId?: string;
      threadId?: string;
      now?: Date;
    } = {}) =>
      discoverDueThreads(appTenant, {
        tenantId: TENANT,
        idleMs: over.idleMs ?? 0,
        limit: over.limit ?? 100,
        now: over.now ?? wayLater(),
        ...(over.channelId !== undefined ? { channelId: over.channelId } : {}),
        ...(over.threadId !== undefined ? { threadId: over.threadId } : {}),
      });

    it("the idle gate: a thread is due only once its newest event is idleMs old", async () => {
      // An hour of idleness demanded, none elapsed.
      const fresh = await due({ idleMs: 3_600_000, now: new Date(), channelId: CH });
      expect(fresh).toEqual([]);

      // The same query with the clock moved past the gate.
      const aged = await due({
        idleMs: 3_600_000,
        now: new Date(Date.now() + 3_600_000 + 60_000),
        channelId: CH,
      });
      expect(aged.map((d) => d.thread.threadId).sort()).toEqual(["due-a", "due-b"]);
    });

    it("returns oldest-idle first", async () => {
      const rows = await due({ channelId: CH });
      // NOT sorted here: the order IS the assertion (`order by last.ts asc`).
      expect(rows.map((d) => d.thread.threadId)).toEqual(["due-a", "due-b"]);
    });

    it("channelId and threadId narrow the sweep", async () => {
      const oneChannel = await due({ channelId: CH });
      expect(oneChannel.map((d) => d.thread.channelId)).toEqual([CH, CH]);

      const oneThread = await due({ channelId: CH, threadId: "due-b" });
      expect(oneThread).toHaveLength(1);
      expect(oneThread[0]!.thread).toEqual(dueB);
    });

    it("a thread whose only events after its cursor are audit-only is NOT due", async () => {
      // First: with material and no cursor, it IS due — otherwise the negative
      // below would pass for the wrong reason.
      await events.append(audited, ingress("extractable material"));
      const before = await due({ channelId: CH_AUDIT });
      expect(before.map((d) => d.thread.threadId)).toEqual(["audit"]);

      // Now cover it with a receipt and append only audit-only types after it.
      await events.appendBatch(audited, [
        extractReceipt(1, 1),
        RESTART_EVENT,
        MEMORY_EVENT,
        ERROR_EVENT,
      ]);
      const after = await due({ channelId: CH_AUDIT });
      expect(after).toEqual([]);

      // And the exclusion really is about TYPE, not about the receipt: one
      // ingress after those audit events makes it due again.
      await events.append(audited, ingress("new material"));
      const again = await due({ channelId: CH_AUDIT });
      expect(again.map((d) => d.thread.threadId)).toEqual(["audit"]);
      // The cursor is the receipt's toSeq (1), not the newest seq (6).
      expect(again[0]!.cursorSeq).toBe(1);
      expect(again[0]!.lastSeq).toBe(6);
    });

    it("`sleep:` channels are excluded — the worker never extracts its own bookkeeping", async () => {
      const seeded = await events.history(workerOwn);
      expect(seeded).toHaveLength(1); // the thread really does have material

      const rows = await due();
      expect(rows.every((d) => !d.thread.channelId.startsWith("sleep:"))).toBe(true);
      expect(rows.some((d) => d.thread.channelId === CH_SLEEP)).toBe(false);
    });

    it("DEFECT: cursorSeq and lastSeq are NUMBERS, not int8 strings", async () => {
      // `seq` and `(data->>'toSeq')::bigint` are both bigint, and postgres.js
      // hands bigint back as a STRING to protect precision. Untouched, the
      // scheduler compares "10" < "9" lexicographically and a cursor arithmetic
      // like `cursor + 1` concatenates — the same root cause as event-store's
      // toSeq() defect. The FakeDb in the unit suite returns strings on purpose,
      // but only the live server proves the coercion sits on the real column.
      const rows = await due({ channelId: CH_CURSOR });
      expect(rows).toHaveLength(1);
      const row = rows[0]!;
      expect(typeof row.cursorSeq).toBe("number");
      expect(typeof row.lastSeq).toBe("number");
      expect(row.cursorSeq).toBe(1);
      expect(row.lastSeq).toBe(3);

      // KNOWN DEFECT, reported to the owner of discovery.ts rather than fixed
      // here: `DueThread.lastTs` is DECLARED `string` ("ISO timestamp") but
      // postgres.js hands `timestamptz` back as a Date and the mapper passes it
      // through unconverted, so `typeof lastTs === "object"` at runtime. It is
      // the same wart `ThreadEvent.ts` already carries, which is why this file
      // pins the property that actually matters — that it names a real instant —
      // instead of pinning the lie.
      expect(Number.isFinite(new Date(row.lastTs).getTime())).toBe(true);
    });

    it("a thread with no cursor reports cursorSeq 0", async () => {
      const rows = await due({ channelId: CH, threadId: "due-a" });
      expect(rows[0]!.cursorSeq).toBe(0);
      expect(rows[0]!.thread).toEqual(dueA);
    });
  });

  // =========================================================================
  describe("runExtractPass: the whole pass, end to end", () => {
    const AGENT = agent("extract");
    const CH = chan("extract");
    const thread = ref(CH, "t1");
    const FACT = "the aurora canary prefers terse answers";
    const logs: string[] = [];
    let first: ExtractReceipt;

    beforeAll(async () => {
      await events.appendBatch(thread, [
        ingress(`remember: ${FACT}`),
        ingress("and how was your afternoon"),
      ]);
      const result = await runExtractPass(
        makeDeps({ provider: createFakeProvider("sleep"), agentId: AGENT, logs }),
        thread,
      );
      if (result.status !== "done") throw new Error(`first pass ${JSON.stringify(result)}`);
      first = result.receipt;
    }, 60_000);

    it("writes the memory row with sleep:extract provenance and the channel it was learned in", async () => {
      const rows = current(await rowsFor(AGENT));
      expect(rows).toHaveLength(1);
      const row = rows[0]!;
      expect(row.text).toBe(FACT);
      expect(row.visibility).toBe("channel");
      expect(row.channel_id).toBe(CH);
      expect(row.user_id).toBeNull();
      expect(row.kind).toBe("semantic");
      expect(row.importance).toBe(7);
      expect(row.meta.source).toBe(EXTRACT_META_SOURCE);
      expect(row.meta.channelId).toBe(CH);
      expect(row.meta.threadId).toBe("t1");
      expect(row.meta.fromSeq).toBe(1);
      expect(row.meta.toSeq).toBe(2);
      // meta is a jsonb OBJECT, not a doubly-encoded string (pg.ts contract) —
      // if it were, every `meta->>` above would be null.
      const typed = await admin.queryOne<{ t: string }>(
        `select jsonb_typeof(meta) as t from memories where id = $1`,
        [row.id],
      );
      expect(typed?.t).toBe("object");
    });

    it("journals the audit memory events first and the receipt LAST", async () => {
      const evts = await events.history(thread);
      expect(typesOf(evts)).toEqual(["ingress", "ingress", "memory", "sleep"]);
      expect(evts[evts.length - 1]!.data.type).toBe("sleep");
      const retained = evts[2]!.data;
      expect(retained.type === "memory" && retained.op).toBe("retain");
    });

    it("the receipt records the range it consumed and what it did", () => {
      expect(first.fromSeq).toBe(1);
      expect(first.toSeq).toBe(2); // the LAST event read, of any type
      expect(first.scanned).toBe(2);
      expect(first.candidates).toBe(1);
      expect(first.added).toBe(1);
      expect(first.updated).toBe(0);
      expect(first.invalidated).toBe(0);
      expect(first.noop).toBe(0);
      expect(first.model).toBe("fake/sleep");
      expect(typeof first.ms).toBe("number");
    });

    it("a second pass over the same thread is skipped with no LLM call and no receipt", async () => {
      // The provider is script-exhausting (an empty array), so any call at all
      // rejects: the skip has to happen BEFORE the model is asked.
      const before = (await events.history(thread)).length;
      const result = await runExtractPass(
        makeDeps({ provider: new FakeProvider([]), agentId: AGENT, logs }),
        thread,
      );
      expect(result).toEqual({ status: "skipped", reason: "no-new-events" });
      expect((await events.history(thread)).length).toBe(before);
    });

    it("re-stating a known fact is a NOOP: a receipt, no duplicate row", async () => {
      await events.append(thread, ingress(`remember: ${FACT}`));
      const result = await runExtractPass(
        makeDeps({ provider: createFakeProvider("sleep"), agentId: AGENT, logs }),
        thread,
      );
      expect(result.status).toBe("done");
      if (result.status !== "done") return;
      const r = result.receipt;

      // fromSeq is the previous receipt's toSeq + 1, which lands ON the audit
      // events that pass wrote — NOT `lastToSeq + 2` and not the receipt's own
      // seq (that only coincides when the pass wrote no memory events).
      expect(r.fromSeq).toBe(first.toSeq + 1);
      expect(r.fromSeq).toBe(3);
      expect(r.toSeq).toBe(5);
      expect(r.scanned).toBe(1); // only the new ingress rendered
      expect(r.candidates).toBe(1);
      expect(r.noop).toBeGreaterThanOrEqual(1);
      expect(r.added).toBe(0);

      // The point of the NOOP branch: the plane did not grow.
      const rows = current(await rowsFor(AGENT));
      expect(rows).toHaveLength(1);
      expect(rows[0]!.text).toBe(FACT);

      // Exactly two receipts, both on this thread.
      expect(extractReceiptsOf(await events.history(thread))).toHaveLength(2);
    });

  });

  // =========================================================================
  describe("§5.1 visibility downgrade (hand-written scripts)", () => {
    const CH = chan("vis");

    /** One pass on a fresh thread + fresh agent with one candidate. */
    async function pass(
      name: string,
      candidate: Record<string, unknown>,
      scope: Partial<SleepScope>,
    ): Promise<RawMemory[]> {
      const AGENT = agent(`vis-${name}`);
      const thread = ref(CH, name);
      await events.append(thread, ingress("a conversation with a real author", "brad"));
      const result = await runExtractPass(
        makeDeps({ provider: proposing([candidate]), agentId: AGENT, scope }),
        thread,
      );
      if (result.status !== "done") throw new Error(`${name}: ${JSON.stringify(result)}`);
      return current(await rowsFor(AGENT));
    }

    it("a `user` candidate naming somebody who never spoke is downgraded to `channel`", async () => {
      // An invented userId would write a row no scope predicate ever matches —
      // invisible forever, which is worse than wrong.
      const rows = await pass(
        "ghost",
        {
          text: "the ghost user likes espresso",
          kind: "semantic",
          importance: 6,
          visibility: "user",
          userId: "nobody-said-this",
        },
        { includeUser: true },
      );
      expect(rows).toHaveLength(1);
      expect(rows[0]!.visibility).toBe("channel");
      expect(rows[0]!.user_id).toBeNull();
      expect(rows[0]!.channel_id).toBe(CH);
    });

    it("a `user` candidate naming a real author survives on a wide surface", async () => {
      const rows = await pass(
        "real",
        {
          text: "brad likes espresso after four",
          kind: "semantic",
          importance: 6,
          visibility: "user",
          userId: "brad",
        },
        { includeUser: true },
      );
      expect(rows).toHaveLength(1);
      expect(rows[0]!.visibility).toBe("user");
      expect(rows[0]!.user_id).toBe("brad");
      // A user fact still records where it was learned (provenance, not scope).
      expect(rows[0]!.channel_id).toBe(CH);
    });

    it("the same candidate on a NARROW surface is downgraded to `channel`", async () => {
      // A shared pipe must not mint rows it could not read back; that is the
      // leak §5.1 exists to prevent.
      const rows = await pass(
        "narrow",
        {
          text: "brad likes espresso before noon",
          kind: "semantic",
          importance: 6,
          visibility: "user",
          userId: "brad",
        },
        { includeUser: false },
      );
      expect(rows).toHaveLength(1);
      expect(rows[0]!.visibility).toBe("channel");
      expect(rows[0]!.user_id).toBeNull();
    });
  });

  // =========================================================================
  describe("DEFECT: a failed pass leaves the plane and the log untouched", () => {
    it("an invalid tool call writes NO rows, NO receipt, and exactly ONE error event", async () => {
      const AGENT = agent("badcall");
      const thread = ref(chan("badcall"), "t");
      await events.append(thread, ingress("remember: something the model will fumble"));

      const logs: string[] = [];
      const result = await runExtractPass(
        makeDeps({
          // `candidates` missing entirely: parseExtract rejects, the pass throws.
          provider: new FakeProvider(() => call("bad", EXTRACT_TOOL_NAME, {})),
          agentId: AGENT,
          logs,
        }),
        thread,
      );

      expect(result.status).toBe("failed");
      expect(await rowsFor(AGENT)).toEqual([]);

      const evts = await events.history(thread);
      expect(typesOf(evts)).toEqual(["ingress", "error"]);
      const err = evts[1]!.data;
      expect(err.type === "error" && err.source).toBe("sleep");
      expect(extractReceiptsOf(evts)).toEqual([]);
      expect(logs.length).toBeGreaterThan(0);
    });

    it("a store failure mid-apply rolls back the ADDs made before it in the same pass", async () => {
      // The claim under test is invariant #1 for this slice: memory rows and the
      // receipt that says they exist commit together or not at all. It is only
      // provable against a real transaction — bind(tx) running on the CALLER's
      // handle is exactly what a FakeDb cannot show.
      const AGENT = agent("rollback");
      const CH = chan("rollback");
      const thread = ref(CH, "t");
      const KNOWN = "the beacon rotates every tuesday";
      const NEW = "pelicans nest on the north jetty";

      const target = await memory.retain({
        agentId: AGENT,
        visibility: "channel",
        channelId: CH,
        kind: "semantic",
        text: KNOWN,
        importance: 5,
      });
      await events.append(thread, ingress("a conversation"));

      interface DecidePayload {
        candidates: { index: number; neighbors: { id: string }[] }[];
      }
      /** An array, not a `let`: TypeScript does not track assignments made
       *  inside the callback below, so a nullable local would narrow to `null`
       *  at every later read. */
      const decidePayloads: DecidePayload[] = [];

      // A hand-written Provider rather than a FakeProvider script: the whole
      // point is to invalidate the target BETWEEN the neighbour search and the
      // transaction, and a FakeScript is synchronous.
      const provider: Provider = {
        name: "racing",
        async complete(opts: CompleteOptions): Promise<AssistantTurn> {
          if (opts.tools[0]?.name === EXTRACT_TOOL_NAME) {
            return call("x", EXTRACT_TOOL_NAME, {
              candidates: [
                { text: NEW, kind: "semantic", importance: 6, visibility: "channel" },
                { text: KNOWN, kind: "semantic", importance: 6, visibility: "channel" },
              ],
            });
          }
          if (opts.tools[0]?.name === DECIDE_TOOL_NAME) {
            decidePayloads.push(JSON.parse(lastUserText(opts.messages)) as DecidePayload);
            // The row was current when the search ran; it is not any more.
            await memory.invalidate(target.id, { reason: "retired by a concurrent editor" });
            return call("d", DECIDE_TOOL_NAME, {
              decisions: [
                { candidate: 0, action: "ADD" },
                { candidate: 1, action: "UPDATE", target: target.id },
              ],
            });
          }
          return UNEXPECTED;
        },
      };

      const logs: string[] = [];
      const result = await runExtractPass(
        makeDeps({ provider, agentId: AGENT, logs }),
        thread,
      );

      // Guard the guard: if FTS had not surfaced the target as a neighbour,
      // parseDecide would have failed the pass for an unrelated reason and every
      // assertion below would pass vacuously.
      expect(decidePayloads).toHaveLength(1);
      const shown = decidePayloads[0]!.candidates[1]!.neighbors.map((n) => n.id);
      expect(shown).toContain(target.id);

      expect(result.status).toBe("failed");
      if (result.status === "failed") expect(result.error).toMatch(/already invalidated/);
      // The UPDATE has to be ATTEMPTED for the rollback to be what is under
      // test. The placement guard added in the fix wave turns an UPDATE whose
      // target lives elsewhere into a logged NOOP, which would make this test
      // pass while proving nothing — so pin that it did not fire. It cannot:
      // the target is `channel` in CH and so is the candidate (no userId,
      // written at `thread.channelId`, which is CH).
      expect(logs.join("\n")).not.toMatch(/refusing to UPDATE/);

      // The ADD that ran BEFORE the throw, in the same transaction, is gone.
      const rows = await rowsFor(AGENT);
      expect(rows.map((r) => r.text)).toEqual([KNOWN]);
      expect(rows[0]!.valid_to).not.toBeNull(); // the test's own invalidation stands
      expect(current(rows)).toEqual([]);

      const evts = await events.history(thread);
      expect(typesOf(evts)).toEqual(["ingress", "error"]);
      expect(extractReceiptsOf(evts)).toEqual([]);
    });

    it("two passes racing on one thread: exactly one receipt, exactly one set of rows", async () => {
      const AGENT = agent("race");
      const thread = ref(chan("race"), "t");
      const FACT = "the harbour siren tests at noon";
      await events.appendBatch(thread, [ingress(`remember: ${FACT}`), ingress("second turn")]);

      const deps = makeDeps({ provider: createFakeProvider("sleep"), agentId: AGENT });
      const [a, b] = await Promise.all([
        runExtractPass(deps, thread),
        runExtractPass(deps, thread),
      ]);

      const statuses = [a.status, b.status].sort();
      expect(statuses).toEqual(["done", "skipped"]);
      const loser = a.status === "skipped" ? a : b.status === "skipped" ? b : null;
      expect(loser).not.toBeNull();
      // BOTH reasons are correct, and which one appears is a scheduling detail:
      //  - `lost-claim` when the loser read its (empty) cursor before the winner
      //    committed and only discovered the receipt under the thread lock;
      //  - `no-new-events` when the winner had already committed by the time the
      //    loser ran its `history()` — the receipt had moved the cursor past
      //    every extractable event, so it never reached the lock at all.
      // Either way the invariant is the same: one receipt, one set of rows.
      // (Observed locally on both server images: `done` + `lost-claim`, i.e.
      // the loser really does reach the thread lock — but that is a timing
      // observation, not something a test may depend on.)
      expect(["lost-claim", "no-new-events"]).toContain(
        loser!.status === "skipped" ? loser!.reason : "",
      );

      expect(extractReceiptsOf(await events.history(thread))).toHaveLength(1);
      const rows = current(await rowsFor(AGENT));
      expect(rows).toHaveLength(1);
      expect(rows[0]!.text).toBe(FACT);
    }, 30_000);
  });

  // =========================================================================
  describe("runReflectPass: consolidation", () => {
    it("below the threshold: skipped, no LLM call, no receipt", async () => {
      const AGENT = agent("refl-empty");
      const deps = makeDeps({
        // An empty array script rejects on the first call, so reaching the
        // provider at all fails the test rather than passing quietly.
        provider: new FakeProvider([]),
        agentId: AGENT,
        settings: { reflectMinMemories: 5 },
      });
      const result = await runReflectPass(deps, { ignoreIdle: true });
      expect(result).toEqual({ status: "skipped", reason: "below-threshold" });
      expect(await events.history(reflectThread(TENANT, AGENT))).toEqual([]);
    });

    it("an insight whose sources span two channels is DROPPED, not widened to the tenant", async () => {
      // §5.1, the rule that matters most here: channel content must never widen
      // to the tenant. `fake/sleep` sources EVERY row in the batch, so a batch
      // drawn from two channels is exactly the case the rule refuses.
      const AGENT = agent("refl-drop");
      const CH_A = chan("refl-drop-a");
      const CH_B = chan("refl-drop-b");
      const seed = async (visibility: "channel" | "tenant", channelId: string | null, text: string) =>
        memory.retain({
          agentId: AGENT,
          visibility,
          ...(channelId ? { channelId } : {}),
          kind: "semantic",
          text,
          importance: 5,
        });
      await seed("channel", CH_A, "drop a1: the ferry leaves at six");
      await seed("channel", CH_A, "drop a2: the ferry is cash only");
      await seed("channel", CH_B, "drop b1: the tram runs until midnight");
      await seed("channel", CH_B, "drop b2: the tram takes cards");
      await seed("tenant", null, "drop t1: invoices are due on the first");

      const logs: string[] = [];
      const result = await runReflectPass(
        makeDeps({
          provider: createFakeProvider("sleep"),
          agentId: AGENT,
          settings: { reflectMinMemories: 5 },
          logs,
        }),
        // The idle gate has its own test below; here it would only decide
        // whether this case runs at all.
        { ignoreIdle: true },
      );

      expect(result.status).toBe("done");
      if (result.status !== "done") return;
      const r = result.receipt;
      expect(r.after).toBeNull();
      expect(r.scanned).toBe(5);
      expect(r.candidates).toBe(1);
      expect(r.added).toBe(0);
      expect(r.noop).toBe(1);
      expect(r.invalidated).toBe(0);

      // Nothing was written: no insight row, and no audit `memory` event.
      const rows = await rowsFor(AGENT);
      expect(rows.some((m) => m.text.startsWith(FAKE_SLEEP_REFLECT_PREFIX))).toBe(false);
      expect(current(rows)).toHaveLength(5);
      const evts = await events.history(reflectThread(TENANT, AGENT));
      expect(typesOf(evts)).toEqual(["sleep"]);
      expect(logs.join("\n")).toMatch(/dropped an insight whose sources span channels/);
    }, 30_000);

    it("a single-channel insight is retained, supersedes only in its own scope, and advances the watermark", async () => {
      const AGENT = agent("refl-keep");
      const CH_A = chan("refl-keep-a");
      const seeded: MemoryRow[] = [];
      const seed = async (visibility: "channel" | "tenant", channelId: string | null, text: string) => {
        const row = await memory.retain({
          agentId: AGENT,
          visibility,
          ...(channelId ? { channelId } : {}),
          kind: "semantic",
          text,
          importance: 5,
        });
        seeded.push(row);
        return row;
      };
      const a1 = await seed("channel", CH_A, "keep a1: the loading bay closes at five");
      const a2 = await seed("channel", CH_A, "keep a2: the loading bay needs a badge");
      await seed("channel", CH_A, "keep a3: the loading bay has two lifts");
      const t1 = await seed("tenant", null, "keep t1: expenses are filed monthly");

      const logs: string[] = [];
      const provider = new FakeProvider((messages: LlmMessage[], opts: CompleteOptions) => {
        if (opts.tools[0]?.name !== REFLECT_TOOL_NAME) return UNEXPECTED;
        return call("r", REFLECT_TOOL_NAME, {
          insights: [
            {
              text: "loading bay access is badge-gated and ends at five",
              importance: 7,
              // Sources: two channel-A rows plus one tenant row => one distinct
              // channel => the insight lands in channel A.
              sources: [a1.id, a2.id, t1.id],
              // Both are its own sources (parseReflect demands that), but only
              // the channel-A one shares the insight's placement.
              supersedes: [a1.id, t1.id],
            },
          ],
        });
      });

      const deps = makeDeps({
        provider,
        agentId: AGENT,
        settings: { reflectMinMemories: 3 },
        logs,
      });
      const result = await runReflectPass(deps, { ignoreIdle: true });
      expect(result.status).toBe("done");
      if (result.status !== "done") return;
      const r = result.receipt;

      expect(r.scanned).toBe(4);
      expect(r.candidates).toBe(1);
      expect(r.added).toBe(1);
      expect(r.invalidated).toBe(1);
      expect(r.updated).toBe(0);
      expect(r.noop).toBe(0);
      expect(r.after).toBeNull();

      const rows = await rowsFor(AGENT);
      const insight = rows.find((m) => m.meta.source === REFLECT_SOURCE);
      expect(insight).toBeDefined();
      expect(insight!.visibility).toBe("channel");
      expect(insight!.channel_id).toBe(CH_A);
      expect(insight!.kind).toBe("semantic");
      expect(insight!.meta.sources).toEqual([a1.id, a2.id, t1.id]);

      const superseded = rows.find((m) => m.id === a1.id)!;
      expect(superseded.valid_to).not.toBeNull();
      expect(superseded.meta.invalidatedReason).toBe(
        `${REFLECT_SOURCE} consolidated into ${insight!.id}`,
      );

      // The tenant row was named in `supersedes` and REFUSED: retiring it
      // because a channel-scoped insight replaced it would lose it in every
      // other channel, where the insight is not readable.
      const tenantRow = rows.find((m) => m.id === t1.id)!;
      expect(tenantRow.valid_to).toBeNull();
      expect(logs.join("\n")).toMatch(/claimed to replace from outside its scope/);

      // `through` is the batch's LAST row in the server's own (recorded_at ms,
      // id) order — asked of the server rather than sorted in JS, because `id`
      // is text and its collation differs between the two CI images.
      const ordered = await admin.query<{ id: string }>(
        `select id from memories
          where tenant_id = $1 and agent_id = $2 and id = any($3::text[])
          order by date_trunc('milliseconds', recorded_at) asc, id asc`,
        [TENANT, AGENT, seeded.map((s) => s.id)],
      );
      expect(r.through.id).toBe(ordered[ordered.length - 1]!.id);

      const evts = await events.history(reflectThread(TENANT, AGENT));
      // retain, invalidate, then the receipt LAST.
      expect(typesOf(evts)).toEqual(["memory", "memory", "sleep"]);

      // A rerun sees only what this pass itself wrote (one insight), which is
      // below the threshold — i.e. the watermark really advanced.
      //
      // `ignoreIdle` is REQUIRED here, and its absence would be a silent hole:
      // the pass just journaled its receipt, so the idle gate would answer
      // `not-idle` first and this assertion would pass without the watermark
      // ever being consulted. The gate itself is tested on its own below.
      const rerun = await runReflectPass(deps, { ignoreIdle: true });
      expect(rerun).toEqual({ status: "skipped", reason: "below-threshold" });
      expect(reflectReceiptsOf(await events.history(reflectThread(TENANT, AGENT)))).toHaveLength(1);
    }, 30_000);

    it("§5.1: `user` and `private` rows never reach the reflect payload", async () => {
      const AGENT = agent("refl-priv");
      const CH = chan("refl-priv");
      const mk = async (
        visibility: "tenant" | "user" | "private",
        text: string,
        extra: { userId?: string; channelId?: string } = {},
      ) =>
        memory.retain({
          agentId: AGENT,
          visibility,
          ...extra,
          kind: "semantic",
          text,
          importance: 5,
        });
      const shared = [
        await mk("tenant", "priv s1: the office wifi rotates quarterly"),
        await mk("tenant", "priv s2: releases ship on thursdays"),
        await mk("tenant", "priv s3: the retro is every other week"),
      ];
      const personal = await mk("user", "priv u1: brad's home address is on file", {
        userId: "brad",
        channelId: CH,
      });
      const scratch = await mk("private", "priv p1: my own scratch note about brad");

      const provider = new FakeProvider((_messages: LlmMessage[], opts: CompleteOptions) => {
        if (opts.tools[0]?.name !== REFLECT_TOOL_NAME) return UNEXPECTED;
        return call("r", REFLECT_TOOL_NAME, { insights: [] });
      });
      const result = await runReflectPass(
        makeDeps({
          provider,
          agentId: AGENT,
          // Deliberately the WIDEST surface: the narrowing is the pass's own,
          // not the caller's, so a trusted CLI cannot widen it back.
          scope: { includeUser: true, includePrivate: true },
          settings: { reflectMinMemories: 3 },
        }),
        { ignoreIdle: true },
      );

      expect(result.status).toBe("done");
      if (result.status !== "done") return;
      expect(result.receipt.scanned).toBe(3);
      expect(result.receipt.candidates).toBe(0);
      expect(result.receipt.added).toBe(0);

      expect(provider.received).toHaveLength(1);
      const payload = lastUserText(provider.received[0]!.messages);
      for (const row of shared) expect(payload).toContain(row.id);
      for (const hidden of [personal, scratch]) {
        expect(payload).not.toContain(hidden.id);
        expect(payload).not.toContain(hidden.text);
      }
    }, 30_000);

    it("the idle gate: a pass that just ran is refused, and an untouched thread is not", async () => {
      // The gate lives on the worker's OWN thread and is the reflect pass's only
      // backoff: a receipt (or an `error`) makes the newest event fresh, so the
      // next sweep is refused for `idleMs`. Both halves matter — a gate that
      // never opens would wedge consolidation forever, and one that never closes
      // would let a dead provider burn two LLM calls every tick.
      const AGENT = agent("refl-gate");
      const CH = chan("refl-gate");
      for (const text of ["gate g1: the depot opens at seven", "gate g2: the depot shuts at four"]) {
        await memory.retain({
          agentId: AGENT,
          visibility: "channel",
          channelId: CH,
          kind: "semantic",
          text,
          importance: 5,
        });
      }
      const deps = makeDeps({
        provider: createFakeProvider("sleep"),
        agentId: AGENT,
        // The default idleMs (600_000) is what the gate compares against; only
        // the threshold is lowered so two seeded rows are a batch.
        settings: { reflectMinMemories: 2 },
      });
      expect(deps.settings.idleMs).toBeGreaterThan(0);

      // FIRST pass, gate ENGAGED: the reflect thread has no events at all, and a
      // thread nobody has ever written to is idle by definition — so the gate
      // opens and the pass runs.
      const first = await runReflectPass(deps);
      expect(first.status).toBe("done");
      if (first.status !== "done") return;
      expect(first.receipt.added).toBe(1);

      // SECOND pass, gate ENGAGED: the receipt above is now the newest event on
      // the thread and it is seconds old, so the pass is refused BEFORE the
      // watermark, the batch query or the provider are consulted.
      const gated = await runReflectPass(deps);
      expect(gated).toEqual({ status: "skipped", reason: "not-idle" });
      // Refused means refused: nothing appended, nothing written.
      expect(reflectReceiptsOf(await events.history(reflectThread(TENANT, AGENT)))).toHaveLength(1);

      // ...and `ignoreIdle` is the documented bypass (`pinky sleep run --now`,
      // smoke), which lands on the real answer for this state instead.
      const bypassed = await runReflectPass(deps, { ignoreIdle: true });
      expect(bypassed).toEqual({ status: "skipped", reason: "below-threshold" });
    }, 30_000);
  });

  // =========================================================================
  describe("MemoryStore.since: the tuple watermark", () => {
    it("DEFECT: rows retained in ONE transaction do not re-appear after the last of them", async () => {
      // The failure this guards is a permanent one, and invisible without a real
      // server: `recorded_at` is timestamptz (microseconds) but postgres.js
      // parses it with `new Date(text)`, which DROPS the sub-millisecond digits.
      // Compared against the raw column, the watermark a caller hands back is
      // strictly LESS than the row it came from, so that row — and, because
      // `recorded_at` defaults to the TRANSACTION's start time, every sibling
      // retained with it — comes back on every pass, forever. The reflect
      // watermark would never advance and the same batch would be consolidated
      // again and again.
      const AGENT = agent("watermark");
      const scope = {
        agentId: AGENT,
        allChannels: true,
        includeUser: false,
        includePrivate: false,
      };

      // Both rows in ONE transaction: they therefore share recorded_at exactly.
      const written = await appTenant.tx(async (tx) => {
        const bound = memory.bind(tx);
        const one = await bound.retain({
          agentId: AGENT,
          visibility: "tenant",
          kind: "semantic",
          text: "watermark row one",
          importance: 5,
        });
        const two = await bound.retain({
          agentId: AGENT,
          visibility: "tenant",
          kind: "semantic",
          text: "watermark row two",
          importance: 5,
        });
        return [one, two];
      });
      expect(written[0]!.recordedAt).toBe(written[1]!.recordedAt);

      const batch = await memory.since({ scope, after: null, limit: 50 });
      expect(batch).toHaveLength(2);
      const last = batch[batch.length - 1]!;

      const next = await memory.since({
        scope,
        after: { recordedAt: last.recordedAt, id: last.id },
        limit: 50,
      });
      expect(next).toEqual([]);
    });

    it("visibilities narrows the scope, and an EMPTY array is refused", async () => {
      const AGENT = agent("since-vis");
      const CH = chan("since-vis");
      await memory.retain({
        agentId: AGENT,
        visibility: "tenant",
        kind: "semantic",
        text: "since vis: a tenant row",
        importance: 5,
      });
      await memory.retain({
        agentId: AGENT,
        visibility: "channel",
        channelId: CH,
        kind: "semantic",
        text: "since vis: a channel row",
        importance: 5,
      });
      const scope = {
        agentId: AGENT,
        allChannels: true,
        includeUser: false,
        includePrivate: false,
      };

      // allChannels is the worker-only read arm: without it the `channel` row
      // is unreachable, because the reflect scope names no channelId at all.
      const all = await memory.since({ scope, after: null, limit: 50 });
      expect(all).toHaveLength(2);
      const narrowed = await memory.since({
        scope,
        after: null,
        limit: 50,
        visibilities: ["tenant"],
      });
      expect(narrowed.map((r) => r.visibility)).toEqual(["tenant"]);

      // "No restriction" is the ABSENT field; an empty array would silently
      // match nothing, which reads as "the plane is empty".
      await expect(
        memory.since({ scope, after: null, limit: 50, visibilities: [] }),
      ).rejects.toThrow(/at least one visibility/);
    });
  });

  // =========================================================================
  describe("EventStore.lockThreadTx", () => {
    it("serializes two transactions on the same thread", async () => {
      // This is what makes the pass's cursor re-read meaningful: the loser
      // cannot look at the cursor until the winner has committed its receipt.
      const thread = ref(chan("lock"), "t");
      await events.append(thread, ingress("seed so the threads row exists"));

      const marks: { name: string; start: number; end: number }[] = [];
      const hold = async (name: string): Promise<void> => {
        await app.tx(async (tx) => {
          await EventStore.lockThreadTx(tx, thread);
          const start = Date.now();
          await new Promise((r) => setTimeout(r, 250));
          marks.push({ name, start, end: Date.now() });
        });
      };

      await Promise.all([hold("a"), hold("b")]);
      expect(marks).toHaveLength(2);
      const [first, second] = [...marks].sort((x, y) => x.start - y.start);
      // The second transaction cannot have entered the critical section before
      // the first left it. A small slack absorbs Date.now() resolution only.
      expect(second!.start).toBeGreaterThanOrEqual(first!.end - 20);
    }, 30_000);
  });
});
