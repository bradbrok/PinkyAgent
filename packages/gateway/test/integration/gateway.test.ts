/**
 * Gateway end-to-end over a real socket and a real event log (DESIGN.md §6:
 * persist → dedup → gate → enqueue).
 *
 * Fake Slack, everything else real: requests are signed the way Slack signs
 * them, served by Bun.serve, and the ingress/decision events are read back out
 * of Postgres through the real EventStore. The unit suite drives the same
 * handler with a fake EventSink, so the SQL underneath it — the thread row, the
 * per-thread seq, the dedup key — is only exercised here.
 *
 * Skipped unless PINKY_INTEGRATION=1. The connection comes from
 * loadEnvConfig() (DATABASE_URL) — local dev is 5544, CI is 5432.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import {
  createDb,
  EventStore,
  loadEnvConfig,
  migrate,
  type Db,
  type EnvConfig,
  type ThreadRef,
} from "@pinky/core";
import { LocalMessenger } from "@pinky/runtime";
import { createGateway, type RawIngress } from "../../src/server";
import { signSlackRequest } from "../../src/slack/verify";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const DB_URL = loadEnvConfig().databaseUrl;
const SCHEMA_DIR = new URL("../../../core/schema", import.meta.url).pathname;

const RUN = crypto.randomUUID().slice(0, 8);
const TENANT = `it-gw-${RUN}`;
const BOT = "UBOT";
const SIGNING_SECRET = "integration-slack-signing-secret";
/** LaneQueue debounces a same-thread burst for 500ms before the single run. */
const DEBOUNCE_MS = 500;

async function until(check: () => boolean | Promise<boolean>, ms = 5000): Promise<void> {
  const deadline = Date.now() + ms;
  for (;;) {
    if (await check()) return;
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await Bun.sleep(15);
  }
}

interface SlackEventOpts {
  eventId: string;
  channel: string;
  ts: string;
  threadTs?: string;
  text: string;
  user?: string;
}

function slackBody(o: SlackEventOpts): string {
  return JSON.stringify({
    type: "event_callback",
    event_id: o.eventId,
    event: {
      type: "message",
      channel: o.channel,
      ts: o.ts,
      ...(o.threadTs === undefined ? {} : { thread_ts: o.threadTs }),
      user: o.user ?? "U0HUMAN",
      text: o.text,
    },
  });
}

suite("gateway end-to-end (live postgres + live HTTP)", () => {
  let db: Db;
  let events: EventStore;
  let server: ReturnType<typeof Bun.serve>;
  let base = "";
  const runs: { thread: ThreadRef; batch: RawIngress[] }[] = [];

  beforeAll(async () => {
    db = createDb(DB_URL, { max: 6 });
    await migrate(db, SCHEMA_DIR);
    await purge();
    events = new EventStore(db);

    // Spread the loaded config so this keeps compiling as EnvConfig grows;
    // only the fields the gateway actually reads are overridden.
    const env: EnvConfig = {
      ...loadEnvConfig(),
      a2aSecret: "integration-a2a-secret",
      slack: { botToken: "xoxb-integration", signingSecret: SIGNING_SECRET },
      port: 0,
    };

    server = Bun.serve({
      port: 0,
      fetch: createGateway({
        env,
        tenantId: TENANT,
        events,
        messenger: new LocalMessenger(db, { nodeId: `gw-${RUN}`, peers: {}, a2aSecret: "" }),
        botUserId: BOT,
        runAgent: async (thread, batch) => {
          runs.push({ thread, batch });
        },
      }),
    });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(async () => {
    server?.stop(true);
    if (db) {
      await purge();
      await db.close();
    }
  });

  /** Scoped delete — this run's tenant plus anything an earlier run left. */
  async function purge(): Promise<void> {
    await db.query(`delete from events where tenant_id like 'it-gw-%'`);
    await db.query(`delete from threads where tenant_id like 'it-gw-%'`);
    await db.query(`delete from ingress_dedup where tenant_id like 'it-gw-%'`);
  }

  /** POST a Slack event, signed the way Slack signs it. */
  function post(body: string, opts?: { sign?: boolean }): Promise<Response> {
    const ts = String(Math.floor(Date.now() / 1000));
    return fetch(`${base}/slack/events`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-slack-request-timestamp": ts,
        "x-slack-signature":
          opts?.sign === false ? "v0=deadbeef" : signSlackRequest(SIGNING_SECRET, ts, body),
      },
      body,
    });
  }

  const ref = (channel: string, threadId: string): ThreadRef => ({
    tenantId: TENANT,
    channelId: `slack:${channel}`,
    threadId,
  });

  it("serves /healthz", async () => {
    const res = await fetch(`${base}/healthz`);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("ok");
  });

  it("a signed mention lands an ingress + decision pair in the events table", async () => {
    const channel = `C${RUN}a`;
    const ts = "1700000000.000100";
    const res = await post(
      slackBody({ eventId: `Ev-${RUN}-a1`, channel, ts, text: `hey <@${BOT}> ship it` }),
    );
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });

    const log = await events.history(ref(channel, ts));
    expect(log.map((e) => e.data.type)).toEqual(["ingress", "decision"]);

    const ingress = log[0]!.data;
    if (ingress.type !== "ingress") throw new Error("expected ingress");
    expect(ingress.platform).toBe("slack");
    expect(ingress.externalId).toBe(`Ev-${RUN}-a1`);
    // The bot's own mention token is stripped before the text is journaled.
    expect(ingress.text).toBe("hey ship it");
    expect(ingress.author).toEqual({ platform: "slack", userId: "U0HUMAN" });

    const decision = log[1]!.data;
    if (decision.type !== "decision") throw new Error("expected decision");
    expect(decision.action).toBe("reply");
    expect(decision.reason).toBe("mention");

    // The thread row was created by the append path, keyed by the ts.
    const thread = await db.queryOne(
      `select thread_id from threads where tenant_id = $1 and channel_id = $2`,
      [TENANT, `slack:${channel}`],
    );
    expect(thread).toEqual({ thread_id: ts });
  });

  it("the engaged message produces exactly one debounced agent run", async () => {
    const channel = `C${RUN}a`;
    const mine = () => runs.filter((r) => r.thread.channelId === `slack:${channel}`);
    await until(() => mine().length === 1);
    expect(mine()[0]!.batch.map((i) => i.text)).toEqual(["hey ship it"]);
    expect(mine()[0]!.thread).toEqual(ref(channel, "1700000000.000100"));
  });

  it("re-POSTing the same event_id is acked and writes nothing new", async () => {
    const channel = `C${RUN}a`;
    const ts = "1700000000.000100";
    const mine = () => runs.filter((r) => r.thread.channelId === `slack:${channel}`);
    const before = await events.history(ref(channel, ts));

    const res = await post(
      slackBody({ eventId: `Ev-${RUN}-a1`, channel, ts, text: `hey <@${BOT}> ship it` }),
    );
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });

    // Same rows, same ids: the retry was recognised before anything was written.
    const after = await events.history(ref(channel, ts));
    expect(after.map((e) => e.id)).toEqual(before.map((e) => e.id));

    // And it does not schedule a second agent run.
    await Bun.sleep(DEBOUNCE_MS + 200);
    expect(mine()).toHaveLength(1);
  });

  it("a 3-message burst on one thread debounces into exactly ONE agent run", async () => {
    const channel = `C${RUN}b`;
    const root = "1700000000.000200";
    for (let i = 1; i <= 3; i++) {
      const res = await post(
        slackBody({
          eventId: `Ev-${RUN}-b${i}`,
          channel,
          ts: `1700000000.00020${i}`,
          threadTs: root,
          text: `<@${BOT}> part ${i}`,
        }),
      );
      expect(res.status).toBe(200);
    }

    const mine = () => runs.filter((r) => r.thread.channelId === `slack:${channel}`);
    await until(() => mine().length > 0);
    // Let a second debounce window elapse: a per-message run would show up here.
    await Bun.sleep(DEBOUNCE_MS + 300);

    expect(mine()).toHaveLength(1);
    const run = mine()[0]!;
    expect(run.thread).toEqual(ref(channel, root));
    expect(run.batch.map((i) => i.text)).toEqual(["part 1", "part 2", "part 3"]);

    // All six events are already in the log by the time the run is invoked —
    // that is what lets one run answer the whole burst.
    const log = await events.history(ref(channel, root));
    expect(log.map((e) => e.data.type)).toEqual([
      "ingress",
      "decision",
      "ingress",
      "decision",
      "ingress",
      "decision",
    ]);
    expect(log.map((e) => Number(e.seq))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("an ambient message is journaled as a silent decision and starts no run", async () => {
    const channel = `C${RUN}c`;
    const ts = "1700000000.000300";
    const res = await post(
      slackBody({ eventId: `Ev-${RUN}-c1`, channel, ts, text: "just chatting, no mention" }),
    );
    expect(res.status).toBe(200);

    const log = await events.history(ref(channel, ts));
    const decision = log[1]!.data;
    if (decision.type !== "decision") throw new Error("expected decision");
    expect(decision.action).toBe("silent");
    expect(decision.reason).toBe("ambient");

    await Bun.sleep(DEBOUNCE_MS + 200);
    expect(runs.filter((r) => r.thread.channelId === `slack:${channel}`)).toHaveLength(0);
  });

  it("a badly signed request is 401 and never reaches the log", async () => {
    const channel = `C${RUN}d`;
    const ts = "1700000000.000400";
    const res = await post(
      slackBody({ eventId: `Ev-${RUN}-d1`, channel, ts, text: `<@${BOT}> forged` }),
      { sign: false },
    );
    expect(res.status).toBe(401);

    expect(await events.history(ref(channel, ts))).toHaveLength(0);
    // The dedup id was never claimed either, so a legitimate retry still works.
    const claimed = await db.query(
      `select 1 from ingress_dedup where tenant_id = $1 and external_id = $2`,
      [TENANT, `Ev-${RUN}-d1`],
    );
    expect(claimed).toHaveLength(0);
  });
});
