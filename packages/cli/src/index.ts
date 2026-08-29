#!/usr/bin/env bun
/**
 * pinky — PinkyAgent CLI (the human-owned control surface; agents never get it).
 *
 *   pinky migrate                          apply schema migrations
 *   pinky config set <key> <value> [--scope global|channel:<id>|agent:<id>]
 *   pinky config get [key] [--scope ...]   read effective settings
 *   pinky gateway                          run the Slack + A2A HTTP gateway
 *   pinky smoke                            end-to-end in-process smoke (FakeProvider, A2A)
 *   pinky prompt "<text>"                  run one agent turn against a local cli thread
 *
 * Two database privilege levels (DESIGN.md §5.1, .env.example): migrations run
 * on DATABASE_ADMIN_URL (DDL + CREATE ROLE), everything else on DATABASE_URL,
 * which should be the NOBYPASSRLS `pinky_app` role so row-level security is
 * actually enforced. Commands that auto-migrate therefore open a short-lived
 * admin handle for that step and close it before opening the app connection.
 */
import {
  loadEnvConfig,
  assertGatewaySecrets,
  createDb,
  migrate,
  withTenant,
  EventStore,
  SettingsStore,
  assertScope,
  threadKey,
  type Db,
  type EnvConfig,
  type LoadOptions,
  type SettingsSnapshot,
  type ThreadRef,
} from "@pinky/core";
import {
  createProvider,
  FakeProvider,
  LocalMessenger,
  runAgentLoop,
  buildSystemPrompt,
  ShedContextTool,
  type AssistantTurn,
} from "@pinky/runtime";
import { createTools } from "@pinky/tools";
import { startGateway, SlackClient } from "@pinky/gateway";

const SCHEMA_DIR = new URL("../../core/schema", import.meta.url).pathname;

/** This process's agent identity: A2A address, settings scope, prompt header. */
const AGENT_ID = "pinky";

/** How often the A2A sender-side retry sweep runs in the gateway. */
const A2A_SWEEP_MS = 30_000;

async function openDb(): Promise<{ db: Db; env: EnvConfig }> {
  const env = loadEnvConfig();
  const db = createDb(env.databaseUrl);
  return { db, env };
}

async function loadSettings(db: Db, opts?: LoadOptions): Promise<SettingsSnapshot> {
  const store = new SettingsStore(db);
  return store.load(opts);
}

/**
 * Apply pending migrations on a short-lived PRIVILEGED connection.
 *
 * The app role cannot do this: 0003_rls.sql creates roles and alters tables,
 * so a pending one-shot migration attempted as `pinky_app` dies with
 * "permission denied". Opened and closed around the call so the long-lived
 * process keeps only its least-privilege pool.
 */
async function migrateAsAdmin(env: EnvConfig): Promise<void> {
  const admin = createDb(env.databaseAdminUrl);
  try {
    await migrate(admin, SCHEMA_DIR);
  } finally {
    await admin.close();
  }
}

async function cmdMigrate(): Promise<void> {
  const env = loadEnvConfig();
  await migrateAsAdmin(env);
  console.log("schema up to date");
}

function parseScope(args: string[]): { scope: string; rest: string[] } {
  const idx = args.indexOf("--scope");
  if (idx === -1) return { scope: "global", rest: args };
  const scope = args[idx + 1];
  if (!scope) throw new Error("--scope requires a value");
  // Fail here with the "expected global|channel:<id>|agent:<id>" message
  // rather than deep inside the store on the first query.
  assertScope(scope);
  return { scope, rest: [...args.slice(0, idx), ...args.slice(idx + 2)] };
}

async function cmdConfig(args: string[]): Promise<void> {
  const [sub, ...raw] = args;
  const { scope, rest } = parseScope(raw);
  const { db } = await openDb();
  try {
    const store = new SettingsStore(db);
    if (sub === "set") {
      const [key, ...valueParts] = rest;
      const rawValue = valueParts.join(" ");
      if (!key || !rawValue) throw new Error('usage: pinky config set <key> <value> [--scope s]');
      let value: unknown;
      try {
        value = JSON.parse(rawValue);
      } catch {
        value = rawValue;
      }
      await store.set(scope, key, value);
      console.log(`set ${key} (${scope})`);
    } else if (sub === "get") {
      const [key] = rest;
      // Only the asked-for scope is overlaid on defaults + global, so
      // `--scope channel:X` never shows channel Y's rows.
      const snapshot = await store.load({ scopes: [scope] });
      if (!key) {
        console.log(JSON.stringify(snapshot, null, 2));
      } else {
        const parts = key.split(".");
        let cur: unknown = snapshot;
        for (const p of parts) cur = (cur as Record<string, unknown>)[p];
        console.log(JSON.stringify(cur ?? null));
      }
    } else {
      throw new Error("usage: pinky config <set|get> ...");
    }
  } finally {
    await db.close();
  }
}

async function cmdGateway(): Promise<void> {
  const env = loadEnvConfig();
  // Before anything binds a port: an empty signing secret or A2A secret does
  // not disable those HMAC checks, it makes every forged request verify.
  assertGatewaySecrets(env);

  // Bot identity. Without it normalizeSlackEvent cannot recognize a mention or
  // a reply to us, so the gate never engages in a channel and the bot looks
  // dead. One client, reused for every delivery below.
  const slack = new SlackClient({ token: env.slack.botToken });
  let botUserId: string;
  try {
    botUserId = (await slack.authTest()).userId;
  } catch (err) {
    throw new Error(
      `Slack auth.test failed — cannot resolve the bot user id, so mentions would never engage. ` +
        `Check SLACK_BOT_TOKEN. (${err instanceof Error ? err.message : String(err)})`,
    );
  }

  await migrateAsAdmin(env);
  const rootDb = createDb(env.databaseUrl);
  // Startup settings are the GLOBAL snapshot: tenant + the model reported in
  // the banner. Per-wake reloads below add the channel/agent scopes.
  const settings = await loadSettings(rootDb);
  // Every runtime handle is tenant-scoped: withTenant sets the
  // `pinky.tenant_id` GUC per transaction, which is what the RLS policy reads.
  // Today only `memories` carries a policy (schema/0003_rls.sql) — the other
  // tables are still app-layer-filtered — so this is load-bearing for the
  // memory plane and forward-looking for the rest. rootDb is kept only to
  // close the pool.
  const db = withTenant(rootDb, settings.tenantId);
  const events = new EventStore(db);
  const messenger = new LocalMessenger(db, {
    nodeId: env.nodeId,
    peers: env.peers,
    a2aSecret: env.a2aSecret,
  });
  // Slack-reachable surface: NO shell. Anything here is callable by anyone who
  // can DM the bot, and bash is arbitrary host execution
  // (packages/tools/src/index.ts). shed_context comes from the runtime.
  const tools = [...createTools({ shell: false }), new ShedContextTool()];

  const server = startGateway({
    env,
    tenantId: settings.tenantId,
    events,
    messenger,
    botUserId,
    runAgent: async (thread, batch) => {
      // `batch` is the whole debounced burst; every message in it is already
      // in the event log, so the projection below sees them all. That is why
      // nothing here iterates it — one run per batch, not one per message.

      // Re-load settings per wake: a `pinky config set` takes effect on the
      // next turn without a restart. The running loop itself cannot mutate it.
      const fresh = await loadSettings(db, {
        scopes: [`channel:${thread.channelId}`, `agent:${AGENT_ID}`],
      });
      await runAgentLoop({
        db,
        provider: createProvider(fresh.model, process.env),
        tools,
        thread,
        agentId: AGENT_ID,
        messenger,
        systemPrompt: buildSystemPrompt({ agentId: AGENT_ID, nodeId: env.nodeId, tools }),
        cwd: process.cwd(),
        settings: fresh,
        deliver: async (text) => {
          const channel = thread.channelId.replace(/^slack:/, "");
          await slack.postMessage({ channel, text, thread_ts: thread.threadId });
        },
      });
    },
  });

  // Sender half of A2A at-least-once: rows a peer refused (or was down for)
  // stay pending, and only a sweep clears them. Once at startup, then on a
  // timer; unref'd so it never keeps the process alive by itself.
  const sweep = async (): Promise<void> => {
    try {
      const { attempted, delivered } = await messenger.flushPending();
      if (attempted > 0) console.log(`[a2a] retry sweep: ${delivered}/${attempted} delivered`);
    } catch (err) {
      console.warn("[a2a] retry sweep failed:", err instanceof Error ? err.message : err);
    }
  };
  await sweep();
  setInterval(sweep, A2A_SWEEP_MS).unref();

  console.log(
    `pinky gateway listening on :${server.port} ` +
      `(node ${env.nodeId}, bot ${botUserId}, model ${settings.model})`,
  );
}

async function cmdSmoke(): Promise<void> {
  const env = loadEnvConfig();
  await migrateAsAdmin(env);
  const rootDb = createDb(env.databaseUrl);
  const settings = await loadSettings(rootDb);
  // Tenant-scoped handle for everything that touches data (see cmdGateway).
  const db = withTenant(rootDb, settings.tenantId);
  const events = new EventStore(db);
  const messenger = new LocalMessenger(db, { nodeId: env.nodeId, peers: {}, a2aSecret: "" });
  const tools = createTools();

  const threadA: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "alpha" };
  const threadB: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "beta" };

  const betaInbox: string[] = [];
  messenger.onMessage("beta", (env2) => {
    betaInbox.push(`${env2.from}: ${env2.text}`);
  });

  const alphaScript: AssistantTurn[] = [
    {
      text: "",
      toolCalls: [
        { id: "c1", name: "a2a_send", args: { to: `beta@${env.nodeId}`, text: "what is 2+2?", kind: "request" } },
      ],
      stopReason: "tool_calls",
    },
    { text: "Asked beta; awaiting reply.", toolCalls: [], stopReason: "stop" },
  ];
  const provider = new FakeProvider(alphaScript);

  const runA = await runAgentLoop({
    db,
    provider,
    tools,
    thread: threadA,
    agentId: "alpha",
    messenger,
    systemPrompt: buildSystemPrompt({ agentId: "alpha", nodeId: env.nodeId, tools }),
    cwd: process.cwd(),
    settings,
  });

  const inbox = await messenger.inbox("beta");
  const historyA = await events.history(threadA);
  const historyB = await events.history(threadB);

  const checks: [string, boolean][] = [
    ["alpha ran to completion", runA.stopReason === "completed"],
    ["a2a message delivered to beta", inbox.length === 1 && inbox[0]!.text === "what is 2+2?"],
    ["live subscriber saw the message", betaInbox.length === 1],
    ["alpha thread logged assistant message", historyA.some((e) => e.data.type === "message")],
    ["alpha thread logged tool_result", historyA.some((e) => e.data.type === "tool_result")],
    ["beta thread untouched (mailbox, not thread)", historyB.length === 0],
    ["thread keys distinct", threadKey(threadA) !== threadKey(threadB)],
  ];

  let failed = 0;
  for (const [name, ok] of checks) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
    if (!ok) failed++;
  }
  await rootDb.close();
  if (failed > 0) {
    console.error(`smoke: ${failed} check(s) failed`);
    process.exit(1);
  }
  console.log("smoke: all checks passed");
}

/** Human-readable tail line + exit code for a finished local run. */
function reportRun(stopReason: string, turns: number): number {
  const label =
    stopReason === "shed" ? "paused at a context restart (resumable)" : stopReason;
  console.error(`[${label} after ${turns} turn(s)]`);
  // `shed` is a clean pause: the continuity document is written and the next
  // wake resumes. The other two are unfinished work, so they must not look
  // like success to a shell or a CI step.
  return stopReason === "shed_failed" || stopReason === "max_turns" ? 1 : 0;
}

async function cmdPrompt(text: string): Promise<void> {
  const env = loadEnvConfig();
  await migrateAsAdmin(env);
  const rootDb = createDb(env.databaseUrl);
  const settings = await loadSettings(rootDb);
  const db = withTenant(rootDb, settings.tenantId);
  const events = new EventStore(db);
  const messenger = new LocalMessenger(db, {
    nodeId: env.nodeId,
    peers: env.peers,
    a2aSecret: env.a2aSecret,
  });
  const provider = createProvider(settings.model, process.env);
  // Local operator surface: the human is at their own terminal running their
  // own agent, so shell access is theirs to grant. (The Slack-reachable
  // gateway above deliberately does not.)
  const tools = [...createTools({ shell: true }), new ShedContextTool()];
  const thread: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:local", threadId: "main" };

  let code = 0;
  try {
    await events.append(thread, {
      type: "ingress",
      platform: "cli",
      author: { platform: "cli", userId: "local" },
      text,
      refs: [],
    });
    const result = await runAgentLoop({
      db,
      provider,
      tools,
      thread,
      agentId: AGENT_ID,
      messenger,
      systemPrompt: buildSystemPrompt({ agentId: AGENT_ID, nodeId: env.nodeId, tools }),
      cwd: process.cwd(),
      settings,
      deliver: async (t) => {
        process.stdout.write(`${t}\n`);
      },
    });
    code = reportRun(result.stopReason, result.turns);
  } finally {
    await rootDb.close();
  }
  if (code !== 0) process.exit(code);
}

const [cmd, ...rest] = process.argv.slice(2);
try {
  switch (cmd) {
    case "migrate":
      await cmdMigrate();
      break;
    case "config":
      await cmdConfig(rest);
      break;
    case "gateway":
      await cmdGateway();
      break;
    case "smoke":
      await cmdSmoke();
      break;
    case "prompt":
      if (!rest[0]) throw new Error('usage: pinky prompt "<text>"');
      await cmdPrompt(rest.join(" "));
      break;
    default:
      console.error("usage: pinky <migrate|config|gateway|smoke|prompt>");
      process.exit(cmd ? 1 : 0);
  }
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
