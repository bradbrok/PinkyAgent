#!/usr/bin/env bun
/**
 * pinky — PinkyAgent CLI (the human-owned control surface; agents never get it).
 *
 *   pinky migrate                          apply schema migrations
 *   pinky config set <key> <value> [--scope global|channel:<id>|agent:<id>]
 *   pinky config get [key] [--scope ...]   read effective settings
 *   pinky config unset <key> [--scope ...] drop one stored row (incl. a pruned one)
 *   pinky memory list [--scope-channel <id>] [--limit N] [--all]
 *   pinky memory search "<query>" [--limit N]
 *   pinky memory show <id-or-prefix>
 *   pinky memory forget <id-or-prefix> [--reason "..."]
 *   pinky sleep run [--now] [--channel <id>] [--thread <id>] [--limit N]  one sweep, now
 *   pinky stats restarts [--channel <id>] [--limit N]   what context restarts cost
 *   pinky stats cache [--channel <id>] [--thread <id>] [--limit N]  prompt-cache hit rate
 *   pinky stats sleep [--channel <id>] [--limit N]      what the sleep worker wrote
 *   pinky mcp list                         configured MCP servers and their state
 *   pinky mcp sync [<server>...] [--timeout-ms N]   connect and republish the catalog
 *   pinky tools list [--scope <scope>...]  head vs deferred, as a run would partition it
 *   pinky smoke                            end-to-end in-process smoke (FakeProvider, A2A, memory)
 *   pinky prompt "<text>"                  run one agent turn against a local cli thread
 *   pinky headless [--shell] [--a2a] [--shared]  long-lived JSONL service on stdin/stdout
 *
 * Two database privilege levels (DESIGN.md §5.1, .env.example): migrations run
 * on DATABASE_ADMIN_URL (DDL + CREATE ROLE), everything else on DATABASE_URL,
 * which should be the NOBYPASSRLS `pinky_app` role so row-level security is
 * actually enforced. Commands that auto-migrate therefore open a short-lived
 * admin handle for that step and close it before opening the app connection.
 *
 * Every command that runs the agent goes through two helpers rather than a
 * copy of the same twelve lines: `bootstrap()` opens the connections, loads
 * settings and builds the memory plane (DESIGN.md §5); `makeRunAgent()` turns
 * that into a `(thread, overrides, batch) => AgentRunResult` call. `pinky
 * headless` (the JSONL service, DESIGN.md §11) is one call to each.
 *
 * STDOUT DISCIPLINE: `pinky headless` owns stdout for the JSONL protocol, so
 * on that path nothing may print there — not a warning, not a Postgres NOTICE
 * (see core/pg.ts `onnotice`), not an A2A sweep line. Everything human goes to
 * stderr. The other commands are human-facing and print normally.
 */
import {
  loadEnvConfig,
  assertGatewaySecrets,
  createDb,
  migrate,
  withTenant,
  EventStore,
  MemoryStore,
  SettingsStore,
  ToolCatalogStore,
  assertScope,
  parseA2AAddress,
  threadKey,
  DEFAULT_SETTINGS,
  type CatalogRecord,
  type Db,
  type EnvConfig,
  type LoadOptions,
  type MemoryRow,
  type RecallScope,
  type SettingsSnapshot,
  type ThreadEventData,
  type ThreadRef,
  type TokenUsage,
} from "@pinky/core";
import {
  createEmbedder,
  createFakeProvider,
  createProvider,
  isEmbeddingsDisabledError,
  partitionTools,
  runAgentLoop,
  buildSystemPrompt,
  FAKE_DEFERRED_MARKER,
  FAKE_SLEEP_REFLECT_PREFIX,
  DeferredToolRegistry,
  FakeEmbedder,
  FakeProvider,
  LocalMessenger,
  ShedContextTool,
  type A2AEnvelope,
  type AgentRunResult,
  type AssistantTurn,
  type Embedder,
  type LlmMessage,
  type MemoryContext,
  type Provider,
  type RunAgentLoopOptions,
  type Tool,
  type ToolSource,
} from "@pinky/runtime";
import { runHeadless, startGateway, type RawIngress, type WakeEnqueue } from "@pinky/gateway";
import { createTools } from "@pinky/tools";
import { McpManager, defaultTransportFactory, type McpServerState } from "@pinky/mcp";
import {
  reflectThread,
  startSleepSweep,
  // Aliased: `startA2ASweep` already has a local `sweep`, and two different
  // sweeps in one file should not be told apart by scope alone.
  sweep as sleepSweep,
  type ExtractReceipt,
  type ReflectReceipt,
  type SleepDeps,
  type SleepScope,
  type SweepReport,
} from "@pinky/sleep";

const SCHEMA_DIR = new URL("../../core/schema", import.meta.url).pathname;

/** This process's agent identity: A2A address, settings scope, prompt header. */
const AGENT_ID = "pinky";

/** Channel id of the local terminal surface — `pinky prompt`'s thread and the
 *  default channel scope of the `pinky memory` commands, so a memory retained
 *  from a prompt run is visible to the human who owns that terminal. */
const CLI_CHANNEL_ID = "cli:local";

/** How often the A2A sender-side retry sweep runs in a long-lived process. */
const A2A_SWEEP_MS = 30_000;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Human log line. Always stderr: on `pinky headless` stdout is the protocol,
 *  and everything below this line is shared with that surface. */
function logStderr(msg: string): void {
  process.stderr.write(`${msg}\n`);
}

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

// ---------------------------------------------------------------------------
// bootstrap — the one place a runtime-bearing command is assembled
// ---------------------------------------------------------------------------

export interface BootstrapOptions {
  /** Apply pending migrations on the admin connection first. Default true. */
  migrate?: boolean;
  /** Extra settings scopes overlaid on the startup snapshot (e.g. agent:pinky). */
  scopes?: string[];
  /** A2A peers. Defaults to env.peers; `{}` keeps the process purely local. */
  peers?: Record<string, string>;
  /** A2A HMAC secret. Defaults to env.a2aSecret. */
  a2aSecret?: string;
  /** Pin the embedder (smoke's FakeEmbedder); `null` forces FTS-only recall.
   *  Omitted => built from `settings.memory.embeddingModel` + the env. */
  embedder?: Embedder | null;
  /**
   * Start the MCP plane. Default true.
   *
   * A read-only query (`pinky stats`, `pinky memory`) has no tool set and no
   * reason to spawn somebody's configured stdio servers as child processes, so
   * it passes false: the manager is still constructed — `boot.mcp` is not
   * optional, and every read on it answers empty — but nothing connects.
   */
  mcp?: boolean;
}

export interface Bootstrap {
  env: EnvConfig;
  /** Startup snapshot. Long-lived surfaces re-`reloadSettings()` per wake. */
  settings: SettingsSnapshot;
  /**
   * Tenant-scoped handle: withTenant sets the `pinky.tenant_id` GUC per
   * transaction, which is what the RLS policy on `memories` reads. Everything
   * that touches data uses this one, never the raw pool.
   */
  db: Db;
  events: EventStore;
  messenger: LocalMessenger;
  memory: MemoryStore;
  /** The deferred-tool catalog (slice 9): what `tool_search`/`tool_describe`
   *  read, and where every MCP sync and `upsertBuiltins` writes. */
  catalog: ToolCatalogStore;
  /**
   * The MCP plane. `servers` is read ONCE, here, from the BOOTSTRAP scopes
   * (global + whatever `opts.scopes` adds — `agent:pinky` on the agent
   * surfaces). A `channel:<id>`-scoped `mcp.servers` row is therefore NOT
   * honored: per-channel servers would need one manager (and one set of child
   * processes) per channel, which is a different design, not a config value.
   * makeRunAgent warns on stderr when a reloaded snapshot disagrees.
   */
  mcp: McpManager;
  /** null => FTS-only recall: no vector voice, and retains store no embedding. */
  embedder: Embedder | null;
  /** Bind the store (+ embedder, when there is one) to one surface's scope. */
  memoryContextFor: (scope: RecallScope) => MemoryContext;
  reloadSettings: (opts?: LoadOptions) => Promise<SettingsSnapshot>;
  /** Closes the underlying pool. `db` is a wrapper and owns nothing itself. */
  close: () => Promise<void>;
}

/**
 * Build the embedder named by `memory.embeddingModel` (DESIGN.md §5.5).
 *
 * Exactly one failure is survivable: the route is configured but its API key
 * is blank. That is a deployment that simply has no embeddings, so it warns
 * once and recall runs on the FTS voice alone. Everything else — an unknown
 * provider, a malformed id — is a misconfiguration the operator must see, so
 * it escapes and the command dies.
 */
function openEmbedder(settings: SettingsSnapshot): Embedder | null {
  try {
    return createEmbedder(settings.memory.embeddingModel, process.env);
  } catch (err) {
    if (!isEmbeddingsDisabledError(err)) throw err;
    console.warn(`[memory] ${errorMessage(err)}; recall runs FTS-only`);
    return null;
  }
}

export async function bootstrap(opts: BootstrapOptions = {}): Promise<Bootstrap> {
  const env = loadEnvConfig();
  const peers = opts.peers ?? env.peers;
  const a2aSecret = opts.a2aSecret ?? env.a2aSecret;
  // Only bites when this process actually has peers: an empty HMAC key does
  // not disable the A2A signature check, it makes every forged envelope
  // verify. A single-node process (or smoke, which passes `peers: {}`) is
  // unaffected.
  assertGatewaySecrets({ ...env, peers, a2aSecret });

  if (opts.migrate !== false) await migrateAsAdmin(env);

  const rootDb = createDb(env.databaseUrl);
  let settings: SettingsSnapshot;
  try {
    settings = await loadSettings(rootDb, opts.scopes ? { scopes: opts.scopes } : undefined);
  } catch (err) {
    await rootDb.close();
    throw err;
  }
  const db = withTenant(rootDb, settings.tenantId);
  const store = new MemoryStore(db, settings.tenantId);
  const embedder = opts.embedder !== undefined ? opts.embedder : openEmbedder(settings);
  const catalog = new ToolCatalogStore(db, settings.tenantId);
  // Logs to STDERR, always: this manager runs inside `pinky headless`, whose
  // stdout is the JSONL protocol, and a connect line on stdout is a corrupt
  // stream (cli/test/integration/mcp-tools.test.ts is the guard).
  const mcp = new McpManager({
    servers: settings.mcp.servers,
    catalog,
    log: logStderr,
    env: process.env,
  });
  // Awaits the per-server catalog trust probes and nothing else: a configured
  // server that is slow, wedged or absent must never delay a boot, so the
  // connects run on background loops (McpManager rule 1).
  if (opts.mcp !== false) await mcp.start();

  return {
    env,
    settings,
    db,
    events: new EventStore(db),
    messenger: new LocalMessenger(db, { nodeId: env.nodeId, peers, a2aSecret }),
    memory: store,
    catalog,
    mcp,
    embedder,
    memoryContextFor: (scope) => ({ store, scope, ...(embedder ? { embedder } : {}) }),
    reloadSettings: (loadOpts) => loadSettings(db, loadOpts),
    close: async () => {
      // MCP first: closing the clients is what reaps the stdio children, and
      // an orphaned child outliving the pool would hold the terminal open.
      await mcp.close();
      await rootDb.close();
    },
  };
}

/**
 * Publish this surface's built-in tools into the catalog (slice 9).
 *
 * Upsert-only and per surface on purpose: `pinky prompt` registers `bash` and
 * `pinky headless` (without `--shell`) does not, so a generational replace
 * would have two processes stamping and clearing each other's rows forever
 * (ToolCatalogStore.upsertBuiltins says the same from the other side).
 *
 * Every built-in goes in, including the ones that can never be deferred
 * (`shed_context`, the three meta-tools): the catalog is also what
 * `pinky tools list` and a `tool_search` with an empty query read, and a model
 * asking "what can I do" should see the whole set, not the subset that happens
 * to be movable.
 */
async function registerBuiltins(boot: Bootstrap, tools: Tool[]): Promise<void> {
  await boot.catalog.upsertBuiltins(
    tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
  );
}

/**
 * Per-run overrides a caller may hand `runAgentLoop`, SUBTRACTED from its
 * option type rather than listed, so the set cannot drift: everything the
 * surface owns (db, tools, the deferred plane, memory, settings…) is fixed by
 * makeRunAgent and the remainder is per-run — today `maxTurns`, `deliver`, `signal` and `onEvent`,
 * and whatever the runtime adds next without an edit here. That is exactly the
 * set `pinky headless` needs (DESIGN.md §11: one abortable run per thread,
 * streaming every appended event and every deliver() to stdout).
 */
export type RunAgentOverrides = Omit<
  Partial<RunAgentLoopOptions>,
  | "db"
  | "provider"
  | "tools"
  | "thread"
  | "agentId"
  | "messenger"
  | "memory"
  | "systemPrompt"
  | "cwd"
  | "settings"
  | "deferred"
>;

export interface RunAgentOptions {
  agentId: string;
  /**
   * The BUILT-IN candidates for this surface — `createTools(...)` plus
   * `ShedContextTool`. Not "the tools the run gets": the MCP tools are added
   * per run from `boot.mcp.tools()` and the whole set is then partitioned
   * (slice 9), so what reaches the provider is the HEAD of that partition and
   * the rest is reachable only through the meta-tools.
   */
  builtins: Tool[];
  /** Fixed provider (smoke's FakeProvider). Default: built per run from the
   *  run's settings, so `pinky config set model` lands on the next wake. */
  provider?: Provider;
  /** Settings for this wake; default the bootstrap snapshot. A long-lived
   *  surface passes a reload scoped to the thread's channel + this agent. */
  settingsFor?: (thread: ThreadRef) => Promise<SettingsSnapshot>;
  /** Recall scope for this thread (DESIGN.md §5.1). Absent => the loop runs
   *  with no memory plane at all (no auto-recall, memory tools refuse).
   *  `batch` is the ingress that woke this run, when the surface has one — it
   *  is the only place the *author* is known, and `user`-visibility recall is
   *  per-user (§5.1), so a scope built from the thread alone cannot see it. */
  scopeFor?: (thread: ThreadRef, batch?: RawIngress[]) => RecallScope;
  cwd?: string;
}

export type RunAgent = (
  thread: ThreadRef,
  overrides?: RunAgentOverrides,
  batch?: RawIngress[],
) => Promise<AgentRunResult>;

/** First line of a tool description — what buildSystemPrompt prints, and
 *  therefore the only part of a description the cached prefix depends on. */
function firstLine(text: string): string {
  return (text.split("\n", 1)[0] ?? "").trim();
}

/** Which `defaultMode` a tool falls under when no list names it. The `mcp__`
 *  prefix is structural (mcpToolName builds it), so this needs no registry. */
const sourceOf = (name: string): ToolSource => (name.startsWith("mcp__") ? "mcp" : "builtin");

/**
 * `buildSystemPrompt`, memoized on the head tool NAMES.
 *
 * The system prompt is the cached prefix (DESIGN.md §4.5/§9) and it lists the
 * head tools, so it can no longer be built once per process: `tools.alwaysOn`
 * is a setting, and a setting can change between wakes. Building it per run is
 * correct and costs nothing; the memo exists so that an UNCHANGED head yields
 * the identical string — the same object, even — which makes "the prefix is
 * stable unless the header moved" visible in the code rather than a property
 * you have to trust.
 */
function memoSystemPrompt(agentId: string, nodeId: string): (tools: Tool[]) => string {
  let key: string | undefined;
  let prompt = "";
  return (tools) => {
    // Name AND the description line the prompt actually prints: an MCP resync
    // can republish a tool under the same name with new prose, and keying on
    // names alone would leave a stale one-liner in the cached prefix forever.
    // Keyed this way the prompt changes exactly when the header does — which
    // is the invalidation the provider is going to charge for anyway.
    // NUL as the separator, written as an escape so this file stays plain text
    // (a literal one makes grep call it binary), because no name or first line
    // can contain it and the two halves can never run together.
    const next = tools.map((t) => `${t.name}\u0000${firstLine(t.description)}`).join("\u0000");
    if (next !== key) {
      key = next;
      prompt = buildSystemPrompt({ agentId, nodeId, tools });
    }
    return prompt;
  };
}

/**
 * One `runAgentLoop` call, pre-bound to a surface. `pinky headless` builds
 * its per-run hooks (deliver/signal/onEvent) as the `overrides` argument.
 *
 * The tool set is assembled PER RUN (slice 9), not once per process:
 *
 *   1. `builtins` + whatever the MCP plane can execute right now;
 *   2. `partitionTools` splits that by the RELOADED settings — `head` is what
 *      renders in the request (and in the system prompt), `deferred` is
 *      reachable only through tool_search/tool_describe/tool_call;
 *   3. the deferred half becomes a `DeferredToolRegistry` over the catalog.
 *
 * `head` is never re-sorted here: partitionTools already ordered it by code
 * unit, and that order is part of the provider cache key.
 */
export function makeRunAgent(boot: Bootstrap, opts: RunAgentOptions): RunAgent {
  const systemPromptFor = memoSystemPrompt(opts.agentId, boot.env.nodeId);
  const cwd = opts.cwd ?? process.cwd();
  // `mcp.servers` was read at bootstrap from the bootstrap scopes; a run's
  // reloaded snapshot may carry a channel-scoped one that nothing acts on.
  // Warned once per distinct value — a per-run line would be a page of
  // identical text on a long-lived process.
  const bootServers = JSON.stringify(boot.settings.mcp.servers);
  const warnedServers = new Set<string>();
  // `sleep.*` is read once at bootstrap for the same reason (slice 6): the
  // sweep timer belongs to the PROCESS and walks every thread of the tenant,
  // so a channel-scoped row has no thread to be about. Warned here rather than
  // in the timer because this is the only place a reloaded snapshot is seen.
  const bootSleep = JSON.stringify(boot.settings.sleep);
  const warnedSleep = new Set<string>();

  return async (thread, overrides = {}, batch) => {
    const settings = opts.settingsFor ? await opts.settingsFor(thread) : boot.settings;
    const scope = opts.scopeFor?.(thread, batch);

    const runServers = JSON.stringify(settings.mcp.servers);
    if (runServers !== bootServers && !warnedServers.has(runServers)) {
      warnedServers.add(runServers);
      logStderr(
        `[mcp] channel ${thread.channelId} overrides mcp.servers; IGNORED — the MCP plane is ` +
          "built once at startup from the global + agent scopes. Set servers there, or run a " +
          "separate process for that channel.",
      );
    }

    const runSleep = JSON.stringify(settings.sleep);
    if (runSleep !== bootSleep && !warnedSleep.has(runSleep)) {
      warnedSleep.add(runSleep);
      logStderr(
        `[sleep] channel ${thread.channelId} overrides sleep.*; IGNORED — the sweep timer is ` +
          "built once at startup from the global + agent scopes. Set it there (and restart), or " +
          "run a separate process for that channel.",
      );
    }

    const { head, deferred } = partitionTools(
      [...opts.builtins, ...boot.mcp.tools()],
      settings.tools,
      sourceOf,
    );

    return runAgentLoop({
      db: boot.db,
      provider: opts.provider ?? createProvider(settings.model, process.env),
      tools: head,
      thread,
      agentId: opts.agentId,
      messenger: boot.messenger,
      systemPrompt: systemPromptFor(head),
      cwd,
      settings,
      deferred: new DeferredToolRegistry({
        catalog: boot.catalog,
        tools: new Map(deferred.map((t) => [t.name, t])),
        // The catalog is tenant-wide and never withdraws a built-in, so it
        // knows names this run either already has in the header or cannot run
        // at all (`bash`, catalogued by `pinky prompt`). Handing over the head
        // set is what lets tool_call answer those two cases precisely instead
        // of blaming an offline server.
        headNames: new Set(head.map((t) => t.name)),
      }),
      ...(scope ? { memory: boot.memoryContextFor(scope) } : {}),
      ...overrides,
    });
  };
}

/**
 * SIGTERM / SIGINT -> the same shutdown EOF gets, then an honest exit code.
 *
 * Without this a signal is fatal by default: Bun terminates the process, the
 * `finally` that calls `boot.close()` never runs, and every MCP stdio child
 * this process spawned is REPARENTED AND LEFT RUNNING. Under a supervisor
 * (systemd, `docker stop`, a k8s rollout) that is one orphan per server per
 * restart, holding its own database handles and file descriptors forever —
 * and signals are exactly how those supervisors stop a long-lived service, so
 * it is the normal path, not an edge case.
 *
 * The shape mirrors the client-gone path (`closeStdout` -> abort -> drain):
 *
 *   1. `drain()` aborts the session so in-flight runs cancel and the surface
 *      stops accepting work — the same abort EOF triggers, so a signal and a
 *      closed pipe end a session identically;
 *   2. `close()` tears the process down in dependency order (`boot.close()`
 *      closes the MCP clients, which is what reaps the children, BEFORE the
 *      connection pool);
 *   3. exit with 128 + the signal number, the convention a supervisor reads.
 *
 * Two safety valves, because a shutdown that hangs is worse than an abrupt
 * one: a SECOND signal exits immediately (the operator asked twice), and a
 * drain or close that has not finished within {@link SHUTDOWN_GRACE_MS} exits
 * anyway. Nothing here writes to stdout — on `pinky headless` that is the
 * protocol, and a shutdown note on it would corrupt the stream.
 *
 * Returns an unsubscribe for the normal exit path, so a completed command does
 * not sit in the event loop holding two listeners.
 */
const SHUTDOWN_SIGNALS = ["SIGTERM", "SIGINT"] as const;
type ShutdownSignal = (typeof SHUTDOWN_SIGNALS)[number];

/** 128 + signal number — what a shell (and a supervisor) reads as "stopped by
 *  that signal" rather than "failed". */
const SIGNAL_EXIT_CODE: Record<ShutdownSignal, number> = { SIGTERM: 143, SIGINT: 130 };

/** Cap on the whole drain+close. A supervisor SIGKILLs after its own timeout;
 *  exiting first, on our own terms, is what keeps the children reaped. */
const SHUTDOWN_GRACE_MS = 10_000;

export function installShutdown(opts: {
  /** Stop accepting work and let in-flight runs settle (abort + await). */
  drain: () => Promise<void> | void;
  /** Release everything, MCP children before the pool. */
  close: () => Promise<void>;
  graceMs?: number;
}): () => void {
  let shuttingDown = false;

  const handle = (signal: ShutdownSignal): void => {
    const code = SIGNAL_EXIT_CODE[signal];
    if (shuttingDown) {
      logStderr(`[shutdown] second ${signal}; exiting now`);
      process.exit(code);
    }
    shuttingDown = true;
    logStderr(`[shutdown] ${signal}: draining, then closing mcp children and the pool`);
    // Never awaited by anyone: this IS the top of the stack for a signal.
    void (async () => {
      // The timer wins if a drain or a close wedges. Unref'd so it is not the
      // reason the process is still alive.
      const timer = setTimeout(() => {
        logStderr(`[shutdown] still closing after ${opts.graceMs ?? SHUTDOWN_GRACE_MS}ms; exiting`);
        process.exit(code);
      }, opts.graceMs ?? SHUTDOWN_GRACE_MS);
      timer.unref();
      try {
        await opts.drain();
      } catch (err) {
        logStderr(`[shutdown] drain failed: ${errorMessage(err)}`);
      }
      try {
        await opts.close();
      } catch (err) {
        logStderr(`[shutdown] close failed: ${errorMessage(err)}`);
      }
      clearTimeout(timer);
      process.exit(code);
    })();
  };

  const listeners = SHUTDOWN_SIGNALS.map((signal) => {
    const listener = (): void => handle(signal);
    process.on(signal, listener);
    return [signal, listener] as const;
  });
  return () => {
    for (const [signal, listener] of listeners) process.off(signal, listener);
  };
}

/**
 * Both halves of A2A at-least-once on one timer (DESIGN.md §7, issue #4).
 * Once now, then every {@link A2A_SWEEP_MS}; unref'd so it never keeps the
 * process alive by itself. Logs to STDERR — a JSONL service owns stdout.
 * Returns a stop function.
 *
 *  - SENDER side: rows a peer refused (or was down for) stay pending, and only
 *    a sweep clears them. No peers configured => flushPending() is a no-op.
 *  - CONSUMER side (`redeliverFor`): every row addressed to this agent with no
 *    consumption receipt is re-fired, so a wake lost to a crashed handler, a
 *    process that died between the delivery claim and the turn, or a peer that
 *    delivered while nothing was subscribed is picked up within one sweep
 *    instead of waiting for the agent to poll `a2a_inbox`. Bounded by the
 *    receipt, not by scheduler state: a re-fire whose message was already
 *    consumed loses the claim and does nothing.
 */
export function startA2ASweep(
  messenger: LocalMessenger,
  log: (msg: string) => void = (msg) => console.error(msg),
  opts: { redeliverFor?: string } = {},
): () => void {
  const sweep = async (): Promise<void> => {
    try {
      const { attempted, delivered } = await messenger.flushPending();
      if (attempted > 0) log(`[a2a] retry sweep: ${delivered}/${attempted} delivered`);
    } catch (err) {
      log(`[a2a] retry sweep failed: ${errorMessage(err)}`);
    }
    if (opts.redeliverFor === undefined) return;
    try {
      const fired = await messenger.redeliverUnconsumed(opts.redeliverFor);
      if (fired > 0) log(`[a2a] redelivered ${fired} unconsumed message(s)`);
    } catch (err) {
      log(`[a2a] redelivery sweep failed: ${errorMessage(err)}`);
    }
  };
  void sweep();
  const timer = setInterval(() => void sweep(), A2A_SWEEP_MS);
  timer.unref();
  return () => clearInterval(timer);
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

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
    } else if (sub === "unset") {
      // The counterpart to `load()` pruning a row it cannot use: the warning
      // names the key, this removes it. Without it the only way to clear a row
      // whose key no longer exists (a renamed setting, a typo) would be SQL,
      // because `set` can only write values that validate.
      const [key] = rest;
      if (!key) throw new Error("usage: pinky config unset <key> [--scope s]");
      const removed = await store.unset(scope, key);
      console.log(removed ? `unset ${key} (${scope})` : `nothing to unset: ${key} (${scope})`);
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
      throw new Error("usage: pinky config <set|get|unset> ...");
    }
  } finally {
    await db.close();
  }
}

// ---------------------------------------------------------------------------
// memory — the human's read/retire surface over the memory plane (DESIGN.md §5)
// ---------------------------------------------------------------------------

/** A full uuid; anything shorter is treated as a prefix (like memory_edit). */
const UUID_LENGTH = 36;
/** A complete uuid — anything else must not reach the uuid column in `get`. */
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
/** Memory text is one line here; `pinky memory show <id>` prints it whole. */
const LINE_TEXT_CHARS = 160;

type Flags = Record<string, string | true>;

/** `--name value`, `--name=value`, and (for `booleans`) a bare `--name`. */
function parseFlags(args: string[], booleans: string[] = []): { flags: Flags; rest: string[] } {
  const flags: Flags = {};
  const rest: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (!arg.startsWith("--")) {
      rest.push(arg);
      continue;
    }
    const eq = arg.indexOf("=");
    if (eq > 2) {
      flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      continue;
    }
    const name = arg.slice(2);
    if (booleans.includes(name)) {
      flags[name] = true;
      continue;
    }
    const value = args[++i];
    if (value === undefined) throw new Error(`--${name} requires a value`);
    flags[name] = value;
  }
  return { flags, rest };
}

function stringFlag(flags: Flags, name: string): string | undefined {
  const raw = flags[name];
  if (raw === undefined) return undefined;
  if (raw === true) throw new Error(`--${name} requires a value`);
  return raw;
}

function intFlag(flags: Flags, name: string, fallback: number): number {
  const raw = stringFlag(flags, name);
  if (raw === undefined) return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1) throw new Error(`--${name} must be a positive integer`);
  return n;
}

/**
 * What the human at this terminal may see (DESIGN.md §5.1). The local surface
 * is trusted: it is the operator's own machine running their own agent, so
 * `user` and `private` rows are both in scope — unlike a shared channel.
 * `channel`-visibility rows come from ONE channel; it defaults to the prompt
 * surface's own so `pinky prompt` memories are listable without a flag.
 */
function cliScope(channelId: string | undefined = CLI_CHANNEL_ID): RecallScope {
  return {
    agentId: AGENT_ID,
    ...(channelId ? { channelId } : {}),
    userId: "local",
    includeUser: true,
    includePrivate: true,
  };
}

/** YYYY-MM-DD from an ISO timestamp. */
function day(iso: string): string {
  return iso.slice(0, 10);
}

/** `<id>  <kind>/<visibility>  imp=<n>  <YYYY-MM-DD>  <text>` (+ `[invalid]`). */
function memoryLine(row: MemoryRow): string {
  const text = row.text.replace(/\s+/g, " ").trim();
  const clipped = text.length > LINE_TEXT_CHARS ? `${text.slice(0, LINE_TEXT_CHARS - 1)}…` : text;
  const invalid = row.validTo ? "[invalid] " : "";
  return `${row.id}  ${row.kind}/${row.visibility}  imp=${row.importance}  ${day(row.recordedAt)}  ${invalid}${clipped}`;
}

/**
 * Resolve what the human typed: a full id takes the direct `get`, a prefix is
 * matched IN SQL and refused when it is ambiguous. Same rule as the
 * memory_edit tool, so an id copied out of a recall result works here too —
 * and for the same reason it must not be a page of `list()` filtered
 * afterwards: `list` is newest-first and capped, so an id printed by a
 * relevance-ranked `memory search` could be one this command then claimed not
 * to exist. `limit: 2` is the whole answer (one hit, or ambiguous).
 */
async function resolveMemory(
  store: MemoryStore,
  scope: RecallScope,
  wanted: string,
): Promise<MemoryRow> {
  if (UUID_PATTERN.test(wanted)) {
    const direct = await store.get(wanted);
    if (direct) return direct;
    throw new Error(`no memory with id ${wanted}`);
  }
  if (wanted.length >= UUID_LENGTH) throw new Error(`no memory with id ${wanted}`);

  // Ids are lowercase hex; the store refuses any other character class
  // because the prefix goes into a LIKE pattern unescaped. Its message for a
  // malformed one ("4..36 characters of lowercase hex") is the right thing for
  // a human to read, so it is deliberately not caught here.
  const candidates = await store.findByIdPrefix(wanted.toLowerCase(), {
    scope,
    includeInvalid: true,
    limit: 2,
  });
  const first = candidates[0];
  if (!first) throw new Error(`no memory with id ${wanted}`);
  if (candidates.length > 1) {
    throw new Error(
      `id ${wanted} is ambiguous (more than one memory shares that prefix); use the full id`,
    );
  }
  return first;
}

/** Embed the query for the vector voice, or undefined for FTS-only search. */
async function embedQuery(boot: Bootstrap, query: string): Promise<number[] | undefined> {
  if (!boot.embedder || query.trim() === "") return undefined;
  if (!(await boot.memory.supportsVectors())) return undefined;
  try {
    const [vector] = await boot.embedder.embed([query]);
    return vector;
  } catch (err) {
    // A dead embedder must not make the CLI useless: the lexical voice alone
    // still answers, and the operator is told why the ranking is thinner.
    console.error(`[memory] embedding failed, searching FTS-only: ${errorMessage(err)}`);
    return undefined;
  }
}

const MEMORY_USAGE =
  "usage: pinky memory <list|search|show|forget> ...\n" +
  "  pinky memory list [--scope-channel <id>] [--limit N] [--all]\n" +
  '  pinky memory search "<query>" [--limit N] [--scope-channel <id>]\n' +
  "  pinky memory show <id-or-prefix>\n" +
  '  pinky memory forget <id-or-prefix> [--reason "..."]';

async function cmdMemory(args: string[]): Promise<void> {
  const [sub, ...raw] = args;
  const { flags, rest } = parseFlags(raw, ["all"]);
  const scope = cliScope(stringFlag(flags, "scope-channel"));

  // Read-only over the memory plane: no tool set, so no reason to spawn a
  // configured MCP server's child processes for a `memory list`.
  const boot = await bootstrap({ mcp: false });
  try {
    if (sub === "list") {
      const rows = await boot.memory.list({
        scope,
        limit: intFlag(flags, "limit", 20),
        ...(flags.all === true ? { includeInvalid: true } : {}),
      });
      if (rows.length === 0) {
        console.log("no memories in scope");
        return;
      }
      for (const row of rows) console.log(memoryLine(row));
    } else if (sub === "search") {
      const query = rest.join(" ").trim();
      if (!query) throw new Error('usage: pinky memory search "<query>" [--limit N]');
      const queryEmbedding = await embedQuery(boot, query);
      const hits = await boot.memory.search({
        scope,
        query,
        limit: intFlag(flags, "limit", 20),
        ...(queryEmbedding ? { queryEmbedding } : {}),
      });
      if (hits.length === 0) {
        console.log("no memories matched");
        return;
      }
      for (const hit of hits) console.log(memoryLine(hit));
    } else if (sub === "show") {
      const [wanted] = rest;
      if (!wanted) throw new Error("usage: pinky memory show <id-or-prefix>");
      const row = await resolveMemory(boot.memory, scope, wanted);
      console.log(`id          ${row.id}`);
      console.log(`agent       ${row.agentId}`);
      console.log(`kind        ${row.kind}`);
      console.log(`visibility  ${row.visibility}`);
      console.log(`importance  ${row.importance}`);
      console.log(`channel     ${row.channelId ?? "-"}`);
      console.log(`user        ${row.userId ?? "-"}`);
      console.log(`recorded    ${row.recordedAt}`);
      console.log(`state       ${row.validTo ? `invalidated ${row.validTo}` : "current"}`);
      console.log(`embedding   ${row.embeddingModel ?? "none"}`);
      console.log(`meta        ${JSON.stringify(row.meta)}`);
      console.log("");
      console.log(row.text);
    } else if (sub === "forget") {
      const [wanted] = rest;
      if (!wanted) throw new Error('usage: pinky memory forget <id-or-prefix> [--reason "..."]');
      const row = await resolveMemory(boot.memory, scope, wanted);
      const reason = stringFlag(flags, "reason");
      // Memories are retired, never deleted (DESIGN.md §5.2): the row stays as
      // history with valid_to stamped, so what the agent believed and when is
      // still reconstructible.
      const ok = await boot.memory.invalidate(row.id, {
        reason: reason ? `forget: ${reason}` : "forget: pinky memory forget",
      });
      console.log(
        ok
          ? `forgot ${row.id} (invalidated, not deleted)`
          : `${row.id} was already invalidated (${row.validTo})`,
      );
    } else {
      throw new Error(MEMORY_USAGE);
    }
  } finally {
    await boot.close();
  }
}

// ---------------------------------------------------------------------------
// stats — the DESIGN.md §13 eval, as a query rather than a study
// ---------------------------------------------------------------------------

/**
 * One `restart` event joined to the assistant turn that paid for it.
 *
 * The successor's FIRST turn is where a restart actually costs money: the
 * fresh window has no cache warmth, so the whole prefix is re-billed as cache
 * *creation* (~1.25x an input token) before anything is read back cheaply.
 * That is why the join reaches forward for the next `message` in the same
 * thread rather than reporting the restart's own numbers alone.
 *
 * Every count comes back through a `::int` cast — `seq` is `bigint`, which
 * postgres.js hands over as a STRING, and the jsonb accessors are text.
 */
interface RestartRow {
  channel_id: string;
  thread_id: string;
  seq: number | string;
  ts: string;
  boundary_seq: number | null;
  tokens_before: number | null;
  tokens_after: number | null;
  recall_tokens: number | null;
  messages: number | null;
  /** null when the provider reported no usage (most OpenAI-compatible routes). */
  usage_input: number | null;
  usage_cache_read: number | null;
  usage_cache_creation: number | null;
  usage_output: number | null;
}

const RESTART_SQL = `
  select r.channel_id,
         r.thread_id,
         r.seq::int                               as seq,
         r.ts,
         (r.data->>'boundarySeq')::int            as boundary_seq,
         (r.data->>'tokensBefore')::int           as tokens_before,
         (r.data->>'tokensAfter')::int            as tokens_after,
         (r.data->>'recallTokens')::int           as recall_tokens,
         (r.data->>'messages')::int               as messages,
         (m.data->'usage'->>'input')::int         as usage_input,
         (m.data->'usage'->>'cacheRead')::int     as usage_cache_read,
         (m.data->'usage'->>'cacheCreation')::int as usage_cache_creation,
         (m.data->'usage'->>'output')::int        as usage_output
    from events r
    left join lateral (
      select n.data
        from events n
       where (n.tenant_id, n.channel_id, n.thread_id) = (r.tenant_id, r.channel_id, r.thread_id)
         and n.seq > r.seq
         and n.type = 'message'
       order by n.seq asc
       limit 1
    ) m on true
   where r.tenant_id = $1
     and r.type = 'restart'
     and ($2::text is null or r.channel_id = $2)
   order by r.ts desc, r.seq desc
   limit $3`;

/** Width of the `channel/thread` column; longer labels are clipped at the head. */
const THREAD_COL = 40;

const padRight = (s: string, n: number): string => (s.length >= n ? s : s + " ".repeat(n - s.length));
const padLeft = (s: string, n: number): string => (s.length >= n ? s : " ".repeat(n - s.length) + s);

/** Keep the TAIL: the distinguishing half of both a thread key and a model id. */
const clip = (s: string, n: number): string => (s.length > n ? `…${s.slice(-(n - 1))}` : s);

/** `<channel>/<thread>`, tail-clipped: the thread id is the distinguishing half. */
function threadLabel(row: RestartRow): string {
  return clip(`${row.channel_id}/${row.thread_id}`, THREAD_COL);
}

const count = (n: number | null): string => (n === null ? "-" : String(n));

/** `-16110 (-93%)` — how much smaller the fresh window is than what it replaced. */
function change(before: number | null, after: number | null): string {
  if (before === null || after === null) return "-";
  const delta = after - before;
  const sign = delta > 0 ? "+" : "";
  if (before <= 0) return `${sign}${delta}`;
  return `${sign}${delta} (${sign}${Math.round((delta / before) * 100)}%)`;
}

/** The successor's first turn, or `n/a` when the provider reported no usage. */
function firstTurn(row: RestartRow): string {
  if (row.usage_input === null && row.usage_output === null) return "n/a";
  return [
    `in ${count(row.usage_input)}`,
    `read ${count(row.usage_cache_read)}`,
    `write ${count(row.usage_cache_creation)}`,
    `out ${count(row.usage_output)}`,
  ].join(" ");
}

/**
 * Share of the successor's first-turn input that was a cache WRITE — the
 * number issue #5 is really about. 1.0 means the restart threw away every byte
 * of warmth and paid the premium to re-create it; a low number means the
 * stable prefix (DESIGN.md §4.5/§9) survived and was read back cheaply.
 *
 * Null when the provider reported no cache counters at all (most
 * OpenAI-compatible routes) rather than 0: "nothing was written" and "nobody
 * counted" are different answers, and averaging the second in as a zero would
 * quietly report a cheap restart that was never measured.
 */
function cacheWriteShare(row: RestartRow): number | null {
  if (row.usage_cache_read === null && row.usage_cache_creation === null) return null;
  const input = row.usage_input ?? 0;
  const read = row.usage_cache_read ?? 0;
  const write = row.usage_cache_creation ?? 0;
  const total = input + read + write;
  if (total <= 0) return null;
  return write / total;
}

const mean = (xs: number[]): number | null =>
  xs.length === 0 ? null : xs.reduce((a, b) => a + b, 0) / xs.length;

const STATS_USAGE =
  "usage: pinky stats <restarts|cache|sleep> ...\n" +
  "  pinky stats restarts [--channel <id>] [--limit N]\n" +
  "  pinky stats cache [--channel <id>] [--thread <id>] [--limit N]\n" +
  "    --limit samples the newest N turns (default 50); warm -> cold\n" +
  "    transitions are detected within that sampled window.\n" +
  "  pinky stats sleep [--channel <id>] [--limit N]\n" +
  "    the newest N sleep-worker receipts (default 50), both phases, every\n" +
  "    thread — reflect receipts live on `sleep:<agentId>`.";

/**
 * `pinky stats restarts` — what context restarts cost this tenant.
 *
 * DESIGN.md §13 lists "restarts discard cache warmth; measure $/task vs a
 * compaction baseline" as an open question. Because the loop journals a
 * `restart` event per boundary and the provider's `usage` on every assistant
 * turn, answering it is one read of the log: no sampling, no instrumentation
 * to switch on, and the time series is already there for a threshold change to
 * be judged against.
 *
 * Read-only, so it does not migrate (`migrate: false`) and takes no embedder.
 */
async function cmdStatsRestarts(raw: string[]): Promise<void> {
  const { flags } = parseFlags(raw);
  const channel = stringFlag(flags, "channel");
  const limit = intFlag(flags, "limit", 20);

  const boot = await bootstrap({ migrate: false, embedder: null, mcp: false });
  try {
    const rows = await boot.db.query<RestartRow>(RESTART_SQL, [
      boot.settings.tenantId,
      channel ?? null,
      limit,
    ]);

    console.log(
      `${padRight("thread", THREAD_COL)}  ${padLeft("bnd", 5)}  ${padLeft("before", 8)}    ` +
        `${padLeft("after", 7)}  ${padLeft("change", 15)}  ${padLeft("recall", 7)}  ` +
        `${padLeft("msgs", 4)}  first turn`,
    );
    for (const row of rows) {
      console.log(
        `${padRight(threadLabel(row), THREAD_COL)}  ` +
          `${padLeft(count(row.boundary_seq), 5)}  ` +
          `${padLeft(count(row.tokens_before), 8)} -> ${padLeft(count(row.tokens_after), 7)}  ` +
          `${padLeft(change(row.tokens_before, row.tokens_after), 15)}  ` +
          `${padLeft(count(row.recall_tokens), 7)}  ${padLeft(count(row.messages), 4)}  ` +
          firstTurn(row),
      );
    }
    if (rows.length === 0) {
      console.log(
        `(no restart events${channel ? ` in channel ${channel}` : ""} for tenant ${boot.settings.tenantId})`,
      );
    }

    const after = rows.map((r) => r.tokens_after).filter((n): n is number => n !== null);
    const shares = rows.map(cacheWriteShare).filter((n): n is number => n !== null);
    const meanAfter = mean(after);
    const meanShare = mean(shares);
    const rebuildCost = after.reduce((a, b) => a + b, 0);
    console.log("");
    console.log(
      `restarts ${rows.length}  ` +
        `mean tokensAfter ${meanAfter === null ? "n/a" : Math.round(meanAfter)}  ` +
        `mean cache-write share ${meanShare === null ? "n/a" : `${Math.round(meanShare * 100)}%`}` +
        ` (${shares.length}/${rows.length} turns reported cache usage)  ` +
        `est. rebuild cost ${rebuildCost} tokens`,
    );
  } finally {
    await boot.close();
  }
}

// --- stats cache -----------------------------------------------------------
//
// `stats restarts` only looks at the FIRST turn after a boundary, which answers
// "what did that restart cost" and nothing else. Prompt caching is a prefix
// match over tools -> system -> messages, so the number that actually decides
// $/task is the steady-state hit rate across every turn — and the moment it
// falls off a cliff. Both are already in the log: the loop journals the
// provider's `usage` on every assistant `message` event.

/**
 * One assistant turn that carried provider usage.
 *
 * `usage.input` is the UNCACHED prompt remainder, DISJOINT from `cacheRead`
 * and `cacheCreation` (Anthropic reports it that way; the OpenAI-compatible
 * route subtracts the cached count from `prompt_tokens` to match), so the
 * billed prompt for a turn is the sum of the three. A `null` here means the
 * provider did not report that counter — which is NOT the same as reporting
 * zero, and the two must never be averaged together.
 *
 * Every count comes back through a `::int` cast (the jsonb accessors are
 * text); `seq` is `bigint`, which postgres.js hands over as a STRING even
 * cast, so it is coerced with `toSeqNumber` before it is ever compared.
 */
interface CacheRow {
  channel_id: string;
  thread_id: string;
  seq: number | string;
  ts: string;
  model: string | null;
  usage_input: number | null;
  usage_output: number | null;
  usage_cache_read: number | null;
  usage_cache_creation: number | null;
}

const CACHE_SQL = `
  select channel_id,
         thread_id,
         seq::int                               as seq,
         ts,
         data->>'model'                         as model,
         (data->'usage'->>'input')::int         as usage_input,
         (data->'usage'->>'output')::int        as usage_output,
         (data->'usage'->>'cacheRead')::int     as usage_cache_read,
         (data->'usage'->>'cacheCreation')::int as usage_cache_creation
    from events
   where tenant_id = $1
     and type = 'message'
     and jsonb_typeof(data->'usage') = 'object'
     and ($2::text is null or channel_id = $2)
     and ($3::text is null or thread_id = $3)
   order by ts desc, seq desc
   limit $4`;

/** A turn's usage, provider-agnostic. `null` = the provider did not count it. */
export interface CacheTurn {
  channelId: string;
  threadId: string;
  seq: number;
  model: string | null;
  input: number | null;
  output: number | null;
  cacheRead: number | null;
  cacheCreation: number | null;
}

/**
 * `seq` is bigint -> string from postgres.js; never compare or sort it raw
 * (core/event-store.ts `toSeq` says what a leaked string breaks). Shared by
 * the cache rows below and by smoke's per-run mark.
 */
function toSeqNumber(value: number | string | null | undefined): number {
  if (value === null || value === undefined) return 0;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function toCacheTurn(row: CacheRow): CacheTurn {
  return {
    channelId: row.channel_id,
    threadId: row.thread_id,
    seq: toSeqNumber(row.seq),
    model: row.model,
    input: row.usage_input,
    output: row.usage_output,
    cacheRead: row.usage_cache_read,
    cacheCreation: row.usage_cache_creation,
  };
}

/**
 * Thread-major, seq-ascending — the order the transition analysis needs, done
 * on the JS side on purpose. `order by <text>` is server-collation dependent
 * (glibc en_US ignores hyphens at the first level, the alpine image's C locale
 * does not), so a SQL sort over channel/thread ids disagrees with a JS one for
 * some ids; keeping the comparison here makes the output identical on both.
 */
export function sortCacheTurns(turns: CacheTurn[]): CacheTurn[] {
  return [...turns].sort((a, b) => {
    if (a.channelId !== b.channelId) return a.channelId < b.channelId ? -1 : 1;
    if (a.threadId !== b.threadId) return a.threadId < b.threadId ? -1 : 1;
    return a.seq - b.seq;
  });
}

/** Did the provider count the cache at all? (Absent on most plain routes.) */
export function hasCacheCounters(turn: CacheTurn): boolean {
  return turn.cacheRead !== null || turn.cacheCreation !== null;
}

/** The billed prompt: uncached remainder + cache reads + cache writes. */
export function promptTokens(turn: CacheTurn): number {
  return (turn.input ?? 0) + (turn.cacheRead ?? 0) + (turn.cacheCreation ?? 0);
}

/**
 * Did the provider count cache WRITES specifically?
 *
 * Anthropic sends `cache_creation_input_tokens`; plain OpenAI and DeepSeek
 * send a hit count and nothing else. Anything derived from the write counter
 * therefore needs its own denominator — scoring a route that cannot report
 * writes as "0% rewritten" is the 0-vs-n/a conflation this module refuses
 * everywhere else.
 */
export function reportsCacheWrite(turn: CacheTurn): boolean {
  return turn.cacheCreation !== null && promptTokens(turn) > 0;
}

/**
 * Share of this turn's prompt served from cache — the headline number.
 *
 * Null (not 0) when the provider reported no cache counters: "nothing was
 * cached" and "nobody counted" are different answers, and averaging the second
 * in as a zero would report a cold agent that was merely unmeasured.
 */
export function cacheHitShare(turn: CacheTurn): number | null {
  if (!hasCacheCounters(turn)) return null;
  const total = promptTokens(turn);
  if (total <= 0) return null;
  return (turn.cacheRead ?? 0) / total;
}

/**
 * Almost the whole prompt was re-written rather than read: the Anthropic docs'
 * signature for "something upstream of the breakpoint changed" — a tool
 * definition, the system prefix, or a mutated messages[0]. This is the
 * actionable line in the summary, because every one of those causes is a bug
 * in how the request is assembled, not a fact about the workload.
 *
 * Only ever true for a turn that REPORTS a write counter, and the summary
 * scores it over exactly that set ({@link reportsCacheWrite}): on a route
 * without one, the honest answer is `n/a`, not 0%.
 */
const PREFIX_REWRITE_SHARE = 0.8;

export function isPrefixRewrite(turn: CacheTurn): boolean {
  if (!reportsCacheWrite(turn)) return false;
  return (turn.cacheCreation ?? 0) >= PREFIX_REWRITE_SHARE * promptTokens(turn);
}

/**
 * A read this size means the entry really was hit (it clears the smallest
 * cacheable prefix any current model has), so dropping to zero on the next
 * turn is an invalidation and not a cold start. It doubles as the floor for
 * the prompt that read nothing: below it there was nothing cacheable to lose.
 */
const COLD_READ_FLOOR = 1024;

/**
 * warm -> cold: the previous turn in this thread read a real cached prefix,
 * and this one read none from a prompt that was big enough to have been
 * cached. the warm→cold "cache-invalidation marker" pattern.
 *
 * Deliberately conditioned on the READ counter alone. Requiring a cache WRITE
 * (`cacheCreation > 0`) would make the marker — and the summary's
 * `cold transitions` — structurally unreachable on plain OpenAI and DeepSeek,
 * which report a hit count and no write count at all: a thread that went cold
 * would print `cold transitions 0`, which reads as "no invalidations" and is
 * really "not measurable". A turn that reports no counter at all after a
 * demonstrably warm one still counts: the route counted a read one turn ago,
 * so the absence now is the absence of the READ, not of the counter.
 */
export function isColdTransition(previous: CacheTurn | undefined, turn: CacheTurn): boolean {
  if (previous === undefined) return false;
  if ((previous.cacheRead ?? 0) < COLD_READ_FLOOR) return false;
  if ((turn.cacheRead ?? 0) !== 0) return false;
  return promptTokens(turn) > COLD_READ_FLOOR;
}

/** One turn with its derived numbers — what both the table and the summary read. */
export interface CacheTurnStats {
  turn: CacheTurn;
  prompt: number;
  hitShare: number | null;
  rewritten: boolean;
  cold: boolean;
}

/**
 * Sort, then walk each thread forward carrying its previous turn.
 *
 * Transitions are detected WITHIN THE SAMPLED WINDOW: `--limit` takes the
 * newest N turns across all threads, so the earliest turn of each thread in it
 * has no predecessor here and is never marked cold. That is the conservative
 * direction on both sides — a transition at the sample edge is missed rather
 * than invented, and no turn is ever compared against one from another thread.
 * Widen `--limit` (or narrow the scan with `--thread`) to see further back.
 */
export function analyzeCacheTurns(turns: CacheTurn[]): CacheTurnStats[] {
  const ordered = sortCacheTurns(turns);
  const previous = new Map<string, CacheTurn>();
  return ordered.map((turn) => {
    // NUL separator, written as an escape so this file stays plain text (a
    // literal NUL makes grep treat it as binary): no id can contain one, so
    // the two halves of the key can never run together into a collision.
    const key = `${turn.channelId}\u0000${turn.threadId}`;
    const stats: CacheTurnStats = {
      turn,
      prompt: promptTokens(turn),
      hitShare: cacheHitShare(turn),
      rewritten: isPrefixRewrite(turn),
      cold: isColdTransition(previous.get(key), turn),
    };
    previous.set(key, turn);
    return stats;
  });
}

export interface CacheSummary {
  /** Turns in the window (the `--limit` sample), measured or not. */
  turns: number;
  /**
   * The ONE denominator: turns whose provider counted the cache over a
   * non-empty prompt. `meanHitShare` and every token total below are computed
   * over exactly this set — never over `turns`, which would average and sum
   * unmeasured turns in as zeros.
   */
  measured: number;
  /** Mean of the per-turn hit shares over the `measured` turns. */
  meanHitShare: number | null;
  /** Token totals over the `measured` turns only (see `measured`). */
  read: number;
  write: number;
  uncached: number;
  prompt: number;
  coldTransitions: number;
  /**
   * Its own denominator: turns that reported a cache-WRITE counter. Plain
   * OpenAI/DeepSeek turns report reads only, so they are `measured` for the
   * hit rate and absent here.
   */
  writeMeasured: number;
  rewritten: number;
  /** `rewritten / writeMeasured` — null when no turn reported a write. */
  rewrittenShare: number | null;
}

export function summarizeCache(stats: CacheTurnStats[]): CacheSummary {
  // `hitShare` is non-null on exactly the turns that carry usable counters, so
  // deriving the count, the mean and the totals from this one array is what
  // keeps the printed denominator from drifting off the number beside it.
  const measured = stats.filter(
    (s): s is CacheTurnStats & { hitShare: number } => s.hitShare !== null,
  );
  const sum = (pick: (s: CacheTurnStats) => number): number =>
    measured.reduce((a, s) => a + pick(s), 0);
  const writeMeasured = stats.filter((s) => reportsCacheWrite(s.turn));
  const rewritten = writeMeasured.filter((s) => s.rewritten).length;
  return {
    turns: stats.length,
    measured: measured.length,
    meanHitShare: mean(measured.map((s) => s.hitShare)),
    read: sum((s) => s.turn.cacheRead ?? 0),
    write: sum((s) => s.turn.cacheCreation ?? 0),
    uncached: sum((s) => s.turn.input ?? 0),
    prompt: sum((s) => s.prompt),
    coldTransitions: stats.filter((s) => s.cold).length,
    writeMeasured: writeMeasured.length,
    rewritten,
    rewrittenShare: writeMeasured.length === 0 ? null : rewritten / writeMeasured.length,
  };
}

/** Width of the model column; a long `provider/vendor/model` id is tail-kept. */
const MODEL_COL = 26;

const pct = (share: number | null): string =>
  share === null ? "n/a" : `${Math.round(share * 100)}%`;

/**
 * `pinky stats cache` — the steady-state prompt-cache hit rate.
 *
 * SAMPLING: `--limit` (default 50) takes the newest N turns across every
 * thread the filters allow, and the analysis then runs per thread inside that
 * sample. So a thread's first SAMPLED turn has no predecessor and is never
 * marked `⊘ cold`, and a real invalidation that happened just before the
 * window is not counted — widen `--limit`, or pin `--thread`, to see further
 * back. Every summary number is scoped to the same sample.
 */
async function cmdStatsCache(raw: string[]): Promise<void> {
  const { flags } = parseFlags(raw);
  const channel = stringFlag(flags, "channel");
  const thread = stringFlag(flags, "thread");
  const limit = intFlag(flags, "limit", 50);

  const boot = await bootstrap({ migrate: false, embedder: null, mcp: false });
  try {
    const rows = await boot.db.query<CacheRow>(CACHE_SQL, [
      boot.settings.tenantId,
      channel ?? null,
      thread ?? null,
      limit,
    ]);
    const stats = analyzeCacheTurns(rows.map(toCacheTurn));

    // The value columns are self-labelled (`prompt N = read R + write W + …`),
    // so the header just sits over the numbers: each width below is the label
    // plus its padded count, exactly as the row below prints them.
    console.log(
      `${padRight("thread", THREAD_COL)}  ${padLeft("seq", 5)}  ${padRight("model", MODEL_COL)}  ` +
        `${padLeft("prompt", 14)}${padLeft("read", 15)}${padLeft("write", 16)}` +
        `${padLeft("uncached", 19)}${padLeft("hit", 10)}`,
    );
    for (const s of stats) {
      const t = s.turn;
      console.log(
        `${padRight(clip(`${t.channelId}/${t.threadId}`, THREAD_COL), THREAD_COL)}  ` +
          `${padLeft(String(t.seq), 5)}  ${padRight(clip(t.model ?? "-", MODEL_COL), MODEL_COL)}  ` +
          `prompt ${padLeft(String(s.prompt), 7)} = read ${padLeft(count(t.cacheRead), 7)}` +
          ` + write ${padLeft(count(t.cacheCreation), 7)}` +
          ` + uncached ${padLeft(count(t.input), 7)}  ` +
          `hit ${padLeft(pct(s.hitShare), 4)}` +
          (s.cold ? "  ⊘ cold" : ""),
      );
    }
    if (stats.length === 0) {
      const where = [channel ? ` in channel ${channel}` : "", thread ? ` thread ${thread}` : ""].join("");
      console.log(
        `(no assistant turns with usage${where} for tenant ${boot.settings.tenantId})`,
      );
    }

    // Three lines, each naming the set it is computed over: `measured` for the
    // hit rate and the token totals, the write-reporting subset for the rewrite
    // share. A number without its denominator is how "unmeasured" gets read as
    // "zero".
    const s = summarizeCache(stats);
    console.log("");
    console.log(
      `turns ${s.turns}  with cache counters ${s.measured}  ` +
        `mean hit ${pct(s.meanHitShare)}  cold transitions ${s.coldTransitions}`,
    );
    console.log(
      `tokens over the ${s.measured} measured turns  ` +
        `read ${s.read}  write ${s.write}  uncached ${s.uncached}  prompt ${s.prompt}`,
    );
    console.log(
      `prefix rewritten (write >= ${Math.round(PREFIX_REWRITE_SHARE * 100)}% of prompt) ` +
        (s.writeMeasured === 0
          ? "n/a (no turn reported a cache-write counter)"
          : `${s.rewritten}/${s.writeMeasured} turns reporting writes (${pct(s.rewrittenShare)})`),
    );
  } finally {
    await boot.close();
  }
}

// --- stats sleep -----------------------------------------------------------
//
// DESIGN.md §12 slice 6 says the sleep-time worker is "measured with `pinky
// stats`", and there is nothing to instrument: every pass journals a receipt
// (`sleep`/`extract` per thread, `sleep`/`reflect` on the worker's own thread)
// inside the transaction that made its memory writes, so what the worker cost
// and what it produced are already two columns of one query.

/** One `sleep` receipt. Both phases share the row; `phase` says which. */
interface SleepRow {
  channel_id: string;
  thread_id: string;
  seq: number | string;
  ts: string;
  phase: string | null;
  from_seq: number | null;
  to_seq: number | null;
  /** reflect only: the watermark this pass advanced the memory plane to. */
  through_at: string | null;
  scanned: number | null;
  candidates: number | null;
  added: number | null;
  updated: number | null;
  invalidated: number | null;
  noop: number | null;
  model: string | null;
  usage_input: number | null;
  usage_output: number | null;
  usage_cache_read: number | null;
  usage_cache_creation: number | null;
  ms: number | null;
}

/**
 * Every receipt, newest first, across EVERY thread — `sleep:<agentId>`
 * included, because that is where the reflect receipts live (the worker
 * journals its cross-thread bookkeeping in the log like everything else, and
 * discovery excludes that channel so it can never extract from itself).
 *
 * Every count comes back through a `::int` cast (the jsonb accessors are
 * text); `seq` is `bigint`, a STRING on the wire, so it goes through
 * `toSeqNumber` before it is compared to anything.
 */
const SLEEP_SQL = `
  select channel_id,
         thread_id,
         seq::int                               as seq,
         ts,
         data->>'phase'                         as phase,
         (data->>'fromSeq')::int                as from_seq,
         (data->>'toSeq')::int                  as to_seq,
         data->'through'->>'recordedAt'         as through_at,
         (data->>'scanned')::int                as scanned,
         (data->>'candidates')::int             as candidates,
         (data->>'added')::int                  as added,
         (data->>'updated')::int                as updated,
         (data->>'invalidated')::int            as invalidated,
         (data->>'noop')::int                   as noop,
         data->>'model'                         as model,
         (data->'usage'->>'input')::int         as usage_input,
         (data->'usage'->>'output')::int        as usage_output,
         (data->'usage'->>'cacheRead')::int     as usage_cache_read,
         (data->'usage'->>'cacheCreation')::int as usage_cache_creation,
         (data->>'ms')::int                     as ms
    from events
   where tenant_id = $1
     and type = 'sleep'
     and ($2::text is null or channel_id = $2)
   order by ts desc, seq desc
   limit $3`;

/**
 * What the worker actually put in the plane, counted from the plane itself
 * rather than summed off the receipts.
 *
 * The two numbers answer different questions and are both worth having: the
 * receipts say how many rows each pass WROTE, this says how many of them are
 * still CURRENT — a row the agent (or a later reflection) has since
 * invalidated is history, not belief (DESIGN.md §5.2).
 */
const SLEEP_MEMORY_SQL = `
  select count(*)::int as n
    from memories
   where tenant_id = $1
     and valid_to is null
     and meta->>'source' like 'sleep:%'`;

/** Width of the range column: `1..240` or `-> 2026-08-29T12:00:00`. */
const RANGE_COL = 24;

/** What range of material this pass consumed — a seq span, or a watermark. */
function sleepRange(row: SleepRow): string {
  if (row.phase === "reflect") return `-> ${(row.through_at ?? "-").slice(0, 19)}`;
  return `${count(row.from_seq)}..${count(row.to_seq)}`;
}

/** Sum of the four numbers `pinky sleep run` prints as `+A ~U -D =N`. */
function sleepCounts(row: SleepRow): string {
  return `+${count(row.added)} ~${count(row.updated)} -${count(row.invalidated)} =${count(row.noop)}`;
}

/** Billed tokens for one receipt, or `-` when no call reported usage. */
function sleepTokens(row: SleepRow): string {
  if (
    row.usage_input === null &&
    row.usage_output === null &&
    row.usage_cache_read === null &&
    row.usage_cache_creation === null
  ) {
    return "-";
  }
  return String(
    (row.usage_input ?? 0) +
      (row.usage_output ?? 0) +
      (row.usage_cache_read ?? 0) +
      (row.usage_cache_creation ?? 0),
  );
}

/** Per-phase totals. `tokens` is summed only over receipts that reported any. */
interface SleepTotals {
  passes: number;
  scanned: number;
  candidates: number;
  added: number;
  updated: number;
  invalidated: number;
  noop: number;
  tokens: number;
  /** Receipts whose pass reported usage — the `tokens` denominator. */
  measured: number;
  ms: number;
}

function totalSleep(rows: SleepRow[]): SleepTotals {
  const totals: SleepTotals = {
    passes: rows.length,
    scanned: 0,
    candidates: 0,
    added: 0,
    updated: 0,
    invalidated: 0,
    noop: 0,
    tokens: 0,
    measured: 0,
    ms: 0,
  };
  for (const row of rows) {
    totals.scanned += row.scanned ?? 0;
    totals.candidates += row.candidates ?? 0;
    totals.added += row.added ?? 0;
    totals.updated += row.updated ?? 0;
    totals.invalidated += row.invalidated ?? 0;
    totals.noop += row.noop ?? 0;
    totals.ms += row.ms ?? 0;
    const tokens = sleepTokens(row);
    if (tokens !== "-") {
      totals.tokens += Number(tokens);
      totals.measured++;
    }
  }
  return totals;
}

function sleepTotalsLine(phase: string, rows: SleepRow[]): string {
  const t = totalSleep(rows);
  return (
    `${padRight(phase, 8)} ${t.passes} pass(es)  scanned ${t.scanned}  candidates ${t.candidates}  ` +
    `+${t.added} ~${t.updated} -${t.invalidated} =${t.noop}  ` +
    `tokens ${t.measured === 0 ? "n/a" : `${t.tokens} over ${t.measured} measured pass(es)`}  ` +
    `ms ${t.ms}`
  );
}

/**
 * `pinky stats sleep` — what the sleep-time worker has done to this tenant.
 *
 * Read-only, so it does not migrate and takes no embedder — but it DOES boot
 * with `agent:pinky` in scope, unlike the other two stats commands: `tenantId`
 * is overlayable, and the receipts were written by `pinky sleep run` /
 * `pinky headless`, both of which resolve it from exactly this overlay.
 * Reading them back under a narrower snapshot could look in the wrong tenant.
 */
async function cmdStatsSleep(raw: string[]): Promise<void> {
  const { flags } = parseFlags(raw);
  const channel = stringFlag(flags, "channel");
  const limit = intFlag(flags, "limit", 50);

  const boot = await bootstrap({
    migrate: false,
    embedder: null,
    mcp: false,
    scopes: [`agent:${AGENT_ID}`],
  });
  try {
    const rows = await boot.db.query<SleepRow>(SLEEP_SQL, [
      boot.settings.tenantId,
      channel ?? null,
      limit,
    ]);

    console.log(
      `${padRight("thread", THREAD_COL)}  ${padRight("phase", 7)}  ${padRight("range", RANGE_COL)}  ` +
        `${padLeft("scanned", 7)}  ${padLeft("cand", 4)}  ${padRight("written", 16)}  ` +
        `${padLeft("tokens", 7)}  ${padLeft("ms", 6)}`,
    );
    for (const row of rows) {
      console.log(
        `${padRight(clip(`${row.channel_id}/${row.thread_id}`, THREAD_COL), THREAD_COL)}  ` +
          `${padRight(row.phase ?? "-", 7)}  ${padRight(sleepRange(row), RANGE_COL)}  ` +
          `${padLeft(count(row.scanned), 7)}  ${padLeft(count(row.candidates), 4)}  ` +
          `${padRight(sleepCounts(row), 16)}  ${padLeft(sleepTokens(row), 7)}  ` +
          `${padLeft(count(row.ms), 6)}`,
      );
    }
    if (rows.length === 0) {
      console.log(
        `(no sleep receipts${channel ? ` in channel ${channel}` : ""} for tenant ${boot.settings.tenantId};` +
          " run `pinky sleep run --now` or enable `sleep.enabled` in `pinky headless`)",
      );
    }

    console.log("");
    console.log(sleepTotalsLine("extract", rows.filter((r) => r.phase === "extract")));
    console.log(sleepTotalsLine("reflect", rows.filter((r) => r.phase === "reflect")));
    // The plane's own count, not a sum of the rows above: this one is not
    // capped by --limit and it excludes everything since invalidated.
    const written = await boot.db.queryOne<{ n: number }>(SLEEP_MEMORY_SQL, [
      boot.settings.tenantId,
    ]);
    // Tenant-wide and NOT filtered to this command's agent, unlike the boot
    // scope: the receipts above are every agent's too (a receipt carries no
    // agent id — the reflect ones only encode it in their `sleep:<agentId>`
    // channel), so filtering one half and not the other would print two
    // numbers that cannot be compared. Say which set it is instead.
    console.log(
      `memories written by the worker: ${written?.n ?? 0} (current rows, all agents, all time)`,
    );
  } finally {
    await boot.close();
  }
}

/** `pinky stats <restarts|cache|sleep>` — the DESIGN.md §13 eval, as queries. */
async function cmdStats(args: string[]): Promise<void> {
  const [sub, ...raw] = args;
  if (sub === "restarts") return cmdStatsRestarts(raw);
  if (sub === "cache") return cmdStatsCache(raw);
  if (sub === "sleep") return cmdStatsSleep(raw);
  throw new Error(STATS_USAGE);
}

// ---------------------------------------------------------------------------
// sleep — the sleep-time worker (DESIGN.md §5.3 item 3, slice 6)
// ---------------------------------------------------------------------------

/**
 * Assemble the worker's dependencies for one surface.
 *
 * Nothing here is stateful — that is the invariant the whole slice rests on
 * (CLAUDE.md #6): the sweep discovers its work by reading the log and journals
 * a receipt inside the transaction that made its memory writes, so this object
 * can be rebuilt from scratch on every process start with nothing to
 * reconcile.
 *
 * The model is `sleep.model` when set and the run model otherwise, and the
 * FULL `provider/model-id` is what gets journaled on every receipt — the log
 * has to answer "which model wrote this memory", and a bare id could not.
 * Splitting a worker model out at all is worth it because extraction is a
 * structured-output job on a transcript, which a small cheap model does well,
 * while the conversation may be on something expensive.
 */
function sleepDepsFor(
  boot: Bootstrap,
  opts: {
    scope: SleepScope;
    /** The snapshot to read `sleep.*` (and the fallback `model`) from. */
    settings: SettingsSnapshot;
    /** `--limit`: threads this sweep may touch, overriding maxThreadsPerSweep. */
    limit?: number;
    /** Smoke pins a per-run agent id so its reflect batch is its own rows. */
    agentId?: string;
    /** Smoke pins the provider (createFakeProvider), like makeRunAgent does. */
    provider?: Provider;
    /**
     * Shutdown switch. Without one every abort guard in the worker is DEAD:
     * a SIGTERM mid-sweep would leave it issuing provider calls for every
     * remaining due thread while `boot.close()` ends the pool underneath, and
     * the resulting failure takes the non-abort path — which journals an
     * `error` event onto a live conversation thread for something that was
     * only a shutdown. Every long-lived surface passes one.
     */
    signal?: AbortSignal;
  },
): SleepDeps {
  const model = opts.settings.sleep.model || opts.settings.model;
  return {
    db: boot.db,
    events: boot.events,
    memory: boot.memory,
    // Absent => FTS-only neighbour search and rows stored without an embedding.
    // Degraded, never fatal (the same ladder runtime/memory-recall.ts walks).
    ...(boot.embedder ? { embedder: boot.embedder } : {}),
    provider: opts.provider ?? createProvider(model, process.env),
    model,
    agentId: opts.agentId ?? AGENT_ID,
    tenantId: opts.settings.tenantId,
    settings:
      opts.limit === undefined
        ? opts.settings.sleep
        : { ...opts.settings.sleep, maxThreadsPerSweep: opts.limit },
    scope: opts.scope,
    ...(opts.signal ? { signal: opts.signal } : {}),
    // STDERR. `pinky sleep run` prints its report on stdout because a human
    // asked for it; everything the worker itself says is a log line, and this
    // same deps object is what runs inside `pinky headless`.
    log: logStderr,
  };
}

const SLEEP_USAGE =
  "usage: pinky sleep run [--now] [--channel <id>] [--thread <id>] [--limit N]\n" +
  "  --now      ignore the idle gate (sleep.idleMs) for this sweep\n" +
  "  --channel  restrict discovery to one channel\n" +
  "  --thread   pin ONE thread (requires --channel); discovery is skipped\n" +
  "  --limit N  threads this sweep may touch (overrides sleep.maxThreadsPerSweep)";

/** `+2 ~1 -0 =3` — what a pass did to the memory plane, in one column. */
function passCounts(r: {
  added: number;
  updated: number;
  invalidated: number;
  noop: number;
}): string {
  return `+${r.added} ~${r.updated} -${r.invalidated} =${r.noop}`;
}

/** Billed tokens for a pass, or `-` when no call reported usage — never 0.
 *  "nothing counted" and "nobody counted" are different facts (CLAUDE.md #7). */
function passTokens(usage: TokenUsage | undefined): string {
  if (!usage) return "-";
  return String(
    usage.input + usage.output + (usage.cacheRead ?? 0) + (usage.cacheCreation ?? 0),
  );
}

/** The bracketed detail both `pinky sleep run` and `pinky stats sleep` print. */
function passDetail(receipt: ExtractReceipt | ReflectReceipt): string {
  return (
    `scanned ${receipt.scanned}, candidates ${receipt.candidates}, ${passCounts(receipt)}, ` +
    `tokens ${passTokens(receipt.usage)}, ms ${receipt.ms}`
  );
}

/** One thread's or the reflect pass's outcome, as the human reads it. */
function passStatus(
  result:
    | SweepReport["threads"][number]["result"]
    | NonNullable<SweepReport["reflect"]>,
): string {
  if (result.status === "done") return `done  [${passDetail(result.receipt)}]`;
  if (result.status === "skipped") return `skipped (${result.reason})`;
  return `failed: ${result.error}`;
}

/**
 * `pinky sleep run` — one sweep, right now, from a terminal or a cron entry.
 *
 * The scheduler holds no state, so this and the `pinky headless` timer are the
 * SAME sweep with a different trigger: whichever runs, the receipts in the log
 * are the only record either consults. Running both is safe — the second takes
 * the thread lock, sees the first's receipt, and reports `lost-claim`.
 *
 * `--thread` pins the thread set outright (discovery is skipped), which is how
 * you re-run one conversation without waiting for its idle gate; it needs
 * `--channel` because a thread id is only unique inside its channel.
 *
 * Exits 1 when any pass FAILED. Skips are the ordinary outcome — most sweeps
 * find nothing new — and never a failure.
 */
async function cmdSleepRun(raw: string[]): Promise<void> {
  const { flags } = parseFlags(raw, ["now"]);
  const channel = stringFlag(flags, "channel");
  const threadId = stringFlag(flags, "thread");
  if (threadId !== undefined && channel === undefined) {
    throw new Error(
      "--thread requires --channel (a thread id is only unique inside its channel)",
    );
  }
  const limit = flags.limit === undefined ? undefined : intFlag(flags, "limit", 1);

  // No MCP plane: the worker has no tool surface of its own — it forces three
  // fixed schemas and reads the answers — so there is nothing to spawn a
  // configured server for. `agent:pinky` because `sleep.*` is the agent's own
  // configuration, the same scope `pinky headless` boots with.
  const boot = await bootstrap({ mcp: false, scopes: [`agent:${AGENT_ID}`] });
  let failed = 0;
  // A cron entry is exactly what a supervisor sends SIGTERM to, and a sweep
  // holds a thread lock across two provider round trips. Aborting is what lets
  // the pass return instead of being killed mid-transaction — and the abort
  // path deliberately journals NO `error` event, because a shutdown is not a
  // broken thread.
  const interrupted = new AbortController();
  let running: Promise<SweepReport> | undefined;
  const stopShutdown = installShutdown({
    drain: async () => {
      interrupted.abort(new Error("shutdown signal"));
      await running?.catch(() => {});
    },
    close: () => boot.close(),
  });
  try {
    const deps = sleepDepsFor(boot, {
      // A trusted local operator, like `pinky memory`: this terminal may read
      // and write `user` rows (DESIGN.md §5.1). `pinky headless --shared`
      // is the surface that narrows.
      scope: { includeUser: true, includePrivate: true },
      settings: boot.settings,
      signal: interrupted.signal,
      ...(limit !== undefined ? { limit } : {}),
    });
    running = sleepSweep(deps, {
      ...(threadId !== undefined && channel !== undefined
        ? {
            threads: [
              { tenantId: boot.settings.tenantId, channelId: channel, threadId },
            ],
          }
        : {}),
      ...(channel !== undefined ? { channelId: channel } : {}),
      ...(flags.now === true ? { ignoreIdle: true } : {}),
    });
    const report = await running;

    for (const { thread, result } of report.threads) {
      const label = clip(`${thread.channelId}/${thread.threadId}`, THREAD_COL);
      console.log(`${padRight(label, THREAD_COL)}  ${passStatus(result)}`);
      if (result.status === "failed") failed++;
    }
    if (report.threads.length === 0) {
      console.log(
        `(no threads due${channel ? ` in channel ${channel}` : ""}` +
          `${flags.now === true ? "" : `; idle gate is ${boot.settings.sleep.idleMs}ms — --now ignores it`})`,
      );
    }
    // Reflection is cross-thread, so it gets its own line rather than a row.
    // `null` means it was not attempted at all, which is not the same as a skip.
    console.log(
      `${padRight("reflect", THREAD_COL)}  ` +
        (report.reflect === null ? "not attempted" : passStatus(report.reflect)),
    );
    if (report.reflect?.status === "failed") failed++;
    console.log(`model ${deps.model}`);
    // The sweep gave up after two CONSECUTIVE threads failed identically —
    // a broken provider or a dead database, not one bad thread. Everything
    // after it was never attempted, so `threads` above is a PARTIAL report and
    // this run must not look like a clean one to a cron entry.
    if (report.halted !== undefined) {
      console.log(`halted: ${report.halted}`);
      failed++;
    }
  } finally {
    stopShutdown();
    // Nothing is in flight on the normal path; this is so the controller never
    // outlives the command with a live listener on it.
    interrupted.abort(new Error("sweep finished"));
    await boot.close();
  }
  if (failed > 0) {
    console.error(
      `sleep run: ${failed} pass(es) failed or halted (see the error events on those threads)`,
    );
    process.exit(1);
  }
}

async function cmdSleep(args: string[]): Promise<void> {
  const [sub, ...rest] = args;
  if (sub === "run") return cmdSleepRun(rest);
  throw new Error(SLEEP_USAGE);
}

// ---------------------------------------------------------------------------
// mcp / tools — the human's window on slice 9
// ---------------------------------------------------------------------------

const MCP_USAGE =
  "usage: pinky mcp <list|sync> ...\n" +
  "  pinky mcp list                              configured servers and their state\n" +
  "  pinky mcp sync [<server>...] [--timeout-ms N]  connect, wait, republish the catalog";

/** Column widths for `pinky mcp list`. Server keys are capped at 32 chars by
 *  their own regex, so nothing here needs clipping except the error. */
const MCP_COL = { server: 18, status: 14, era: 7, protocol: 12, name: 24 };
/** One line of `lastError` is enough to recognize the failure; the full text
 *  is in the stderr log the manager already wrote. */
const MCP_ERROR_CHARS = 60;

function mcpRow(state: McpServerState): string {
  return (
    `${padRight(state.server, MCP_COL.server)}  ` +
    `${padRight(state.status, MCP_COL.status)}  ` +
    `${padRight(state.era ?? "-", MCP_COL.era)}  ` +
    `${padRight(state.protocolVersion ?? "-", MCP_COL.protocol)}  ` +
    `${padRight(clip(state.serverName ?? "-", MCP_COL.name), MCP_COL.name)}  ` +
    `${padLeft(String(state.toolCount), 5)}  ` +
    clip(state.lastError ?? "-", MCP_ERROR_CHARS)
  );
}

/**
 * `pinky mcp list` — what the MCP plane thinks of each configured server.
 *
 * A snapshot of a process that has just started, so `trusted-cache` and
 * `connecting` are the normal answers here: bootstrap does not wait for a
 * server (that is the point), and the catalog is what makes request 1
 * answerable anyway. `pinky mcp sync` is the command that waits.
 */
async function cmdMcpList(): Promise<void> {
  // No embedder: nothing here recalls anything, and a missing API key should
  // not print a memory warning over a table about MCP.
  const boot = await bootstrap({ scopes: [`agent:${AGENT_ID}`], embedder: null });
  try {
    const states = boot.mcp.states();
    console.log(
      `${padRight("server", MCP_COL.server)}  ${padRight("status", MCP_COL.status)}  ` +
        `${padRight("era", MCP_COL.era)}  ${padRight("protocol", MCP_COL.protocol)}  ` +
        `${padRight("server name", MCP_COL.name)}  ${padLeft("tools", 5)}  last error`,
    );
    for (const state of states) console.log(mcpRow(state));
    if (states.length === 0) {
      console.log(
        "(no mcp servers configured; add one with " +
          `\`pinky config set mcp.servers.<name> '{"transport":"stdio","command":"..."}'\`)`,
      );
    } else if (states.some((s) => s.status === "trusted-cache" || s.status === "connecting")) {
      // Not a defect: this process started a moment ago and never waits for a
      // server. Say so, so "connecting" does not read as "stuck".
      console.log("");
      console.log(
        "(a server is still connecting — this command never waits; `pinky mcp sync` does)",
      );
    }
  } finally {
    await boot.close();
  }
}

/**
 * `pinky mcp sync [<server>...]` — connect, wait, and say what the catalog holds.
 *
 * The one place the CLI blocks on an MCP server, because it is the one place a
 * human asked it to: after adding or repointing a server you want to know
 * whether it answers and what it published, not to discover it later through a
 * `tool_search` that returns nothing. Exits 1 if any named server errored, so
 * it is usable as a deployment step.
 */
async function cmdMcpSync(raw: string[]): Promise<void> {
  const { flags, rest } = parseFlags(raw);
  const timeoutMs = intFlag(flags, "timeout-ms", 15_000);
  const boot = await bootstrap({ scopes: [`agent:${AGENT_ID}`], embedder: null });
  let failed = 0;
  try {
    const configured = boot.mcp.states().map((s) => s.server);
    const wanted = rest.length > 0 ? rest : configured;
    const unknown = wanted.filter((s) => !configured.includes(s));
    if (unknown.length > 0) {
      throw new Error(
        `not configured: ${unknown.join(", ")}` +
          (configured.length > 0 ? ` (configured: ${configured.join(", ")})` : ""),
      );
    }
    if (wanted.length === 0) {
      console.log("no mcp servers configured; nothing to sync");
      return;
    }
    // Concurrently: two slow servers should cost one timeout, not two.
    const settled = await Promise.all(
      wanted.map((server) => waitForServer(boot.mcp, server, timeoutMs)),
    );
    for (const [i, state] of settled.entries()) {
      const server = wanted[i]!;
      if (!state) {
        console.log(`${padRight(server, MCP_COL.server)}  not configured`);
        failed++;
        continue;
      }
      // The catalog, not the client: what a NEXT process would be served.
      const published = await boot.catalog.serverState(server);
      console.log(
        `${padRight(server, MCP_COL.server)}  ${padRight(state.status, MCP_COL.status)}  ` +
          `${padRight(state.era ?? "-", MCP_COL.era)}  ` +
          `${padRight(state.protocolVersion ?? "-", MCP_COL.protocol)}  ` +
          `catalog ${padLeft(String(published?.count ?? 0), 4)} tool(s)` +
          (state.lastError ? `  ${clip(state.lastError, MCP_ERROR_CHARS)}` : ""),
      );
      if (state.status !== "connected") failed++;
    }
  } finally {
    await boot.close();
  }
  if (failed > 0) {
    console.error(`mcp sync: ${failed} server(s) did not connect`);
    process.exit(1);
  }
}

async function cmdMcp(args: string[]): Promise<void> {
  const [sub, ...rest] = args;
  if (sub === "list") return cmdMcpList();
  if (sub === "sync") return cmdMcpSync(rest);
  throw new Error(MCP_USAGE);
}

const TOOLS_USAGE =
  "usage: pinky tools list [--scope <scope>]...\n" +
  "  --scope may be repeated (global | channel:<id> | agent:<id>);\n" +
  `  default: the scopes an agent surface boots with (global + agent:${AGENT_ID}).`;

/** Every `--scope X` / `--scope=X` in `args`, validated, with the rest of the
 *  argv returned. parseFlags keeps only the LAST value of a repeated flag, and
 *  this one is legitimately repeatable — an overlay is a list. */
function collectScopes(args: string[]): { scopes: string[]; rest: string[] } {
  const scopes: string[] = [];
  const rest: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--scope") {
      const value = args[++i];
      if (value === undefined) throw new Error("--scope requires a value");
      scopes.push(assertScope(value));
      continue;
    }
    if (arg.startsWith("--scope=")) {
      const value = arg.slice("--scope=".length);
      if (!value) throw new Error("--scope requires a value");
      scopes.push(assertScope(value));
      continue;
    }
    rest.push(arg);
  }
  return { scopes, rest };
}

/** A catalog row as `partitionTools` sees it. The execute() is never called —
 *  this command reports the PARTITION, and the partition is a pure function of
 *  the names and the settings. Building a real tool would mean connecting to
 *  every server just to print a table. */
function catalogAsTool(record: CatalogRecord): Tool {
  return {
    name: record.name,
    description: record.description,
    parameters: record.parameters,
    execute: () =>
      Promise.resolve({ text: `${record.name} is not executable from \`pinky tools list\``, isError: true }),
  };
}

/**
 * `pinky tools list` — the header/catalog split, exactly as a run computes it.
 *
 * Not a re-implementation of the rule: it calls the same `partitionTools` with
 * the same settings overlay a wake would load, so what it prints is the
 * PARTITION a run with these scopes would compute. The header is the provider
 * cache key and the most expensive thing in a request, and until now the only
 * way to see the split was to run the agent and read a log.
 *
 * It is a view of the CATALOG, not of one process's tool objects, so the
 * counts are a superset of any single request: catalogued built-ins from
 * OTHER surfaces are included (`bash`, registered by `pinky prompt`, is
 * `head` here even in a headless deployment that never loads it), and so are
 * tools from servers that are currently offline — which a run would still
 * find via tool_search, since the catalog outlives a connection.
 */
async function cmdToolsList(raw: string[]): Promise<void> {
  const { scopes } = collectScopes(raw);
  const wanted = scopes.length > 0 ? scopes : [`agent:${AGENT_ID}`];
  // No MCP plane: the catalog already knows every name, and listing tools must
  // not spawn a server (nor wait for one).
  const boot = await bootstrap({ scopes: wanted, mcp: false, embedder: null });
  try {
    // `boot.settings` IS the overlay for `wanted` (bootstrap loaded it with
    // exactly those scopes), so this is the snapshot a run in that scope reads.
    const settings = boot.settings;
    const records = await boot.catalog.entries();
    const { head, deferred } = partitionTools(records.map(catalogAsTool), settings.tools, sourceOf);
    const mode = new Map<string, string>();
    for (const tool of head) mode.set(tool.name, "head");
    for (const tool of deferred) mode.set(tool.name, "deferred");

    console.log(
      `${padRight("mode", 8)}  ${padRight("source", 7)}  ${padRight("server", MCP_COL.server)}  name`,
    );
    for (const record of records) {
      console.log(
        `${padRight(mode.get(record.name) ?? "?", 8)}  ${padRight(record.source, 7)}  ` +
          `${padRight(record.server ?? "-", MCP_COL.server)}  ${record.name}`,
      );
    }
    if (records.length === 0) {
      console.log("(the tool catalog is empty; run `pinky prompt`, `pinky headless` or `pinky mcp sync` once)");
    }
    console.log("");
    console.log(
      `scopes ${wanted.join(" + ")}  head ${head.length} (rendered in every request)  ` +
        `deferred ${deferred.length} (reached via tool_search/tool_describe/tool_call)  ` +
        `defaultMode builtin=${settings.tools.defaultMode.builtin} mcp=${settings.tools.defaultMode.mcp}`,
    );
  } finally {
    await boot.close();
  }
}

async function cmdTools(args: string[]): Promise<void> {
  const [sub, ...rest] = args;
  if (sub === "list") return cmdToolsList(rest);
  throw new Error(TOOLS_USAGE);
}

// ---------------------------------------------------------------------------
// smoke
// ---------------------------------------------------------------------------

/** Matches the `memories.embedding vector(1536)` column on pgvector images —
 *  a narrower fake vector is rejected by the column type, so smoke would pass
 *  on alpine and fail in CI. */
const SMOKE_EMBEDDING_DIMENSIONS = 1536;
/** Agent id owning every memory smoke writes, so cleanup is one exact delete. */
const SMOKE_MEMORY_AGENT = "smoke";
/** Distinctive enough that the FTS voice can find it and nothing else. */
const SMOKE_CANARY = "The smoke canary passphrase is zebra-quartz.";
const SMOKE_QUERY = "smoke canary passphrase";

/** Settings key the smoke MCP server is registered under, hence the prefix of
 *  every tool it publishes (`mcp__smokefx__…`). Not in the settings table:
 *  smoke builds its own manager so a dev machine's real servers stay out. */
const SMOKE_MCP_SERVER = "smokefx";
/** The modern (2026-07-28) stdio fixture, spawned as a real child process —
 *  this leg is worth having precisely because it is not a mock. */
const SMOKE_MCP_FIXTURE = new URL("../../mcp/test/fixtures/modern-server.ts", import.meta.url).pathname;
/** How long smoke waits for that child to spawn, negotiate and publish. */
const SMOKE_MCP_TIMEOUT_MS = 20_000;
/** How long a closed manager gets to reap its child before it is an orphan. */
const SMOKE_MCP_REAP_MS = 3_000;
const SMOKE_MCP_PROMPT = "please echo pinky-smoke-42 through a catalog tool";

/**
 * Agent id owning every memory row and receipt the sleep leg writes (slice 6).
 *
 * Deliberately NOT `SMOKE_MEMORY_AGENT`: the reflect pass reads across every
 * channel (RecallScope.allChannels), so an agent id shared with the other legs
 * would put their canaries in its batch and make `added` unpredictable. A
 * smoke-only id pins the batch to what this run just extracted, and makes the
 * cleanup one exact delete — the same trick `SMOKE_MEMORY_AGENT` plays.
 */
const SMOKE_SLEEP_AGENT = "smoke-sleep";
/** `fake/sleep` extracts whatever follows `remember:` verbatim, so this is the
 *  exact text the pass must end up storing. */
const SMOKE_SLEEP_CANARY = "the smoke sleep canary is amber-falcon";

/**
 * Newest seq in a thread, or 0 when it has none — smoke's per-run mark.
 *
 * NOT `(await events.history(ref)).length`. `history()` is a FORWARD page
 * capped at `DEFAULT_HISTORY_PAGE` (500) rows from the oldest event, so on a
 * long-lived database the fixed `cli:smoke/*` threads outgrow one page and the
 * length stops moving: the mark sticks at 500, `history(...).slice(500)` is
 * empty, and every check reading this run's events fails on an empty slice
 * (which is exactly how a red `bun run smoke` showed up on a dev DB while CI's
 * fresh database hid it). A seq mark does not grow with the log, and
 * `history(ref, { afterSeq })` pages forward from it.
 *
 * One query on the shared `Db` rather than a new EventStore seam: the store is
 * append-and-read-forward on purpose, and smoke is the only caller that needs
 * "where does the log end right now".
 */
async function latestSeq(db: Db, ref: ThreadRef): Promise<number> {
  const row = await db.queryOne<{ seq: number | string }>(
    `select seq from events
      where (tenant_id, channel_id, thread_id) = ($1, $2, $3)
      order by seq desc limit 1`,
    [ref.tenantId, ref.channelId, ref.threadId],
  );
  // bigint -> string on the wire; a raw string would compare lexicographically.
  return toSeqNumber(row?.seq);
}

/**
 * Wait for one MCP server to SETTLE — `connected` or `error` — or give up.
 *
 * The manager deliberately has no "await my servers" method: a run must never
 * block on a child process spawning (McpManager rule 1), and the catalog is
 * what makes request 1 answerable without one. But a human typing
 * `pinky mcp sync` and a smoke check are the two callers that do want the
 * answer, so the waiting lives here, in the surface that asked, rather than in
 * the plane that must not do it.
 *
 * Polled rather than event-driven on purpose: `states()` is the manager's only
 * public read, and a poll cannot miss a transition that happened between two
 * awaits. Returns the last state seen; `undefined` only when the server is not
 * configured at all.
 */
async function waitForServer(
  mcp: McpManager,
  server: string,
  timeoutMs: number,
): Promise<McpServerState | undefined> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const state = mcp.state(server);
    if (!state) return undefined;
    if (state.status === "connected" || state.status === "error") return state;
    if (Date.now() >= deadline) return state;
    await Bun.sleep(50);
  }
}

/** True once `pid` is gone (or was never there). A closed stdio transport has
 *  killed its child and awaited its exit, so this normally answers on the
 *  first probe; the poll covers the SIGKILL path, which does not wait. */
async function childReaped(pid: number | null | undefined, timeoutMs: number): Promise<boolean> {
  if (pid === null || pid === undefined) return true;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      // Signal 0 tests for existence without delivering anything.
      process.kill(pid, 0);
    } catch {
      return true;
    }
    if (Date.now() >= deadline) return false;
    await Bun.sleep(50);
  }
}

async function cmdSmoke(): Promise<void> {
  // Deterministic, offline, and dimension-matched to the real column.
  const embedder = new FakeEmbedder({ dimensions: SMOKE_EMBEDDING_DIMENSIONS });
  // peers/secret emptied: smoke is a single-node in-process check and must not
  // start POSTing to a peer configured in someone's .env.
  // `mcp: false`: smoke brings its OWN manager (leg 4) pointed at the fixture,
  // and must neither depend on nor disturb whatever servers this machine has
  // configured — spawning them here would make a local `pinky config` decide
  // whether smoke passes.
  const boot = await bootstrap({ peers: {}, a2aSecret: "", embedder, mcp: false });
  const { env, settings, db, events, messenger } = boot;
  const builtins = createTools();
  // Slice 9: what this surface can run is what the catalog should say it can.
  await registerBuiltins(boot, builtins);
  // The memory plane is the thing under test here, so nothing about it is left
  // to the dev database's settings rows.
  const smokeSettings: SettingsSnapshot = {
    ...settings,
    memory: { ...settings.memory, autoRecall: true },
  };

  const threadA: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "alpha" };
  const threadB: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "beta" };
  const threadM: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "memory" };
  const threadR: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "memory-recall" };
  const threadT: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "mcp-tools" };
  const threadS: ThreadRef = { tenantId: settings.tenantId, channelId: "cli:smoke", threadId: "sleep" };
  /** Where the reflect receipt lands: the worker's own thread (slice 6). */
  const threadSR = reflectThread(settings.tenantId, SMOKE_SLEEP_AGENT);

  // `alpha` and `beta` are smoke's own addresses on this node, and the receipt
  // counts below are exact counts — so anything an interrupted earlier run
  // left in this node's partition is cleared first.
  await db.query(
    `delete from a2a_messages where node_to = $1 and to_agent in ('alpha', 'beta')`,
    [env.nodeId],
  );

  // Smoke reuses fixed thread ids, so the log carries every previous run.
  // Everything asserted below is read from these marks FORWARD — otherwise a
  // run that journaled nothing would still "pass" on yesterday's events.
  const [markA, markB, markM, markT, markS, markSR] = await Promise.all([
    latestSeq(db, threadA),
    latestSeq(db, threadB),
    latestSeq(db, threadM),
    latestSeq(db, threadT),
    latestSeq(db, threadS),
    latestSeq(db, threadSR),
  ]);

  const betaHeard: A2AEnvelope[] = [];
  messenger.onMessage("beta", (env2) => {
    betaHeard.push(env2);
  });

  // --- 1. A2A round trip -------------------------------------------------
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
  const runA = await makeRunAgent(boot, {
    agentId: "alpha",
    builtins,
    provider: new FakeProvider(alphaScript),
    settingsFor: () => Promise.resolve(smokeSettings),
  })(threadA);

  // --- 1b. the consumption receipt (issue #4) ----------------------------
  // The row alpha just sent is DELIVERED (local send marks it) but nothing has
  // consumed it: the subscriber above only appended to an array. That is
  // exactly the state a crash between the delivery claim and the agent's turn
  // leaves behind, and `delivered_at` cannot tell the two apart — so recovery
  // keys on the receipt (`read_at`) instead and re-fires everything unread.
  const heardAfterSend = betaHeard.length;
  const firedUnread = await messenger.redeliverUnconsumed("beta");
  const heardAfterRedeliver = betaHeard.length;
  // inbox() stamps the receipt (same column claimConsumption writes), so the
  // very next redelivery has nothing left to fire.
  const inbox = await messenger.inbox("beta");
  const firedAfterReceipt = await messenger.redeliverUnconsumed("beta");
  const claimedAfterReceipt =
    betaHeard[0] !== undefined && (await messenger.claimConsumption(betaHeard[0].id));

  // --- 2. retain -> recall round trip through the tools ------------------
  const memoryScope: RecallScope = {
    agentId: SMOKE_MEMORY_AGENT,
    channelId: threadM.channelId,
    userId: "local",
    includeUser: true,
    includePrivate: true,
  };
  const memoryScript: AssistantTurn[] = [
    {
      text: "",
      toolCalls: [
        { id: "m1", name: "retain", args: { text: SMOKE_CANARY, kind: "semantic", importance: 7 } },
      ],
      stopReason: "tool_calls",
    },
    {
      text: "",
      toolCalls: [{ id: "m2", name: "recall", args: { query: SMOKE_QUERY } }],
      stopReason: "tool_calls",
    },
    { text: "Retained and recalled.", toolCalls: [], stopReason: "stop" },
  ];
  const runM = await makeRunAgent(boot, {
    agentId: SMOKE_MEMORY_AGENT,
    builtins,
    provider: new FakeProvider(memoryScript),
    settingsFor: () => Promise.resolve(smokeSettings),
    scopeFor: () => memoryScope,
  })(threadM);

  // --- 3. auto-recall on a FRESH thread ----------------------------------
  // Nothing of the run above is in this thread's log, so a `<memories>` block
  // at messages[0] can only have come from the memory plane (DESIGN.md §5.4).
  await events.append(threadR, {
    type: "ingress",
    platform: "cli",
    author: { platform: "cli", userId: "local" },
    text: `what is the ${SMOKE_QUERY}?`,
    refs: [],
  });
  const prompts: LlmMessage[][] = [];
  const recallProvider = new FakeProvider((messages) => {
    // Snapshot: the loop mutates the array it hands the provider.
    prompts.push(messages.map((m) => ({ ...m })));
    return { text: "Recalled.", toolCalls: [], stopReason: "stop" };
  });
  const runR = await makeRunAgent(boot, {
    agentId: SMOKE_MEMORY_AGENT,
    builtins,
    provider: recallProvider,
    settingsFor: () => Promise.resolve(smokeSettings),
    scopeFor: () => ({ ...memoryScope, channelId: threadR.channelId }),
  })(threadR);

  // --- 4. deferred tools over a REAL MCP server (slice 9) ----------------
  //
  // The whole point of the slice in one leg: a tool that is NOT in the request
  // header is found by search, read by describe, executed by call, and its
  // output comes back in the assistant's answer — while the header itself is
  // unchanged (three meta-tools in, zero `mcp__` tools). The server is the
  // modern stdio fixture spawned as a child process, because a mock cannot
  // fail the way a real one can.
  //
  // Its own manager, not `boot.mcp`: smoke must not depend on (or disturb) the
  // servers a dev machine has configured in its settings table.
  const spawnedTransports: { pid?: number | null }[] = [];
  const smokeMcp = new McpManager({
    servers: {
      [SMOKE_MCP_SERVER]: { transport: "stdio", command: "bun", args: ["run", SMOKE_MCP_FIXTURE] },
    },
    catalog: boot.catalog,
    // stderr: smoke prints its own PASS/FAIL lines on stdout and the manager's
    // connect chatter is not one of them.
    log: logStderr,
    env: process.env,
    // The default factory, wrapped only to keep the child's pid so the orphan
    // check below can be an assertion instead of a hope.
    transportFactory: (server, config, mcpEnv) => {
      const transport = defaultTransportFactory(server, config, mcpEnv);
      spawnedTransports.push(transport as unknown as { pid?: number | null });
      return transport;
    },
  });

  let mcpNames: string[] = [];
  let headNames: string[] = [];
  let runT: AgentRunResult | undefined;
  let historyT: Awaited<ReturnType<typeof events.history>> = [];
  let mcpState: McpServerState | undefined;
  let reaped = false;
  try {
    await smokeMcp.start();
    mcpState = await waitForServer(smokeMcp, SMOKE_MCP_SERVER, SMOKE_MCP_TIMEOUT_MS);
    mcpNames = await boot.catalog.listNames({ server: SMOKE_MCP_SERVER });

    await events.append(threadT, {
      type: "ingress",
      platform: "cli",
      author: { platform: "cli", userId: "local" },
      text: SMOKE_MCP_PROMPT,
      refs: [],
    });
    // `fake/deferred` scripts the four turns; `received` is how the header the
    // provider actually saw is read back (a FakeProvider records every
    // CompleteOptions), which is the only honest way to assert on a cache key.
    const deferredProvider = createFakeProvider("deferred");
    runT = await makeRunAgent(
      { ...boot, mcp: smokeMcp },
      {
        agentId: SMOKE_MEMORY_AGENT,
        builtins,
        provider: deferredProvider,
        settingsFor: () => Promise.resolve(smokeSettings),
      },
    )(threadT);
    headNames = deferredProvider.received[0]?.tools.map((t) => t.name) ?? [];
    historyT = await events.history(threadT, { afterSeq: markT });
  } finally {
    // Closing the manager must reap the child; an orphaned `bun run` holding
    // this process's pipes is exactly the failure this leg exists to catch.
    await smokeMcp.close();
    reaped = (
      await Promise.all(
        spawnedTransports.map((t) => childReaped(t.pid, SMOKE_MCP_REAP_MS)),
      )
    ).every(Boolean);
  }
  const toolResultNames = historyT
    .filter((e) => e.data.type === "tool_result")
    .map((e) => (e.data.type === "tool_result" ? e.data.name : ""));
  const finalText = [...historyT]
    .reverse()
    .find((e) => e.data.type === "message")
    ?.data;
  const answer = finalText?.type === "message" ? finalText.text : "";
  // Cleanup, like the memories below: DELETE, not the store's usual withdrawal.
  // These rows are a fixture's residue in a shared dev database, and a
  // `mcp__smokefx__*` row outliving the process that could run it would offer
  // every later tool_search a tool nothing can execute.
  await db.query(`delete from tool_catalog where tenant_id = $1 and server = $2`, [
    settings.tenantId,
    SMOKE_MCP_SERVER,
  ]);

  // --- 5. the sleep-time worker (slice 6) --------------------------------
  //
  // The whole slice in one leg: events the agent already lived through become
  // memory-plane rows while nothing is talking, and the pass's RECEIPT is
  // journaled in the same transaction as the rows (DESIGN.md §5.3 item 3,
  // CLAUDE.md invariant #6) — so "did it run" and "what did it write" are the
  // same question asked of the log. `fake/sleep` answers the three forced tool
  // calls, which is what makes this keyless and deterministic.
  //
  // Nothing here touches live context: every event a pass appends is
  // audit-only, so this thread's rendered prompt is byte-identical before and
  // after (DESIGN.md §3, §4.5).
  await events.appendBatch(threadS, [
    {
      type: "ingress",
      platform: "cli",
      author: { platform: "cli", userId: "local" },
      text: `remember: ${SMOKE_SLEEP_CANARY}`,
      refs: [],
    },
    // A second event so the consumed range is a real span, and so the
    // transcript renderer has more than one line to render.
    { type: "message", role: "assistant", text: "Noted.", toolCalls: [], model: "fake/sleep" },
  ]);
  const sleepSettings: SettingsSnapshot = {
    ...settings,
    // Not the dev database's rows — this leg IS the thing under test.
    // `reflectMinMemories: 1` is what makes the single row this pass extracts
    // enough to trigger the consolidation pass in the same sweep.
    sleep: { ...DEFAULT_SETTINGS.sleep, model: "fake/sleep", reflectMinMemories: 1 },
  };
  const sleepDeps = sleepDepsFor(boot, {
    // The local surface is trusted (DESIGN.md §5.1), like `pinky sleep run`.
    scope: { includeUser: true, includePrivate: true },
    settings: sleepSettings,
    agentId: SMOKE_SLEEP_AGENT,
    provider: createFakeProvider("sleep"),
  });
  const sleepReport = await sleepSweep(sleepDeps, { threads: [threadS], ignoreIdle: true });
  // Idempotence, which is the property that makes a stateless scheduler safe:
  // the cursor moved past everything extractable, and everything the pass
  // itself appended (`memory`, `sleep`) is material the worker does not read.
  // `reflect: false` because the first sweep's insight is itself a new memory
  // row, so a second reflection would consolidate the consolidation — legal
  // (and bounded by reflectMinMemories), but not what this check is about.
  const sleepAgain = await sleepSweep(sleepDeps, {
    threads: [threadS],
    ignoreIdle: true,
    reflect: false,
  });
  const sleepSecond = sleepAgain.threads[0]?.result;

  // Read back through the plane's own scope predicate, not by id: a row the
  // §5.1 predicate cannot see is a row the agent will never recall.
  const sleepHits = await boot.memory.search({
    scope: {
      agentId: SMOKE_SLEEP_AGENT,
      channelId: threadS.channelId,
      userId: "local",
      includeUser: true,
      includePrivate: true,
    },
    query: "smoke sleep canary amber-falcon",
    limit: 10,
  });
  const isExtractReceipt = (d: ThreadEventData): d is ExtractReceipt =>
    d.type === "sleep" && d.phase === "extract";
  const isReflectReceipt = (d: ThreadEventData): d is ReflectReceipt =>
    d.type === "sleep" && d.phase === "reflect";
  const extractReceipt = (await events.history(threadS, { afterSeq: markS }))
    .map((e) => e.data)
    .find(isExtractReceipt);
  const reflectReceipt = (await events.history(threadSR, { afterSeq: markSR }))
    .map((e) => e.data)
    .find(isReflectReceipt);

  // From this run's marks forward: one forward page each, which is orders of
  // magnitude more than a run appends.
  const historyA = await events.history(threadA, { afterSeq: markA });
  const historyB = await events.history(threadB, { afterSeq: markB });
  const historyM = await events.history(threadM, { afterSeq: markM });
  const recallResult = historyM.find(
    (e) => e.data.type === "tool_result" && e.data.name === "recall",
  );
  const recallText = recallResult?.data.type === "tool_result" ? recallResult.data.text : "";
  const injected = prompts[0]?.[0];

  // Cleanup before reporting: a failed check must not leave rows behind. The
  // sleep agent's rows go too — the next run's reflect batch reads across
  // channels, and yesterday's canary in it would make `added` unpredictable.
  await db.query(`delete from memories where tenant_id = $1 and agent_id = any($2)`, [
    settings.tenantId,
    [SMOKE_MEMORY_AGENT, SMOKE_SLEEP_AGENT],
  ]);

  const checks: [string, boolean][] = [
    ["alpha ran to completion", runA.stopReason === "completed"],
    ["a2a message delivered to beta", inbox.length === 1 && inbox[0]!.text === "what is 2+2?"],
    ["live subscriber saw the message", heardAfterSend === 1],
    [
      "delivered-but-unconsumed is re-fired (crash recovery)",
      firedUnread === 1 && heardAfterRedeliver === heardAfterSend + 1,
    ],
    [
      "the receipt stops redelivery",
      firedAfterReceipt === 0 && betaHeard.length === heardAfterRedeliver,
    ],
    ["a consumed message cannot be claimed twice", claimedAfterReceipt === false],
    ["alpha thread logged assistant message", historyA.some((e) => e.data.type === "message")],
    ["alpha thread logged tool_result", historyA.some((e) => e.data.type === "tool_result")],
    ["beta thread untouched this run (mailbox, not thread)", historyB.length === 0],
    ["thread keys distinct", threadKey(threadA) !== threadKey(threadB)],
    ["memory run completed", runM.stopReason === "completed"],
    [
      "retain journaled a memory event",
      historyM.some((e) => e.data.type === "memory" && e.data.op === "retain"),
    ],
    ["recall returned the retained text", recallText.includes("zebra-quartz")],
    [
      "recall journaled a memory event",
      historyM.some((e) => e.data.type === "memory" && e.data.op === "recall"),
    ],
    ["auto-recall run completed", runR.stopReason === "completed"],
    [
      "auto-recall injected <memories> at messages[0]",
      injected?.role === "user" && (injected.text ?? "").includes("<memories>"),
    ],
    [
      "auto-recall block carried the memory across threads",
      (injected?.text ?? "").includes("zebra-quartz"),
    ],
    // --- slice 9 -----------------------------------------------------------
    ["mcp fixture connected over real stdio", mcpState?.status === "connected"],
    [
      "mcp fixture negotiated the modern era",
      mcpState?.era === "modern" && (mcpState?.protocolVersion ?? "").startsWith("2026-"),
    ],
    [
      "catalog holds the server's namespaced tools",
      mcpNames.length > 0 && mcpNames.every((n) => n.startsWith(`mcp__${SMOKE_MCP_SERVER}__`)),
    ],
    [
      "the three meta-tools are in the request header",
      ["tool_search", "tool_describe", "tool_call"].every((n) => headNames.includes(n)),
    ],
    // The header is the cache key: an MCP tool in it would be a schema paid
    // for on every request, which is the whole cost this slice avoids.
    ["no mcp tool is in the request header (default mode: deferred)", !headNames.some((n) => n.startsWith("mcp__"))],
    ["deferred run completed", runT?.stopReason === "completed"],
    [
      "the model searched, described and called a catalog tool",
      ["tool_search", "tool_describe", "tool_call"].every((n) => toolResultNames.includes(n)),
    ],
    [
      "the mcp tool's output reached the assistant's answer",
      answer.includes("pinky-smoke-42") && answer.includes(FAKE_DEFERRED_MARKER),
    ],
    ["closing the mcp plane reaped its child process", reaped],
    // --- slice 6 -----------------------------------------------------------
    ["sleep extraction pass completed", sleepReport.threads[0]?.result.status === "done"],
    [
      "the worker turned an event into a memory row",
      sleepHits.some((h) => h.text.includes("amber-falcon") && h.meta.source === "sleep:extract"),
    ],
    // The receipt is the pass's only durable record, and it committed with the
    // rows above or not at all.
    ["the extract pass journaled its receipt", (extractReceipt?.added ?? 0) >= 1],
    [
      "a second sweep over the same thread has nothing to extract",
      sleepSecond?.status === "skipped" && sleepSecond.reason === "no-new-events",
    ],
    [
      "reflection journaled its receipt on the worker's own thread",
      reflectReceipt?.added === 1,
    ],
    [
      "the consolidated insight is itself a memory row",
      sleepHits.some((h) => h.text.startsWith(FAKE_SLEEP_REFLECT_PREFIX)),
    ],
  ];

  let failed = 0;
  for (const [name, ok] of checks) {
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
    if (!ok) failed++;
  }
  await boot.close();
  if (failed > 0) {
    console.error(`smoke: ${failed} check(s) failed`);
    process.exit(1);
  }
  console.log("smoke: all checks passed");
}

// ---------------------------------------------------------------------------
// prompt
// ---------------------------------------------------------------------------

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
  const boot = await bootstrap({ scopes: [`agent:${AGENT_ID}`] });
  // Local operator surface: the human is at their own terminal running their
  // own agent, so shell access is theirs to grant — and for the same reason
  // the recall scope below trusts it with `user` and `private` memories.
  const builtins = [...createTools({ shell: true }), new ShedContextTool()];
  // This surface's built-ins go into the catalog before the first run, so a
  // deferred built-in (`tools.deferred: ["bash"]`) is still findable and
  // `pinky tools list` sees what this terminal can actually do.
  await registerBuiltins(boot, builtins);
  const thread: ThreadRef = {
    tenantId: boot.settings.tenantId,
    channelId: CLI_CHANNEL_ID,
    threadId: "main",
  };
  const runAgent = makeRunAgent(boot, {
    agentId: AGENT_ID,
    builtins,
    scopeFor: () => ({
      agentId: AGENT_ID,
      channelId: CLI_CHANNEL_ID,
      userId: "local",
      includeUser: true,
      includePrivate: true,
    }),
  });

  // Ctrl-C at the operator's own terminal is the common case here, and it must
  // still reap the MCP children this process spawned — same handler, same
  // order (abort the run, close the plane, exit 130).
  const interrupted = new AbortController();
  let run: Promise<AgentRunResult> | undefined;
  const stopShutdown = installShutdown({
    drain: async () => {
      interrupted.abort(new Error("shutdown signal"));
      await run?.catch(() => {});
    },
    close: () => boot.close(),
  });

  let code = 0;
  try {
    await boot.events.append(thread, {
      type: "ingress",
      platform: "cli",
      author: { platform: "cli", userId: "local" },
      text,
      refs: [],
    });
    run = runAgent(thread, {
      deliver: async (t) => {
        process.stdout.write(`${t}\n`);
      },
      signal: interrupted.signal,
    });
    const result = await run;
    code = reportRun(result.stopReason, result.turns);
  } finally {
    stopShutdown();
    await boot.close();
  }
  if (code !== 0) process.exit(code);
}

// ---------------------------------------------------------------------------
// headless — the JSONL service (DESIGN.md §11); the primary interface
// ---------------------------------------------------------------------------

/**
 * Where a peer's message lands in the log.
 *
 * `channelId` is `a2a:<agentId@nodeId>` — the SENDER's normalized address, so
 * every peer gets its own channel: settings (model, thresholds, `selfConfig`)
 * are per-channel, which means a peer can be given a different model or budget
 * than the human pipe without a new config layer, and one chatty peer cannot
 * crowd another out of a shared thread.
 *
 * `threadId` is the sender's `threadHint` (a2a_send sets it to the thread the
 * sending agent was working in) and `main` when it names none. Namespaced by
 * the sender's channel, that gives one conversation per (peer, their thread) —
 * a request and its response project into the same window, which is what makes
 * a reply intelligible to the model, while an unrelated peer conversation
 * never contaminates it.
 */
function a2aThreadFor(
  tenantId: string,
  env: A2AEnvelope,
  nodeId: string,
): { thread: ThreadRef; from: string } {
  const addr = parseA2AAddress(env.from, nodeId);
  // Normalized once, here: a locally-fired envelope may carry a bare `alpha`
  // while the same row read back from the mailbox says `alpha@local`, and one
  // peer must not end up with two channels.
  const from = `${addr.agentId}@${addr.nodeId}`;
  return {
    thread: { tenantId, channelId: `a2a:${from}`, threadId: env.threadHint ?? "main" },
    from,
  };
}

/**
 * A2A as a headless WAKE SOURCE (DESIGN.md §7 wake-on-message, §8.1 "one
 * wake(thread_id, cause) entry point"; issue #4).
 *
 * Until this existed, a message that arrived while `pinky headless` was
 * running only reached the agent if it happened to call the `a2a_inbox` tool:
 * `LocalMessenger` fired subscribers and nothing in production subscribed. The
 * mailbox row said `delivered_at`, which is scheduler bookkeeping — proof that
 * this node took custody, not that anyone acted.
 *
 * So the consumer, and only the consumer, decides a message is done:
 *
 *   1. ONE transaction claims the receipt (`read_at`) and appends the `a2a`
 *      event. Both commit or neither does — there is no state where a message
 *      is marked consumed with nothing in the log to show for it, and none
 *      where the event exists twice.
 *   2. Only the transaction that won the claim enqueues the run. The batch is
 *      just the author hint; the run reads the event log, where step 1 put the
 *      message (core/projection.ts renders `a2a` as a user turn).
 *   3. Recovery is `redeliverUnconsumed` — at startup, before the session
 *      accepts input, and again on the 30s sweep. Re-firing is free: a message
 *      already consumed loses the claim in step 1 and enqueues nothing.
 *
 * A handler that throws leaves the row unread, so the next sweep retries it.
 * Everything logs to stderr; stdout is the protocol.
 */
function a2aWakeSource(
  boot: Bootstrap,
  log: (msg: string) => void,
): (enqueue: WakeEnqueue) => Promise<() => void> {
  const { tenantId } = boot.settings;
  const { nodeId } = boot.env;

  const consume = async (env: A2AEnvelope, enqueue: WakeEnqueue): Promise<void> => {
    const { thread, from } = a2aThreadFor(tenantId, env, nodeId);
    // Normalized like `from`: a live fire carries whatever the sender typed
    // (`pinky`), a redelivered row carries the stored form (`pinky@local`),
    // and the journaled event should read the same either way.
    const toAddr = parseA2AAddress(env.to, nodeId);
    const to = `${toAddr.agentId}@${toAddr.nodeId}`;
    const consumed = await boot.db.tx(async (tx) => {
      // The receipt is the idempotency hinge: lose the claim and another
      // consumer (or an earlier delivery of the same row) already has this.
      if (!(await boot.messenger.claimConsumption(env.id, tx))) return false;
      await EventStore.appendTx(tx, thread, [
        {
          type: "a2a",
          from,
          to,
          kind: env.kind,
          text: env.text,
          msgId: env.id,
        },
      ]);
      return true;
    });
    if (!consumed) return;
    enqueue(
      thread,
      [{ text: env.text, author: { platform: "a2a", userId: from }, externalId: env.id }],
      "a2a",
    );
  };

  return async (enqueue) => {
    // Handlers are fired synchronously but finish asynchronously, so the
    // startup drain below has to wait for the turns it started, not just for
    // the fires: the session reads its first command only after this resolves.
    const inFlight = new Set<Promise<void>>();
    const unsubscribe = boot.messenger.onMessage(AGENT_ID, (env) => {
      const done = consume(env, enqueue).catch((err) => {
        log(`[a2a] wake for ${env.id} failed (left unconsumed for the sweep): ${errorMessage(err)}`);
      });
      inFlight.add(done);
      void done.finally(() => inFlight.delete(done));
    });
    try {
      const fired = await boot.messenger.redeliverUnconsumed(AGENT_ID);
      await Promise.all([...inFlight]);
      if (fired > 0) log(`[a2a] startup recovery: re-fired ${fired} unconsumed message(s)`);
    } catch (err) {
      // Recovery failing is not a reason to refuse the session: the sweep
      // retries in 30s and stdin works meanwhile.
      log(`[a2a] startup recovery failed: ${errorMessage(err)}`);
    }
    return unsubscribe;
  };
}

/**
 * `pinky headless [--shell] [--a2a]` — one command object per stdin line, one
 * event object per stdout line, for as long as the caller keeps the pipe open.
 *
 * Three deliberate differences from `pinky prompt`:
 *
 * 1. No shell unless `--shell`. Headless is driven by ANOTHER PROGRAM, so it
 *    is treated like the gateway, not like a human at their own terminal.
 * 2. Settings are re-read per run, scoped to the run's channel and this agent,
 *    so a `pinky config set` (or a settings_set tool write) lands on the next
 *    run of a long-lived process instead of at the next restart. That is also
 *    why `ready` reports `defaultModel`: the bootstrap snapshot, not a promise
 *    about what any particular run will use.
 * 3. No HTTP listener unless `--a2a`. The protocol needs no socket; only
 *    inbound A2A from another MACHINE does. The sweep runs either way: its
 *    sender half is idle without peers, but its consumer half (redelivering
 *    unconsumed messages) matters on a single node too.
 *
 * The sleep-time worker's timer (slice 6) also lives here, and only here, when
 * the BOOTSTRAP snapshot says `sleep.enabled`. Like `mcp.servers` it is read
 * once — it sweeps every thread of the tenant, so it belongs to the process,
 * not to a run — which means a `sleep.*` change lands on the next restart.
 *
 * Wake-on-message is wired here and nowhere else: `wakes` (a2aWakeSource)
 * subscribes this process to its own mailbox, so a peer's message journals an
 * `a2a` event and wakes a run on that peer's thread — `run_started` carries
 * `"cause":"a2a"` — instead of waiting for the agent to poll `a2a_inbox`.
 *
 * The recall scope is the trusted-local one by default (`user` + `private`
 * visible, DESIGN.md §5.1): the process reads its commands from a pipe its
 * operator opened, and the subject user is the prompt's author, so
 * `user`-visibility memories follow whoever the client says is speaking.
 *
 * `--shared` drops both flags. Use it when the driving program is a BRIDGE for
 * several people rather than one operator's own tool: `userId` then comes from
 * whatever the client puts in `author`, and trusting that would let one
 * party's claimed identity recall another's `user` memories — and every
 * party's prompt recall the agent's `private` ones — into a shared thread,
 * which §5.1 says never happens.
 */
async function cmdHeadless(args: string[]): Promise<void> {
  const { flags } = parseFlags(args, ["shell", "a2a", "shared"]);
  const shared = flags.shared === true;

  // Losing stdout is a session-ending EVENT here, not an exception. A client
  // that closes the pipe makes the next write fail with EPIPE — and on Node's
  // streams that failure arrives asynchronously, as an 'error' event that
  // kills the process (exit 1, no `exiting`, no drain, boot never closed)
  // unless something is listening. So: listen, stop writing, and end the
  // session the same way `exit --abort` would, through the normal exit path.
  let stdoutOpen = true;
  const clientGone = new AbortController();
  const closeStdout = (err: unknown): void => {
    if (!stdoutOpen) return;
    stdoutOpen = false;
    clientGone.abort(err);
    logStderr(`[headless] stdout closed by the client (${String(err)}); draining and exiting`);
  };
  const isEpipe = (err: unknown): boolean => {
    const code = (err as NodeJS.ErrnoException | null)?.code;
    return code === "EPIPE" || code === "ERR_STREAM_DESTROYED";
  };
  process.stdout.on("error", (err: NodeJS.ErrnoException) => {
    if (!isEpipe(err)) throw err; // not "the reader left" — that is a real bug
    closeStdout(err);
  });

  const write = (line: string): void => {
    if (!stdoutOpen) return; // further writes would just re-raise EPIPE
    try {
      // Backpressure is deliberately ignored: a false return only means Bun is
      // buffering the line, and protocol lines are small. A client that reads
      // slower than the agent writes grows this process's memory — acceptable
      // for a pipe the operator owns; revisit if that ever stops being true.
      process.stdout.write(line);
    } catch (err) {
      if (!isEpipe(err)) throw err;
      closeStdout(err); // the synchronous half of the same failure
    }
  };
  const writeEvent = (obj: Record<string, unknown>): void => {
    write(`${JSON.stringify(obj)}\n`);
  };

  let boot: Bootstrap | undefined;
  let server: ReturnType<typeof startGateway> | undefined;
  let stopSweep: (() => void) | undefined;
  let stopSleep: (() => Promise<void>) | undefined;
  let session: Promise<void> | undefined;
  let released = false;
  /**
   * The sleep worker's shutdown switch, aborted BEFORE the pool closes.
   *
   * Without it the worker's abort guards never fire: a sweep in flight would
   * keep opening provider round trips for every remaining due thread while the
   * pool went away underneath, and the failure that followed would take the
   * NON-abort path — journaling an `error` event onto a live conversation
   * thread for what was only a SIGTERM.
   */
  const sleepStopping = new AbortController();
  // Everything this process holds, released once. Called by the shutdown
  // handler on a signal and by the `finally` on the normal path.
  const release = async (): Promise<void> => {
    if (released) return;
    released = true;
    stopSweep?.();
    // Order matters: ABORT first, then await. `stopSleep()` clears the timer
    // and waits for the sweep in flight, so awaiting it without the abort
    // would wait for every remaining due thread's LLM round trip. With it, the
    // pass returns at its next guard and the pool closes on a settled sweep —
    // no transaction torn mid-flight. `installShutdown`'s grace timer is the
    // valve if a provider hangs past it.
    sleepStopping.abort(new Error("headless is shutting down"));
    await stopSleep?.();
    server?.stop();
    await boot?.close();
  };
  // Installed BEFORE bootstrap: `mcp.start()` spawns stdio children inside it,
  // so a signal arriving during startup must already have somewhere to land.
  const stopShutdown = installShutdown({
    drain: async () => {
      // The same switch the client-gone path throws (DESIGN.md §11): the
      // session stops reading stdin, in-flight runs abort, `exiting` is
      // written, and runHeadless returns.
      clientGone.abort(new Error("shutdown signal"));
      await session?.catch(() => {});
    },
    close: release,
  });
  try {
    boot = await bootstrap({ scopes: [`agent:${AGENT_ID}`] });
    const builtins = [...createTools({ shell: flags.shell === true }), new ShedContextTool()];
    await registerBuiltins(boot, builtins);

    const started = boot;
    const run = makeRunAgent(started, {
      agentId: AGENT_ID,
      builtins,
      settingsFor: (thread) =>
        started.reloadSettings({ scopes: [`channel:${thread.channelId}`, `agent:${AGENT_ID}`] }),
      scopeFor: (thread, batch) => ({
        agentId: AGENT_ID,
        channelId: thread.channelId,
        userId: batch?.[batch.length - 1]?.author.userId ?? "local",
        includeUser: !shared,
        includePrivate: !shared,
      }),
    });

    // Inbound A2A is the one thing here that needs a port. An empty secret
    // makes /a2a/deliver answer 503 to everything, so a listener would be
    // theatre.
    if (flags.a2a === true) {
      if (started.env.a2aSecret.trim() === "") {
        logStderr("[a2a] --a2a ignored: A2A_SECRET is empty (inbound delivery would 503)");
      } else {
        server = startGateway({ env: started.env, messenger: started.messenger });
        logStderr(`[a2a] listening on :${server.port} (POST /a2a/deliver, GET /healthz)`);
      }
    }
    // Always swept, peers or not: the sender-side retry has nothing to do
    // without peers, but the consumer-side redelivery does — an unconsumed
    // message is a single-node failure mode too (issue #4).
    stopSweep = startA2ASweep(started.messenger, logStderr, { redeliverFor: AGENT_ID });

    // The sleep-time worker's timer (slice 6, DESIGN.md §5.3 item 3, §8.1).
    //
    // `sleep.*` is read ONCE here, from the BOOTSTRAP snapshot, exactly like
    // `mcp.servers`: this is one process-lifetime timer sweeping EVERY thread
    // of the tenant, not a per-run decision, so a `channel:`-scoped
    // `sleep.enabled` has no thread to belong to and a change of any kind
    // takes effect on the next restart rather than the next wake.
    //
    // A misconfigured worker must not cost the session: `createProvider`
    // throws on an unroutable `sleep.model`, and refusing to serve stdin
    // because a background sweep could not be built is exactly the
    // "bad value stops the boot" failure the settings table exists to avoid
    // (CLAUDE.md #3). So it degrades to no sweep, loudly, on stderr.
    if (started.settings.sleep.enabled) {
      try {
        const deps = sleepDepsFor(started, {
          // The worker inherits the surface's width (DESIGN.md §5.1): a shared
          // bridge must not mint `user` rows it could not read back.
          scope: { includeUser: !shared, includePrivate: !shared },
          settings: started.settings,
          signal: sleepStopping.signal,
        });
        stopSleep = startSleepSweep(deps, { intervalMs: started.settings.sleep.intervalMs });
        logStderr(
          `[sleep] sweep every ${started.settings.sleep.intervalMs}ms with ${deps.model} ` +
            `(idle gate ${started.settings.sleep.idleMs}ms, ` +
            `<= ${started.settings.sleep.maxThreadsPerSweep} thread(s) per sweep)`,
        );
      } catch (err) {
        logStderr(`[sleep] disabled: ${errorMessage(err)}`);
      }
    }

    session = runHeadless({
      tenantId: started.settings.tenantId,
      agentId: AGENT_ID,
      nodeId: started.env.nodeId,
      defaultModel: started.settings.model,
      events: started.events,
      runAgent: (thread, batch, hooks) =>
        run(
          thread,
          { deliver: hooks.deliver, signal: hooks.signal, onEvent: hooks.onEvent },
          batch,
        ),
      // stdin is not the only way to reach this agent: a peer's message wakes
      // a run through the same lane (DESIGN.md §7, §8.1).
      wakes: a2aWakeSource(started, logStderr),
      stdin: Bun.stdin.stream(),
      write,
      log: logStderr,
      signal: clientGone.signal,
    });
    await session;
  } catch (err) {
    // Startup failures land here too — a bad DATABASE_URL, a failed migration,
    // a port already bound. A client parsing stdout would otherwise see the
    // pipe close with nothing on it, so the failure gets one protocol line
    // before the usual exit(1).
    writeEvent({
      type: "error",
      message: `headless failed: ${err instanceof Error ? err.message : String(err)}`,
    });
    throw err;
  } finally {
    stopShutdown();
    await release();
  }
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
    case "memory":
      await cmdMemory(rest);
      break;
    case "stats":
      await cmdStats(rest);
      break;
    case "mcp":
      await cmdMcp(rest);
      break;
    case "tools":
      await cmdTools(rest);
      break;
    case "sleep":
      await cmdSleep(rest);
      break;
    case "smoke":
      await cmdSmoke();
      break;
    case "prompt":
      if (!rest[0]) throw new Error('usage: pinky prompt "<text>"');
      await cmdPrompt(rest.join(" "));
      break;
    case "headless":
      await cmdHeadless(rest);
      break;
    default:
      console.error(
        "usage: pinky <migrate|config|memory|stats|mcp|tools|sleep|smoke|prompt|headless>",
      );
      process.exit(cmd ? 1 : 0);
  }
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
