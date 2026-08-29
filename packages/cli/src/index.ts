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
  assertScope,
  threadKey,
  type Db,
  type EnvConfig,
  type LoadOptions,
  type MemoryRow,
  type RecallScope,
  type SettingsSnapshot,
  type ThreadRef,
} from "@pinky/core";
import {
  createEmbedder,
  createProvider,
  isEmbeddingsDisabledError,
  FakeEmbedder,
  FakeProvider,
  LocalMessenger,
  runAgentLoop,
  buildSystemPrompt,
  ShedContextTool,
  type AgentRunResult,
  type AssistantTurn,
  type Embedder,
  type LlmMessage,
  type MemoryContext,
  type Provider,
  type RunAgentLoopOptions,
  type Tool,
} from "@pinky/runtime";
import { runHeadless, startGateway, type RawIngress } from "@pinky/gateway";
import { createTools } from "@pinky/tools";

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

  return {
    env,
    settings,
    db,
    events: new EventStore(db),
    messenger: new LocalMessenger(db, { nodeId: env.nodeId, peers, a2aSecret }),
    memory: store,
    embedder,
    memoryContextFor: (scope) => ({ store, scope, ...(embedder ? { embedder } : {}) }),
    reloadSettings: (loadOpts) => loadSettings(db, loadOpts),
    close: () => rootDb.close(),
  };
}

/**
 * Per-run overrides a caller may hand `runAgentLoop`, SUBTRACTED from its
 * option type rather than listed, so the set cannot drift: everything the
 * surface owns (db, tools, memory, settings…) is fixed by makeRunAgent and the
 * remainder is per-run — today `maxTurns`, `deliver`, `signal` and `onEvent`,
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
>;

export interface RunAgentOptions {
  agentId: string;
  tools: Tool[];
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

/** One `runAgentLoop` call, pre-bound to a surface. `pinky headless` builds
 *  its per-run hooks (deliver/signal/onEvent) as the `overrides` argument. */
export function makeRunAgent(boot: Bootstrap, opts: RunAgentOptions): RunAgent {
  const systemPrompt = buildSystemPrompt({
    agentId: opts.agentId,
    nodeId: boot.env.nodeId,
    tools: opts.tools,
  });
  const cwd = opts.cwd ?? process.cwd();

  return async (thread, overrides = {}, batch) => {
    const settings = opts.settingsFor ? await opts.settingsFor(thread) : boot.settings;
    const scope = opts.scopeFor?.(thread, batch);
    return runAgentLoop({
      db: boot.db,
      provider: opts.provider ?? createProvider(settings.model, process.env),
      tools: opts.tools,
      thread,
      agentId: opts.agentId,
      messenger: boot.messenger,
      systemPrompt,
      cwd,
      settings,
      ...(scope ? { memory: boot.memoryContextFor(scope) } : {}),
      ...overrides,
    });
  };
}

/**
 * Sender half of A2A at-least-once (DESIGN.md §7): rows a peer refused (or was
 * down for) stay pending, and only a sweep clears them. Once now, then on a
 * timer; unref'd so it never keeps the process alive by itself. Logs to
 * STDERR — a JSONL service owns stdout. Returns a stop function.
 */
export function startA2ASweep(
  messenger: LocalMessenger,
  log: (msg: string) => void = (msg) => console.error(msg),
): () => void {
  const sweep = async (): Promise<void> => {
    try {
      const { attempted, delivered } = await messenger.flushPending();
      if (attempted > 0) log(`[a2a] retry sweep: ${delivered}/${attempted} delivered`);
    } catch (err) {
      log(`[a2a] retry sweep failed: ${errorMessage(err)}`);
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

  const boot = await bootstrap();
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

async function cmdSmoke(): Promise<void> {
  // Deterministic, offline, and dimension-matched to the real column.
  const embedder = new FakeEmbedder({ dimensions: SMOKE_EMBEDDING_DIMENSIONS });
  // peers/secret emptied: smoke is a single-node in-process check and must not
  // start POSTing to a peer configured in someone's .env.
  const boot = await bootstrap({ peers: {}, a2aSecret: "", embedder });
  const { env, settings, db, events, messenger } = boot;
  const tools = createTools();
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

  const betaInbox: string[] = [];
  messenger.onMessage("beta", (env2) => {
    betaInbox.push(`${env2.from}: ${env2.text}`);
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
    tools,
    provider: new FakeProvider(alphaScript),
    settingsFor: () => Promise.resolve(smokeSettings),
  })(threadA);

  // --- 2. retain -> recall round trip through the tools ------------------
  const memoryScope: RecallScope = {
    agentId: SMOKE_MEMORY_AGENT,
    channelId: threadM.channelId,
    userId: "local",
    includeUser: true,
    includePrivate: true,
  };
  // Smoke reuses fixed thread ids, so the log carries every previous run.
  // Everything asserted below is sliced from this mark forward — otherwise a
  // run that retained nothing would still "pass" on yesterday's events.
  const markM = (await events.history(threadM)).length;
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
    tools,
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
    tools,
    provider: recallProvider,
    settingsFor: () => Promise.resolve(smokeSettings),
    scopeFor: () => ({ ...memoryScope, channelId: threadR.channelId }),
  })(threadR);

  const inbox = await messenger.inbox("beta");
  const historyA = await events.history(threadA);
  const historyB = await events.history(threadB);
  const historyM = (await events.history(threadM)).slice(markM);
  const recallResult = historyM.find(
    (e) => e.data.type === "tool_result" && e.data.name === "recall",
  );
  const recallText = recallResult?.data.type === "tool_result" ? recallResult.data.text : "";
  const injected = prompts[0]?.[0];

  // Cleanup before reporting: a failed check must not leave rows behind.
  await db.query(`delete from memories where tenant_id = $1 and agent_id = $2`, [
    settings.tenantId,
    SMOKE_MEMORY_AGENT,
  ]);

  const checks: [string, boolean][] = [
    ["alpha ran to completion", runA.stopReason === "completed"],
    ["a2a message delivered to beta", inbox.length === 1 && inbox[0]!.text === "what is 2+2?"],
    ["live subscriber saw the message", betaInbox.length === 1],
    ["alpha thread logged assistant message", historyA.some((e) => e.data.type === "message")],
    ["alpha thread logged tool_result", historyA.some((e) => e.data.type === "tool_result")],
    ["beta thread untouched (mailbox, not thread)", historyB.length === 0],
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
  const tools = [...createTools({ shell: true }), new ShedContextTool()];
  const thread: ThreadRef = {
    tenantId: boot.settings.tenantId,
    channelId: CLI_CHANNEL_ID,
    threadId: "main",
  };
  const runAgent = makeRunAgent(boot, {
    agentId: AGENT_ID,
    tools,
    scopeFor: () => ({
      agentId: AGENT_ID,
      channelId: CLI_CHANNEL_ID,
      userId: "local",
      includeUser: true,
      includePrivate: true,
    }),
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
    const result = await runAgent(thread, {
      deliver: async (t) => {
        process.stdout.write(`${t}\n`);
      },
    });
    code = reportRun(result.stopReason, result.turns);
  } finally {
    await boot.close();
  }
  if (code !== 0) process.exit(code);
}

// ---------------------------------------------------------------------------
// headless — the JSONL service (DESIGN.md §11); the primary interface
// ---------------------------------------------------------------------------

/** Human log line. Always stderr: stdout is the protocol. */
function logStderr(msg: string): void {
  process.stderr.write(`${msg}\n`);
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
 *    inbound A2A does. Outbound A2A retries need no listener, so the sweep
 *    runs whenever peers are configured either way.
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
  try {
    boot = await bootstrap({ scopes: [`agent:${AGENT_ID}`] });
    const tools = [...createTools({ shell: flags.shell === true }), new ShedContextTool()];

    const started = boot;
    const run = makeRunAgent(started, {
      agentId: AGENT_ID,
      tools,
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
    const peers = Object.keys(started.env.peers);
    stopSweep = peers.length > 0 ? startA2ASweep(started.messenger, logStderr) : undefined;

    await runHeadless({
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
      stdin: Bun.stdin.stream(),
      write,
      log: logStderr,
      signal: clientGone.signal,
    });
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
    stopSweep?.();
    server?.stop();
    await boot?.close();
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
      console.error("usage: pinky <migrate|config|memory|smoke|prompt|headless>");
      process.exit(cmd ? 1 : 0);
  }
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
