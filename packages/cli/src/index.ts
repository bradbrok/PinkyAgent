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
 *   pinky stats restarts [--channel <id>] [--limit N]   what context restarts cost
 *   pinky stats cache [--channel <id>] [--thread <id>] [--limit N]  prompt-cache hit rate
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
  parseA2AAddress,
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
  type A2AEnvelope,
  type AgentRunResult,
  type AssistantTurn,
  type Embedder,
  type LlmMessage,
  type MemoryContext,
  type Provider,
  type RunAgentLoopOptions,
  type Tool,
} from "@pinky/runtime";
import { runHeadless, startGateway, type RawIngress, type WakeEnqueue } from "@pinky/gateway";
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
  "usage: pinky stats <restarts|cache> ...\n" +
  "  pinky stats restarts [--channel <id>] [--limit N]\n" +
  "  pinky stats cache [--channel <id>] [--thread <id>] [--limit N]\n" +
  "    --limit samples the newest N turns (default 50); warm -> cold\n" +
  "    transitions are detected within that sampled window.";

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

  const boot = await bootstrap({ migrate: false, embedder: null });
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

  const boot = await bootstrap({ migrate: false, embedder: null });
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

/** `pinky stats <restarts|cache>` — the DESIGN.md §13 eval, as two queries. */
async function cmdStats(args: string[]): Promise<void> {
  const [sub, ...raw] = args;
  if (sub === "restarts") return cmdStatsRestarts(raw);
  if (sub === "cache") return cmdStatsCache(raw);
  throw new Error(STATS_USAGE);
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
  const [markA, markB, markM] = await Promise.all([
    latestSeq(db, threadA),
    latestSeq(db, threadB),
    latestSeq(db, threadM),
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
    tools,
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

  // Cleanup before reporting: a failed check must not leave rows behind.
  await db.query(`delete from memories where tenant_id = $1 and agent_id = $2`, [
    settings.tenantId,
    SMOKE_MEMORY_AGENT,
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
    // Always swept, peers or not: the sender-side retry has nothing to do
    // without peers, but the consumer-side redelivery does — an unconsumed
    // message is a single-node failure mode too (issue #4).
    stopSweep = startA2ASweep(started.messenger, logStderr, { redeliverFor: AGENT_ID });

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
      // stdin is not the only way to reach this agent: a peer's message wakes
      // a run through the same lane (DESIGN.md §7, §8.1).
      wakes: a2aWakeSource(started, logStderr),
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
    case "stats":
      await cmdStats(rest);
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
      console.error("usage: pinky <migrate|config|memory|stats|smoke|prompt|headless>");
      process.exit(cmd ? 1 : 0);
  }
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
