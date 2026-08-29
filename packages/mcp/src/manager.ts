/**
 * The MCP client plane (DESIGN.md slice 9): one `McpManager` owns every
 * configured MCP server, keeps the Postgres tool catalog in sync with what
 * those servers publish, and dispatches `tools/call`.
 *
 * Five rules, in the order they bite:
 *
 * 1. **A server never blocks startup.** `start()` does one cheap catalog probe
 *    per server and returns; connecting happens on a background loop. Request 1
 *    of a session lands long before a stdio child has spawned, so the catalog
 *    is the source of truth for *what tools exist* and the client is only the
 *    source of truth for *calling them*.
 * 2. **The catalog is trusted by config hash.** If the rows the catalog holds
 *    for a server were written under the same `hashServerConfig(config)`, they
 *    are served immediately (`status: "trusted-cache"`). A changed hash means
 *    the operator repointed the server, and the rows are stale until the sync
 *    lands.
 * 3. **A generation is replaced, never patched, and never flapped.** Every
 *    successful sync is one `catalog.replaceServer(...)` — deterministically
 *    sorted, deduped, with the config hash stamped on it. A DISCONNECT writes
 *    nothing: an outage must not empty the catalog and rewrite every prompt.
 * 4. **Version negotiation is `auto`, always.** The SDK defaults to `'legacy'`
 *    (the 2025 `initialize` handshake, byte-identical to a pre-2026 client);
 *    `{ mode: "auto" }` probes `server/discover` first and falls back. Without
 *    it a 2026-07-28 server is talked to as if it were a 2025 one and every
 *    modern affordance (ttl hints, `subscriptions/listen`) is invisible.
 * 5. **`call()` never throws.** Its two outcomes are a rendered `ToolResult`
 *    and an `isError` `ToolResult`, because it is invoked from inside a
 *    journaled agent turn.
 *
 * stdout is protocol-only in this repo (CLAUDE.md #5), so every line this file
 * emits goes through the injected `log` — the CLI points it at stderr.
 */
import { Client, type McpSubscription, type ProtocolEra, type Transport } from "@modelcontextprotocol/client";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/client/stdio";
import type { McpServerConfig } from "@pinky/core";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import {
  canonicalJson,
  compareNames,
  hashServerConfig,
  isValidServerKey,
  mcpToolNames,
  resolveEnvPlaceholders,
  sortObjectKeysDeep,
  splitMcpToolName,
} from "./naming";
import { renderCallToolResult } from "./render";

// ---------------------------------------------------------------------------
// Contracts with the outside world
// ---------------------------------------------------------------------------

/** One tool as the catalog wants it written (core's `CatalogToolInput`, named
 *  locally so this package does not import the store it writes through). */
export interface McpCatalogTool {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  /** The server's own spelling — what `tools/call` must send back. */
  rawName?: string;
}

/** One live catalog row as this package reads it back. Structurally a subset
 *  of core's `CatalogRecord`, so `ToolCatalogStore.entries()` satisfies it. */
export interface McpCatalogRow {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  rawName?: string;
}

/** What the catalog looks like from here. Deliberately the three methods this
 *  file uses and nothing else, so `ToolCatalogStore` (core) satisfies it
 *  structurally and the wiring layer needs no adapter. */
export interface CatalogSink {
  replaceServer(
    server: string,
    configHash: string | null,
    tools: McpCatalogTool[],
  ): Promise<{ upserted: number; removed: number }>;
  serverState(
    server: string,
  ): Promise<{ configHash: string | null; count: number; updatedAt: string } | null>;
  /** Live rows for one server, name-ordered — the trusted generation, read
   *  back so the tool HEADER is identical before and after the first sync. */
  entries(opts: { server: string }): Promise<McpCatalogRow[]>;
}

/**
 * - `trusted-cache` — the catalog's rows match the configured hash and are
 *   being served; the connection is still coming up.
 * - `connecting` — no trusted rows (or a reconnect in flight).
 * - `connected` — live, synced.
 * - `error` — the last connect attempt failed; a retry is scheduled and the
 *   previous generation (if any) is still in the catalog.
 */
export type McpServerStatus = "trusted-cache" | "connecting" | "connected" | "error";

/** What `pinky mcp list` prints. Optional fields are absent, never undefined
 *  (exactOptionalPropertyTypes). */
export interface McpServerState {
  server: string;
  status: McpServerStatus;
  era?: ProtocolEra;
  protocolVersion?: string;
  serverName?: string;
  toolCount: number;
  lastError?: string;
  configHash: string;
}

/** Injectable timers — the ttl refresh is scheduled through this so a unit
 *  test can drive an hour of clock in a microtask. */
export interface McpTimers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

/** Injectable sleep for the reconnect backoff. RESOLVES (never rejects) when
 *  the signal fires, so the loop's own `closed` check is the only exit test. */
export type McpSleep = (ms: number, signal?: AbortSignal) => Promise<void>;

/** Builds the transport for one server. The seam unit tests use to hand the
 *  manager an `InMemoryTransport` instead of a child process. */
export type McpTransportFactory = (
  server: string,
  config: McpServerConfig,
  env: Record<string, string | undefined>,
) => Transport | Promise<Transport>;

export interface McpManagerOptions {
  /** `settings.mcp.servers`: key -> config. */
  servers: Record<string, McpServerConfig>;
  catalog: CatalogSink;
  /** Every diagnostic line. The CLI points this at stderr — stdout is protocol. */
  log?: (line: string) => void;
  /** Source for `${ENV}` placeholders in stdio `env` / http `headers`. */
  env?: Record<string, string | undefined>;
  timers?: McpTimers;
  sleep?: McpSleep;
  /** Jitter source for the reconnect backoff; pinned in tests. */
  random?: () => number;
  transportFactory?: McpTransportFactory;
  clientName?: string;
  clientVersion?: string;
  /** Cap on the reconnect backoff. */
  maxReconnectDelayMs?: number;
  baseReconnectDelayMs?: number;
  /** Per-request timeout handed to the SDK (connect, list, call). */
  requestTimeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Floor on a ttl-driven refresh. A server is free to say `ttlMs: 0` ("
 * immediately stale"), and honoring that literally would put a `tools/list` on
 * a hot loop; the freshness hint is advice about cacheability, not a schedule.
 */
export const MIN_REFRESH_MS = 60_000;
/**
 * Ceiling on a ttl-driven refresh: 24h.
 *
 * A server is free to send a nonsense `ttlMs` (2^32, say). `setTimeout` takes
 * a SIGNED 32-BIT delay: anything above 2_147_483_647 overflows and Bun/Node
 * fire it on the next tick — so an absurdly long ttl becomes a `tools/list` +
 * `replaceServer` HOT LOOP, the exact opposite of what the hint asked for.
 * Clamping is what makes a hostile or buggy hint harmless.
 */
export const MAX_REFRESH_MS = 24 * 60 * 60 * 1000;
/** The `setTimeout` delay ceiling; every scheduled delay is clamped to it. */
export const MAX_TIMER_MS = 2_147_483_647;
/** A tool description longer than this is truncated before the catalog write.
 *  The catalog's generated `tsvector` has a hard 1 MB limit, and ONE oversized
 *  description aborts the whole transaction — losing every row of the
 *  generation and, before the sync fix below, respawning the server forever. */
export const MAX_DESCRIPTION_CHARS = 8_000;
/** A schema whose canonical JSON exceeds this is replaced by a stub. Same
 *  reason, plus: a 64 KB schema at prefix position 0 is its own problem. */
export const MAX_SCHEMA_BYTES = 64 * 1024;
/** Backoff floor/ceiling for retrying a failed catalog write. */
export const SYNC_RETRY_BASE_MS = 5_000;
export const SYNC_RETRY_MAX_MS = 5 * 60_000;
export const DEFAULT_BASE_RECONNECT_MS = 1_000;
export const DEFAULT_MAX_RECONNECT_MS = 30_000;
const DEFAULT_CLIENT_NAME = "pinky";
const DEFAULT_CLIENT_VERSION = "0.0.1";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  settled: boolean;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  const d: Deferred<T> = {
    promise,
    settled: false,
    resolve: (value: T) => {
      if (d.settled) return;
      d.settled = true;
      resolve(value);
    },
  };
  return d;
}

/** setTimeout/clearTimeout, unref'd where the runtime allows it: a background
 *  reconnect timer must not be the reason `pinky headless` refuses to exit. */
const realTimers: McpTimers = {
  setTimeout(fn: () => void, ms: number): unknown {
    const handle = setTimeout(fn, ms) as unknown as { unref?: () => void };
    handle.unref?.();
    return handle;
  },
  clearTimeout(handle: unknown): void {
    clearTimeout(handle as ReturnType<typeof setTimeout>);
  },
};

const realSleep: McpSleep = (ms, signal) =>
  new Promise<void>((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms) as unknown as { unref?: () => void };
    timer.unref?.();
    const onAbort = (): void => {
      clearTimeout(timer as unknown as ReturnType<typeof setTimeout>);
      resolve();
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });

/** Full-jitter backoff, the same shape as providers/retry.ts uses for HTTP. */
export function reconnectDelayMs(
  attempt: number,
  opts: { base: number; cap: number; random: () => number },
): number {
  const window = Math.min(opts.cap, opts.base * 2 ** Math.max(0, attempt - 1));
  return Math.floor(opts.random() * window);
}

/** The wire's `ttlMs`, if the server sent a usable one. `ListToolsResult` is a
 *  loose object in the SDK's types, so the field arrives as `unknown`. */
export function readTtlMs(result: unknown): number | undefined {
  if (typeof result !== "object" || result === null) return undefined;
  const value = (result as Record<string, unknown>).ttlMs;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return undefined;
  return value;
}

/**
 * The `RequestInit` for an http server. Exported because the security rule in
 * it is worth testing directly.
 *
 * `redirect: "error"` WHENEVER custom headers are configured: those headers
 * carry the credentials `${ENV}` placeholders resolve to, and `fetch` replays
 * request headers on a redirect. A server (or anything that can answer for its
 * hostname) could therefore answer 302 with a `Location` on another origin and
 * be handed the bearer token. Refusing to follow redirects costs nothing —
 * an MCP endpoint is a fixed URL — and closes a credential-exfiltration path.
 * With no custom headers there is nothing to leak, so redirects stay default.
 */
export function httpRequestInit(
  config: Extract<McpServerConfig, { transport: "http" }>,
  env: Record<string, string | undefined>,
): RequestInit {
  const headers = resolveEnvPlaceholders(config.headers, env);
  return {
    headers,
    ...(Object.keys(headers).length > 0 ? { redirect: "error" as const } : {}),
  };
}

/** The default transport for a config. `${ENV}` placeholders resolve HERE and
 *  nowhere else — never on the way into the settings table, never into a hash. */
export const defaultTransportFactory: McpTransportFactory = (server, config, env) => {
  if (config.transport === "http") {
    return new StreamableHTTPClientTransport(new URL(config.url), {
      requestInit: httpRequestInit(config, env),
    });
  }
  // A child that inherits nothing has no PATH; merge over the SDK's own
  // safe-to-inherit set rather than replacing it.
  const childEnv = { ...getDefaultEnvironment(), ...resolveEnvPlaceholders(config.env, env) };
  return new StdioClientTransport({
    command: config.command,
    ...(config.args ? { args: config.args } : {}),
    env: childEnv,
    ...(config.cwd ? { cwd: config.cwd } : {}),
    // The child's stderr is the operator's window into a misbehaving server,
    // and our stderr is already the non-protocol channel.
    stderr: "inherit",
  });
};

// ---------------------------------------------------------------------------
// Per-server runtime
// ---------------------------------------------------------------------------

interface ToolEntry {
  rawName: string;
  description: string;
  parameters: Record<string, unknown>;
}

interface ServerRuntime {
  server: string;
  config: McpServerConfig;
  configHash: string;
  status: McpServerStatus;
  era: ProtocolEra | undefined;
  protocolVersion: string | undefined;
  serverName: string | undefined;
  lastError: string | undefined;
  toolCount: number;
  /** The live generation, keyed by FINAL name. Empty until the first sync. */
  tools: Map<string, ToolEntry>;
  client: Client | undefined;
  /** Held from the moment the factory returns it, so `close()` can abort a
   *  connect that is still in flight (the client cannot close what it has not
   *  been handed yet). */
  transport: Transport | undefined;
  /** Connected AND synced. `call()` gates on this, not on `client`, because a
   *  client exists for the whole connect handshake and for a reconnect. */
  live: boolean;
  subscription: McpSubscription | undefined;
  refreshTimer: unknown;
  /** Settles once the first connect attempt has succeeded or failed, so a
   *  `call()` racing startup waits for an answer instead of a stale map. */
  firstAttempt: Deferred<void>;
  /** Resolved by `client.onclose`; the connect loop parks on it. */
  dropped: Deferred<void> | undefined;
  /** Serializes syncs so a ttl refresh and a list-changed cannot interleave.
   *  ALWAYS a `.catch`-terminated promise: a stored rejection would poison
   *  every later `.then()` (and surface as an unhandled rejection from the
   *  fire-and-forget `void this.resync(...)` call sites). */
  syncChain: Promise<void>;
  /** Consecutive failed syncs; drives the retry backoff. Reset on success. */
  syncFailures: number;
}

// ---------------------------------------------------------------------------
// The manager
// ---------------------------------------------------------------------------

export class McpManager {
  private readonly serverConfigs: Record<string, McpServerConfig>;
  private readonly catalog: CatalogSink;
  private readonly log: (line: string) => void;
  private readonly env: Record<string, string | undefined>;
  private readonly timers: McpTimers;
  private readonly sleep: McpSleep;
  private readonly random: () => number;
  private readonly transportFactory: McpTransportFactory;
  private readonly clientName: string;
  private readonly clientVersion: string;
  private readonly baseReconnectMs: number;
  private readonly maxReconnectMs: number;
  private readonly requestTimeoutMs: number | undefined;

  private readonly runtimes = new Map<string, ServerRuntime>();
  private readonly loops: Promise<void>[] = [];
  private readonly stop = new AbortController();
  private closed = false;
  private started = false;

  constructor(opts: McpManagerOptions) {
    this.serverConfigs = opts.servers;
    this.catalog = opts.catalog;
    // The log sink is injected by the caller and writes to stderr. A sink that
    // throws (a closed pipe, a wrapper with a bug) must not be able to fail a
    // connect, a sync, or a fire-and-forget `void this.resync(...)` on a timer
    // — the diagnostics channel is never load-bearing.
    const sink = opts.log;
    this.log = sink
      ? (line: string): void => {
          try {
            sink(line);
          } catch {
            /* a broken log sink is not a reason to lose a server */
          }
        }
      : (): void => {};
    this.env = opts.env ?? process.env;
    this.timers = opts.timers ?? realTimers;
    this.sleep = opts.sleep ?? realSleep;
    this.random = opts.random ?? Math.random;
    this.transportFactory = opts.transportFactory ?? defaultTransportFactory;
    this.clientName = opts.clientName ?? DEFAULT_CLIENT_NAME;
    this.clientVersion = opts.clientVersion ?? DEFAULT_CLIENT_VERSION;
    this.baseReconnectMs = opts.baseReconnectDelayMs ?? DEFAULT_BASE_RECONNECT_MS;
    this.maxReconnectMs = opts.maxReconnectDelayMs ?? DEFAULT_MAX_RECONNECT_MS;
    this.requestTimeoutMs = opts.requestTimeoutMs;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Register every configured server, decide per server whether the catalog's
   * rows can be trusted, and start the background connect loops. Awaits only
   * the catalog probes — never a server.
   */
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;

    for (const [server, config] of Object.entries(this.serverConfigs)) {
      if (!isValidServerKey(server)) {
        this.log(`[mcp] ${JSON.stringify(server)}: skipped — not a legal server key`);
        continue;
      }
      this.runtimes.set(server, {
        server,
        config,
        configHash: hashServerConfig(config),
        status: "connecting",
        era: undefined,
        protocolVersion: undefined,
        serverName: undefined,
        lastError: undefined,
        toolCount: 0,
        tools: new Map(),
        client: undefined,
        transport: undefined,
        live: false,
        subscription: undefined,
        refreshTimer: undefined,
        firstAttempt: deferred<void>(),
        dropped: undefined,
        syncChain: Promise.resolve(),
        syncFailures: 0,
      });
    }

    await Promise.all([...this.runtimes.values()].map((rt) => this.probeCatalog(rt)));
    for (const rt of this.runtimes.values()) this.loops.push(this.runServer(rt));
  }

  /** Tear everything down: timers, subscriptions, clients, loops. Idempotent. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.stop.abort();
    for (const rt of this.runtimes.values()) {
      this.clearRefresh(rt);
      rt.firstAttempt.resolve();
      await rt.subscription?.close().catch(() => {});
      rt.subscription = undefined;
      rt.live = false;
      await rt.client?.close().catch(() => {});
      rt.client = undefined;
      // A connect still in flight has a transport but no closed client yet;
      // closing it is what unsticks the handshake.
      await rt.transport?.close().catch(() => {});
      rt.transport = undefined;
      rt.dropped?.resolve();
    }
    await Promise.allSettled(this.loops);
  }

  // -------------------------------------------------------------------------
  // Read surfaces
  // -------------------------------------------------------------------------

  /** Every configured server's state, in configuration order. */
  states(): McpServerState[] {
    return [...this.runtimes.values()].map((rt) => ({
      server: rt.server,
      status: rt.status,
      ...(rt.era ? { era: rt.era } : {}),
      ...(rt.protocolVersion ? { protocolVersion: rt.protocolVersion } : {}),
      ...(rt.serverName ? { serverName: rt.serverName } : {}),
      toolCount: rt.toolCount,
      ...(rt.lastError ? { lastError: rt.lastError } : {}),
      configHash: rt.configHash,
    }));
  }

  /** One server's state, or undefined when it is not configured. */
  state(server: string): McpServerState | undefined {
    return this.states().find((s) => s.server === server);
  }

  /**
   * One `Tool` per row of the live generation, sorted by final name — the
   * candidates for the always-on partition. Empty for a server that has not
   * synced yet (a trusted cache serves `tool_search`/`tool_call` through the
   * catalog, which does not need a `Tool` object).
   */
  tools(): Tool[] {
    const out: Tool[] = [];
    for (const rt of this.runtimes.values()) {
      for (const [name, entry] of rt.tools) {
        out.push(new McpTool(name, entry.description, entry.parameters, this));
      }
    }
    out.sort((a, b) => compareNames(a.name, b.name));
    return out;
  }

  // -------------------------------------------------------------------------
  // Dispatch
  // -------------------------------------------------------------------------

  /**
   * Execute a namespaced MCP tool. Never throws: a bad name, a dead server and
   * a server-side failure all come back as `isError` text.
   */
  async call(
    name: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<ToolResult> {
    if (this.closed) return { text: "mcp plane is shut down", isError: true };

    const server = this.locate(name);
    if (server === undefined) {
      return { text: `unknown MCP tool ${JSON.stringify(name)}`, isError: true };
    }
    const rt = this.runtimes.get(server);
    if (!rt) {
      return { text: `MCP server ${JSON.stringify(server)} is not configured`, isError: true };
    }

    // A trusted cache can hand the model a name before the child process has
    // finished spawning; wait out the first attempt rather than lying about it.
    await this.awaitFirstAttempt(rt, signal);
    if (signal?.aborted) return { text: "mcp tool call aborted", isError: true };

    const client = rt.live ? rt.client : undefined;
    const entry = rt.tools.get(name);
    if (!client) {
      const because = rt.lastError ? `: ${rt.lastError}` : "";
      return { text: `MCP server ${JSON.stringify(server)} is not connected${because}`, isError: true };
    }
    if (!entry) {
      return {
        text: `MCP server ${JSON.stringify(server)} does not publish ${JSON.stringify(name)}`,
        isError: true,
      };
    }

    try {
      const result = await client.callTool(
        { name: entry.rawName, arguments: args },
        {
          ...(signal ? { signal } : {}),
          ...(this.requestTimeoutMs ? { timeout: this.requestTimeoutMs } : {}),
          // Manual MRTR mode: an `input_required` answer reaches the renderer
          // as a value instead of a typed throw, and becomes a clean tool error.
          allowInputRequired: true,
        },
      );
      return renderCallToolResult(result);
    } catch (err) {
      return { text: `mcp tool call failed: ${errorMessage(err)}`, isError: true };
    }
  }

  /**
   * Which server publishes `name`: the live map first (which the trusted-cache
   * path has already filled from the catalog), then the name's own server
   * segment — which survives sanitizing and truncation by construction.
   */
  private locate(name: string): string | undefined {
    for (const rt of this.runtimes.values()) if (rt.tools.has(name)) return rt.server;
    const split = splitMcpToolName(name);
    if (split && this.runtimes.has(split.server)) return split.server;
    return undefined;
  }

  private async awaitFirstAttempt(rt: ServerRuntime, signal?: AbortSignal): Promise<void> {
    if (rt.firstAttempt.settled) return;
    if (!signal) {
      await rt.firstAttempt.promise;
      return;
    }
    await new Promise<void>((resolve) => {
      const done = (): void => {
        signal.removeEventListener("abort", done);
        resolve();
      };
      signal.addEventListener("abort", done, { once: true });
      void rt.firstAttempt.promise.then(done);
    });
  }

  // -------------------------------------------------------------------------
  // Connect / reconnect
  // -------------------------------------------------------------------------

  /**
   * Decide whether the catalog's rows for this server can be trusted, and if
   * so ADOPT THEM AS THE LIVE GENERATION before anything is spawned.
   *
   * Loading the rows is not a nicety: tool schemas render at prefix position 0
   * (tools -> system -> messages), so a tool that is absent from the header on
   * run 1 and present on run 2 changes the cached prefix and invalidates every
   * provider cache tier — and shows the model a different tool set on
   * consecutive wakes. The catalog's generation is exactly the one the last
   * sync wrote under this same config hash, so serving it immediately makes
   * the header byte-identical across the connect.
   */
  private async probeCatalog(rt: ServerRuntime): Promise<void> {
    try {
      const state = await this.catalog.serverState(rt.server);
      if (state && state.configHash === rt.configHash) {
        rt.status = "trusted-cache";
        rt.tools = await this.loadCachedGeneration(rt);
        rt.toolCount = rt.tools.size;
        this.log(
          `[mcp] ${rt.server}: serving ${rt.toolCount} cached tools (config hash matches); connecting in background`,
        );
        return;
      }
      if (state) {
        this.log(`[mcp] ${rt.server}: cached tools are from a different config; waiting for a sync`);
      }
    } catch (err) {
      this.log(`[mcp] ${rt.server}: catalog probe failed: ${errorMessage(err)}`);
    }
    rt.status = "connecting";
  }

  /**
   * The trusted generation, read back from the catalog into the same
   * `name -> entry` map a sync produces. `rawName` falls back to the name's
   * own suffix only when the row has none (a pre-`raw_name` row); a wrong raw
   * name is caught by the server, and dropping the tool would be the header
   * change this whole path exists to avoid.
   */
  private async loadCachedGeneration(rt: ServerRuntime): Promise<Map<string, ToolEntry>> {
    const map = new Map<string, ToolEntry>();
    try {
      const rows = await this.catalog.entries({ server: rt.server });
      for (const row of [...rows].sort((a, b) => compareNames(a.name, b.name))) {
        if (typeof row.name !== "string" || row.name === "") continue;
        map.set(row.name, {
          rawName: row.rawName ?? splitMcpToolName(row.name)?.raw ?? row.name,
          description: row.description ?? "",
          parameters: normalizeSchema(row.parameters),
        });
      }
    } catch (err) {
      // A cache we cannot read is a cache we do not have; the connect is
      // already in flight and will write the generation itself.
      this.log(`[mcp] ${rt.server}: could not load cached tools: ${errorMessage(err)}`);
    }
    return map;
  }

  /** The per-server supervisor: connect, serve until the link drops, back off,
   *  repeat. Exits only on `close()`. */
  private async runServer(rt: ServerRuntime): Promise<void> {
    let attempt = 0;
    while (!this.closed) {
      const dropped = deferred<void>();
      rt.dropped = dropped;
      try {
        await this.connectOnce(rt, dropped);
        attempt = 0;
        rt.firstAttempt.resolve();
        await dropped.promise;
        if (this.closed) break;
        this.log(`[mcp] ${rt.server}: connection closed; reconnecting`);
        rt.status = "connecting";
      } catch (err) {
        rt.firstAttempt.resolve();
        // A connect torn down BY `close()` is not a failure. `close()` closes
        // the transport out from under an in-flight handshake, and the SDK
        // rejects it ("transport was closed during the server/discover
        // probe"); logging that as an error and flipping the server to
        // `error` would make a clean shutdown — `pinky mcp list`, which
        // bootstraps, prints and exits — look like an outage. The last known
        // state is kept, so a post-mortem `states()` still reads true.
        if (!this.closed) {
          rt.lastError = errorMessage(err);
          rt.status = "error";
          this.log(`[mcp] ${rt.server}: connect failed: ${rt.lastError}`);
        }
      }
      // The catalog keeps the previous generation on purpose; only the live
      // client handle goes away.
      await this.dropClient(rt);
      if (this.closed) break;
      attempt += 1;
      const delay = clampDelay(
        reconnectDelayMs(attempt, {
          base: this.baseReconnectMs,
          cap: this.maxReconnectMs,
          random: this.random,
        }),
      );
      this.log(`[mcp] ${rt.server}: retrying in ${delay}ms (attempt ${attempt})`);
      await this.sleep(delay, this.stop.signal);
    }
  }

  private async connectOnce(rt: ServerRuntime, dropped: Deferred<void>): Promise<void> {
    const transport = await this.transportFactory(rt.server, rt.config, this.env);
    rt.transport = transport;
    const client = new Client(
      { name: this.clientName, version: this.clientVersion },
      {
        // The SDK default is 'legacy'. Without this a 2026-07-28 server is
        // spoken to as a 2025 one; see the module header, rule 4.
        versionNegotiation: { mode: "auto" },
        // We implement no elicitation / sampling / roots handlers (all three
        // are deprecated in 2026-07-28), so auto-fulfilment has nothing to
        // fulfil with; manual mode plus `allowInputRequired` on the call site
        // turns an `input_required` answer into a readable tool error.
        inputRequired: { autoFulfill: false },
        // Era-transparent list-change delivery: on a legacy connection this is
        // the unsolicited `notifications/tools/list_changed` handler, on a
        // modern one the SDK auto-opens the `subscriptions/listen` stream that
        // carries it. `autoRefresh: false` because we resync through our own
        // pagination + catalog write, not the SDK's list cache.
        listChanged: {
          tools: {
            autoRefresh: false,
            debounceMs: 0,
            onChanged: (): void => {
              void this.resync(rt, "list_changed");
            },
          },
        },
      },
    );
    client.onclose = (): void => dropped.resolve();
    client.onerror = (err: Error): void => {
      if (this.closed) return; // teardown noise, not a condition to report
      rt.lastError = err.message;
      this.log(`[mcp] ${rt.server}: ${err.message}`);
    };

    rt.client = client;
    await client.connect(transport, {
      ...(this.requestTimeoutMs ? { timeout: this.requestTimeoutMs } : {}),
    });
    if (this.closed) throw new Error("manager closed during connect");

    rt.era = client.getProtocolEra();
    rt.protocolVersion = client.getNegotiatedProtocolVersion();
    rt.serverName = client.getServerVersion()?.name;
    rt.lastError = undefined;
    rt.status = "connected";
    this.log(
      `[mcp] ${rt.server}: connected era=${rt.era ?? "?"} protocol=${rt.protocolVersion ?? "?"}` +
        ` server=${rt.serverName ?? "anonymous"}`,
    );

    await this.syncNow(rt);
    await this.subscribeListChanged(rt);
    rt.live = true;
  }

  private async dropClient(rt: ServerRuntime): Promise<void> {
    this.clearRefresh(rt);
    const { subscription, client, transport } = rt;
    rt.live = false;
    rt.subscription = undefined;
    rt.client = undefined;
    rt.transport = undefined;
    if (subscription) await subscription.close().catch(() => {});
    if (client) {
      // Detach before closing: `close()` fires `onclose`, and the connect loop
      // is already past its `dropped` await (a second resolve is a no-op, but
      // an error callback firing during teardown is noise in the log).
      client.onclose = (): void => {};
      client.onerror = (): void => {};
      await client.close().catch(() => {});
    } else if (transport) {
      await transport.close().catch(() => {});
    }
  }

  /**
   * Modern era only. The `listChanged` client option already auto-opens the
   * `subscriptions/listen` stream when the server advertises the capability;
   * this covers the case where it did not (and the server nonetheless
   * advertises it) by opening the stream explicitly and registering the
   * notification handler the auto-open would have installed.
   */
  private async subscribeListChanged(rt: ServerRuntime): Promise<void> {
    const client = rt.client;
    if (!client) return;
    if (rt.era !== "modern") {
      this.log(`[mcp] ${rt.server}: list-changed via legacy notifications`);
      return;
    }
    const auto = client.autoOpenedSubscription;
    if (auto) {
      rt.subscription = auto;
      this.log(`[mcp] ${rt.server}: list-changed via auto-opened subscription`);
      return;
    }
    if (client.getServerCapabilities()?.tools?.listChanged !== true) {
      this.log(`[mcp] ${rt.server}: server does not advertise tools.listChanged`);
      return;
    }
    try {
      client.setNotificationHandler("notifications/tools/list_changed", (): void => {
        void this.resync(rt, "list_changed");
      });
      rt.subscription = await client.listen({ toolsListChanged: true });
      this.log(`[mcp] ${rt.server}: list-changed via subscriptions/listen`);
    } catch (err) {
      this.log(`[mcp] ${rt.server}: subscriptions/listen failed: ${errorMessage(err)}`);
    }
  }

  // -------------------------------------------------------------------------
  // Catalog sync
  // -------------------------------------------------------------------------

  /**
   * Queue a resync behind whatever sync is already running for this server.
   *
   * `syncChain` must never be left holding a rejection: every later `.then()`
   * would inherit it and the server would stop resyncing for the life of the
   * process, silently. So the stored chain is always the CAUGHT promise, and
   * only the returned one can reject.
   */
  private resync(rt: ServerRuntime, reason: string): Promise<void> {
    return this.enqueueSync(rt, reason, { refresh: true, announce: true });
  }

  /**
   * The first sync of a connection. It does NOT propagate: a catalog write
   * that fails (an oversized row aborting the transaction, the database
   * briefly away) is not a reason to tear down a healthy MCP connection and
   * respawn the child — that turns one bad tool description into an infinite
   * spawn loop. The connection stays up, the previous generation stays in the
   * catalog, `lastError` records why, and a backed-off retry is scheduled.
   */
  private syncNow(rt: ServerRuntime): Promise<void> {
    return this.enqueueSync(rt, "connect", { refresh: false, announce: false });
  }

  /** Queue a sync behind whatever sync is already running for this server, and
   *  own the failure policy in one place. */
  private enqueueSync(
    rt: ServerRuntime,
    reason: string,
    opts: { refresh: boolean; announce: boolean },
  ): Promise<void> {
    // Everything — the announcement included — runs INSIDE the chained
    // function, so a `log` that throws cannot escape synchronously into a
    // timer callback or an SDK notification handler, where nothing would
    // catch it.
    const run = rt.syncChain.then(async () => {
      if (this.closed || !rt.client) return;
      try {
        if (opts.announce) this.log(`[mcp] ${rt.server}: resync (${reason})`);
        await this.syncTools(rt, opts);
        rt.syncFailures = 0;
      } catch (err) {
        if (this.closed) return;
        rt.syncFailures += 1;
        rt.lastError = errorMessage(err);
        this.log(`[mcp] ${rt.server}: sync failed (${reason}): ${rt.lastError}`);
        this.scheduleSyncRetry(rt);
      }
    });
    // The STORED chain is always caught, so neither a later `.then()` nor a
    // fire-and-forget caller can inherit a rejection — even if `log` throws.
    rt.syncChain = run.catch(() => {});
    return rt.syncChain;
  }

  /** Retry a failed catalog write with full-jitter backoff, on the same timer
   *  slot the ttl refresh uses (a retry supersedes a refresh). */
  private scheduleSyncRetry(rt: ServerRuntime): void {
    this.clearRefresh(rt);
    if (this.closed) return;
    const delay = clampDelay(
      reconnectDelayMs(rt.syncFailures, {
        base: SYNC_RETRY_BASE_MS,
        cap: SYNC_RETRY_MAX_MS,
        random: this.random,
      }),
    );
    this.log(`[mcp] ${rt.server}: retrying sync in ${delay}ms (failure ${rt.syncFailures})`);
    rt.refreshTimer = this.timers.setTimeout(() => {
      rt.refreshTimer = undefined;
      void this.resync(rt, "sync-retry");
    }, delay);
  }

  private async syncTools(rt: ServerRuntime, opts: { refresh: boolean }): Promise<void> {
    const client = rt.client;
    if (!client || this.closed) return;

    const pages: { tools: unknown[]; ttlMs: number | undefined } = await listAllTools(
      client,
      opts.refresh,
      this.requestTimeoutMs,
    );

    const byRaw = new Map<string, { description: string; parameters: Record<string, unknown> }>();
    for (const raw of pages.tools) {
      const tool = raw as { name?: unknown; description?: unknown; title?: unknown; inputSchema?: unknown };
      if (typeof tool.name !== "string" || tool.name === "") continue;
      const described =
        typeof tool.description === "string"
          ? tool.description
          : typeof tool.title === "string"
            ? tool.title
            : "";
      byRaw.set(tool.name, this.boundTool(rt, tool.name, described, tool.inputSchema));
    }

    // Names are assigned for the WHOLE generation at once so a sanitization
    // collision can be disambiguated rather than dropped, and the outcome does
    // not depend on the order the server listed its tools in.
    const named = mcpToolNames(rt.server, [...byRaw.keys()]);
    const rows: McpCatalogTool[] = [];
    const entries = new Map<string, ToolEntry>();
    // Code-unit order on the FINAL name: 2026-07-28 asks for a deterministic
    // `tools/list` order "to improve LLM prompt cache hit rates", and the
    // header we build from it has the same requirement.
    named.sort((a, b) => compareNames(a.name, b.name));
    for (const { raw, name } of named) {
      const bound = byRaw.get(raw);
      if (!bound) continue;
      entries.set(name, { rawName: raw, ...bound });
      rows.push({ name, rawName: raw, ...bound });
    }

    const result = await this.catalog.replaceServer(rt.server, rt.configHash, rows);
    rt.tools = entries;
    rt.toolCount = entries.size;
    this.log(
      `[mcp] ${rt.server}: ${result.upserted} tools synced, ${result.removed} withdrawn` +
        (pages.ttlMs === undefined ? "" : ` (ttlMs=${pages.ttlMs})`),
    );
    this.scheduleRefresh(rt, pages.ttlMs);
  }

  private scheduleRefresh(rt: ServerRuntime, ttlMs: number | undefined): void {
    this.clearRefresh(rt);
    if (ttlMs === undefined || this.closed) return;
    const delay = clampDelay(Math.min(MAX_REFRESH_MS, Math.max(MIN_REFRESH_MS, ttlMs)));
    rt.refreshTimer = this.timers.setTimeout(() => {
      rt.refreshTimer = undefined;
      void this.resync(rt, "ttl");
    }, delay);
  }

  /**
   * One tool's description and schema, bounded.
   *
   * The catalog's search column is a generated `tsvector`, and Postgres caps a
   * tsvector at 1 MB. A single 1.35 MB description therefore aborts the WHOLE
   * `replaceServer` transaction — every row of the generation lost — so the
   * cap belongs on this side of the wire, not in a hope that servers behave.
   * A stub replaces an over-large schema for the same reason (and because a
   * 64 KB schema at prefix position 0 is its own problem); the description
   * says so, which is the only channel the model has.
   */
  private boundTool(
    rt: ServerRuntime,
    raw: string,
    description: string,
    schema: unknown,
  ): { description: string; parameters: Record<string, unknown> } {
    let text = description;
    if (text.length > MAX_DESCRIPTION_CHARS) {
      this.log(`[mcp] ${rt.server}: description for ${raw} truncated (${text.length} chars)`);
      text = `${text.slice(0, MAX_DESCRIPTION_CHARS)} […truncated from ${description.length} chars]`;
    }
    let parameters = normalizeSchema(schema);
    const size = Buffer.byteLength(canonicalJson(parameters), "utf8");
    if (size > MAX_SCHEMA_BYTES) {
      this.log(`[mcp] ${rt.server}: schema for ${raw} replaced by a stub (${size} bytes)`);
      parameters = { type: "object" };
      text = `${text}\n\n[schema omitted: ${size} bytes exceeds the ${MAX_SCHEMA_BYTES}-byte limit; pass arguments as documented above]`;
    }
    return { description: text, parameters };
  }

  private clearRefresh(rt: ServerRuntime): void {
    if (rt.refreshTimer === undefined) return;
    this.timers.clearTimeout(rt.refreshTimer);
    rt.refreshTimer = undefined;
  }
}

// ---------------------------------------------------------------------------
// listTools pagination
// ---------------------------------------------------------------------------

/**
 * `tools/list`, aggregated, plus its `ttlMs`.
 *
 * `listTools()` called WITHOUT a `cursor` walks every page inside the SDK and
 * returns the complete list with no `nextCursor`, capped by the client's
 * `listMaxPages` (64) — so there is nothing left here to paginate. That cap is
 * a throw, and the caller treats it exactly like any other sync failure: the
 * connection survives, the previous generation stands, a retry is scheduled.
 *
 * `cacheMode: "refresh"` on a resync, because the whole reason we are here is
 * that the cached list is no longer trusted.
 *
 * `cacheScope` is read and NOT acted on: the SDK's response cache lives inside
 * this client, and our catalog is per-tenant and per-deployment — never a
 * shared intermediary — so a `"private"` result has nowhere to leak to. If
 * this ever fronted a multi-principal cache, `cachePartition` would have to be
 * set to the auth subject.
 */
async function listAllTools(
  client: Client,
  refresh: boolean,
  timeoutMs: number | undefined,
): Promise<{ tools: unknown[]; ttlMs: number | undefined }> {
  const result = await client.listTools(undefined, {
    ...(refresh ? { cacheMode: "refresh" as const } : {}),
    ...(timeoutMs ? { timeout: timeoutMs } : {}),
  });
  return { tools: [...result.tools], ttlMs: readTtlMs(result) };
}

/**
 * A tool's `inputSchema`, defended AND canonicalized.
 *
 * Defended: MCP requires an object schema, but a server that sends nothing at
 * all must still produce a callable tool.
 *
 * Canonicalized: this value goes verbatim onto the provider wire as
 * `input_schema`, and it reaches us from two sources that disagree about key
 * order — Postgres `jsonb` re-sorts object keys at every level, a live
 * `tools/list` does not. Run 1 (trusted cache) and run 2 (post-sync) would
 * otherwise send different bytes for the same schema, changing the cached
 * prefix at position 0 and busting every provider cache tier on alternate
 * wakes. Both paths go through here, and so does the catalog write, so all
 * three agree byte for byte. See `sortObjectKeysDeep` in naming.ts.
 */
function normalizeSchema(schema: unknown): Record<string, unknown> {
  if (typeof schema === "object" && schema !== null && !Array.isArray(schema)) {
    return sortObjectKeysDeep(schema) as Record<string, unknown>;
  }
  return { properties: {}, type: "object" };
}

/** Every scheduled delay goes through here: `setTimeout` takes a signed 32-bit
 *  delay and silently fires on the next tick when it overflows. */
function clampDelay(ms: number): number {
  if (!Number.isFinite(ms) || ms < 0) return 0;
  return Math.min(MAX_TIMER_MS, Math.floor(ms));
}

// ---------------------------------------------------------------------------
// The Tool adapter
// ---------------------------------------------------------------------------

/** What `McpTool` needs from the manager — the seam a test fakes. */
export interface McpCaller {
  call(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<ToolResult>;
}

/**
 * One always-on MCP tool, as the agent loop's header sees it. The schema is
 * the server's own `inputSchema` verbatim (JSON Schema 2020-12 is legal on the
 * wire as of 2026-07-28), because rewriting it would make the model's
 * arguments disagree with what the server validates.
 */
export class McpTool implements Tool {
  constructor(
    readonly name: string,
    readonly description: string,
    readonly parameters: Record<string, unknown>,
    private readonly caller: McpCaller,
  ) {}

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    return await this.caller.call(this.name, args, ctx.signal);
  }
}
