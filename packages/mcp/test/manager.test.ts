/**
 * McpManager, driven through two real SDK paths and no network:
 *
 *  - a `ScriptedTransport` that answers raw JSON-RPC. Because it answers
 *    `server/discover`, `versionNegotiation: { mode: "auto" }` selects the
 *    MODERN era, which is the only way to exercise `ttlMs` (a 2026-07-28
 *    field), `subscriptions/listen`, and multi-page `tools/list` in a unit
 *    test. It is a *server*, not a stub of the client: the SDK's real Client
 *    parses everything it sends, which is how the "servers implementing
 *    2026-07-28 MUST include resultType" rule showed up here rather than in
 *    production.
 *  - `InMemoryTransport.createLinkedPair()` against a real
 *    `@modelcontextprotocol/server` `McpServer`, which negotiates the LEGACY
 *    era and carries `notifications/tools/list_changed` the 2025 way.
 *
 * The catalog is a hand-rolled fake that records generations, in the house
 * style (assert on what was written, not on how).
 */
import { describe, expect, it } from "bun:test";
import { InMemoryTransport, type Transport } from "@modelcontextprotocol/client";
import { McpServer, type StandardSchemaWithJSON } from "@modelcontextprotocol/server";
import type { McpServerConfig } from "@pinky/core";
import {
  MAX_REFRESH_MS,
  MIN_REFRESH_MS,
  McpManager,
  McpTool,
  type CatalogSink,
  type McpCatalogRow,
  type McpCatalogTool,
  type McpTimers,
  hashServerConfig,
  httpRequestInit,
  reconnectDelayMs,
  sortObjectKeysDeep,
} from "../src/index";

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/** Records every generation written, plus the trusted rows to hand back. */
class FakeCatalog implements CatalogSink {
  readonly generations: { server: string; configHash: string | null; tools: McpCatalogTool[] }[] = [];
  state: { configHash: string | null; count: number; updatedAt: string } | null = null;
  /** The live rows a trusted read gets back, keyed by server. */
  cached = new Map<string, McpCatalogRow[]>();
  serverStateCalls: string[] = [];
  entriesCalls: string[] = [];

  async replaceServer(
    server: string,
    configHash: string | null,
    tools: McpCatalogTool[],
  ): Promise<{ upserted: number; removed: number }> {
    this.generations.push({ server, configHash, tools });
    return { upserted: tools.length, removed: 0 };
  }

  async serverState(
    server: string,
  ): Promise<{ configHash: string | null; count: number; updatedAt: string } | null> {
    this.serverStateCalls.push(server);
    return this.state;
  }

  async entries(opts: { server: string }): Promise<McpCatalogRow[]> {
    this.entriesCalls.push(opts.server);
    return this.cached.get(opts.server) ?? [];
  }

  /** The names of the newest generation, in the order they were written. */
  latest(): string[] {
    return (this.generations.at(-1)?.tools ?? []).map((t) => t.name);
  }
}

type Msg = Record<string, unknown>;

/** A JSON-RPC server on a Transport. `handle` returns the `result` body for a
 *  request, `null` to answer with a method-not-found error, and `undefined`
 *  when it already replied out of band (the listen ack). */
class ScriptedTransport implements Transport {
  onclose?: (() => void) | undefined;
  onerror?: ((error: Error) => void) | undefined;
  onmessage?: (<T extends Msg>(message: T) => void) | undefined;
  readonly received: Msg[] = [];
  started = false;
  startError: Error | undefined;
  closeCount = 0;

  constructor(private readonly handle: (msg: Msg, push: (m: Msg) => void) => Msg | null | undefined) {}

  async start(): Promise<void> {
    if (this.startError) throw this.startError;
    this.started = true;
  }

  async close(): Promise<void> {
    this.closeCount += 1;
    this.onclose?.();
  }

  async send(message: Msg): Promise<void> {
    this.received.push(message);
    const id = message.id;
    if (id === undefined) return; // a notification: nothing to answer
    queueMicrotask(() => {
      const push = (m: Msg): void => this.onmessage?.(m);
      const result = this.handle(message, push);
      if (result === undefined) return;
      if (result === null) {
        push({ jsonrpc: "2.0", id, error: { code: -32601, message: `no ${String(message.method)}` } });
        return;
      }
      // 2026-07-28 requires `resultType` on every result; absent is only
      // legal for earlier revisions.
      push({ jsonrpc: "2.0", id, result: { resultType: "complete", ...result } });
    });
  }

  /** Deliver a server-initiated notification (a list_changed, say). */
  push(message: Msg): void {
    this.onmessage?.(message);
  }
}

interface ScriptOptions {
  pages?: { tools: Msg[]; nextCursor?: string }[];
  /** `tools/list` cache hint. REQUIRED on the modern era — the 2026-07-28 wire
   *  schema makes `ttlMs`/`cacheScope` mandatory on every cacheable result, so
   *  a script that omits them is rejected by the client before we see it. */
  ttlMs?: number;
  callResult?: Msg;
}

/** A modern (2026-07-28) server: answers `server/discover`, pages `tools/list`,
 *  acknowledges `subscriptions/listen`. */
function modernScript(opts: ScriptOptions = {}): ScriptedTransport {
  const ttlMs = opts.ttlMs ?? 90_000;
  const pages = opts.pages ?? [
    {
      tools: [
        { name: "beta", description: "second", inputSchema: { type: "object", properties: { b: { type: "string" } } } },
      ],
      nextCursor: "p2",
    },
    {
      tools: [{ name: "alpha.one", description: "first", inputSchema: { type: "object", properties: {} } }],
    },
  ];
  const transport: ScriptedTransport = new ScriptedTransport((msg, push) => {
    switch (msg.method) {
      case "server/discover":
        return {
          supportedVersions: ["2026-07-28"],
          capabilities: { tools: { listChanged: true } },
          ttlMs: 0,
          cacheScope: "private",
          _meta: { "io.modelcontextprotocol/serverInfo": { name: "scripted", version: "1.0.0" } },
        };
      case "tools/list": {
        const params = (msg.params ?? {}) as { cursor?: string };
        const index = params.cursor === undefined ? 0 : pages.findIndex((p) => p.nextCursor === undefined && params.cursor === "p2");
        const page = params.cursor === undefined ? pages[0] : pages[Math.max(1, index)];
        return {
          tools: page?.tools ?? [],
          ...(page?.nextCursor ? { nextCursor: page.nextCursor } : {}),
          ttlMs,
          cacheScope: "public",
        };
      }
      case "tools/call":
        return opts.callResult ?? {
          content: [{ type: "text", text: `called ${String((msg.params as { name?: string })?.name)}` }],
        };
      case "subscriptions/listen":
        push({
          jsonrpc: "2.0",
          method: "notifications/subscriptions/acknowledged",
          params: {
            notifications: { toolsListChanged: true },
            _meta: { "io.modelcontextprotocol/subscriptionId": msg.id },
          },
        });
        return undefined;
      default:
        return null;
    }
  });
  return transport;
}

/** Timers that never fire on their own — the test decides when. */
class FakeTimers implements McpTimers {
  readonly scheduled: { fn: () => void; ms: number; cancelled: boolean }[] = [];

  setTimeout(fn: () => void, ms: number): unknown {
    this.scheduled.push({ fn, ms, cancelled: false });
    return this.scheduled.length - 1;
  }

  clearTimeout(handle: unknown): void {
    const entry = this.scheduled[handle as number];
    if (entry) entry.cancelled = true;
  }

  /** Run the newest live timer. */
  fireLatest(): void {
    for (let i = this.scheduled.length - 1; i >= 0; i--) {
      const entry = this.scheduled[i];
      if (entry && !entry.cancelled) {
        entry.cancelled = true;
        entry.fn();
        return;
      }
    }
    throw new Error("no live timer to fire");
  }
}

const HTTP_CONFIG: McpServerConfig = { transport: "http", url: "https://example.invalid/mcp" };

/**
 * What Postgres `jsonb` does to a stored object on the way back out: keys are
 * re-sorted by LENGTH first, then bytes, at every level. The catalog column is
 * jsonb, so this is what a trusted-cache read actually returns — and why the
 * cached and live schemas must both be canonicalized before they reach the
 * provider as `input_schema`.
 */
function jsonbReorder(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(jsonbReorder);
  if (typeof value !== "object" || value === null) return value;
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  const keys = Object.keys(record).sort((a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0));
  for (const key of keys) out[key] = jsonbReorder(record[key]);
  return out;
}

/** Spin the event loop until `predicate` holds, or fail loudly. */
async function waitFor(predicate: () => boolean, label: string, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 5));
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("reconnectDelayMs", () => {
  it("is full-jitter, exponential, and capped", () => {
    const random = () => 1; // the top of the jitter window
    expect(reconnectDelayMs(1, { base: 1000, cap: 30_000, random })).toBe(1000);
    expect(reconnectDelayMs(2, { base: 1000, cap: 30_000, random })).toBe(2000);
    expect(reconnectDelayMs(6, { base: 1000, cap: 30_000, random })).toBe(30_000);
    expect(reconnectDelayMs(99, { base: 1000, cap: 30_000, random })).toBe(30_000);
    // Jitter is the whole window, so a 0 draw is a legal (immediate) retry.
    expect(reconnectDelayMs(3, { base: 1000, cap: 30_000, random: () => 0 })).toBe(0);
  });
});

describe("McpManager — start and catalog trust", () => {
  it("trusted-cache exposes tools() from the catalog before connect, and a call through it awaits the connection", async () => {
    // The header is the cached prefix (tools -> system -> messages). A tool
    // that is absent on run 1 and present on run 2 invalidates every provider
    // cache tier AND shows the model a different tool set on consecutive
    // wakes, so the trusted generation has to be complete before anything is
    // spawned — not after the first sync.
    const catalog = new FakeCatalog();
    catalog.state = { configHash: hashServerConfig(HTTP_CONFIG), count: 2, updatedAt: "2026-08-29T00:00:00Z" };
    catalog.cached.set("s", [
      // Deliberately out of order, and with a raw name the final name cannot
      // be reversed into: both have to come back from the ROW, not the name.
      { name: "mcp__s__beta", description: "second", parameters: { type: "object" }, rawName: "beta" },
      {
        name: "mcp__s__alpha_one",
        description: "first",
        parameters: { type: "object", properties: {} },
        rawName: "alpha.one",
      },
    ]);

    const transport = modernScript();
    let released = false;
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      // Hold the connect open so the assertions below are unambiguously
      // "before the server answered".
      transportFactory: async () => {
        await waitFor(() => released, "release");
        return transport;
      },
      timers: new FakeTimers(),
    });

    await mgr.start();
    expect(mgr.state("s")).toMatchObject({ status: "trusted-cache", toolCount: 2 });
    expect(catalog.entriesCalls).toEqual(["s"]);
    expect(catalog.generations).toHaveLength(0);
    // The whole header, complete, sorted, with the catalog's descriptions and
    // schemas — identical to what the post-sync header will be.
    const header = mgr.tools();
    expect(header.map((t) => t.name)).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    expect(header[0]?.description).toBe("first");
    expect(header[0]?.parameters).toEqual({ type: "object", properties: {} });

    // ... but calling one still waits for the connection: a cached schema is
    // not a live server.
    let settled = false;
    const pending = mgr.call("mcp__s__alpha_one", {}).then((r) => {
      settled = true;
      return r;
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(settled).toBe(false);

    released = true;
    expect(await pending).toEqual({ text: "called alpha.one" });
    // The RAW name from the cached row is what went on the wire.
    expect((transport.received.find((m) => m.method === "tools/call")?.params as { name?: string })?.name).toBe(
      "alpha.one",
    );
    expect(mgr.state("s")?.status).toBe("connected");
    // The sync produced the same header the cache did: no change, no cache bust.
    expect(mgr.tools().map((t) => t.name)).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    await mgr.close();
  });

  it("does not trust rows written under a different config hash", async () => {
    const catalog = new FakeCatalog();
    catalog.state = { configHash: "someotherhash", count: 7, updatedAt: "2026-08-29T00:00:00Z" };
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript(),
      timers: new FakeTimers(),
    });
    await mgr.start();
    expect(mgr.state("s")?.status).toBe("connecting");
    expect(mgr.state("s")?.toolCount).toBe(0);
    await mgr.close();
  });

  it("skips a server whose settings key is not legal, and never connects it", async () => {
    const catalog = new FakeCatalog();
    const lines: string[] = [];
    let factoryCalls = 0;
    const mgr = new McpManager({
      servers: { "Bad Key": HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => {
        factoryCalls += 1;
        return modernScript();
      },
    });
    await mgr.start();
    expect(mgr.states()).toEqual([]);
    expect(factoryCalls).toBe(0);
    expect(lines.join("\n")).toContain("not a legal server key");
    await mgr.close();
  });
});

describe("McpManager — sync", () => {
  it("negotiates the modern era, walks every page, and writes one sorted generation", async () => {
    const catalog = new FakeCatalog();
    const lines: string[] = [];
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => modernScript({ ttlMs: 90_000 }),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");

    expect(mgr.state("s")).toMatchObject({
      server: "s",
      status: "connected",
      era: "modern",
      protocolVersion: "2026-07-28",
      serverName: "scripted",
      toolCount: 2,
      configHash: hashServerConfig(HTTP_CONFIG),
    });
    // Page 2's tool sorts before page 1's: the order is the FINAL name's, not
    // the wire's, and `.` is sanitized to `_` on the way in.
    expect(catalog.latest()).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    expect(catalog.generations[0]?.configHash).toBe(hashServerConfig(HTTP_CONFIG));
    // The server's own spelling is preserved for `tools/call`.
    expect(catalog.generations[0]?.tools[0]).toMatchObject({
      rawName: "alpha.one",
      description: "first",
    });
    // The schema is passed through verbatim.
    expect(catalog.generations[0]?.tools[1]?.parameters).toEqual({
      type: "object",
      properties: { b: { type: "string" } },
    });
    expect(lines.join("\n")).toContain("era=modern protocol=2026-07-28");
    await mgr.close();
  });

  it("disambiguates a sanitization collision instead of dropping the loser", async () => {
    // `a.b` and `a/b` both sanitize to `a_b`. Dropping the second means the
    // model never learns that tool exists — it is simply absent from the
    // catalog, with only a log line the model cannot read. The loser gets the
    // hashed spelling instead, the same mechanism truncation uses.
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () =>
        modernScript({
          pages: [
            {
              tools: [
                { name: "a/b", description: "slash", inputSchema: { type: "object" } },
                { name: "a.b", description: "dot", inputSchema: { type: "object" } },
              ],
            },
          ],
        }),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");

    const names = catalog.latest();
    expect(names).toHaveLength(2);
    // Raw names are sorted first, so "a.b" (0x2E) keeps the plain spelling
    // regardless of the order the server listed them in.
    expect(names[0]).toBe("mcp__s__a_b");
    expect(names[1]).toMatch(/^mcp__s__a_b_[0-9a-f]{8}$/);
    const rows = catalog.generations[0]?.tools ?? [];
    expect(rows.find((t) => t.name === "mcp__s__a_b")?.rawName).toBe("a.b");
    expect(rows.find((t) => t.name === names[1])?.rawName).toBe("a/b");
    // ... and both are callable, with the server's own spelling on the wire.
    expect(await mgr.call(names[1] as string, {})).toEqual({ text: "called a/b" });
    await mgr.close();
  });

  it("serves byte-identical schemas from the cache and from a live sync", async () => {
    // `McpTool.parameters` goes VERBATIM onto the provider wire as
    // `input_schema`, at prefix position 0. The cache path reads it back
    // through jsonb (keys re-sorted by length then bytes); the sync path uses
    // the server's own key order. If the two disagree, run 1 and run 2 send
    // different bytes for the same tool and every provider cache tier is
    // invalidated on alternate wakes.
    const nested = {
      type: "object",
      properties: {
        outer: {
          type: "object",
          properties: { inner: { type: "string", description: "d" }, count: { type: "integer" } },
          required: ["inner"],
        },
      },
      required: ["outer"],
    };

    // --- run 1: a live sync ---
    const live = new FakeCatalog();
    const mgrLive = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog: live,
      transportFactory: () =>
        modernScript({
          pages: [{ tools: [{ name: "nest", description: "n", inputSchema: nested }] }],
        }),
      timers: new FakeTimers(),
    });
    await mgrLive.start();
    await waitFor(() => live.generations.length === 1, "live sync");
    const fromSync = mgrLive.tools()[0]?.parameters;
    await mgrLive.close();

    // --- run 2: the same rows, read back the way Postgres hands them over ---
    const cached = new FakeCatalog();
    cached.state = { configHash: hashServerConfig(HTTP_CONFIG), count: 1, updatedAt: "x" };
    cached.cached.set(
      "s",
      (live.generations[0]?.tools ?? []).map((t) => ({
        name: t.name,
        description: t.description ?? "",
        parameters: jsonbReorder(t.parameters) as Record<string, unknown>,
        ...(t.rawName ? { rawName: t.rawName } : {}),
      })),
    );
    const mgrCached = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog: cached,
      transportFactory: () => modernScript(),
      timers: new FakeTimers(),
    });
    await mgrCached.start();
    const fromCache = mgrCached.tools()[0]?.parameters;
    await mgrCached.close();

    // The reorder really did change the bytes — otherwise this proves nothing.
    expect(JSON.stringify(jsonbReorder(nested))).not.toBe(JSON.stringify(nested));
    expect(JSON.stringify(fromCache)).toBe(JSON.stringify(fromSync));
    expect(JSON.stringify(fromSync)).toBe(JSON.stringify(sortObjectKeysDeep(nested)));
  });

  it("caps a giant description and stubs a giant schema rather than losing the generation", async () => {
    // The catalog's search column is a generated tsvector, capped at 1 MB by
    // Postgres. One oversized description aborts the WHOLE replaceServer
    // transaction, so every row of the generation is lost — and before the
    // sync fix, the failure respawned the server forever.
    const catalog = new FakeCatalog();
    const lines: string[] = [];
    const huge = "x".repeat(1_350_000);
    const fatSchema = {
      type: "object",
      properties: { blob: { type: "string", description: "y".repeat(70_000) } },
    };
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () =>
        modernScript({
          pages: [
            {
              tools: [
                { name: "wordy", description: huge, inputSchema: { type: "object" } },
                { name: "fat", description: "ok", inputSchema: fatSchema },
              ],
            },
          ],
        }),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");

    const rows = catalog.generations[0]?.tools ?? [];
    const wordy = rows.find((t) => t.name === "mcp__s__wordy");
    expect(wordy?.description?.length).toBeLessThan(8_100);
    expect(wordy?.description).toContain("truncated from 1350000 chars");

    const fat = rows.find((t) => t.name === "mcp__s__fat");
    expect(fat?.parameters).toEqual({ type: "object" });
    expect(fat?.description).toContain("schema omitted");
    expect(lines.join("\n")).toContain("description for wordy truncated");
    expect(lines.join("\n")).toContain("schema for fat replaced by a stub");
    await mgr.close();
  });

  it("clamps an absurd ttlMs to 24h (setTimeout overflows past 2^31 and fires immediately)", async () => {
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript({ ttlMs: 2 ** 32 }),
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");
    const live = timers.scheduled.filter((t) => !t.cancelled);
    expect(live[0]?.ms).toBe(MAX_REFRESH_MS);
    expect(live[0]?.ms).toBeLessThanOrEqual(2_147_483_647);
    await mgr.close();
  });

  it("schedules a ttl refresh at max(60s, ttlMs) and resyncs when it fires", async () => {
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript({ ttlMs: 90_000 }),
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");

    const live = timers.scheduled.filter((t) => !t.cancelled);
    expect(live).toHaveLength(1);
    expect(live[0]?.ms).toBe(90_000);

    timers.fireLatest();
    await waitFor(() => catalog.generations.length === 2, "ttl resync");
    expect(catalog.latest()).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    // ... and the next refresh is scheduled again.
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(1);
    await mgr.close();
  });

  it("floors a server's aggressive ttl at 60s (a freshness hint is not a schedule)", async () => {
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript({ ttlMs: 0 }),
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    expect(timers.scheduled.filter((t) => !t.cancelled)[0]?.ms).toBe(MIN_REFRESH_MS);
    await mgr.close();
  });

});

describe("McpManager — list_changed", () => {
  it("resyncs on a modern subscriptions/listen notification", async () => {
    const catalog = new FakeCatalog();
    const transport = modernScript();
    const lines: string[] = [];
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => transport,
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    expect(lines.join("\n")).toContain("list-changed via auto-opened subscription");
    // The subscription id the client used, echoed the way the stream carries it.
    const listen = transport.received.find((m) => m.method === "subscriptions/listen");
    transport.push({
      jsonrpc: "2.0",
      method: "notifications/tools/list_changed",
      params: { _meta: { "io.modelcontextprotocol/subscriptionId": listen?.id } },
    });
    await waitFor(() => catalog.generations.length === 2, "resync");
    await mgr.close();
  });

  it("resyncs on a LEGACY unsolicited notification (real McpServer over InMemoryTransport)", async () => {
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const lines: string[] = [];
    const server = new McpServer(
      { name: "unit-legacy", version: "9.9.9" },
      { capabilities: { tools: { listChanged: true } } },
    );
    server.registerTool(
      "zeta",
      { description: "z", inputSchema: rawSchema({ type: "object", properties: {} }) },
      () => ({ content: [{ type: "text" as const, text: "z!" }] }),
    );
    const alpha = server.registerTool(
      "alpha",
      { description: "a", inputSchema: rawSchema({ type: "object", properties: {} }) },
      () => ({ content: [{ type: "text" as const, text: "a!" }] }),
    );
    const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
    await server.connect(serverSide);

    const mgr = new McpManager({
      servers: { u: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => clientSide,
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");

    expect(mgr.state("u")).toMatchObject({ status: "connected", era: "legacy", serverName: "unit-legacy" });
    expect(catalog.latest()).toEqual(["mcp__u__alpha", "mcp__u__zeta"]);
    expect(lines.join("\n")).toContain("list-changed via legacy notifications");
    // A 2025-era `tools/list` carries no `ttlMs` (the field is 2026-07-28
    // only), so there is nothing to schedule a refresh from.
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(0);

    alpha.remove();
    await waitFor(() => catalog.generations.length === 2, "legacy resync");
    expect(catalog.latest()).toEqual(["mcp__u__zeta"]);

    await mgr.close();
    await server.close();
  });
});

describe("McpManager — resilience", () => {
  it("a failed catalog write keeps the connection, the generation, and retries with backoff", async () => {
    // Two failure modes at once. (1) A catalog write that fails used to throw
    // out of `connectOnce`, so ONE oversized row — enough to abort the whole
    // transaction — meant close the client, respawn the child, fail again,
    // forever. (2) `syncChain` is a promise chain, and a stored rejection
    // poisons every later `.then()`, so the retry would reject WITHOUT EVER
    // RUNNING. The connection must survive, and the retry must be a timer.
    const catalog = new FakeCatalog();
    let spawns = 0;
    let failNext = false;
    const original = catalog.replaceServer.bind(catalog);
    catalog.replaceServer = async (server, hash, tools) => {
      if (failNext) {
        failNext = false;
        throw new Error("tsvector is too long");
      }
      return await original(server, hash, tools);
    };
    const timers = new FakeTimers();
    const lines: string[] = [];
    failNext = true;
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => {
        spawns += 1;
        return modernScript();
      },
      timers,
      random: () => 1, // top of the jitter window, so the delay is predictable
    });
    await mgr.start();
    await waitFor(() => lines.some((l) => l.includes("sync failed")), "the failed sync");

    // The client is untouched: no reconnect, no respawn.
    expect(spawns).toBe(1);
    expect(mgr.state("s")?.status).toBe("connected");
    expect(mgr.state("s")?.lastError).toContain("tsvector is too long");
    expect(catalog.generations).toHaveLength(0);

    // A retry is scheduled on a timer, backed off, not a hot loop.
    const live = timers.scheduled.filter((t) => !t.cancelled);
    expect(live).toHaveLength(1);
    expect(live[0]?.ms).toBe(5_000);
    expect(lines.join("\n")).toContain("retrying sync in 5000ms");

    timers.fireLatest();
    await waitFor(() => catalog.generations.length === 1, "the retry");
    expect(catalog.latest()).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    expect(spawns).toBe(1);
    await mgr.close();
  });

  it("survives a log sink that throws (a fire-and-forget resync must not become an unhandled rejection)", async () => {
    const catalog = new FakeCatalog();
    const transport = modernScript();
    const timers = new FakeTimers();
    let calls = 0;
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: () => {
        calls += 1;
        throw new Error("stderr is gone");
      },
      transportFactory: () => transport,
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    expect(calls).toBeGreaterThan(0);

    // The ttl timer fires `void this.resync(...)`; every log inside it throws.
    // Nothing may escape into the timer callback.
    expect(() => timers.fireLatest()).not.toThrow();
    await waitFor(() => catalog.generations.length === 2, "resync despite the broken log");
    await mgr.close();
  });

  it("keeps resyncing after a REsync throws", async () => {
    const catalog = new FakeCatalog();
    const transport = modernScript();
    let failNext = false;
    const original = catalog.replaceServer.bind(catalog);
    catalog.replaceServer = async (server, hash, tools) => {
      if (failNext) {
        failNext = false;
        throw new Error("catalog write failed");
      }
      return await original(server, hash, tools);
    };
    const timers = new FakeTimers();
    const lines: string[] = [];
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => transport,
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");

    failNext = true;
    timers.fireLatest();
    await waitFor(() => lines.some((l) => l.includes("sync failed")), "failed resync");
    expect(catalog.generations).toHaveLength(1);

    const listen = transport.received.find((m) => m.method === "subscriptions/listen");
    transport.push({
      jsonrpc: "2.0",
      method: "notifications/tools/list_changed",
      params: { _meta: { "io.modelcontextprotocol/subscriptionId": listen?.id } },
    });
    await waitFor(() => catalog.generations.length === 2, "recovered resync");
    await mgr.close();
  });
});

describe("McpManager — outages", () => {
  it("keeps the previous generation when a reconnect fails (never flaps the list)", async () => {
    const catalog = new FakeCatalog();
    const good = modernScript();
    let calls = 0;
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => {
        calls += 1;
        if (calls === 1) return good;
        const dead = modernScript();
        dead.startError = new Error("spawn ENOENT");
        return dead;
      },
      timers: new FakeTimers(),
      // Instant, but a macrotask, so the reconnect loop cannot starve the test.
      sleep: () => new Promise<void>((r) => setTimeout(r, 0)),
      random: () => 0,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    const before = catalog.latest();
    expect(before).toHaveLength(2);

    // The link drops. The catalog must NOT be rewritten.
    await good.close();
    await waitFor(() => mgr.state("s")?.status === "error", "error status");
    expect(catalog.generations).toHaveLength(1);
    expect(mgr.state("s")?.lastError).toContain("spawn ENOENT");
    // The in-memory generation is still serving, so a name still resolves.
    expect(mgr.tools().map((t) => t.name)).toEqual(before);

    await mgr.close();
  });

  it("stays silent when close() lands mid-connect: no error line, no reconnect, transport closed", async () => {
    // `pinky mcp list` bootstraps, prints and exits, so `close()` routinely
    // lands while a handshake is in flight. Closing the transport out from
    // under it makes the SDK reject the probe; that is OUR teardown, not an
    // outage, and it must not print, must not schedule a retry, and must not
    // rewrite the state the listing just showed.
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const lines: string[] = [];
    let sleeps = 0;
    // A server that accepts the probe and never answers it.
    const transport = new ScriptedTransport(() => undefined);
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      log: (l) => lines.push(l),
      transportFactory: () => transport,
      timers,
      sleep: async () => {
        sleeps += 1;
      },
    });
    await mgr.start();
    await waitFor(
      () => transport.received.some((m) => m.method === "server/discover"),
      "the probe to go out",
    );
    const statusDuringConnect = mgr.state("s")?.status;
    expect(statusDuringConnect).toBe("connecting");

    await mgr.close();

    expect(transport.closeCount).toBeGreaterThan(0);
    expect(lines.filter((l) => l.includes("connect failed"))).toEqual([]);
    expect(lines.filter((l) => l.includes("retrying in"))).toEqual([]);
    expect(sleeps).toBe(0);
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(0);
    // The last known state survives the shutdown; nothing flipped to "error".
    expect(mgr.state("s")).toMatchObject({ status: "connecting" });
    expect(mgr.state("s")?.lastError).toBeUndefined();
    expect(catalog.generations).toHaveLength(0);
  });

  it("cancels the listen subscription and the refresh timer on close", async () => {
    // A `subscriptions/listen` stream is a long-lived POST. Leaking one per
    // manager is a slow file-descriptor leak against a real HTTP server, so
    // the teardown has to reach the wire while the transport is still alive.
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const transport = modernScript();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => transport,
      timers,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    const listenId = transport.received.find((m) => m.method === "subscriptions/listen")?.id;
    expect(listenId).toBeDefined();
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(1);

    await mgr.close();

    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(0);
    expect(
      transport.received.some(
        (m) =>
          m.method === "notifications/cancelled" &&
          (m.params as { requestId?: unknown } | undefined)?.requestId === listenId,
      ),
    ).toBe(true);
    expect(transport.closeCount).toBeGreaterThan(0);
  });

  it("cancels the refresh timer when the link drops and the reconnect fails", async () => {
    const catalog = new FakeCatalog();
    const timers = new FakeTimers();
    const first = modernScript();
    let calls = 0;
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => {
        calls += 1;
        if (calls === 1) return first;
        const dead = modernScript();
        dead.startError = new Error("still down");
        return dead;
      },
      timers,
      sleep: () => new Promise<void>((r) => setTimeout(r, 0)),
      random: () => 0,
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "first sync");
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(1);

    await first.close();
    await waitFor(() => mgr.state("s")?.status === "error", "reconnect failure");
    // No timer left pointing at a dead client (it would resync into nothing).
    expect(timers.scheduled.filter((t) => !t.cancelled)).toHaveLength(0);
    await mgr.close();
  });

  it("surfaces a first-connect failure as status error and answers calls cleanly", async () => {
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => {
        const dead = modernScript();
        dead.startError = new Error("spawn ENOENT");
        return dead;
      },
      timers: new FakeTimers(),
      sleep: () => new Promise<void>((r) => setTimeout(r, 0)),
      random: () => 0,
    });
    await mgr.start();
    await waitFor(() => mgr.state("s")?.status === "error", "error status");
    expect(catalog.generations).toHaveLength(0);

    const result = await mgr.call("mcp__s__anything", {});
    expect(result.isError).toBe(true);
    expect(result.text).toContain("is not connected");
    await mgr.close();
  });
});

describe("McpManager — call", () => {
  it("sends the server's RAW name and renders the result", async () => {
    const catalog = new FakeCatalog();
    const transport = modernScript();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => transport,
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");

    const result = await mgr.call("mcp__s__alpha_one", { x: 1 });
    expect(result).toEqual({ text: "called alpha.one" });
    const call = transport.received.find((m) => m.method === "tools/call");
    expect((call?.params as { name?: string } | undefined)?.name).toBe("alpha.one");
    await mgr.close();
  });

  it("renders structuredContent and passes isError through", async () => {
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () =>
        modernScript({
          callResult: {
            content: [{ type: "text", text: "nope" }],
            structuredContent: { ok: false },
            isError: true,
          },
        }),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");
    const result = await mgr.call("mcp__s__beta", {});
    expect(result.isError).toBe(true);
    expect(result.text).toBe('nope\n```json\n{\n  "ok": false\n}\n```');
    await mgr.close();
  });

  it("answers an unknown name without throwing, and names the server from the name itself", async () => {
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript(),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");

    const unknown = await mgr.call("not_even_an_mcp_name", {});
    expect(unknown).toEqual({ text: 'unknown MCP tool "not_even_an_mcp_name"', isError: true });

    // A well-formed name for a server that is configured but does not publish
    // it: the server segment survives sanitizing and truncation, so the
    // manager can say honestly WHICH server is missing WHAT.
    const gone = await mgr.call("mcp__s__gone", {});
    expect(gone.isError).toBe(true);
    expect(gone.text).toContain("does not publish");
    await mgr.close();
  });

  it("refuses after close instead of hanging", async () => {
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript(),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");
    await mgr.close();
    expect(await mgr.call("mcp__s__beta", {})).toEqual({ text: "mcp plane is shut down", isError: true });
  });
});

describe("httpRequestInit", () => {
  it("refuses redirects when credentials are configured, so a 302 cannot forward them", async () => {
    // fetch replays request headers on a redirect: a server answering 302 with
    // a Location on another origin would be handing that origin the bearer
    // token. An MCP endpoint is a fixed URL, so refusing costs nothing.
    const withToken = httpRequestInit(
      { transport: "http", url: "https://example.com/mcp", headers: { Authorization: "${TOK}" } },
      { TOK: "Bearer secret" },
    );
    expect(withToken.headers).toEqual({ Authorization: "Bearer secret" });
    expect(withToken.redirect).toBe("error");

    // Nothing to leak, nothing to restrict.
    const bare = httpRequestInit({ transport: "http", url: "https://example.com/mcp" }, {});
    expect(bare.redirect).toBeUndefined();
  });
});

describe("McpTool", () => {
  it("carries the catalog's schema and forwards ctx.signal to the manager", async () => {
    const seen: { name: string; args: Record<string, unknown>; signal?: AbortSignal }[] = [];
    const tool = new McpTool("mcp__s__thing", "does a thing", { type: "object", properties: { a: {} } }, {
      async call(name, args, signal) {
        seen.push({ name, args, ...(signal ? { signal } : {}) });
        return { text: "ok" };
      },
    });
    const controller = new AbortController();
    const result = await tool.execute({ a: 1 }, {
      cwd: "/tmp",
      db: {} as never,
      thread: { tenantId: "t", channelId: "c", threadId: "th" },
      emit: async () => {},
      signal: controller.signal,
    });
    expect(result).toEqual({ text: "ok" });
    expect(seen[0]?.name).toBe("mcp__s__thing");
    expect(seen[0]?.args).toEqual({ a: 1 });
    expect(seen[0]?.signal).toBe(controller.signal);
    expect(tool.parameters).toEqual({ type: "object", properties: { a: {} } });
  });

  it("is what tools() hands the always-on partition, sorted by final name", async () => {
    const catalog = new FakeCatalog();
    const mgr = new McpManager({
      servers: { s: HTTP_CONFIG },
      catalog,
      transportFactory: () => modernScript(),
      timers: new FakeTimers(),
    });
    await mgr.start();
    await waitFor(() => catalog.generations.length === 1, "sync");
    const tools = mgr.tools();
    expect(tools.map((t) => t.name)).toEqual(["mcp__s__alpha_one", "mcp__s__beta"]);
    expect(tools[0]).toBeInstanceOf(McpTool);
    expect(tools[0]?.description).toBe("first");
    await mgr.close();
  });
});

// ---------------------------------------------------------------------------

/** A raw JSON Schema wearing just enough Standard Schema for `registerTool`.
 *  Keeps the fixture free of a schema-library dependency (see
 *  test/fixtures/modern-server.ts, which uses the same trick). */
function rawSchema(schema: Record<string, unknown>): StandardSchemaWithJSON<unknown, unknown> {
  return {
    "~standard": {
      version: 1,
      vendor: "pinky-test",
      validate: (value: unknown) => ({ value }),
      jsonSchema: { input: () => schema, output: () => schema },
    },
  };
}
