/**
 * McpManager against REAL child processes over REAL stdio.
 *
 * The unit suite drives the manager through in-process transports, which
 * proves the SDK's message handling and nothing about spawning. Everything
 * that only exists once there is a child process is untested until it runs
 * here:
 *
 *  - `versionNegotiation: { mode: "auto" }` on stdio spawns a DISPOSABLE
 *    SIBLING process for the `server/discover` probe and starts the caller's
 *    transport only after the era is known. A modern server therefore has to
 *    tolerate being started twice, and a legacy one has to survive the probe
 *    sibling being killed. Neither is observable in memory.
 *  - the era verdict itself: `modern` + `2026-07-28` against the SDK-built
 *    fixture, `legacy` + `2025-06-18` against the hand-rolled one that answers
 *    `server/discover` with -32601.
 *  - a spawn that fails (the DEFECT test): a bad command must land as
 *    `status: "error"` with the previously cached generation still served,
 *    not as a throw out of `start()`.
 *
 * Skipped unless PINKY_INTEGRATION=1 — the same gate as the rest of the
 * integration suite. It needs no database: the catalog is the same in-memory
 * fake the unit tests use, because what is under test here is the WIRE.
 *
 *   bun run test:integration
 */
import { describe, expect, it } from "bun:test";
import type { McpServerConfig } from "@pinky/core";
import {
  McpManager,
  hashServerConfig,
  type CatalogSink,
  type McpCatalogRow,
  type McpCatalogTool,
} from "../../src/index";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const FIXTURES = new URL("../fixtures/", import.meta.url).pathname;
const MODERN = `${FIXTURES}modern-server.ts`;
const LEGACY = `${FIXTURES}legacy-server.ts`;

function stdio(file: string): McpServerConfig {
  return { transport: "stdio", command: "bun", args: ["run", file] };
}

class FakeCatalog implements CatalogSink {
  readonly generations: { server: string; configHash: string | null; tools: McpCatalogTool[] }[] = [];
  state: { configHash: string | null; count: number; updatedAt: string } | null = null;
  cached = new Map<string, McpCatalogRow[]>();

  async replaceServer(
    server: string,
    configHash: string | null,
    tools: McpCatalogTool[],
  ): Promise<{ upserted: number; removed: number }> {
    this.generations.push({ server, configHash, tools });
    return { upserted: tools.length, removed: 0 };
  }

  async serverState(): Promise<{ configHash: string | null; count: number; updatedAt: string } | null> {
    return this.state;
  }

  async entries(opts: { server: string }): Promise<McpCatalogRow[]> {
    return this.cached.get(opts.server) ?? [];
  }

  latest(): string[] {
    return (this.generations.at(-1)?.tools ?? []).map((t) => t.name);
  }
}

/** What Postgres `jsonb` does on the way back out: keys re-sorted by LENGTH
 *  then bytes, at every level. The catalog's `parameters` column is jsonb. */
function jsonbReorder(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(jsonbReorder);
  if (typeof value !== "object" || value === null) return value;
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(record).sort((a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0))) {
    out[key] = jsonbReorder(record[key]);
  }
  return out;
}

/** Spawning a Bun child is not instant; give it room but fail loudly. */
async function waitFor(predicate: () => boolean, label: string, ms = 20_000): Promise<void> {
  const deadline = Date.now() + ms;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}

suite("McpManager over real stdio", () => {
  it("negotiates the MODERN era (2026-07-28) against the SDK fixture and syncs its tools", async () => {
    const catalog = new FakeCatalog();
    const lines: string[] = [];
    const config = stdio(MODERN);
    const mgr = new McpManager({
      servers: { modern: config },
      catalog,
      log: (l) => lines.push(l),
    });
    try {
      await mgr.start();
      await waitFor(() => catalog.generations.length >= 1, "first sync");

      expect(mgr.state("modern")).toMatchObject({
        server: "modern",
        status: "connected",
        era: "modern",
        protocolVersion: "2026-07-28",
        serverName: "pinky-modern-fixture",
        configHash: hashServerConfig(config),
      });
      // Sorted by FINAL name, code-unit order, `.` sanitized to `_`.
      expect(catalog.latest()).toEqual([
        "mcp__modern__always_fails",
        "mcp__modern__echo_nested",
        "mcp__modern__grow_toolset",
        "mcp__modern__report_stats",
      ]);
      // The nested object schema survives the wire intact.
      const echo = catalog.generations[0]?.tools.find((t) => t.name === "mcp__modern__echo_nested");
      expect(echo?.rawName).toBe("echo.nested");
      expect(echo?.parameters).toMatchObject({
        type: "object",
        properties: {
          outer: {
            type: "object",
            properties: { inner: { type: "string", description: "the inner string" } },
            required: ["inner"],
          },
        },
        required: ["outer"],
      });

      // Calls: the raw name goes back on the wire, arguments arrive intact.
      expect(await mgr.call("mcp__modern__echo_nested", { outer: { inner: "hi", count: 2 } })).toEqual({
        text: "echo:hi:2",
      });
      // structuredContent renders as a fenced JSON block after the text blocks.
      const stats = await mgr.call("mcp__modern__report_stats", { label: "x" });
      expect(stats.isError).toBeUndefined();
      expect(stats.text).toBe(
        'stats follow\n```json\n{\n  "label": "x",\n  "count": 3,\n  "ok": true\n}\n```',
      );
      // isError is a passthrough, not a harness failure.
      expect(await mgr.call("mcp__modern__always_fails", {})).toEqual({
        text: "deliberate failure",
        isError: true,
      });

      // The modern era carries `ttlMs` on `tools/list` (SEP-2549).
      expect(lines.join("\n")).toContain("ttlMs=300000");
      expect(lines.join("\n")).toContain("era=modern protocol=2026-07-28");

      // A tools/list_changed over `subscriptions/listen` triggers a resync.
      const before = catalog.generations.length;
      expect(await mgr.call("mcp__modern__grow_toolset", {})).toEqual({ text: "bonus_tool published" });
      await waitFor(() => catalog.generations.length > before, "list_changed resync");
      expect(catalog.latest()).toContain("mcp__modern__bonus_tool");
      expect(await mgr.call("mcp__modern__bonus_tool", {})).toEqual({ text: "bonus" });
    } finally {
      await mgr.close();
    }
  }, 60_000);

  it("DEFECT: two managers over the same fixture send byte-identical schemas (cache vs live)", async () => {
    // `McpTool.parameters` goes verbatim onto the provider wire as
    // `input_schema`, at prefix position 0. Manager 1 gets it from a live
    // `tools/list`; manager 2 gets the same rows back through jsonb, which
    // re-sorts object keys. Different bytes for the same tool would change the
    // cached prefix and bust every provider cache tier on alternate wakes.
    const config = stdio(MODERN);
    const live = new FakeCatalog();
    const first = new McpManager({ servers: { modern: config }, catalog: live });
    let fromSync: Record<string, unknown> | undefined;
    try {
      await first.start();
      await waitFor(() => live.generations.length >= 1, "live sync");
      fromSync = first.tools().find((t) => t.name === "mcp__modern__echo_nested")?.parameters;
    } finally {
      await first.close();
    }
    expect(fromSync).toBeDefined();

    const cached = new FakeCatalog();
    cached.state = { configHash: hashServerConfig(config), count: 4, updatedAt: new Date().toISOString() };
    cached.cached.set(
      "modern",
      (live.generations[0]?.tools ?? []).map((t) => ({
        name: t.name,
        description: t.description ?? "",
        parameters: jsonbReorder(t.parameters) as Record<string, unknown>,
        ...(t.rawName ? { rawName: t.rawName } : {}),
      })),
    );
    const second = new McpManager({ servers: { modern: config }, catalog: cached });
    try {
      await second.start();
      const fromCache = second.tools().find((t) => t.name === "mcp__modern__echo_nested")?.parameters;
      // The reorder really did move keys — otherwise this proves nothing.
      expect(JSON.stringify(jsonbReorder(fromSync))).not.toBe(JSON.stringify(fromSync));
      expect(JSON.stringify(fromCache)).toBe(JSON.stringify(fromSync));
      // ... and the whole header agrees, name for name.
      expect(second.tools().map((t) => t.name)).toEqual(
        (live.generations[0]?.tools ?? []).map((t) => t.name),
      );
    } finally {
      await second.close();
    }
  }, 60_000);

  it("falls back to the LEGACY era (2025-06-18) against a server with no server/discover", async () => {
    const catalog = new FakeCatalog();
    const lines: string[] = [];
    const mgr = new McpManager({
      servers: { legacy: stdio(LEGACY) },
      catalog,
      log: (l) => lines.push(l),
    });
    try {
      await mgr.start();
      await waitFor(() => catalog.generations.length >= 1, "first sync");

      expect(mgr.state("legacy")).toMatchObject({
        status: "connected",
        era: "legacy",
        protocolVersion: "2025-06-18",
        serverName: "pinky-legacy-fixture",
        toolCount: 2,
      });
      expect(catalog.latest()).toEqual(["mcp__legacy__legacy_add", "mcp__legacy__legacy_echo"]);
      expect(catalog.generations[0]?.tools[1]?.rawName).toBe("legacy.echo");

      expect(await mgr.call("mcp__legacy__legacy_echo", { text: "yo" })).toEqual({
        text: "legacy-echo:yo",
      });
      expect(await mgr.call("mcp__legacy__legacy_add", { a: 2, b: 3 })).toEqual({ text: "sum=5" });
      expect(lines.join("\n")).toContain("era=legacy protocol=2025-06-18");
      // No 2026 freshness hint on this era, so no refresh was scheduled.
      expect(lines.join("\n")).not.toContain("ttlMs=");
    } finally {
      await mgr.close();
    }
  }, 60_000);

  it("DEFECT: a server that cannot be spawned becomes status=error and leaves the cached generation alone", async () => {
    // Regression guard for the flap: an outage (or a typo'd command) must not
    // empty the catalog. `start()` must not throw either — one broken server
    // cannot stop the process from booting.
    const catalog = new FakeCatalog();
    const config: McpServerConfig = {
      transport: "stdio",
      command: `/nonexistent/pinky-mcp-${crypto.randomUUID().slice(0, 8)}`,
    };
    // The catalog already holds a generation written under this very config.
    catalog.state = {
      configHash: hashServerConfig(config),
      count: 2,
      updatedAt: new Date().toISOString(),
    };
    catalog.cached.set("broken", [
      { name: "mcp__broken__one", description: "one", parameters: { type: "object" }, rawName: "one" },
      { name: "mcp__broken__two", description: "two", parameters: { type: "object" }, rawName: "two" },
    ]);
    const lines: string[] = [];
    const mgr = new McpManager({
      servers: { broken: config },
      catalog,
      log: (l) => lines.push(l),
      // One retry's worth of loop, instantly, so the test does not sit through
      // the real backoff.
      sleep: () => new Promise<void>((r) => setTimeout(r, 5)),
      random: () => 0,
    });
    try {
      await mgr.start();
      // Trusted immediately, before anything was spawned.
      expect(mgr.state("broken")).toMatchObject({ status: "trusted-cache", toolCount: 2 });
      expect(mgr.tools().map((t) => t.name)).toEqual(["mcp__broken__one", "mcp__broken__two"]);

      await waitFor(() => mgr.state("broken")?.status === "error", "error status");
      const state = mgr.state("broken");
      expect(state?.lastError ?? "").not.toBe("");
      expect(state?.configHash).toBe(hashServerConfig(config));
      // The cached generation is untouched: no replaceServer, no empty list,
      // and the header the model sees is the same one it saw before the spawn
      // failed.
      expect(catalog.generations).toHaveLength(0);
      expect(mgr.tools().map((t) => t.name)).toEqual(["mcp__broken__one", "mcp__broken__two"]);
      expect(lines.join("\n")).toContain("connect failed");

      const result = await mgr.call("mcp__broken__one", {});
      expect(result.isError).toBe(true);
      expect(result.text).toContain("is not connected");
    } finally {
      await mgr.close();
    }
  }, 60_000);
});
