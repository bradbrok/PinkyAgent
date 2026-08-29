/**
 * FIXTURE — a MODERN (MCP 2026-07-28) stdio server, built with the real
 * `@modelcontextprotocol/server` SDK.
 *
 * Run as a child process by the integration test:
 *
 *     bun run packages/mcp/test/fixtures/modern-server.ts
 *
 * What it exists to prove, on the wire and not in a mock:
 *  - `server/discover` answers, so `versionNegotiation: { mode: "auto" }`
 *    selects the MODERN era and negotiates 2026-07-28 (era detection);
 *  - a nested-object `inputSchema` survives the round trip into the catalog;
 *  - `structuredContent` renders as a fenced JSON block;
 *  - an `isError` tool result is passed through, not swallowed;
 *  - `tools/list` carries a `ttlMs` cache hint (SEP-2549), which is what the
 *    manager's refresh schedule keys on;
 *  - a `notifications/tools/list_changed` reaches the client over
 *    `subscriptions/listen` and triggers a resync.
 *
 * The tool schemas are hand-written `StandardSchemaWithJSON` objects rather
 * than zod: the fixture then needs no dependency of its own, and the JSON
 * Schema the client sees is byte-for-byte what this file wrote (a converted
 * schema would make an assertion about `properties` a test of zod's converter).
 */
import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import type { StandardSchemaWithJSON } from "@modelcontextprotocol/server";

/** A raw JSON Schema wearing just enough Standard Schema to be registrable.
 *  `validate` accepts everything: argument validation is the harness's job in
 *  this slice (agent D's registry), not the fixture's. */
function rawSchema<T>(schema: Record<string, unknown>): StandardSchemaWithJSON<T, T> {
  return {
    "~standard": {
      version: 1,
      vendor: "pinky-fixture",
      validate: (value: unknown) => ({ value: value as T }),
      jsonSchema: { input: () => schema, output: () => schema },
    },
  };
}

const server = new McpServer(
  { name: "pinky-modern-fixture", version: "1.0.0" },
  {
    capabilities: { tools: { listChanged: true } },
    // A real freshness hint on `tools/list` (2026-07-28 only): the manager
    // schedules its refresh at max(60s, ttlMs) from this.
    cacheHints: { "tools/list": { ttlMs: 300_000, cacheScope: "public" } },
  },
);

// 1. Nested object schema — the shape a naive flattener would lose.
server.registerTool(
  "echo.nested",
  {
    description: "Echo a nested payload back as text",
    inputSchema: rawSchema<{ outer: { inner: string; count?: number } }>({
      type: "object",
      properties: {
        outer: {
          type: "object",
          properties: {
            inner: { type: "string", description: "the inner string" },
            count: { type: "integer", minimum: 0 },
          },
          required: ["inner"],
        },
      },
      required: ["outer"],
    }),
  },
  (args) => ({
    content: [{ type: "text" as const, text: `echo:${args.outer?.inner ?? ""}:${args.outer?.count ?? 0}` }],
  }),
);

// 2. structuredContent.
server.registerTool(
  "report_stats",
  {
    description: "Return a small structured payload",
    inputSchema: rawSchema<{ label?: string }>({
      type: "object",
      properties: { label: { type: "string" } },
    }),
  },
  (args) => {
    const payload = { label: args.label ?? "none", count: 3, ok: true };
    return {
      content: [{ type: "text" as const, text: "stats follow" }],
      structuredContent: payload,
    };
  },
);

// 3. isError passthrough.
server.registerTool(
  "always_fails",
  {
    description: "Always returns a tool-level error",
    inputSchema: rawSchema<Record<string, never>>({ type: "object", properties: {} }),
  },
  () => ({
    content: [{ type: "text" as const, text: "deliberate failure" }],
    isError: true,
  }),
);

// 5, declared before 4 so the handler can close over it: the tool that only
// appears after a list change.
const bonus = server.registerTool(
  "bonus_tool",
  {
    description: "Only exists after grow_toolset runs",
    inputSchema: rawSchema<Record<string, never>>({ type: "object", properties: {} }),
  },
  () => ({ content: [{ type: "text" as const, text: "bonus" }] }),
);
bonus.disable();

// 4. The trigger. `enable()` already emits the list-changed event; the explicit
// send makes the intent obvious and is harmless if it double-fires (the client
// debounces and a resync is idempotent).
server.registerTool(
  "grow_toolset",
  {
    description: "Publish one more tool and announce the change",
    inputSchema: rawSchema<Record<string, never>>({ type: "object", properties: {} }),
  },
  () => {
    bonus.enable();
    server.sendToolListChanged();
    return { content: [{ type: "text" as const, text: "bonus_tool published" }] };
  },
);

serveStdio(() => server, {
  onerror: (err) => {
    process.stderr.write(`[modern-fixture] ${err.message}\n`);
  },
});
