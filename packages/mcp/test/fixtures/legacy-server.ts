/**
 * FIXTURE — a HAND-ROLLED LEGACY (MCP 2025-06-18) stdio server. No SDK.
 *
 *     bun run packages/mcp/test/fixtures/legacy-server.ts
 *
 * The point of hand-rolling it: the SDK's `versionNegotiation: { mode: "auto" }`
 * probes `server/discover` first and falls back to the 2025 `initialize`
 * handshake on "definitive legacy signals". A server built with the 2.0 server
 * SDK would answer the probe, so the fallback would never run. This one knows
 * exactly four methods and answers `server/discover` with -32601 — the shape of
 * every server written before 2026-07-28 existed.
 *
 * Wire rules it obeys, all of them load-bearing for the fallback:
 *  - newline-delimited JSON on stdin/stdout, nothing else on stdout ever;
 *  - `initialize` -> `{ protocolVersion, capabilities, serverInfo }`;
 *  - `notifications/initialized` -> no response (it is a notification);
 *  - `tools/list`, `tools/call`, `ping`;
 *  - anything else -> a JSON-RPC error, never a crash and never silence
 *    (silence on stdio is read by the SDK as "legacy" too, but by TIMEOUT —
 *    a much slower and less honest path to the same verdict).
 */

const PROTOCOL_VERSION = "2025-06-18";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: string | number;
  method: string;
  params?: Record<string, unknown>;
}

const TOOLS = [
  {
    name: "legacy.echo",
    description: "Echo the text argument back",
    inputSchema: {
      type: "object",
      properties: { text: { type: "string", description: "what to echo" } },
      required: ["text"],
    },
  },
  {
    name: "legacy_add",
    description: "Add two numbers",
    inputSchema: {
      type: "object",
      properties: { a: { type: "number" }, b: { type: "number" } },
      required: ["a", "b"],
    },
  },
];

function write(message: Record<string, unknown>): void {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function reply(id: string | number, result: Record<string, unknown>): void {
  write({ jsonrpc: "2.0", id, result });
}

function fail(id: string | number, code: number, message: string): void {
  write({ jsonrpc: "2.0", id, error: { code, message } });
}

function callTool(id: string | number, params: Record<string, unknown> | undefined): void {
  const name = typeof params?.name === "string" ? params.name : "";
  const args = (params?.arguments ?? {}) as Record<string, unknown>;
  if (name === "legacy.echo") {
    reply(id, { content: [{ type: "text", text: `legacy-echo:${String(args.text ?? "")}` }] });
    return;
  }
  if (name === "legacy_add") {
    const sum = Number(args.a ?? 0) + Number(args.b ?? 0);
    reply(id, { content: [{ type: "text", text: `sum=${sum}` }] });
    return;
  }
  // Tool-level failure, per spec, is a RESULT with isError — not a JSON-RPC error.
  reply(id, { content: [{ type: "text", text: `unknown tool ${name}` }], isError: true });
}

function handle(message: JsonRpcRequest): void {
  const { id, method, params } = message;

  // Notifications carry no id and get no response.
  if (id === undefined) return;

  switch (method) {
    case "initialize":
      reply(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: "pinky-legacy-fixture", version: "0.1.0" },
      });
      return;
    case "ping":
      reply(id, {});
      return;
    case "tools/list":
      reply(id, { tools: TOOLS });
      return;
    case "tools/call":
      callTool(id, params);
      return;
    default:
      // Includes `server/discover` — the definitive legacy signal.
      fail(id, -32601, `Method not found: ${method}`);
      return;
  }
}

let buffer = "";
process.stdin.on("data", (chunk: Buffer | string) => {
  buffer += typeof chunk === "string" ? chunk : chunk.toString("utf8");
  for (;;) {
    const newline = buffer.indexOf("\n");
    if (newline < 0) break;
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line === "") continue;
    try {
      handle(JSON.parse(line) as JsonRpcRequest);
    } catch (err) {
      process.stderr.write(`[legacy-fixture] bad line: ${String(err)}\n`);
    }
  }
});
process.stdin.on("end", () => process.exit(0));
