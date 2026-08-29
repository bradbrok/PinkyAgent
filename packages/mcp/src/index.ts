/**
 * `@pinky/mcp` — the MCP client plane (DESIGN.md slice 9).
 *
 * Depends on `@modelcontextprotocol/client@2.0.0` (the 2026-07-28 stable line;
 * NOT the legacy monolithic `@modelcontextprotocol/sdk@1.x`) and on nothing
 * else outside the workspace. Everything a caller needs is here:
 *
 *   naming   — the pure `mcp__<server>__<raw>` transform + the config hash
 *   render   — CallToolResult -> ToolResult
 *   manager  — McpManager (connect, sync, dispatch) and the McpTool adapter
 */
export * from "./naming";
export * from "./render";
export * from "./manager";
