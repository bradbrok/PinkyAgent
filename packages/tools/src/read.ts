/**
 * read: read a file with 1-based offset/limit and return numbered lines.
 * Defaults to the whole file, capped at 2000 lines.
 */
import { existsSync, statSync } from "node:fs";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import { sandboxResolve } from "./sandbox";

const DEFAULT_CAP_LINES = 2000;

export class ReadTool implements Tool {
  readonly name = "read";
  readonly description =
    "Read a file inside the sandbox. Returns numbered lines; optional 1-based offset and limit. Defaults to the whole file (up to 2000 lines).";
  readonly parameters = {
    type: "object",
    properties: {
      path: { type: "string", description: "Path relative to the sandbox root" },
      offset: { type: "number", description: "1-based line offset" },
      limit: { type: "number", description: "Max lines to return" },
    },
    required: ["path"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (typeof args.path !== "string" || args.path.length === 0) {
      return { text: "read: 'path' must be a non-empty string", isError: true };
    }
    const resolved = sandboxResolve(ctx.cwd, args.path);
    if (!resolved.ok) {
      return { text: `read: ${resolved.error}`, isError: true };
    }
    if (!existsSync(resolved.abs)) {
      return { text: `read: no such file: ${args.path}`, isError: true };
    }
    if (statSync(resolved.abs).isDirectory()) {
      return { text: `read: ${args.path} is a directory; use the glob tool instead`, isError: true };
    }

    let offset = 1;
    if (args.offset !== undefined) {
      const n = Number(args.offset);
      if (!Number.isInteger(n) || n < 1) {
        return { text: "read: 'offset' must be a positive integer", isError: true };
      }
      offset = n;
    }
    let limit: number | undefined;
    if (args.limit !== undefined) {
      const n = Number(args.limit);
      if (!Number.isInteger(n) || n < 1) {
        return { text: "read: 'limit' must be a positive integer", isError: true };
      }
      limit = n;
    }

    const text = await Bun.file(resolved.abs).text();
    const lines = text.split("\n");
    if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
    const start = offset - 1;
    const end = limit === undefined ? start + DEFAULT_CAP_LINES : start + limit;
    const slice = lines.slice(start, end);
    const truncated = limit === undefined && end < lines.length;

    if (slice.length === 0) {
      return { text: "(empty)" };
    }
    const body = slice.map((line, i) => `${start + i + 1}: ${line}`).join("\n");
    return { text: truncated ? `${body}\n(truncated)` : body };
  }
}
