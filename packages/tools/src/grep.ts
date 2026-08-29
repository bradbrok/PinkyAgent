/**
 * grep: regex search across files under a root. JS RegExp (no Rust regex).
 * Skips binary-ish files (>1MB or NUL in first 8KB). Returns `file:line: text`.
 */
import { Glob } from "bun";
import { closeSync, openSync, readSync, statSync } from "node:fs";
import path from "node:path";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import { sandboxResolve } from "./sandbox";

const DEFAULT_LIMIT = 100;
const MAX_FILE_BYTES = 1024 * 1024;
const BINARY_SNIFF_BYTES = 8 * 1024;

function isBinaryish(abs: string): boolean {
  try {
    const st = statSync(abs);
    if (st.size > MAX_FILE_BYTES || !st.isFile()) return true;
    const fd = openSync(abs, "r");
    try {
      const head = Buffer.alloc(BINARY_SNIFF_BYTES);
      const n = readSync(fd, head, 0, BINARY_SNIFF_BYTES, 0);
      return head.subarray(0, n).includes(0);
    } finally {
      closeSync(fd);
    }
  } catch {
    return true;
  }
}

export class GrepTool implements Tool {
  readonly name = "grep";
  readonly description =
    "Search files under the sandbox with a JavaScript RegExp. Optional 'path' root, 'glob' filter, and 'limit' (default 100).";
  readonly parameters = {
    type: "object",
    properties: {
      pattern: { type: "string", description: "RegExp pattern" },
      path: { type: "string", description: "Root directory (defaults to sandbox cwd)" },
      glob: { type: "string", description: "Filter which files are searched" },
      limit: { type: "number", description: "Max matches (default 100)" },
    },
    required: ["pattern"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (typeof args.pattern !== "string" || args.pattern.length === 0) {
      return { text: "grep: 'pattern' must be a non-empty string", isError: true };
    }
    if (args.path !== undefined && typeof args.path !== "string") {
      return { text: "grep: 'path' must be a string", isError: true };
    }
    if (args.glob !== undefined && typeof args.glob !== "string") {
      return { text: "grep: 'glob' must be a string", isError: true };
    }
    let limit = DEFAULT_LIMIT;
    if (args.limit !== undefined) {
      const n = Number(args.limit);
      if (!Number.isInteger(n) || n < 1) {
        return { text: "grep: 'limit' must be a positive integer", isError: true };
      }
      limit = n;
    }
    let re: RegExp;
    try {
      re = new RegExp(args.pattern);
    } catch (err) {
      return { text: `grep: invalid pattern: ${(err as Error).message}`, isError: true };
    }

    const resolved = sandboxResolve(ctx.cwd, args.path);
    if (!resolved.ok) {
      return { text: `grep: ${resolved.error}`, isError: true };
    }
    const root = resolved.abs;
    const filter = typeof args.glob === "string" ? args.glob : "**/*";

    const matches: string[] = [];
    outer: for await (const rel of new Glob(filter).scan({ cwd: root, dot: false })) {
      const abs = path.join(root, rel);
      if (isBinaryish(abs)) continue;
      let text: string;
      try {
        text = await Bun.file(abs).text();
      } catch {
        continue;
      }
      const lines = text.split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (re.test(lines[i]!)) {
          matches.push(`${rel}:${i + 1}: ${lines[i]!}`);
          if (matches.length >= limit) break outer;
        }
      }
    }

    if (matches.length === 0) {
      return { text: "(no matches)" };
    }
    return { text: matches.join("\n") };
  }
}
