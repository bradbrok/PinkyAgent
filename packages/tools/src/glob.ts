/**
 * glob: match files under a root with Bun's Glob. Sorted, capped at 500.
 */
import { Glob } from "bun";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import { sandboxResolve } from "./sandbox";

const MAX_MATCHES = 500;

export class GlobTool implements Tool {
  readonly name = "glob";
  readonly description =
    "Match files under the sandbox with a glob pattern. Optional 'path' sets the root (defaults to cwd). Sorted matches, capped at 500.";
  readonly parameters = {
    type: "object",
    properties: {
      pattern: { type: "string", description: "Glob pattern" },
      path: { type: "string", description: "Root directory (defaults to sandbox cwd)" },
    },
    required: ["pattern"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (typeof args.pattern !== "string" || args.pattern.length === 0) {
      return { text: "glob: 'pattern' must be a non-empty string", isError: true };
    }
    if (args.path !== undefined && typeof args.path !== "string") {
      return { text: "glob: 'path' must be a string", isError: true };
    }
    const resolved = sandboxResolve(ctx.cwd, args.path);
    if (!resolved.ok) {
      return { text: `glob: ${resolved.error}`, isError: true };
    }

    const matches: string[] = [];
    for await (const match of new Glob(args.pattern).scan({ cwd: resolved.abs, dot: false })) {
      matches.push(match);
    }
    matches.sort();

    const capped = matches.length > MAX_MATCHES;
    const shown = capped ? matches.slice(0, MAX_MATCHES) : matches;
    if (shown.length === 0) {
      return { text: "(no matches)" };
    }
    return { text: shown.join("\n") + (capped ? "\n(capped at 500 matches)" : "") };
  }
}
