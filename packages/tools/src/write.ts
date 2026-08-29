/**
 * write: write content to a file, creating parent directories as needed.
 */
import { dirname } from "node:path";
import { mkdirSync } from "node:fs";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import { sandboxResolve } from "./sandbox";

export class WriteTool implements Tool {
  readonly name = "write";
  readonly description = "Write string content to a file inside the sandbox, creating parent directories. Returns bytes written.";
  readonly parameters = {
    type: "object",
    properties: {
      path: { type: "string", description: "Path relative to the sandbox root" },
      content: { type: "string", description: "File content" },
    },
    required: ["path", "content"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (typeof args.path !== "string" || args.path.length === 0) {
      return { text: "write: 'path' must be a non-empty string", isError: true };
    }
    if (typeof args.content !== "string") {
      return { text: "write: 'content' must be a string", isError: true };
    }
    const resolved = sandboxResolve(ctx.cwd, args.path);
    if (!resolved.ok) {
      return { text: `write: ${resolved.error}`, isError: true };
    }
    mkdirSync(dirname(resolved.abs), { recursive: true });
    const bytes = await Bun.write(resolved.abs, args.content);
    return { text: `write: wrote ${bytes} bytes to ${args.path}` };
  }
}
