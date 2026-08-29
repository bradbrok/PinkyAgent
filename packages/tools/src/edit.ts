/**
 * edit: literal single-occurrence string replacement in a file.
 */
import { existsSync, statSync } from "node:fs";
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";
import { sandboxResolve } from "./sandbox";

export class EditTool implements Tool {
  readonly name = "edit";
  readonly description =
    "Replace a literal string in a file. Fails if the string is absent or occurs more than once.";
  readonly parameters = {
    type: "object",
    properties: {
      path: { type: "string", description: "Path relative to the sandbox root" },
      old: { type: "string", description: "Literal string to find (must occur exactly once)" },
      new: { type: "string", description: "Replacement string" },
    },
    required: ["path", "old", "new"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (typeof args.path !== "string" || args.path.length === 0) {
      return { text: "edit: 'path' must be a non-empty string", isError: true };
    }
    if (typeof args.old !== "string" || args.old.length === 0) {
      return { text: "edit: 'old' must be a non-empty string", isError: true };
    }
    if (typeof args.new !== "string") {
      return { text: "edit: 'new' must be a string", isError: true };
    }
    const resolved = sandboxResolve(ctx.cwd, args.path);
    if (!resolved.ok) {
      return { text: `edit: ${resolved.error}`, isError: true };
    }
    if (!existsSync(resolved.abs) || statSync(resolved.abs).isDirectory()) {
      return { text: `edit: no such file: ${args.path}`, isError: true };
    }

    const oldStr = args.old;
    const newStr = args.new;
    const before = await Bun.file(resolved.abs).text();

    let count = 0;
    let at = -1;
    let idx = 0;
    while ((idx = before.indexOf(oldStr, idx)) !== -1) {
      count++;
      at = idx;
      idx += oldStr.length;
    }
    if (count === 0) {
      return { text: `edit: 'old' not found in ${args.path}`, isError: true };
    }
    if (count > 1) {
      return { text: `edit: ambiguous, ${count} occurrences of 'old' in ${args.path}`, isError: true };
    }

    // Splice manually: String.replace would interpret $-patterns in `new`.
    const after = before.slice(0, at) + newStr + before.slice(at + oldStr.length);
    await Bun.write(resolved.abs, after);

    const delta = Buffer.byteLength(after) - Buffer.byteLength(before);
    const sign = delta >= 0 ? "+" : "";
    return { text: `edit: replaced 1 occurrence in ${args.path} (${sign}${delta} bytes)` };
  }
}
