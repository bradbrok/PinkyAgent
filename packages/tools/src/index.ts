/**
 * MVP tool set. Intentionally absent: shed_context, write_continuity — those
 * belong to the runtime package.
 *
 * P8 as revised (DESIGN.md, "human-granted self-configuration"): behavioral
 * config lives ONLY in the `settings` table, the human CLI is the default
 * write path, and the agent's own write path is exactly one validated tool —
 * `settings_set` — restricted to the keys a human allow-listed under
 * `selfConfig` and journaled as a `config` event. No tool edits a config
 * file, so a value an agent gets wrong is a tool error it can read and retry,
 * never a process that fails to start. `tenantId` and `selfConfig` itself are
 * never delegable, and the system prompt is not a setting: there is still no
 * self-lobotomy path.
 *
 * The memory tools (recall/retain/memory_edit) and the settings tools
 * (settings_get/settings_set) are always registered: each degrades to a clean
 * error when the runtime hands it no memory context / no settings snapshot,
 * so a surface without those planes costs a few unusable tool descriptions and
 * nothing else. A memory is what the agent records about the world; a setting
 * is what it may adjust about itself — and only where a human said so.
 *
 * bash is OPT-IN (`createTools({ shell: true })`) and off by default. Every
 * other tool here is either path-contained by sandboxResolve (read/write/edit/
 * glob/grep) or a narrow mailbox call (a2a). bash is neither: it is arbitrary
 * host execution, so whether it exists at all is a property of the *caller*,
 * not of the tool set. A trusted local operator surface (`pinky prompt`, run
 * by the human at their own terminal) opts in; a surface reachable by any
 * program driving `pinky headless` over a pipe does not, unless it passes
 * `--shell`. Even when enabled the child process
 * gets a minimal env — see ./bash.ts — so DATABASE_URL and friends never leak.
 * That is what keeps the settings table reachable only through `settings_set`
 * and its allow-list, instead of through a psql the agent shelled out to.
 */
import type { Tool } from "@pinky/runtime";
import { BashTool } from "./bash";
import { ReadTool } from "./read";
import { WriteTool } from "./write";
import { EditTool } from "./edit";
import { GlobTool } from "./glob";
import { GrepTool } from "./grep";
import { A2ASendTool, A2AInboxTool } from "./a2a";
import { RecallTool, RetainTool, MemoryEditTool } from "./memory";
import { SettingsGetTool, SettingsSetTool } from "./settings";

export { BashTool, type BashToolOptions } from "./bash";
export { ReadTool } from "./read";
export { WriteTool } from "./write";
export { EditTool } from "./edit";
export { GlobTool } from "./glob";
export { GrepTool } from "./grep";
export { A2ASendTool, A2AInboxTool } from "./a2a";
export { RecallTool, RetainTool, MemoryEditTool, visibleInScope, allowedVisibility } from "./memory";
export { SettingsGetTool, SettingsSetTool, readSettingPath } from "./settings";

export interface CreateToolsOptions {
  /** Include the bash tool. Default false — callers must opt in deliberately. */
  shell?: boolean;
  /** Extra env allowlist for the bash tool (ignored unless `shell` is true). */
  shellEnv?: Record<string, string>;
}

export function createTools(opts: CreateToolsOptions = {}): Tool[] {
  const tools: Tool[] = [];
  if (opts.shell === true) {
    tools.push(new BashTool(opts.shellEnv ? { env: opts.shellEnv } : {}));
  }
  tools.push(
    new ReadTool(),
    new WriteTool(),
    new EditTool(),
    new GlobTool(),
    new GrepTool(),
    new A2ASendTool(),
    new A2AInboxTool(),
    new RecallTool(),
    new RetainTool(),
    new MemoryEditTool(),
    new SettingsGetTool(),
    new SettingsSetTool(),
  );
  return tools;
}
