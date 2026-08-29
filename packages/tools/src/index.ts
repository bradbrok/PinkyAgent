/**
 * MVP tool set. Note the anti-self-lobotomy rule: no tool may write settings
 * (behavioral config is human-CLI-only) or mutate the system prompt/model.
 * Intentionally absent: shed_context, write_continuity — those belong to the
 * runtime package.
 *
 * bash is OPT-IN (`createTools({ shell: true })`) and off by default. Every
 * other tool here is either path-contained by sandboxResolve (read/write/edit/
 * glob/grep) or a narrow mailbox call (a2a). bash is neither: it is arbitrary
 * host execution, so whether it exists at all is a property of the *caller*,
 * not of the tool set. A trusted local operator surface (`pinky prompt`, run
 * by the human at their own terminal) opts in; a surface reachable by any
 * Slack user who can DM the bot does not. Even when enabled the child process
 * gets a minimal env — see ./bash.ts — so DATABASE_URL and friends never leak,
 * which is what keeps P8 ("agents cannot self-lobotomize") true.
 */
import type { Tool } from "@pinky/runtime";
import { BashTool } from "./bash";
import { ReadTool } from "./read";
import { WriteTool } from "./write";
import { EditTool } from "./edit";
import { GlobTool } from "./glob";
import { GrepTool } from "./grep";
import { A2ASendTool, A2AInboxTool } from "./a2a";

export { BashTool, type BashToolOptions } from "./bash";
export { ReadTool } from "./read";
export { WriteTool } from "./write";
export { EditTool } from "./edit";
export { GlobTool } from "./glob";
export { GrepTool } from "./grep";
export { A2ASendTool, A2AInboxTool } from "./a2a";

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
  );
  return tools;
}
