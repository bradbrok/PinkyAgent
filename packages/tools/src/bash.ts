/**
 * bash: run a shell command via `sh -c` with a timeout. stdout and stderr are
 * combined and capped at 50KB. Non-zero exit codes and timeouts are errors.
 *
 * Containment (DESIGN.md P8 + §8.3): the child NEVER inherits the host
 * process env. The gateway process holds DATABASE_URL, SLACK_BOT_TOKEN,
 * OPENROUTER_API_KEY, ANTHROPIC_API_KEY, A2A_SECRET and SLACK_SIGNING_SECRET;
 * inheriting them would put `psql $DATABASE_URL -c "update settings ..."` one
 * tool call away and let the agent rewrite its own behavioral config. So the
 * child gets an explicit minimal env (PATH/HOME/LANG/LC_ALL/TERM/TMPDIR, only
 * those actually present) plus whatever the caller explicitly allowlists via
 * `new BashTool({ env })`.
 *
 * Note this is containment of *secrets*, not of the filesystem: unlike the
 * path tools, bash is not constrained by sandboxResolve, which is exactly why
 * createTools() leaves it off by default (see ./index.ts).
 */
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";

const DEFAULT_TIMEOUT_SEC = 30;
const MAX_TIMEOUT_SEC = 120;
const MAX_OUTPUT_BYTES = 50 * 1024;
const TIMEOUT = Symbol("timeout");

/** The only host env vars copied into the child, and only when set. */
const INHERITED_ENV_KEYS = ["PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR"] as const;

export interface BashToolOptions {
  /** Extra variables the caller deliberately exposes to the shell. */
  env?: Record<string, string>;
}

export class BashTool implements Tool {
  readonly name = "bash";
  readonly description =
    "Execute a shell command string via `sh -c`. Combines stdout+stderr, capped at 50KB. Timeout in seconds (default 30, max 120). Runs with a minimal environment: host secrets are not visible.";
  readonly parameters = {
    type: "object",
    properties: {
      command: { type: "string", description: "Command passed to sh -c" },
      timeout: { type: "number", description: "Timeout in seconds (default 30, max 120)" },
    },
    required: ["command"],
  };

  private readonly extraEnv: Record<string, string>;

  constructor(options: BashToolOptions = {}) {
    this.extraEnv = { ...(options.env ?? {}) };
  }

  /** Minimal env: allowlisted host vars, then caller-supplied ones (which win). */
  private childEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const key of INHERITED_ENV_KEYS) {
      const value = process.env[key];
      if (typeof value === "string") env[key] = value;
    }
    for (const [key, value] of Object.entries(this.extraEnv)) {
      env[key] = value;
    }
    return env;
  }

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    const command = args.command;
    if (typeof command !== "string" || command.length === 0) {
      return { text: "bash: 'command' must be a non-empty string", isError: true };
    }
    let tsec = DEFAULT_TIMEOUT_SEC;
    if (args.timeout !== undefined) {
      const n = Number(args.timeout);
      if (!Number.isFinite(n)) {
        return { text: "bash: 'timeout' must be a number (seconds)", isError: true };
      }
      tsec = Math.min(Math.max(n, 0), MAX_TIMEOUT_SEC);
    }

    const proc = Bun.spawn(["sh", "-c", command], {
      cwd: ctx.cwd,
      // Explicit, never inherited. See the module header.
      env: this.childEnv(),
      stdout: "pipe",
      stderr: "pipe",
    });
    // Start draining immediately so large outputs never deadlock on the pipe.
    const stdoutP = new Response(proc.stdout).text();
    const stderrP = new Response(proc.stderr).text();

    let resolveTimeout!: (value: typeof TIMEOUT) => void;
    const timeoutP = new Promise<typeof TIMEOUT>((resolve) => {
      resolveTimeout = resolve;
    });
    const timer = setTimeout(() => resolveTimeout(TIMEOUT), tsec * 1000);
    const out = await Promise.race([proc.exited, timeoutP]);
    clearTimeout(timer);

    if (out === TIMEOUT) {
      proc.kill(9);
      await proc.exited.catch(() => {});
      return { text: `bash: timed out after ${tsec}s`, isError: true };
    }

    const stdout = await stdoutP;
    const stderr = await stderrP;
    let combined = stdout;
    if (stderr.length > 0) {
      combined += (stdout.length > 0 ? "\n" : "") + stderr;
    }
    if (combined.length > MAX_OUTPUT_BYTES) {
      combined = combined.slice(0, MAX_OUTPUT_BYTES) + "\n[truncated]";
    }

    if (out !== 0) {
      return { text: `bash: exit code ${out}\n${combined}`, isError: true };
    }
    return { text: combined };
  }
}
