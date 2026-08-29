/**
 * Slice 9 end to end, through the CLI, as real child processes: a configured
 * MCP server, the catalog it publishes, and a `pinky headless` run that finds a
 * tool it was never given, reads its schema, calls it, and answers with the
 * result.
 *
 * Why this file exists next to headless.test.ts rather than inside it:
 *
 *  - Nothing else proves the MCP plane's diagnostics stay OFF STDOUT. The
 *    manager logs a line per server (era, protocol version, tools synced) and
 *    a stdio child inherits stderr; one `console.log` on that path corrupts the
 *    JSONL protocol, and only a real process can show it. Hence the same blunt
 *    assertion as headless.test.ts: EVERY stdout line must parse as JSON.
 *  - Nothing else proves a DEFERRED tool is reachable. The unit suite fakes the
 *    catalog, the manager suite fakes the loop; here a tool that is absent from
 *    the request header is executed by name and its output lands in a `reply`.
 *
 * Keyless: the channel's model is `fake/deferred` (runtime/providers/fake.ts),
 * which scripts tool_search -> tool_describe -> tool_call -> a final answer
 * carrying the tool's own output plus a marker. No API key is consulted.
 *
 * The catalog is warmed by `pinky mcp sync` BEFORE the headless session, which
 * is not a workaround but the design: `start()` never blocks on a server, so a
 * process whose catalog is cold would race the child's spawn. A warmed catalog
 * is what a second process is supposed to inherit (config-hash trust), and
 * this is the only test that exercises that hand-off between two real
 * processes.
 *
 * Skipped unless PINKY_INTEGRATION=1:
 *
 *   bun run db:up && bun run migrate
 *   PINKY_INTEGRATION=1 bun test packages/cli/test/integration/mcp-tools.test.ts
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { createDb, loadEnvConfig, SettingsStore, type Db } from "@pinky/core";
import { FAKE_DEFERRED_MARKER } from "@pinky/runtime";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const ENV = loadEnvConfig();
const CLI = new URL("../../src/index.ts", import.meta.url).pathname;
/** The modern (2026-07-28) stdio fixture, spawned by the CLI under test. */
const FIXTURE = new URL("../../../mcp/test/fixtures/modern-server.ts", import.meta.url).pathname;

const RUN = crypto.randomUUID().replace(/-/g, "").slice(0, 8);
/**
 * The settings key of the server, hence the `mcp__<server>__` prefix of every
 * tool it publishes. Two properties matter:
 *  - it matches `^[a-z0-9][a-z0-9_-]{0,31}$` (the manager skips anything else);
 *  - the shared `mcptest` prefix is what beforeAll can safely sweep, since no
 *    other surface ever writes catalog rows under it.
 */
const SERVER = `mcptest${RUN}`;
const CHANNEL = `jsonl:mcp-${RUN}`;
const SCOPE = `channel:${CHANNEL}`;
/** Distinctive enough to prove the ECHOED text came from the prompt. */
const CANARY = `canary-${RUN}`;

const DEADLINE_MS = 60_000;
const TEST_TIMEOUT_MS = 120_000;

interface Run {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/** Spawn one `pinky <args>` child and collect it whole. */
async function runCli(args: string[], stdinLines: string[] = []): Promise<Run> {
  const proc = Bun.spawn(["bun", "run", CLI, ...args], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PINKY_INTEGRATION: undefined },
  });
  const timer = setTimeout(() => proc.kill(), DEADLINE_MS);
  try {
    for (const line of stdinLines) proc.stdin.write(`${line}\n`);
    await proc.stdin.flush();
    await proc.stdin.end();
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    return { stdout, stderr, exitCode: await proc.exited };
  } finally {
    clearTimeout(timer);
  }
}

interface Session extends Run {
  lines: Record<string, unknown>[];
  /** Anything on stdout that was NOT valid JSON — must always be empty. */
  garbage: string[];
}

/** `pinky headless`, fed `commands`, with stdout split into protocol lines. */
async function headless(commands: string[]): Promise<Session> {
  const run = await runCli(["headless"], commands);
  const lines: Record<string, unknown>[] = [];
  const garbage: string[] = [];
  for (const raw of run.stdout.split("\n")) {
    if (raw.trim() === "") continue;
    try {
      lines.push(JSON.parse(raw) as Record<string, unknown>);
    } catch {
      garbage.push(raw);
    }
  }
  return { ...run, lines, garbage };
}

/** The `data.type` of each streamed `event` line, and their tool names. */
function toolResults(session: Session): { name: string; text: string; isError: boolean }[] {
  return session.lines
    .filter((l) => l.type === "event")
    .map((l) => (l.event as { data?: Record<string, unknown> })?.data ?? {})
    .filter((d) => d.type === "tool_result")
    .map((d) => ({
      name: String(d.name ?? ""),
      text: String(d.text ?? ""),
      isError: d.isError === true,
    }));
}

/**
 * Pids of THIS run's fixture children, from `ps`.
 *
 * Matched on the unique server key that beforeAll put on the child's command
 * line, not on the fixture path (which smoke and every other run share) and
 * not on a parent pid (`bun run <cli>` may or may not be the direct parent
 * depending on how bun launches a script). `-o command=` is the portable
 * spelling on both macOS and Linux.
 */
async function fixturePids(): Promise<number[]> {
  const proc = Bun.spawn(["ps", "-A", "-o", "pid=,command="], { stdout: "pipe", stderr: "ignore" });
  const out = await new Response(proc.stdout).text();
  await proc.exited;
  return out
    .split("\n")
    .filter((line) => line.includes(SERVER) && line.includes("modern-server.ts"))
    // `ps` itself is never a match, but a grep-like self-match would be: the
    // filter above needs both tokens, and this process has neither.
    .map((line) => Number.parseInt(line.trim().split(/\s+/)[0] ?? "", 10))
    .filter((pid) => Number.isInteger(pid) && pid > 0);
}

/** Poll until `check()` is true, or give up. Returns whether it became true. */
async function until(check: () => Promise<boolean> | boolean, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await check()) return true;
    if (Date.now() >= deadline) return false;
    await Bun.sleep(100);
  }
}

/** True once the process is gone. Signal 0 tests existence without delivering. */
function isGone(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return false;
  } catch {
    return true;
  }
}

suite("pinky + mcp deferred tools (live processes, live db)", () => {
  let db: Db;
  let sync: Run;

  beforeAll(async () => {
    db = createDb(ENV.databaseAdminUrl);
    // A killed earlier run leaves two kinds of residue, and the settings row is
    // by far the worse one: `mcp.servers.mcptest…` is GLOBAL, so until it is
    // removed every later `pinky prompt|headless|smoke|mcp list` on this
    // machine spawns the fixture as a child process. The catalog row is milder
    // (a stale name this run's tool_search could answer with, which nothing
    // can execute) but is swept for the same reason. Only this file ever
    // writes `mcptest%`, so both deletes stay inside "delete what you created".
    await db.query(`delete from settings where scope = 'global' and key like 'mcp.servers.mcptest%'`);
    await db.query(`delete from tool_catalog where server like 'mcptest%'`);

    const settings = new SettingsStore(db);
    // GLOBAL scope on purpose: `mcp.servers` is read once, at bootstrap, from
    // the global + agent scopes — a channel-scoped server is not honored (the
    // manager and its child processes are per PROCESS, not per channel).
    await settings.set("global", `mcp.servers.${SERVER}`, {
      transport: "stdio",
      command: "bun",
      // The trailing argument is ignored by the fixture and exists only to put
      // this run's unique server key on the child's COMMAND LINE, so the
      // SIGTERM test below can find (and then prove the death of) exactly its
      // own child in `ps` — never a concurrent run's, never smoke's.
      args: ["run", FIXTURE, SERVER],
    });
    // The agent id is fixed ("pinky"), so the keyless model is pinned on the
    // channel this run's prompts land in.
    await settings.set(SCOPE, "model", "fake/deferred");

    // Warm the catalog with the same command an operator would run.
    sync = await runCli(["mcp", "sync", SERVER]);
  }, TEST_TIMEOUT_MS);

  afterAll(async () => {
    if (!db) return;
    await db.query(`delete from events where channel_id = $1`, [CHANNEL]);
    await db.query(`delete from threads where channel_id = $1`, [CHANNEL]);
    await db.query(`delete from ingress_dedup where external_id like $1`, [`${RUN}%`]);
    await db.query(`delete from settings where scope = $1`, [SCOPE]);
    await db.query(`delete from settings where scope = 'global' and key = $1`, [
      `mcp.servers.${SERVER}`,
    ]);
    await db.query(`delete from tool_catalog where server = $1`, [SERVER]);
    await db.close();
  });

  it(
    "`pinky mcp sync` connects the server and publishes its tools to the catalog",
    async () => {
      expect(sync.exitCode).toBe(0);
      // One line per server: status, era, negotiated version, catalog count.
      expect(sync.stdout).toContain(SERVER);
      expect(sync.stdout).toContain("connected");
      expect(sync.stdout).toContain("modern");
      expect(sync.stdout).toContain("2026-");

      const rows = await db.query<{ name: string; source: string; raw_name: string | null }>(
        `select name, source, raw_name from tool_catalog
          where server = $1 and removed_at is null order by name collate "C"`,
        [SERVER],
      );
      expect(rows.length).toBeGreaterThan(0);
      for (const row of rows) {
        expect(row.source).toBe("mcp");
        expect(row.name.startsWith(`mcp__${SERVER}__`)).toBe(true);
        // The server's own spelling is kept: `tools/call` has to send it back,
        // and it is not recoverable from the sanitized namespaced name.
        expect(row.raw_name).not.toBeNull();
      }
      // `echo.nested` -> `echo_nested`: the dot is not legal in a tool name.
      expect(rows.map((r) => r.name)).toContain(`mcp__${SERVER}__echo_nested`);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "`pinky tools list` shows the mcp tools as deferred, never in the header",
    async () => {
      const listed = await runCli(["tools", "list", "--scope", SCOPE]);
      expect(listed.exitCode).toBe(0);
      const rows = listed.stdout
        .split("\n")
        .filter((l) => l.includes(`mcp__${SERVER}__`));
      expect(rows.length).toBeGreaterThan(0);
      // Default mode for an MCP tool is deferred: hundreds of schemas must not
      // ride in the cached prefix of every request.
      for (const row of rows) expect(row.startsWith("deferred")).toBe(true);
      // The meta-tools are the door, and are always in the header.
      for (const meta of ["tool_search", "tool_describe", "tool_call"]) {
        const line = listed.stdout.split("\n").find((l) => l.trimEnd().endsWith(` ${meta}`));
        expect(line).toBeDefined();
        expect(line!.startsWith("head")).toBe(true);
      }
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "answers one prompt by finding, describing and calling a tool it was never given",
    async () => {
      const session = await headless([
        JSON.stringify({
          type: "prompt",
          text: `please echo ${CANARY} back to me`,
          channelId: CHANNEL,
          threadId: "t1",
          id: `${RUN}:p1`,
        }),
        JSON.stringify({ type: "exit" }),
      ]);

      // The point of the file: the MCP plane's connect/sync chatter, and the
      // stdio child's inherited stderr, stayed off the protocol stream.
      expect(session.garbage).toEqual([]);
      expect(session.exitCode).toBe(0);
      const types = session.lines.map((l) => String(l.type));
      expect(types[0]).toBe("ready");
      expect(types).not.toContain("error");
      expect(types.at(-1)).toBe("exiting");

      // The manager DID log — to stderr, where it belongs.
      expect(session.stderr).toContain(`[mcp] ${SERVER}`);

      // Three tool results, in order: the whole deferred round trip.
      const results = toolResults(session);
      expect(results.map((r) => r.name)).toEqual(["tool_search", "tool_describe", "tool_call"]);
      for (const result of results) expect(result.isError).toBe(false);
      expect(results[0]!.text).toContain(`mcp__${SERVER}__echo_nested`);
      // tool_describe hands back the server's own JSON Schema, nested intact.
      expect(results[1]!.text).toContain('"inner"');
      // ...which is what the arguments were built from: the fixture echoes
      // `outer.inner`, so the canary can only be here if the schema was read.
      expect(results[2]!.text).toContain(CANARY);

      const reply = session.lines.find((l) => l.type === "reply");
      expect(reply).toBeDefined();
      expect(String(reply!.text)).toContain(CANARY);
      expect(String(reply!.text)).toContain(FAKE_DEFERRED_MARKER);
      expect(reply!.channelId).toBe(CHANNEL);

      const finished = session.lines.find((l) => l.type === "run_finished")!;
      expect(finished.stopReason).toBe("completed");
      expect(finished.turns).toBe(4);
    },
    TEST_TIMEOUT_MS,
  );
  it(
    "SIGTERM drains the session and reaps the stdio child instead of orphaning it",
    async () => {
      // The failure this guards: Bun's default action for a signal TERMINATES
      // the process, so the `finally` that closes the MCP plane never runs and
      // every stdio child is reparented and left alive. Under systemd, `docker
      // stop` or a k8s rollout — which is how a long-lived service is always
      // stopped — that leaks one child per server per restart. Only a real
      // process can show it, and only `ps` can prove the child is gone.
      const proc = Bun.spawn(["bun", "run", CLI, "headless"], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, PINKY_INTEGRATION: undefined },
      });
      const timer = setTimeout(() => proc.kill("SIGKILL"), DEADLINE_MS);
      let pids: number[] = [];
      try {
        // Wait for `ready`, then for the child: `start()` never blocks a boot,
        // so the fixture spawns on a background loop some time after it.
        const reader = proc.stdout.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (!buf.includes("\n")) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
        }
        expect(JSON.parse(buf.split("\n")[0] ?? "{}").type).toBe("ready");
        void reader.cancel();

        const spawned = await until(async () => (pids = await fixturePids()).length > 0, 30_000);
        expect(spawned).toBe(true);

        // stdin stays OPEN: this session ends because of the signal and for no
        // other reason (an EOF would have drained it the old way).
        //
        // `process.kill(pid)` rather than `proc.kill()`: Bun's subprocess kill
        // signals the child's whole PROCESS GROUP, which would deliver SIGTERM
        // to the fixture directly and make the reaping assertion below pass
        // whether or not the parent ever cleaned up. A supervisor stopping one
        // service signals the service; so does this.
        process.kill(proc.pid, "SIGTERM");
        const exitCode = await proc.exited;
        // 128 + SIGTERM: stopped as asked, not crashed and not "success".
        expect(exitCode).toBe(143);

        // This fixture happens to be well behaved — `serveStdio` exits on the
        // stdin EOF a dying parent produces — so on its own this assertion
        // would pass even with no handler at all. It is here for the servers
        // that are NOT well behaved (a wrapper script, anything that keeps
        // running when its stdin closes), which is the population that
        // actually orphans. The DISCRIMINATING assertions are the two below:
        // the handler ran, and the process exited as stopped rather than dead.
        const reaped = await until(() => pids.every(isGone), 10_000);
        const survivors = pids.filter((pid) => !isGone(pid));
        expect({ reaped, survivors }).toEqual({ reaped: true, survivors: [] });

        const stderr = await new Response(proc.stderr).text();
        // Verified to fail when the handler is removed: without it the process
        // is terminated by the default action and this line never appears.
        expect(stderr).toContain("[shutdown] SIGTERM");
        // A clean shutdown, not a crash: no stack frames on the way out.
        expect(stderr).not.toMatch(/\n\s+at /);
      } finally {
        clearTimeout(timer);
        // Belt and braces: never leave a fixture behind for the next test.
        for (const pid of pids) {
          try {
            process.kill(pid, "SIGKILL");
          } catch {
            // already gone — the outcome under test
          }
        }
        try {
          await proc.stdin.end();
        } catch {
          // stdin closed with the process
        }
      }
    },
    TEST_TIMEOUT_MS,
  );
});
