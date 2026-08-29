/**
 * `pinky headless` end to end, as a real child process (DESIGN.md §11).
 *
 * This is the only test that proves the JSONL contract as a USER of the CLI
 * sees it, and the only one that can catch the failure mode the unit suite is
 * structurally blind to: something on the startup path printing to stdout.
 * gateway/test/headless.test.ts drives runHeadless with an injected `write`,
 * so a stray console.log — a Postgres NOTICE, a bootstrap warning, a sweep
 * line — is invisible there and corrupts the stream here. Hence the blunt
 * assertion below: EVERY stdout line must parse as JSON.
 *
 * Keyless by construction: the model for this run's channel scope is set to
 * `fake/echo` (runtime/providers/fake.ts), so no API key is consulted.
 *
 * Skipped unless PINKY_INTEGRATION=1:
 *
 *   bun run db:up && bun run migrate
 *   PINKY_INTEGRATION=1 bun test packages/cli/test/integration/headless.test.ts
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { createDb, loadEnvConfig, SettingsStore, type Db } from "@pinky/core";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const ENV = loadEnvConfig();
/** Path to the CLI entry, so the test does not depend on the runner's cwd. */
const CLI = new URL("../../src/index.ts", import.meta.url).pathname;

/** Unique per run: two runs (or a rerun after a crash) never share a channel,
 *  a settings row or a dedup id. */
const RUN = `headless-test-${crypto.randomUUID().slice(0, 8)}`;
const CHANNEL = `jsonl:${RUN}`;
const SCOPE = `channel:${CHANNEL}`;

/** A hung child must fail the test, not block the suite. macOS has no
 *  `timeout(1)`, so the deadline is a timer that kills the process; the pipes
 *  then close and the assertions run against whatever was produced. */
const DEADLINE_MS = 60_000;
/** Room for `bun install`-cold start + migrate + the run itself. */
const TEST_TIMEOUT_MS = 90_000;

interface Session {
  lines: Record<string, unknown>[];
  /** Anything on stdout that was NOT valid JSON — must always be empty. */
  garbage: string[];
  stderr: string;
  exitCode: number;
}

/** Spawn `pinky headless`, feed it `commands`, and collect the whole session. */
async function headless(commands: string[]): Promise<Session> {
  const proc = Bun.spawn(["bun", "run", CLI, "headless"], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, PINKY_INTEGRATION: undefined },
  });

  const timer = setTimeout(() => proc.kill(), DEADLINE_MS);
  try {
    for (const command of commands) proc.stdin.write(`${command}\n`);
    await proc.stdin.flush();
    await proc.stdin.end();

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    const exitCode = await proc.exited;

    const lines: Record<string, unknown>[] = [];
    const garbage: string[] = [];
    for (const raw of stdout.split("\n")) {
      if (raw.trim() === "") continue;
      try {
        lines.push(JSON.parse(raw) as Record<string, unknown>);
      } catch {
        garbage.push(raw);
      }
    }
    return { lines, garbage, stderr, exitCode };
  } finally {
    clearTimeout(timer);
  }
}

const types = (s: Session): string[] => s.lines.map((l) => String(l.type));
const first = (s: Session, type: string): Record<string, unknown> | undefined =>
  s.lines.find((l) => l.type === type);

suite("pinky headless (live process, live db)", () => {
  let db: Db;

  beforeAll(async () => {
    // Admin url: the cleanup below touches events/threads/ingress_dedup, and
    // settings has no RLS, so one privileged handle covers both jobs.
    db = createDb(ENV.databaseAdminUrl);
    // The headless agent id is fixed ("pinky"), so the model is pinned on the
    // CHANNEL scope instead — this run's threads get fake/echo, nothing else
    // in the dev database changes.
    await new SettingsStore(db).set(SCOPE, "model", "fake/echo");
  });

  afterAll(async () => {
    if (!db) return;
    await db.query(`delete from events where channel_id = $1`, [CHANNEL]);
    await db.query(`delete from threads where channel_id = $1`, [CHANNEL]);
    await db.query(`delete from ingress_dedup where external_id like $1`, [`${RUN}%`]);
    await db.query(`delete from settings where scope = $1`, [SCOPE]);
    await db.close();
  });

  it(
    "answers one prompt and exits cleanly, emitting only protocol lines",
    async () => {
      const session = await headless([
        JSON.stringify({
          type: "prompt",
          text: "hello there",
          channelId: CHANNEL,
          threadId: "t1",
          id: `${RUN}:p1`,
        }),
        JSON.stringify({ type: "exit" }),
      ]);

      // The point of the whole file: nothing but the protocol reached stdout.
      expect(session.garbage).toEqual([]);
      expect(session.exitCode).toBe(0);

      const seen = types(session);
      expect(seen[0]).toBe("ready");
      expect(seen).toContain("run_started");
      expect(seen).toContain("event");
      expect(seen).toContain("reply");
      expect(seen).toContain("run_finished");
      expect(seen[seen.length - 1]).toBe("exiting");
      // No error line anywhere in a clean session.
      expect(seen).not.toContain("error");

      const ready = first(session, "ready")!;
      expect(ready.agentId).toBe("pinky");
      expect(typeof ready.nodeId).toBe("string");
      expect(typeof ready.tenantId).toBe("string");
      // `defaultModel`, not `model`: the bootstrap snapshot, while the run
      // itself re-resolves the model from channel + agent scopes.
      expect(typeof ready.defaultModel).toBe("string");
      expect("model" in ready).toBe(false);

      // Ordering per thread: run_started -> (event|reply)* -> run_finished.
      expect(seen.indexOf("run_started")).toBeLessThan(seen.indexOf("reply"));
      expect(seen.indexOf("reply")).toBeLessThan(seen.indexOf("run_finished"));

      // The live event stream carried the assistant `message` event, which is
      // what proves onEvent is wired to the loop's append path.
      const events = session.lines.filter((l) => l.type === "event");
      const kinds = events.map(
        (l) => ((l.event as { data?: { type?: string } })?.data?.type ?? "") as string,
      );
      // The ingress itself is NOT here: ingest() writes it before the run
      // starts, so it predates the loop's onEvent hook. What the stream
      // carries is everything the run appended.
      expect(kinds).toContain("message");
      expect(kinds).toContain("egress");
      expect(kinds).not.toContain("ingress");
      for (const line of events) {
        expect(line.channelId).toBe(CHANNEL);
        expect(line.threadId).toBe("t1");
      }

      // fake/echo replies with the projected text of the last user message,
      // which for an ingress is "[<platform> <displayName??userId>]: <text>".
      const reply = first(session, "reply")!;
      expect(String(reply.text).startsWith("echo:")).toBe(true);
      expect(String(reply.text)).toContain("hello there");
      expect(reply.channelId).toBe(CHANNEL);

      const finished = first(session, "run_finished")!;
      expect(finished.stopReason).toBe("completed");
      expect(finished.turns).toBe(1);
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "reports a malformed line and keeps serving the next prompt",
    async () => {
      const session = await headless([
        "{not json at all",
        JSON.stringify({
          type: "prompt",
          text: "still alive?",
          channelId: CHANNEL,
          threadId: "t2",
          id: `${RUN}:p2`,
        }),
        JSON.stringify({ type: "exit" }),
      ]);

      expect(session.garbage).toEqual([]);
      expect(session.exitCode).toBe(0);

      const seen = types(session);
      // A bad line is a normal protocol event, not a crash: error, then the
      // session carries on and answers the prompt that followed it.
      expect(seen[0]).toBe("ready");
      expect(seen[1]).toBe("error");
      expect(seen.indexOf("error")).toBeLessThan(seen.indexOf("run_started"));

      const error = first(session, "error")!;
      expect(String(error.message)).toContain("invalid JSON");
      expect(error.line).toBe("{not json at all");

      const reply = first(session, "reply")!;
      expect(String(reply.text)).toContain("still alive?");
      expect(first(session, "run_finished")!.stopReason).toBe("completed");
      expect(seen[seen.length - 1]).toBe("exiting");
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "exits cleanly when the client stops reading stdout (EPIPE)",
    async () => {
      // The failure this guards: process.stdout.write fails asynchronously
      // once the reader is gone, and an unlistened 'error' event on that
      // stream kills the process with exit 1 — no `exiting`, no drain, the
      // database handle never closed. Only a real pipe can produce it.
      const proc = Bun.spawn(["bun", "run", CLI, "headless"], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, PINKY_INTEGRATION: undefined },
      });
      const timer = setTimeout(() => proc.kill(), DEADLINE_MS);
      try {
        // Read just the `ready` line, then walk away from stdout while
        // keeping stdin open and prompts coming.
        const reader = proc.stdout.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (!buf.includes("\n")) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
        }
        const ready = JSON.parse(buf.split("\n")[0] ?? "{}") as Record<string, unknown>;
        expect(ready.type).toBe("ready");
        await reader.cancel();

        for (let i = 0; i < 3; i++) {
          try {
            proc.stdin.write(
              `${JSON.stringify({
                type: "prompt",
                text: `still typing ${i}`,
                channelId: CHANNEL,
                threadId: "epipe",
                id: `${RUN}:epipe-${i}`,
              })}\n`,
            );
            await proc.stdin.flush();
          } catch {
            // The child is already gone; that is the outcome under test.
          }
          await Bun.sleep(200);
        }

        const exitCode = await proc.exited;
        const stderr = await new Response(proc.stderr).text();
        expect(exitCode).toBe(0);
        // A clean shutdown, not a crash: one stderr note and no stack frames.
        expect(stderr).toContain("stdout closed by the client");
        expect(stderr).not.toMatch(/\n\s+at /);
      } finally {
        clearTimeout(timer);
        try {
          await proc.stdin.end();
        } catch {
          // stdin is already closed with the process.
        }
      }
    },
    TEST_TIMEOUT_MS,
  );
});
