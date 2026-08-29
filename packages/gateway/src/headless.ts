/**
 * JSONL headless mode — the primary interface (DESIGN.md §11 gateways, §3 log).
 *
 * A long-lived process speaks JSON Lines: one command object per stdin line,
 * one event object per stdout line. stderr is for human logs only (`opts.log`);
 * stdout is protocol-only, and `write` is the single output path so nothing
 * can interleave a stray console.log into the stream.
 *
 * There are no process globals here on purpose: stdin is any
 * `AsyncIterable<string | Uint8Array>` and output is a `write(line)` callback,
 * so a whole session is drivable from an array of chunks in a unit test.
 *
 * Shape of a session:
 *
 *   ready                                             (once, before any command)
 *   run_started -> (event | reply)* -> run_finished   (per run, per thread)
 *   exiting                                           (once, last line written)
 *
 * Ordering is guaranteed PER THREAD, not globally: different threads run
 * concurrently and their lines interleave. Every run line carries threadId +
 * channelId so a client can demultiplex.
 *
 * Concurrency model: one promise chain per (channelId, threadId) lane. A
 * prompt is persisted immediately (ingest is the dedup claim + the ingress
 * append in ONE transaction — see EventSink) and then chained, so a burst on
 * one thread is answered one run at a time, in arrival order, while other
 * threads proceed in parallel.
 *
 * RUN ACCOUNTING: every enqueued prompt produces exactly one
 * `run_started` … (`run_finished` | `error` with `"run":"failed"`) pair, so a
 * client can balance them. A run that is cancelled before it ever reaches the
 * agent — `abort` on its thread, `exit --abort`, or `opts.signal` firing while
 * it was still parked on the chain — is reported as
 * `run_started` + `run_finished {stopReason:"aborted", turns:0}` and never
 * executes.
 */
import type { Principal, ThreadEvent, ThreadRef } from "@pinky/core";
import type { AgentRunResult } from "@pinky/runtime";
import type { EventSink, RawIngress } from "./server";

/** Defaults for the fields a client may omit (see parseCommand). */
const DEFAULT_THREAD_ID = "main";
const DEFAULT_CHANNEL_ID = "jsonl:local";
const DEFAULT_USER_ID = "local";
/** The platform tag written onto every ingress event and author principal. */
const PLATFORM = "jsonl";
/** How much of an unparseable line is echoed back in the error event. */
const MAX_ECHOED_LINE = 200;
/**
 * Largest single stdin line accepted. Without a cap, a client that never sends
 * a newline turns this process's memory into its buffer; over it, the line is
 * dropped (reported once) and the framer resynchronises on the next newline.
 */
const MAX_LINE_BYTES = 1024 * 1024;

// ---------------------------------------------------------------------------
// Commands (stdin -> service)
// ---------------------------------------------------------------------------

export type HeadlessCommand =
  | {
      type: "prompt";
      text: string;
      threadId: string;
      channelId: string;
      /** Ingest dedup key; generated when the client omitted one. */
      id?: string;
      author: Principal;
    }
  /** Cancels the run in flight on that thread AND drops the ones queued behind it. */
  | { type: "abort"; threadId: string }
  /** `abort: true` cancels in-flight and queued runs instead of waiting. */
  | { type: "exit"; abort: boolean };

// ---------------------------------------------------------------------------
// Session options
// ---------------------------------------------------------------------------

/** Per-run callbacks the session hands the caller's runAgent. */
export interface HeadlessRunHooks {
  /** Fires when `abort` (or `exit --abort`, or `opts.signal`) targets this run. */
  signal: AbortSignal;
  /** Every event the loop appends, streamed as it lands. */
  onEvent: (event: ThreadEvent) => void;
  /** The loop's deliver(): assistant text bound for the client. */
  deliver: (text: string) => Promise<void>;
}

export interface HeadlessOpts {
  tenantId: string;
  agentId: string;
  nodeId: string;
  /**
   * Reported in the `ready` line as `defaultModel`. Informational ONLY: it is
   * the bootstrap snapshot, while every run re-resolves the model from
   * `channel:<id>` + `agent:<id>` settings, so a given run may use another one.
   */
  defaultModel: string;
  events: EventSink;
  /**
   * Resolves one ingress batch into exactly ONE agent run. The batch is
   * already in the event log, so the run's projection sees it; nothing here
   * re-reads it.
   */
  runAgent: (
    thread: ThreadRef,
    batch: RawIngress[],
    hooks: HeadlessRunHooks,
  ) => Promise<AgentRunResult>;
  stdin: AsyncIterable<string | Uint8Array>;
  /** The ONLY stdout path. Called with a complete line, newline included. */
  write: (line: string) => void;
  /** Optional human log (stderr). Never stdout. */
  log?: (msg: string) => void;
  /**
   * Ends the session from the outside, exactly like `exit --abort`: stdin is
   * abandoned mid-read, in-flight runs are cancelled, queued runs are reported
   * aborted, and the normal exit path (drain -> `exiting`) still runs. The CLI
   * fires this when the client closes stdout (EPIPE) — the protocol has no
   * reader left, so there is nothing to finish.
   */
  signal?: AbortSignal;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

type ParseResult = { ok: true; cmd: HeadlessCommand } | { ok: false; error: string };

function fail(error: string): ParseResult {
  return { ok: false, error };
}

function optionalString(
  value: unknown,
  field: string,
  fallback: string,
): { ok: true; value: string } | { ok: false; error: string } {
  if (value === undefined) return { ok: true, value: fallback };
  if (typeof value !== "string") return { ok: false, error: `${field} must be a string` };
  return { ok: true, value };
}

/**
 * One stdin line -> one command, with every default already applied, so the
 * session never has to reason about absent fields. Rejections are values, not
 * throws: a malformed line is a normal event in this protocol.
 */
export function parseCommand(line: string): ParseResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (err) {
    return fail(`invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return fail("command must be a JSON object");
  }
  const obj = parsed as Record<string, unknown>;
  if (typeof obj.type !== "string") return fail("command.type must be a string");

  switch (obj.type) {
    case "prompt": {
      if (typeof obj.text !== "string") return fail("prompt.text must be a string");
      if (obj.text.length === 0) return fail("prompt.text must not be empty");

      const threadId = optionalString(obj.threadId, "prompt.threadId", DEFAULT_THREAD_ID);
      if (!threadId.ok) return fail(threadId.error);
      const channelId = optionalString(obj.channelId, "prompt.channelId", DEFAULT_CHANNEL_ID);
      if (!channelId.ok) return fail(channelId.error);
      if (obj.id !== undefined && typeof obj.id !== "string") {
        return fail("prompt.id must be a string");
      }

      // author is the client's identity claim; the platform tag is ours.
      let userId = DEFAULT_USER_ID;
      let displayName: string | undefined;
      if (obj.author !== undefined) {
        if (typeof obj.author !== "object" || obj.author === null || Array.isArray(obj.author)) {
          return fail("prompt.author must be an object");
        }
        const author = obj.author as Record<string, unknown>;
        const uid = optionalString(author.userId, "prompt.author.userId", DEFAULT_USER_ID);
        if (!uid.ok) return fail(uid.error);
        userId = uid.value;
        if (author.displayName !== undefined) {
          if (typeof author.displayName !== "string") {
            return fail("prompt.author.displayName must be a string");
          }
          displayName = author.displayName;
        }
      }

      return {
        ok: true,
        cmd: {
          type: "prompt",
          text: obj.text,
          threadId: threadId.value,
          channelId: channelId.value,
          ...(typeof obj.id === "string" ? { id: obj.id } : {}),
          author: {
            platform: PLATFORM,
            userId,
            ...(displayName === undefined ? {} : { displayName }),
          },
        },
      };
    }
    case "abort": {
      const threadId = optionalString(obj.threadId, "abort.threadId", DEFAULT_THREAD_ID);
      if (!threadId.ok) return fail(threadId.error);
      return { ok: true, cmd: { type: "abort", threadId: threadId.value } };
    }
    case "exit": {
      if (obj.abort !== undefined && typeof obj.abort !== "boolean") {
        return fail("exit.abort must be a boolean");
      }
      return { ok: true, cmd: { type: "exit", abort: obj.abort === true } };
    }
    default:
      return fail(`unknown command type: ${JSON.stringify(obj.type)}`);
  }
}

// ---------------------------------------------------------------------------
// Line framing
// ---------------------------------------------------------------------------

const STOP = Symbol("stop");

/**
 * `source`, stopped short when `signal` fires. The read in progress is
 * abandoned — a pipe that nobody writes to again would never resolve it, and
 * this is the path that ends a session whose client has gone away — and the
 * source's own return() is invoked so the underlying stream is released.
 * One abort listener for the whole loop, removed on the way out.
 */
async function* untilAborted<T>(
  source: AsyncIterable<T>,
  signal: AbortSignal,
): AsyncGenerator<T> {
  const iterator = source[Symbol.asyncIterator]();
  let onAbort: (() => void) | undefined;
  const stopped = new Promise<typeof STOP>((resolve) => {
    if (signal.aborted) {
      resolve(STOP);
      return;
    }
    onAbort = (): void => resolve(STOP);
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    for (;;) {
      const pending = iterator.next();
      // The abandoned read must not surface as an unhandled rejection when the
      // stream is cancelled out from under it.
      void pending.catch(() => {});
      const next = await Promise.race([pending, stopped]);
      if (next === STOP) return;
      if (next.done === true) return;
      yield next.value;
    }
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
    try {
      const returned = iterator.return?.();
      if (returned) void Promise.resolve(returned).catch(() => {});
    } catch {
      // The source is going away regardless; closing it is best-effort.
    }
  }
}

/**
 * Raw chunks -> lines. Chunk boundaries fall wherever the OS pipe put them,
 * including mid-line and mid-UTF-8-sequence, so bytes go through a streaming
 * TextDecoder and the tail is buffered until its newline arrives. A final
 * line with no trailing newline (EOF mid-write) is still yielded.
 *
 * A line over MAX_LINE_BYTES is never yielded: the buffer is released the
 * moment it goes over, `onOverflow` fires ONCE for that line, and the rest of
 * it is discarded up to (and including) the next newline, so the very next
 * command in the stream is still read normally.
 */
async function* toLines(
  stdin: AsyncIterable<string | Uint8Array>,
  onOverflow: () => void,
): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buf = "";
  let bufBytes = 0;
  /** True while the tail of an oversized line is being thrown away. */
  let discarding = false;

  for await (const chunk of stdin) {
    let rest = typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true });
    while (rest.length > 0) {
      const nl = rest.indexOf("\n");
      if (discarding) {
        if (nl < 0) break; // the whole chunk belongs to the dropped line
        discarding = false;
        rest = rest.slice(nl + 1);
        continue;
      }
      if (nl < 0) {
        buf += rest;
        bufBytes += Buffer.byteLength(rest);
        rest = "";
        if (bufBytes > MAX_LINE_BYTES) {
          buf = "";
          bufBytes = 0;
          discarding = true;
          onOverflow();
        }
        break;
      }
      const head = rest.slice(0, nl);
      const complete = bufBytes + Buffer.byteLength(head) <= MAX_LINE_BYTES;
      if (complete) yield buf + head;
      else onOverflow();
      buf = "";
      bufBytes = 0;
      rest = rest.slice(nl + 1);
    }
  }

  buf += decoder.decode(); // flush a dangling multi-byte sequence
  if (discarding || buf.length === 0) return;
  if (Buffer.byteLength(buf) > MAX_LINE_BYTES) onOverflow();
  else yield buf;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/** One conversation's serial executor. */
interface Lane {
  threadId: string;
  /** Tail of the promise chain: the next run awaits it before starting. */
  chain: Promise<void>;
  /** The controller of the run currently executing, or null between runs. */
  inFlight: AbortController | null;
  /**
   * Runs enqueued and not yet settled, the in-flight one included. At zero the
   * lane is idle and is dropped from the map, so a long session does not keep
   * one entry per thread it has ever seen.
   */
  queued: number;
  /**
   * Bumped by `abort`. A run carries the epoch it was queued in; a run whose
   * epoch is behind the lane's was cancelled while parked and never executes.
   * Prompts that arrive AFTER the abort carry the new epoch and run normally.
   */
  epoch: number;
}

/** One queued unit of work, with the lane epoch it was queued in. */
interface QueuedRun {
  thread: ThreadRef;
  ingress: RawIngress;
  epoch: number;
  /** The ingress was already in the log (a duplicate id nothing ever ran). */
  replay: boolean;
}

const laneKey = (channelId: string, threadId: string): string => `${channelId} ${threadId}`;

/**
 * Point a run's controller at the session's, with an explicit dispose. Cheaper
 * and tidier than `AbortSignal.any` here: the session signal outlives every
 * run, and a per-run dependent signal it never releases is a slow leak.
 */
function forwardAbort(
  session: AbortSignal,
  run: AbortController,
): { dispose: () => void } {
  if (session.aborted) {
    run.abort(session.reason);
    return { dispose: () => {} };
  }
  const onAbort = (): void => run.abort(session.reason);
  session.addEventListener("abort", onAbort, { once: true });
  return { dispose: () => session.removeEventListener("abort", onAbort) };
}

/**
 * Run the JSONL session. Resolves after `exit`, stdin EOF (EOF == exit) or
 * `opts.signal`, once every in-flight run has settled and `exiting` has been
 * written.
 */
export async function runHeadless(opts: HeadlessOpts): Promise<void> {
  const emit = (obj: Record<string, unknown>): void => {
    opts.write(`${JSON.stringify(obj)}\n`);
  };
  const log = (msg: string): void => opts.log?.(msg);
  const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err));

  /**
   * The whole session's cancel switch: `exit --abort` and `opts.signal` fire
   * it, every in-flight run forwards from it, and every queued run checks it
   * before starting. Without it, cancellation would only ever reach the runs
   * that happened to be executing at that instant.
   */
  const sessionAbort = new AbortController();
  if (opts.signal) {
    const outer = opts.signal;
    if (outer.aborted) sessionAbort.abort(outer.reason);
    else outer.addEventListener("abort", () => sessionAbort.abort(outer.reason), { once: true });
  }

  emit({
    type: "ready",
    nodeId: opts.nodeId,
    agentId: opts.agentId,
    tenantId: opts.tenantId,
    defaultModel: opts.defaultModel,
  });

  const lanes = new Map<string, Lane>();

  /**
   * One run, start to finish. NEVER REJECTS: the lane chain is built from it,
   * so a rejection here would park every later prompt on this thread forever
   * and make drain() (and therefore `exiting`) fail. A failing agent is an
   * `error` line; a failing `write` is a stderr line and nothing else.
   */
  const runOne = async (lane: Lane, run: QueuedRun): Promise<void> => {
    const { channelId, threadId } = run.thread;
    const at = { threadId, channelId };
    try {
      emit({ type: "run_started", ...at, ...(run.replay ? { replay: true } : {}) });

      if (sessionAbort.signal.aborted || run.epoch < lane.epoch) {
        // Cancelled while parked on the chain. It gets the closing half of its
        // pair anyway so the client's run accounting balances, and the agent
        // is never entered.
        emit({ type: "run_finished", ...at, stopReason: "aborted", turns: 0 });
        return;
      }

      const controller = new AbortController();
      const link = forwardAbort(sessionAbort.signal, controller);
      lane.inFlight = controller;
      try {
        const result = await opts.runAgent(run.thread, [run.ingress], {
          signal: controller.signal,
          onEvent: (event) => emit({ type: "event", ...at, event }),
          deliver: async (text) => {
            emit({ type: "reply", ...at, text });
          },
        });
        emit({
          type: "run_finished",
          ...at,
          stopReason: result.stopReason,
          turns: result.turns,
        });
      } catch (err) {
        // A thrown run has no result to report, so it gets no run_finished:
        // `error` closes it instead, tagged `run: "failed"` so a client can
        // tell a run-closing error from an ingest/duplicate one without
        // matching on the message. The lane stays usable.
        emit({ type: "error", ...at, run: "failed", message: `run failed: ${errText(err)}` });
        log(`run failed on ${channelId}/${threadId}: ${errText(err)}`);
      } finally {
        link.dispose();
        lane.inFlight = null;
      }
    } catch (err) {
      // Only `write` itself can land here — the run could not be reported, so
      // there is nowhere to report that either. stderr, and carry on.
      log(`headless: dropped a protocol line for ${channelId}/${threadId}: ${errText(err)}`);
    }
  };

  const enqueue = (thread: ThreadRef, ingress: RawIngress, replay = false): void => {
    const key = laneKey(thread.channelId, thread.threadId);
    let lane = lanes.get(key);
    if (!lane) {
      lane = {
        threadId: thread.threadId,
        chain: Promise.resolve(),
        inFlight: null,
        queued: 0,
        epoch: 0,
      };
      lanes.set(key, lane);
    }
    const current = lane;
    current.queued++;
    const run: QueuedRun = { thread, ingress, epoch: current.epoch, replay };
    const step = async (): Promise<void> => {
      try {
        await runOne(current, run);
      } finally {
        current.queued--;
        // Idle lanes are dropped; a later prompt on this thread just makes a
        // fresh one (and a fresh epoch, so an old `abort` cannot reach it).
        if (current.queued === 0 && lanes.get(key) === current) lanes.delete(key);
      }
    };
    // Serial per lane, and the chain survives a rejection: `step` is both the
    // fulfil and the reject handler, so one bad run cannot poison the lane.
    current.chain = current.chain.then(step, step).catch(() => {});
  };

  /** Runs cancelled: the one in flight, plus everything parked behind it. */
  const abortThread = (threadId: string): number => {
    let aborted = 0;
    // `abort` addresses a thread, not a (channel, thread) lane — the command
    // carries no channelId — so every lane on that threadId is hit.
    for (const lane of lanes.values()) {
      if (lane.threadId !== threadId) continue;
      aborted += lane.queued;
      lane.epoch++; // everything already queued is cancelled
      lane.inFlight?.abort();
    }
    return aborted;
  };

  /** `exit --abort`: cancel the session — in flight and queued alike. */
  const abortAll = (): void => {
    sessionAbort.abort();
    // Runs already executing forward from the session signal, so nothing else
    // is needed here; queued runs see `sessionAbort.signal.aborted` and skip.
  };

  const handlePrompt = async (cmd: Extract<HeadlessCommand, { type: "prompt" }>): Promise<void> => {
    const thread: ThreadRef = {
      tenantId: opts.tenantId,
      channelId: cmd.channelId,
      threadId: cmd.threadId,
    };
    const externalId = cmd.id ?? `${PLATFORM}:${crypto.randomUUID()}`;
    const at = { threadId: cmd.threadId, channelId: cmd.channelId };

    // Persist before enqueueing (DESIGN.md §11 "persist -> dedup -> enqueue"):
    // the dedup claim and the ingress append are one transaction, so a failure
    // leaves the id unclaimed and the client may retry it verbatim.
    let written: unknown[] | null;
    try {
      written = await opts.events.ingest(thread, externalId, [
        {
          type: "ingress",
          platform: PLATFORM,
          author: cmd.author,
          text: cmd.text,
          refs: [],
          externalId,
        },
      ]);
    } catch (err) {
      emit({ type: "error", ...at, message: `ingest failed: ${errText(err)}` });
      return;
    }

    if (written === null) {
      // Already in the log under this id. Whether that is a client bug or a
      // recovery depends on what this thread is doing RIGHT NOW:
      const lane = lanes.get(laneKey(cmd.channelId, cmd.threadId));
      if (lane && lane.queued > 0) {
        // …a run is in flight or queued for it, so the resend would double the
        // reply. Refused, and the client is told, because it is usually a bug.
        emit({
          type: "error",
          ...at,
          message: `duplicate id ${externalId}; already ingested, run already pending`,
        });
        return;
      }
      // …nothing is running it. The ingress was persisted but never answered
      // (a process that died between ingest and run), and refusing forever
      // would make that unrecoverable, so it runs — flagged `replay` so the
      // client knows this run answers a prompt it already sent.
      enqueue(thread, { text: cmd.text, author: cmd.author, externalId }, true);
      return;
    }

    enqueue(thread, { text: cmd.text, author: cmd.author, externalId });
  };

  /** Every lane's chain as it stands now (no new work is accepted by then). */
  const drain = async (): Promise<void> => {
    await Promise.all([...lanes.values()].map((lane) => lane.chain));
  };

  const onOverflow = (): void => {
    emit({ type: "error", message: `line exceeds ${MAX_LINE_BYTES} bytes; dropped` });
  };

  for await (const rawLine of toLines(untilAborted(opts.stdin, sessionAbort.signal), onOverflow)) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line.trim().length === 0) continue;

    const parsed = parseCommand(line);
    if (!parsed.ok) {
      emit({ type: "error", message: parsed.error, line: line.slice(0, MAX_ECHOED_LINE) });
      continue;
    }

    const cmd = parsed.cmd;
    if (cmd.type === "exit") {
      if (cmd.abort) abortAll();
      break;
    }
    if (cmd.type === "abort") {
      if (abortThread(cmd.threadId) === 0) {
        emit({
          type: "error",
          threadId: cmd.threadId,
          message: `no run in flight or queued for thread ${JSON.stringify(cmd.threadId)}`,
        });
      }
      continue;
    }
    await handlePrompt(cmd);
  }

  // EOF is an exit: finish what is running (an aborted run still settles, and
  // a cancelled queued one reports itself without running), then say so.
  await drain();
  emit({ type: "exiting" });
}
