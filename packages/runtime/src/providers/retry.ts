/**
 * Shared retry + per-request timeout policy for the streaming HTTP providers.
 *
 * DESIGN.md §8.1 makes the LLM call an activity boundary: it either produces a
 * recorded result or a loud failure. A single 429/529/5xx or socket blip must
 * not end a conversation, so every provider request runs inside `withRetry`.
 *
 * Semantics:
 *  - Retry on HTTP 408, 409, 429 and any 5xx, plus network-level fetch failures
 *    (TypeError / ECONNRESET-family) and per-attempt timeouts.
 *  - Never retry other 4xx (400/401/403/404/413/422 ...) — those are bugs, not weather.
 *  - Never retry once the response stream has yielded an event: tokens already
 *    reached the caller and a restart would duplicate output.
 *  - Exponential backoff with *full* jitter (`random() * min(cap, base*2^n)`),
 *    overridden by a `retry-after` header (delta-seconds or HTTP-date) when present.
 *  - The caller's AbortSignal wins immediately, in flight or mid-backoff: the
 *    abort reason is rethrown and no further attempt is made.
 *  - Each attempt gets its own `timeoutMs` budget covering connect *and* the whole
 *    streamed body; the attempt is raced against the deadline so a provider (or a
 *    test double) that ignores signals still fails loudly instead of hanging.
 */

/** Injectable for tests; the default honors the abort signal. */
export type SleepFn = (ms: number, signal?: AbortSignal) => Promise<void>;

export const DEFAULT_MAX_RETRIES = 3;
/** Whole-request budget per attempt, including streaming. */
export const DEFAULT_TIMEOUT_MS = 120_000;
export const DEFAULT_BASE_DELAY_MS = 500;
export const DEFAULT_MAX_DELAY_MS = 8_000;
/** Upper bound honored for a server-sent `retry-after`. */
export const MAX_RETRY_AFTER_MS = 60_000;
/** 4xx codes that are worth retrying (timeout / conflict / rate limit). */
export const RETRYABLE_STATUS = new Set([408, 409, 429]);
/** Error bodies are truncated to this many characters in messages. */
export const ERROR_BODY_CHARS = 500;

export class HttpStatusError extends Error {
  readonly status: number;
  readonly body: string;
  readonly retryAfterMs: number | undefined;

  constructor(
    message: string,
    init: { status: number; body: string; retryAfterMs?: number | undefined },
  ) {
    super(message);
    this.name = "HttpStatusError";
    this.status = init.status;
    this.body = init.body;
    this.retryAfterMs = init.retryAfterMs;
  }
}

export class RequestTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(label: string, timeoutMs: number) {
    super(`${label} request timed out after ${timeoutMs}ms`);
    this.name = "RequestTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

/** Throw the provider's standard error for a non-2xx response (body truncated). */
export async function assertOk(res: Response, label: string): Promise<void> {
  if (res.ok) return;
  const body = await res.text().catch(() => "");
  throw new HttpStatusError(`${label} API error ${res.status}: ${body.slice(0, ERROR_BODY_CHARS)}`, {
    status: res.status,
    body,
    retryAfterMs: parseRetryAfter(res.headers.get("retry-after")),
  });
}

/** `retry-after` as delta-seconds or HTTP-date -> milliseconds (undefined if unusable). */
export function parseRetryAfter(value: string | null | undefined, now = Date.now()): number | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  if (/^\d+(\.\d+)?$/.test(trimmed)) {
    const seconds = Number(trimmed);
    return Number.isFinite(seconds) ? Math.max(0, Math.round(seconds * 1000)) : undefined;
  }
  const at = Date.parse(trimmed);
  if (Number.isNaN(at)) return undefined;
  return Math.max(0, at - now);
}

export function isRetryableStatus(status: number): boolean {
  return RETRYABLE_STATUS.has(status) || status >= 500;
}

const NETWORK_ERROR_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ECONNABORTED",
  "EPIPE",
  "ETIMEDOUT",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "ENETDOWN",
  "EAI_AGAIN",
  "ENOTFOUND",
  "ConnectionClosed",
  "ConnectionRefused",
  "ConnectionReset",
  "FailedToOpenSocket",
]);

/** fetch network failures: WHATWG throws TypeError; Bun/Node attach a `code`. */
export function isNetworkError(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  if (typeof err !== "object" || err === null) return false;
  const e = err as { code?: unknown; name?: unknown; message?: unknown };
  if (typeof e.code === "string" && NETWORK_ERROR_CODES.has(e.code)) return true;
  if (typeof e.name === "string" && NETWORK_ERROR_CODES.has(e.name)) return true;
  const message = typeof e.message === "string" ? e.message : "";
  return /econnreset|socket hang up|fetch failed|network error|connection (closed|reset|refused)/i.test(
    message,
  );
}

export function isRetryableError(err: unknown): boolean {
  if (err instanceof HttpStatusError) return isRetryableStatus(err.status);
  if (err instanceof RequestTimeoutError) return true;
  if (isAbortError(err)) return false; // caller-driven; handled before we get here
  return isNetworkError(err);
}

export function isAbortError(err: unknown): boolean {
  return typeof err === "object" && err !== null && (err as { name?: unknown }).name === "AbortError";
}

function abortReason(signal: AbortSignal): unknown {
  const reason: unknown = signal.reason;
  if (reason !== undefined && reason !== null) return reason;
  return new DOMException("The operation was aborted.", "AbortError");
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortReason(signal);
}

/** `AbortSignal.any` when the runtime has it (Bun 1.4 does), else manual fan-in. */
export function combineSignals(caller: AbortSignal | undefined, deadline: AbortSignal): AbortSignal {
  if (!caller) return deadline;
  const any = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any;
  if (typeof any === "function") return any.call(AbortSignal, [caller, deadline]);
  const controller = new AbortController();
  for (const signal of [caller, deadline]) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
  }
  return controller.signal;
}

/** A promise that rejects with the signal's reason when it aborts. */
function rejectOnAbort(signal: AbortSignal | undefined): {
  promise: Promise<never>;
  dispose: () => void;
} {
  if (!signal) return { promise: new Promise<never>(() => {}), dispose: () => {} };
  let listener: (() => void) | undefined;
  const promise = new Promise<never>((_resolve, reject) => {
    if (signal.aborted) {
      reject(abortReason(signal));
      return;
    }
    listener = () => reject(abortReason(signal));
    signal.addEventListener("abort", listener, { once: true });
  });
  // Nobody may end up awaiting this race arm; keep it from surfacing as an
  // unhandled rejection without changing what the race observes.
  void promise.catch(() => {});
  return {
    promise,
    dispose: () => {
      if (listener) signal.removeEventListener("abort", listener);
    },
  };
}

/** Default sleep: a timer that unwinds early when the signal aborts. */
export const realSleep: SleepFn = (ms, signal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortReason(signal));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(abortReason(signal!));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });

export interface RetryConfig {
  /** Provider label used in timeout/error messages ("Anthropic", "OpenAI"). */
  label: string;
  maxRetries: number;
  timeoutMs: number;
  sleep: SleepFn;
  baseDelayMs?: number | undefined;
  maxDelayMs?: number | undefined;
  /** Jitter source; injectable so tests can pin the backoff. */
  random?: (() => number) | undefined;
}

export interface AttemptContext {
  /** Caller signal combined with this attempt's deadline — pass to fetch and the SSE reader. */
  signal: AbortSignal;
  /** Call once the response stream yields its first event; disables further retries. */
  markStreamProgress: () => void;
}

/** Backoff for one failure: honor `retry-after`, else full jitter. */
export function backoffDelayMs(
  err: unknown,
  attemptIndex: number,
  cfg: Pick<RetryConfig, "baseDelayMs" | "maxDelayMs" | "random">,
): number {
  if (err instanceof HttpStatusError && err.retryAfterMs !== undefined) {
    return Math.min(err.retryAfterMs, MAX_RETRY_AFTER_MS);
  }
  const base = cfg.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
  const cap = cfg.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
  const random = cfg.random ?? Math.random;
  const window = Math.min(cap, base * 2 ** attemptIndex);
  return Math.floor(random() * window);
}

async function sleepOrAbort(sleep: SleepFn, ms: number, signal: AbortSignal | undefined): Promise<void> {
  throwIfAborted(signal);
  if (ms <= 0) return;
  const guard = rejectOnAbort(signal);
  try {
    const pending = sleep(ms, signal);
    void pending.catch(() => {});
    await Promise.race([pending, guard.promise]);
  } finally {
    guard.dispose();
  }
}

/**
 * Run one HTTP attempt under a deadline, retrying transient failures.
 * `attempt` owns the fetch, the status check and the stream consumption.
 */
export async function withRetry<T>(
  cfg: RetryConfig,
  callerSignal: AbortSignal | undefined,
  attempt: (ctx: AttemptContext) => Promise<T>,
): Promise<T> {
  const maxRetries = Math.max(0, cfg.maxRetries);

  for (let attemptIndex = 0; ; attemptIndex++) {
    throwIfAborted(callerSignal);
    const deadline = AbortSignal.timeout(cfg.timeoutMs);
    const signal = combineSignals(callerSignal, deadline);
    let progressed = false;
    const guard = rejectOnAbort(signal);
    try {
      const running = attempt({ signal, markStreamProgress: () => (progressed = true) });
      void running.catch(() => {});
      return await Promise.race([running, guard.promise]);
    } catch (raw) {
      // The caller's abort always wins, and never retries.
      if (callerSignal?.aborted) throw abortReason(callerSignal);
      const err =
        deadline.aborted && !(raw instanceof HttpStatusError)
          ? new RequestTimeoutError(cfg.label, cfg.timeoutMs)
          : raw;
      if (progressed || attemptIndex >= maxRetries || !isRetryableError(err)) throw err;
      await sleepOrAbort(cfg.sleep, backoffDelayMs(err, attemptIndex, cfg), callerSignal);
    } finally {
      guard.dispose();
    }
  }
}
