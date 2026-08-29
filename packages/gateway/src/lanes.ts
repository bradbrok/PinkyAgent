/**
 * Per-conversation serial executor with debounce (DESIGN.md §6: "debounce
 * ~500ms same-author burst → one batched turn").
 *
 * - Items for the same key coalesce into one batch during the debounce window.
 * - Batches for one key run strictly serially: the next batch never starts
 *   while the previous handler invocation is running.
 * - Different keys run in parallel.
 */

type TimerHandle = ReturnType<typeof setTimeout>;

export interface LaneQueueOpts<T> {
  debounceMs?: number;
  /** Handler errors are reported here if given, else logged and swallowed. */
  onError?: (error: unknown, key: string, batch: T[]) => void;
}

export class LaneQueue<T = unknown> {
  private readonly handler: (key: string, batch: T[]) => Promise<void>;
  private readonly debounceMs: number;
  private readonly onError: (error: unknown, key: string, batch: T[]) => void;
  private readonly queues = new Map<string, T[]>();
  private readonly timers = new Map<string, TimerHandle>();
  private readonly running = new Set<string>();
  private queuedCount = 0;

  constructor(handler: (key: string, batch: T[]) => Promise<void>, opts: LaneQueueOpts<T> = {}) {
    this.handler = handler;
    this.debounceMs = opts.debounceMs ?? 500;
    this.onError =
      opts.onError ??
      ((error, key) => {
        console.error(`LaneQueue handler failed for key ${key}:`, error);
      });
  }

  /** Items enqueued but not yet handed to the handler (test introspection). */
  get pending(): number {
    return this.queuedCount;
  }

  enqueue(key: string, item: T): void {
    const q = this.queues.get(key) ?? [];
    q.push(item);
    this.queues.set(key, q);
    this.queuedCount++;

    clearTimeout(this.timers.get(key));
    this.timers.set(key, setTimeout(() => this.flush(key), this.debounceMs));
  }

  private flush(key: string): void {
    this.timers.delete(key);
    // Serial per key: if a previous batch is still running, leave items queued;
    // the completion path re-flushes when the handler settles.
    if (this.running.has(key)) return;

    const batch = this.queues.get(key);
    if (!batch || batch.length === 0) return;
    this.queues.delete(key);
    this.queuedCount -= batch.length;
    this.running.add(key);

    void this.handler(key, batch)
      .catch((error) => this.onError(error, key, batch))
      .finally(() => {
        this.running.delete(key);
        // Items enqueued during the run are debounce-ready: flush immediately.
        if ((this.queues.get(key)?.length ?? 0) > 0) this.flush(key);
      });
  }
}
