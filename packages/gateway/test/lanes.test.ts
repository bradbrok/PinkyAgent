import { afterEach, describe, expect, test, vi } from "bun:test";
import { LaneQueue } from "../src/lanes";

interface Batch<T> {
  key: string;
  items: T[];
}

/** Microtask drain: lets a settled promise's .finally chain run. */
async function flushAsync(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

afterEach(() => {
  vi.useRealTimers();
});

describe("LaneQueue", () => {
  test("debounce coalesces a burst into one batch", async () => {
    vi.useFakeTimers();
    const batches: Batch<string>[] = [];
    const lane = new LaneQueue<string>(async (key, batch) => {
      batches.push({ key, items: batch });
    }, { debounceMs: 30 });

    lane.enqueue("conv1", "a");
    lane.enqueue("conv1", "b");
    lane.enqueue("conv1", "c");
    expect(lane.pending).toBe(3);

    vi.advanceTimersByTime(30);
    await flushAsync();
    expect(batches).toEqual([{ key: "conv1", items: ["a", "b", "c"] }]);
    expect(lane.pending).toBe(0);
  });

  test("per-key serial: second batch waits for the first handler", async () => {
    vi.useFakeTimers();
    const gate = Promise.withResolvers<void>();
    const order: string[] = [];
    const lane = new LaneQueue<string>(async (key, batch) => {
      order.push(`start:${key}:${batch.join(",")}`);
      if (batch[0] === "a") await gate.promise;
      order.push(`end:${key}:${batch.join(",")}`);
    }, { debounceMs: 20 });

    lane.enqueue("k", "a");
    vi.advanceTimersByTime(20); // first batch running, blocked
    await flushAsync();
    lane.enqueue("k", "b"); // queued while running → completion-flush picks it
    vi.advanceTimersByTime(20);
    await flushAsync();
    expect(order).toEqual(["start:k:a"]);

    gate.resolve();
    await flushAsync();
    vi.advanceTimersByTime(1);
    await flushAsync();
    expect(order).toEqual(["start:k:a", "end:k:a", "start:k:b", "end:k:b"]);
  });

  test("different keys run in parallel", async () => {
    vi.useFakeTimers();
    const gate = Promise.withResolvers<void>();
    const started: string[] = [];
    const lane = new LaneQueue<string>(async (key, batch) => {
      started.push(key);
      if (key === "slow") await gate.promise;
    }, { debounceMs: 20 });

    lane.enqueue("slow", "1");
    lane.enqueue("fast", "1");
    vi.advanceTimersByTime(20);
    await flushAsync();
    // Both started even though "slow" is still blocked.
    expect(started).toContain("slow");
    expect(started).toContain("fast");
    gate.resolve();
    await flushAsync();
  });

  test("an item enqueued during the run starts as its own batch", async () => {
    vi.useFakeTimers();
    const batches: Batch<string>[] = [];
    const gate = Promise.withResolvers<void>();
    let blocked = true;
    const lane = new LaneQueue<string>(async (key, batch) => {
      batches.push({ key, items: batch });
      if (blocked) await gate.promise;
    }, { debounceMs: 20 });

    lane.enqueue("k", "a");
    vi.advanceTimersByTime(20);
    await flushAsync(); // first batch blocked in handler
    lane.enqueue("k", "b");
    blocked = false;
    gate.resolve();
    await flushAsync();
    vi.advanceTimersByTime(1);
    await flushAsync();
    expect(batches.map((b) => b.items)).toEqual([["a"], ["b"]]);
  });

  test("handler errors reach onError and do not wedge the lane", async () => {
    vi.useFakeTimers();
    const errors: string[] = [];
    let calls = 0;
    const lane = new LaneQueue<string>(async () => {
      calls++;
      if (calls === 1) throw new Error("boom");
    }, {
      debounceMs: 20,
      onError: (e) => {
        errors.push(e instanceof Error ? e.message : String(e));
      },
    });

    lane.enqueue("k", "a");
    vi.advanceTimersByTime(20);
    await flushAsync();
    expect(errors).toEqual(["boom"]);

    lane.enqueue("k", "b"); // lane still usable after the failure
    vi.advanceTimersByTime(20);
    await flushAsync();
    expect(calls).toBe(2);
  });
});
