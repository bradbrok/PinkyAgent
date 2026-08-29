/**
 * Shared SSE (server-sent events) parsing for streaming LLM providers.
 * Exported for tests; providers feed it a Response body and handle events.
 */

export interface SseEvent {
  /** Value of the `event:` line, or null when absent. */
  event: string | null;
  /** Concatenated `data:` lines for the event. */
  data: string;
}

/**
 * Iterate SSE events from a byte stream. Handles CRLF, multi-line `data:`,
 * comment (`:`) lines, and chunks split anywhere in the byte stream.
 */
export async function* iterateSse(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType: string | null = null;
  let dataLines: string[] = [];

  const flush = (): SseEvent | null => {
    if (dataLines.length === 0) {
      eventType = null;
      return null;
    }
    const evt: SseEvent = { event: eventType, data: dataLines.join("\n") };
    eventType = null;
    dataLines = [];
    return evt;
  };

  const pending: SseEvent[] = [];
  const handleLine = (raw: string): void => {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line === "") {
      const evt = flush();
      if (evt) pending.push(evt);
      return;
    }
    if (line.startsWith(":")) return; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventType = value;
    else if (field === "data") dataLines.push(value);
  };

  try {
    for (;;) {
      if (signal?.aborted) {
        throw new DOMException("The operation was aborted.", "AbortError");
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        handleLine(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
      }
      while (pending.length > 0) yield pending.shift()!;
    }
    buffer += decoder.decode();
    if (buffer.length > 0) handleLine(buffer);
    const tail = flush();
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

/** Convenience: build a ReadableStream from a string (used by tests). */
export function sseStreamFromText(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}
