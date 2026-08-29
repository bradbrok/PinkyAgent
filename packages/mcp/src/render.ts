/**
 * `CallToolResult` (MCP wire) -> `ToolResult` (what the agent loop journals as
 * a `tool_result` event and the projection renders into the prompt).
 *
 * The whole job is lossy on purpose. A tool result is TEXT in this runtime:
 * the loop has no multimodal path, the event log stores what the model saw,
 * and a base64 image inlined into a `tool_result` would blow a context window
 * that DESIGN §4 spends a whole subsystem defending. So binary blocks become
 * one-line placeholders that say what was there and how big it was — enough
 * for the model to decide what to do next, none of the bytes.
 *
 * Three protocol details this file is the only place that knows about:
 *
 *  - **`resultType` (MRTR, 2026-07-28).** A result may be `input_required`
 *    instead of `complete`: the server wants elicitation / sampling / roots
 *    from the client mid-call. We implement none of those (all three are
 *    deprecated, and none has a use here), so that is a clean tool error, not
 *    a crash. A result with NO `resultType` is complete — that is every
 *    2025-era server.
 *  - **`structuredContent` is any JSON** (SEP-2106), including `null`, a
 *    number or an array. Presence is `!== undefined`, never truthiness.
 *  - **`isError` is passthrough.** A tool that ran and reported a problem is
 *    not a harness failure; the model gets the text and decides.
 */
import type { ToolResult } from "@pinky/runtime";

/** What `renderCallToolResult` accepts: the SDK's `CallToolResult`, plus the
 *  wire fields the SDK's TS type only exposes through its index signature. */
export interface RenderableCallToolResult {
  content?: unknown;
  structuredContent?: unknown;
  isError?: boolean | undefined;
  resultType?: unknown;
  [key: string]: unknown;
}

/** Returned for an `input_required` result — we implement no elicitation,
 *  sampling or roots (all deprecated in 2026-07-28, all unused here). */
export const INPUT_REQUIRED_TEXT =
  "server requires additional input (elicitation) — unsupported by this harness";

/** A result with no content and no structured payload still has to say something. */
export const EMPTY_RESULT_TEXT = "(tool returned no content)";

/**
 * Ceiling on a rendered result, matching `bash`'s 50 KB (tools/src/bash.ts).
 *
 * Not a tidiness rule. A tool result is journaled into the event log AND
 * rendered into the next prompt, so an unbounded one is an unbounded prompt:
 * past the context window the loop escalates to a forced shed, and the shed's
 * own `complete()` call carries the same oversized message — unsendable, so
 * the thread wedges with no way out (DESIGN.md §4). An MCP server is a third
 * party we do not control; the cap is the boundary.
 */
export const MAX_MCP_RESULT_BYTES = 50 * 1024;

/** UTF-8 bytes, not UTF-16 units — the cap is about what goes on the wire. */
function utf8Length(text: string): number {
  return Buffer.byteLength(text, "utf8");
}

/**
 * Cut `text` to {@link MAX_MCP_RESULT_BYTES} on a CHARACTER boundary and say
 * how big it really was, so the model can decide to ask for less rather than
 * wonder whether the tool half-worked.
 */
export function truncateResultText(text: string, max: number = MAX_MCP_RESULT_BYTES): string {
  const bytes = utf8Length(text);
  if (bytes <= max) return text;
  // Byte-slicing UTF-8 can split a code point; walk back from an over-long
  // character slice instead, which is cheap and always lands on a boundary.
  let end = Math.min(text.length, max);
  while (end > 0 && utf8Length(text.slice(0, end)) > max) end -= 1;
  return `${text.slice(0, end)}\n[truncated: ${bytes} bytes of tool output, ${max} kept]`;
}

/**
 * Bytes behind a base64 payload, without decoding it. `length/4*3` minus the
 * padding: an exact count for well-formed base64 and a sane estimate for the
 * rest, at zero allocation — the point of the placeholder is to avoid
 * materializing the blob at all.
 */
export function base64ByteLength(data: string): number {
  const clean = data.replace(/[^A-Za-z0-9+/=]/g, "");
  if (clean === "") return 0;
  const padding = clean.endsWith("==") ? 2 : clean.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((clean.length * 3) / 4) - padding);
}

/** `[image image/png, 1234 bytes]` — the shape of every binary placeholder. */
function mediaPlaceholder(kind: string, mimeType: unknown, data: unknown): string {
  const mime = typeof mimeType === "string" && mimeType !== "" ? mimeType : "application/octet-stream";
  const bytes = typeof data === "string" ? base64ByteLength(data) : 0;
  return `[${kind} ${mime}, ${bytes} bytes]`;
}

/** One content block -> its text form. Unknown block types are named, not
 *  dropped: a silently swallowed block is a debugging dead end. */
export function renderContentBlock(block: unknown): string {
  if (typeof block !== "object" || block === null) return "[unrenderable content block]";
  const b = block as Record<string, unknown>;
  switch (b.type) {
    case "text":
      return typeof b.text === "string" ? b.text : "";
    case "image":
      return mediaPlaceholder("image", b.mimeType, b.data);
    case "audio":
      return mediaPlaceholder("audio", b.mimeType, b.data);
    case "resource_link": {
      const uri = typeof b.uri === "string" ? b.uri : "?";
      const name = typeof b.name === "string" && b.name !== "" ? ` ${b.name}` : "";
      return `[resource_link${name} ${uri}]`;
    }
    case "resource": {
      const resource = (typeof b.resource === "object" && b.resource !== null
        ? b.resource
        : {}) as Record<string, unknown>;
      // A text resource IS its text — that is the whole point of embedding it.
      if (typeof resource.text === "string") return resource.text;
      const uri = typeof resource.uri === "string" ? resource.uri : "?";
      const mime = typeof resource.mimeType === "string" ? `, ${resource.mimeType}` : "";
      const bytes = typeof resource.blob === "string" ? `, ${base64ByteLength(resource.blob)} bytes` : "";
      return `[resource ${uri}${mime}${bytes}]`;
    }
    default:
      return `[unsupported content block ${JSON.stringify(b.type ?? null)}]`;
  }
}

/** `structuredContent` as a fenced JSON block, so the model can see it is
 *  machine-readable and quote it back. Unserializable payloads (a cycle) are
 *  reported rather than thrown — a render must never fail a journaled turn. */
function renderStructured(value: unknown): string {
  let json: string;
  try {
    json = JSON.stringify(value, null, 2) ?? "null";
  } catch {
    json = '"<unserializable structuredContent>"';
  }
  return `\`\`\`json\n${json}\n\`\`\``;
}

/**
 * The renderer. Never throws: a malformed result becomes an `isError` string,
 * because the caller is inside a tool call whose only two outcomes are "text"
 * and "text flagged as an error".
 */
export function renderCallToolResult(result: RenderableCallToolResult | null | undefined): ToolResult {
  if (result === null || result === undefined) {
    return { text: "mcp server returned no result", isError: true };
  }
  // Missing resultType means complete (every 2025-era server).
  if (result.resultType === "input_required") {
    return { text: INPUT_REQUIRED_TEXT, isError: true };
  }

  const parts: string[] = [];
  const content = Array.isArray(result.content) ? result.content : [];
  for (const block of content) {
    const rendered = renderContentBlock(block);
    if (rendered !== "") parts.push(rendered);
  }
  if (result.structuredContent !== undefined) parts.push(renderStructured(result.structuredContent));

  // Truncate the JOINED text, structuredContent block included: the budget is
  // the whole result, not per block.
  const text = truncateResultText(parts.length > 0 ? parts.join("\n") : EMPTY_RESULT_TEXT);
  return result.isError === true ? { text, isError: true } : { text };
}
