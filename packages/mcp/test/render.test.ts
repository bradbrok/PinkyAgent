/**
 * CallToolResult -> ToolResult. The renderer is the only place in the runtime
 * that knows the MCP content vocabulary, and its output is what gets journaled
 * as a `tool_result` event, so "what does the model actually see" is decided
 * here and nowhere else.
 */
import { describe, expect, it } from "bun:test";
import {
  EMPTY_RESULT_TEXT,
  INPUT_REQUIRED_TEXT,
  MAX_MCP_RESULT_BYTES,
  base64ByteLength,
  renderCallToolResult,
  renderContentBlock,
  truncateResultText,
} from "../src/render";

const text = (t: string) => ({ type: "text", text: t });

describe("base64ByteLength", () => {
  it("counts the decoded bytes without decoding", () => {
    expect(base64ByteLength(Buffer.from("hello").toString("base64"))).toBe(5);
    expect(base64ByteLength(Buffer.from("hi").toString("base64"))).toBe(2);
    expect(base64ByteLength(Buffer.alloc(300).toString("base64"))).toBe(300);
    expect(base64ByteLength("")).toBe(0);
  });
});

describe("renderContentBlock", () => {
  it("renders text verbatim", () => {
    expect(renderContentBlock(text("hello"))).toBe("hello");
  });

  it("replaces image and audio with a sized placeholder, never the bytes", () => {
    const data = Buffer.alloc(1024).toString("base64");
    expect(renderContentBlock({ type: "image", mimeType: "image/png", data })).toBe(
      "[image image/png, 1024 bytes]",
    );
    expect(renderContentBlock({ type: "audio", mimeType: "audio/wav", data })).toBe(
      "[audio audio/wav, 1024 bytes]",
    );
    // The base64 must not appear anywhere in the rendered text.
    expect(renderContentBlock({ type: "image", mimeType: "image/png", data })).not.toContain(
      data.slice(0, 20),
    );
  });

  it("renders an embedded TEXT resource as its text", () => {
    expect(
      renderContentBlock({
        type: "resource",
        resource: { uri: "file:///a.txt", mimeType: "text/plain", text: "contents" },
      }),
    ).toBe("contents");
  });

  it("renders an embedded BLOB resource as a locator line", () => {
    expect(
      renderContentBlock({
        type: "resource",
        resource: { uri: "file:///a.bin", mimeType: "application/octet-stream", blob: Buffer.alloc(9).toString("base64") },
      }),
    ).toBe("[resource file:///a.bin, application/octet-stream, 9 bytes]");
  });

  it("renders a resource link as a locator line", () => {
    expect(renderContentBlock({ type: "resource_link", name: "readme", uri: "file:///README" })).toBe(
      "[resource_link readme file:///README]",
    );
  });

  it("names an unknown block instead of silently dropping it", () => {
    expect(renderContentBlock({ type: "tool_use", name: "x" })).toBe(
      '[unsupported content block "tool_use"]',
    );
    expect(renderContentBlock(null)).toBe("[unrenderable content block]");
  });
});

describe("renderCallToolResult", () => {
  it("joins text blocks with newlines", () => {
    expect(renderCallToolResult({ content: [text("one"), text("two")] })).toEqual({
      text: "one\ntwo",
    });
  });

  it("appends structuredContent as a fenced JSON block", () => {
    const result = renderCallToolResult({
      content: [text("stats follow")],
      structuredContent: { bmi: 22.86 },
    });
    expect(result.text).toBe('stats follow\n```json\n{\n  "bmi": 22.86\n}\n```');
    expect(result.isError).toBeUndefined();
  });

  it("treats structuredContent presence as !== undefined (SEP-2106: any JSON)", () => {
    expect(renderCallToolResult({ content: [], structuredContent: null }).text).toBe(
      "```json\nnull\n```",
    );
    expect(renderCallToolResult({ content: [], structuredContent: 0 }).text).toBe("```json\n0\n```");
  });

  it("passes isError through", () => {
    expect(renderCallToolResult({ content: [text("boom")], isError: true })).toEqual({
      text: "boom",
      isError: true,
    });
  });

  it("treats a MISSING resultType as complete (every 2025-era server)", () => {
    expect(renderCallToolResult({ content: [text("fine")] }).isError).toBeUndefined();
    expect(renderCallToolResult({ content: [text("fine")], resultType: "complete" }).isError).toBeUndefined();
  });

  it("turns an MRTR input_required result into a clean tool error", () => {
    // We implement no elicitation / sampling / roots (all deprecated in
    // 2026-07-28), so the only honest answer is "unsupported here".
    expect(
      renderCallToolResult({ resultType: "input_required", inputRequests: { confirm: {} } }),
    ).toEqual({ text: INPUT_REQUIRED_TEXT, isError: true });
  });

  it("says something when the server says nothing", () => {
    expect(renderCallToolResult({ content: [] })).toEqual({ text: EMPTY_RESULT_TEXT });
    expect(renderCallToolResult({})).toEqual({ text: EMPTY_RESULT_TEXT });
  });

  it("never throws on a malformed result", () => {
    expect(renderCallToolResult(null).isError).toBe(true);
    expect(renderCallToolResult({ content: "not an array" }).text).toBe(EMPTY_RESULT_TEXT);
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(renderCallToolResult({ content: [], structuredContent: cyclic }).text).toContain(
      "unserializable",
    );
  });
});

describe("truncation", () => {
  it("caps an oversized result and names the original size", () => {
    // The loop journals a tool result into the event log AND renders it into
    // the next prompt. An unbounded result is an unbounded prompt: past the
    // window the loop forces a shed, and the shed's own request carries the
    // same oversized message — unsendable, so the thread wedges (DESIGN §4).
    const huge = "z".repeat(MAX_MCP_RESULT_BYTES * 3);
    const result = renderCallToolResult({ content: [{ type: "text", text: huge }] });
    expect(Buffer.byteLength(result.text, "utf8")).toBeLessThan(MAX_MCP_RESULT_BYTES + 100);
    expect(result.text).toContain(`[truncated: ${MAX_MCP_RESULT_BYTES * 3} bytes of tool output`);
    expect(result.isError).toBeUndefined();
  });

  it("budgets the WHOLE result, structuredContent block included", () => {
    const result = renderCallToolResult({
      content: [{ type: "text", text: "a".repeat(MAX_MCP_RESULT_BYTES - 10) }],
      structuredContent: { blob: "b".repeat(MAX_MCP_RESULT_BYTES) },
    });
    expect(Buffer.byteLength(result.text, "utf8")).toBeLessThan(MAX_MCP_RESULT_BYTES + 100);
    expect(result.text).toContain("[truncated:");
  });

  it("keeps isError while truncating", () => {
    const result = renderCallToolResult({
      content: [{ type: "text", text: "e".repeat(MAX_MCP_RESULT_BYTES * 2) }],
      isError: true,
    });
    expect(result.isError).toBe(true);
    expect(result.text).toContain("[truncated:");
  });

  it("leaves anything under the cap byte-identical", () => {
    expect(truncateResultText("short")).toBe("short");
    const exact = "x".repeat(MAX_MCP_RESULT_BYTES);
    expect(truncateResultText(exact)).toBe(exact);
  });

  it("cuts on a character boundary, never mid-code-point", () => {
    // Each emoji is 4 UTF-8 bytes; a byte-slice at an odd offset would leave a
    // lone surrogate and produce mojibake in the event log.
    const text = "😀".repeat(100);
    const out = truncateResultText(text, 50);
    const kept = out.slice(0, out.indexOf("\n[truncated"));
    expect(Buffer.byteLength(kept, "utf8")).toBeLessThanOrEqual(50);
    expect(kept).toBe("😀".repeat(12));
    expect([...kept].every((c) => c === "😀")).toBe(true);
  });
});
