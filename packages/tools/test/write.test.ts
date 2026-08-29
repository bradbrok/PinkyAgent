import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { WriteTool } from "../src/write";
import { makeCtx, makeTmpDir } from "./helpers";

describe("write", () => {
  const tool = new WriteTool();

  test("creates parent directories and reports bytes", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { path: "nested/deep/file.txt", content: "hello" },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("5 bytes");
      expect(readFileSync(path.join(dir, "nested/deep/file.txt"), "utf8")).toBe("hello");
    } finally {
      cleanup();
    }
  });

  test("non-string content is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { path: "f.txt", content: 42 },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
    } finally {
      cleanup();
    }
  });
});
