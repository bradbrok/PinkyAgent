import { describe, expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { ReadTool } from "../src/read";
import { makeCtx, makeTmpDir } from "./helpers";

describe("read", () => {
  const tool = new ReadTool();

  test("whole file with numbered lines", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "one\ntwo\nthree\n");
      const res = await tool.execute({ path: "a.txt" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toBe("1: one\n2: two\n3: three");
    } finally {
      cleanup();
    }
  });

  test("offset and limit slice the range", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "one\ntwo\nthree\nfour\n");
      const res = await tool.execute(
        { path: "a.txt", offset: 2, limit: 2 },
        makeCtx(dir),
      );
      expect(res.text).toBe("2: two\n3: three");
    } finally {
      cleanup();
    }
  });

  test("missing file is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ path: "gone.txt" }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("no such file");
    } finally {
      cleanup();
    }
  });

  test("directory is an error that suggests glob", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ path: "." }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("glob");
    } finally {
      cleanup();
    }
  });

  test("long file notes truncation beyond 2000 lines", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const content = Array.from({ length: 2100 }, (_, i) => `line${i}`).join("\n");
      writeFileSync(path.join(dir, "big.txt"), content);
      const res = await tool.execute({ path: "big.txt" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("(truncated)");
    } finally {
      cleanup();
    }
  });
});
