import { describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { EditTool } from "../src/edit";
import { makeCtx, makeTmpDir } from "./helpers";

describe("edit", () => {
  const tool = new EditTool();

  test("single occurrence is replaced", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "alpha beta gamma");
      const res = await tool.execute(
        { path: "a.txt", old: "beta", new: "BE" },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(readFileSync(path.join(dir, "a.txt"), "utf8")).toBe("alpha BE gamma");
      expect(res.text).toContain("-2 bytes");
    } finally {
      cleanup();
    }
  });

  test("ambiguous replacement reports occurrence count", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "x x x");
      const res = await tool.execute(
        { path: "a.txt", old: "x", new: "y" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("3 occurrences");
      expect(readFileSync(path.join(dir, "a.txt"), "utf8")).toBe("x x x");
    } finally {
      cleanup();
    }
  });

  test("missing needle is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "hello");
      const res = await tool.execute(
        { path: "a.txt", old: "nope", new: "yep" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("not found");
    } finally {
      cleanup();
    }
  });

  test("missing file is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { path: "gone.txt", old: "a", new: "b" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("no such file");
    } finally {
      cleanup();
    }
  });

  test("replacement containing $ patterns is literal", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "before");
      const res = await tool.execute(
        { path: "a.txt", old: "before", new: "$`$&after" },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(readFileSync(path.join(dir, "a.txt"), "utf8")).toBe("$`$&after");
    } finally {
      cleanup();
    }
  });
});
