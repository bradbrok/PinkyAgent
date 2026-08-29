import { describe, expect, test } from "bun:test";
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { GlobTool } from "../src/glob";
import { makeCtx, makeTmpDir } from "./helpers";

describe("glob", () => {
  const tool = new GlobTool();

  test("matches files by pattern", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.ts"), "");
      writeFileSync(path.join(dir, "b.js"), "");
      const res = await tool.execute({ pattern: "*.ts" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toBe("a.ts");
    } finally {
      cleanup();
    }
  });

  test("matches are sorted", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "z.txt"), "");
      writeFileSync(path.join(dir, "a.txt"), "");
      const res = await tool.execute({ pattern: "*.txt" }, makeCtx(dir));
      expect(res.text).toBe("a.txt\nz.txt");
    } finally {
      cleanup();
    }
  });

  test("path argument scopes the root", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const sub = path.join(dir, "sub");
      mkdirSync(sub);
      writeFileSync(path.join(sub, "inside.txt"), "");
      const res = await tool.execute({ pattern: "*.txt", path: "sub" }, makeCtx(dir));
      expect(res.text).toBe("inside.txt");
    } finally {
      cleanup();
    }
  });

  test("more than 500 matches are capped with a note", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      for (let i = 0; i < 600; i++) {
        writeFileSync(path.join(dir, `f${i}.txt`), "");
      }
      const res = await tool.execute({ pattern: "*.txt" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("capped at 500");
    } finally {
      cleanup();
    }
  });

  test("no matches returns a notice", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ pattern: "*.nope" }, makeCtx(dir));
      expect(res.text).toBe("(no matches)");
    } finally {
      cleanup();
    }
  });
});
