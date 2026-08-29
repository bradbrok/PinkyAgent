import { describe, expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { GrepTool } from "../src/grep";
import { makeCtx, makeTmpDir } from "./helpers";

describe("grep", () => {
  const tool = new GrepTool();

  test("finds matching lines as file:line: text", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "a.txt"), "alpha\nbeta\ngamma\n");
      const res = await tool.execute({ pattern: "beta" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toBe("a.txt:2: beta");
    } finally {
      cleanup();
    }
  });

  test("glob filter restricts which files are searched", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "keep.txt"), "needle");
      writeFileSync(path.join(dir, "skip.log"), "needle");
      const res = await tool.execute(
        { pattern: "needle", glob: "*.txt" },
        makeCtx(dir),
      );
      expect(res.text).toBe("keep.txt:1: needle");
    } finally {
      cleanup();
    }
  });

  test("searches recursively into subdirectories", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      mkdirSync(path.join(dir, "sub"));
      writeFileSync(path.join(dir, "sub", "deep.txt"), "needle here");
      const res = await tool.execute({ pattern: "needle" }, makeCtx(dir));
      expect(res.text).toBe(`sub/deep.txt:1: needle here`);
    } finally {
      cleanup();
    }
  });

  test("binary files are skipped", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      writeFileSync(path.join(dir, "bin.dat"), Buffer.concat([
        Buffer.from("needle"),
        Buffer.from([0, 1, 2, 0]),
      ]));
      const res = await tool.execute({ pattern: "needle" }, makeCtx(dir));
      expect(res.text).toBe("(no matches)");
    } finally {
      cleanup();
    }
  });

  test("limit caps the number of matches", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const content = Array.from({ length: 20 }, (_, i) => `hit${i}`).join("\n");
      writeFileSync(path.join(dir, "a.txt"), content);
      const res = await tool.execute({ pattern: "hit", limit: 5 }, makeCtx(dir));
      expect(res.text.split("\n").length).toBe(5);
    } finally {
      cleanup();
    }
  });

  test("invalid regex is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ pattern: "([unclosed" }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("invalid pattern");
    } finally {
      cleanup();
    }
  });
});
