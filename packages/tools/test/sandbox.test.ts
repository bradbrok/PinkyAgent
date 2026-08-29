import { describe, expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { EditTool } from "../src/edit";
import { GlobTool } from "../src/glob";
import { GrepTool } from "../src/grep";
import { ReadTool } from "../src/read";
import { WriteTool } from "../src/write";
import { makeCtx, makeTmpDir } from "./helpers";

describe("sandbox", () => {
  const reader = new ReadTool();
  const writer = new WriteTool();
  const editor = new EditTool();
  const globber = new GlobTool();
  const greper = new GrepTool();

  test("read rejects ../.. escape", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await reader.execute({ path: "../../../etc/passwd" }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("escapes sandbox");
    } finally {
      cleanup();
    }
  });

  test("write rejects ../.. escape", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await writer.execute(
        { path: "../escape.txt", content: "x" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("escapes sandbox");
    } finally {
      cleanup();
    }
  });

  test("edit rejects ../.. escape", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await editor.execute(
        { path: "../../victim.txt", old: "a", new: "b" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("escapes sandbox");
    } finally {
      cleanup();
    }
  });

  test("glob rejects escape via path", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await globber.execute(
        { pattern: "*", path: "../../" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("escapes sandbox");
    } finally {
      cleanup();
    }
  });

  test("grep rejects escape via path", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await greper.execute(
        { pattern: "x", path: "../../" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("escapes sandbox");
    } finally {
      cleanup();
    }
  });

  test("in-sandbox subdirectory paths still work", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      mkdirSync(path.join(dir, "sub"));
      writeFileSync(path.join(dir, "sub", "f.txt"), "ok");
      const res = await reader.execute({ path: "sub/f.txt" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toBe("1: ok");
    } finally {
      cleanup();
    }
  });
});
