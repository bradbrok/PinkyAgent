import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { BashTool } from "../src/bash";
import { makeCtx, makeTmpDir } from "./helpers";

describe("bash", () => {
  const tool = new BashTool();

  test("echo captures stdout", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ command: "echo hello" }, makeCtx(dir));
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("hello");
    } finally {
      cleanup();
    }
  });

  test("non-zero exit is an error with the exit code", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({ command: "exit 3" }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("exit code 3");
    } finally {
      cleanup();
    }
  });

  test("timeout kills a sleep", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const start = Date.now();
      const res = await tool.execute({ command: "sleep 5", timeout: 1 }, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("timed out");
      expect(Date.now() - start).toBeLessThan(4000);
    } finally {
      cleanup();
    }
  });

  test("stderr is combined with stdout", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: "echo out && echo err >&2" },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("out");
      expect(res.text).toContain("err");
    } finally {
      cleanup();
    }
  });

  test("output is capped at 50KB with a truncated marker", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: "seq 1 200000" },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("[truncated]");
      expect(res.text.length).toBeLessThanOrEqual(50 * 1024 + 16);
    } finally {
      cleanup();
    }
  });

  test("timeout above the max is clamped to 120s", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: "echo ok", timeout: 9999 },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("ok");
    } finally {
      cleanup();
    }
  });
});

describe("bash env containment", () => {
  // Stands in for the gateway's real secrets (DATABASE_URL, SLACK_BOT_TOKEN,
  // OPENROUTER_API_KEY, A2A_SECRET, ...) that live in the host process env.
  const SECRET_VALUE = "leak-9f3a2c";

  beforeAll(() => {
    process.env.PINKY_TEST_SECRET = SECRET_VALUE;
  });

  afterAll(() => {
    delete process.env.PINKY_TEST_SECRET;
  });

  test("host process env is not inherited by the shell", async () => {
    const tool = new BashTool();
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: 'env; echo "secret=[$PINKY_TEST_SECRET]"' },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).not.toContain(SECRET_VALUE);
      expect(res.text).not.toContain("PINKY_TEST_SECRET");
      expect(res.text).toContain("secret=[]");
    } finally {
      cleanup();
    }
  });

  test("caller-supplied env values ARE visible", async () => {
    const tool = new BashTool({ env: { PINKY_ALLOWED: "allowed-1b7d" } });
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: 'echo "allowed=[$PINKY_ALLOWED]"; echo "secret=[$PINKY_TEST_SECRET]"' },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("allowed=[allowed-1b7d]");
      expect(res.text).toContain("secret=[]");
    } finally {
      cleanup();
    }
  });

  test("the minimal env still carries PATH and HOME", async () => {
    const tool = new BashTool();
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { command: 'echo "path=[${PATH:+set}]"; echo "home=[${HOME:+set}]"' },
        makeCtx(dir),
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("path=[set]");
      expect(res.text).toContain("home=[set]");
    } finally {
      cleanup();
    }
  });
});
