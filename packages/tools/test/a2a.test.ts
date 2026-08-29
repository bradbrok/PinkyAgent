import { describe, expect, test } from "bun:test";
import { A2AInboxTool, A2ASendTool } from "../src/a2a";
import { makeCtx, makeFakeMessenger, makeTmpDir } from "./helpers";
import type { A2AEnvelope } from "@pinky/runtime";

describe("a2a_send", () => {
  const tool = new A2ASendTool();

  test("sends an envelope with derived from and thread hint", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const messenger = makeFakeMessenger({ nodeId: "node-9" });
      const ctx = makeCtx(dir, { messenger, agentId: "pinky" });
      const res = await tool.execute(
        { to: "peer@node-1", text: "hello peer", kind: "request" },
        ctx,
      );
      expect(res.isError).toBeUndefined();
      expect(res.text).toContain("id-1");
      expect(messenger.sent.length).toBe(1);
      const env = messenger.sent[0]!;
      expect(env.from).toBe("pinky@node-9");
      expect(env.to).toBe("peer@node-1");
      expect(env.kind).toBe("request");
      expect(env.text).toBe("hello peer");
      expect(env.threadHint).toBe("thread-test");
    } finally {
      cleanup();
    }
  });

  test("kind defaults to message", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const messenger = makeFakeMessenger();
      const ctx = makeCtx(dir, { messenger, agentId: "pinky" });
      await tool.execute({ to: "broadcast", text: "hi all" }, ctx);
      expect(messenger.sent[0]!.kind).toBe("message");
    } finally {
      cleanup();
    }
  });

  test("absent messenger is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute(
        { to: "peer@node-1", text: "hi" },
        makeCtx(dir),
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("A2A not enabled");
    } finally {
      cleanup();
    }
  });

  test("invalid kind is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const messenger = makeFakeMessenger();
      const ctx = makeCtx(dir, { messenger, agentId: "pinky" });
      const res = await tool.execute(
        { to: "peer@node-1", text: "hi", kind: "telepathy" },
        ctx,
      );
      expect(res.isError).toBe(true);
      expect(res.text).toContain("'kind'");
    } finally {
      cleanup();
    }
  });
});

describe("a2a_inbox", () => {
  const tool = new A2AInboxTool();

  test("returns canned envelopes as JSON with from/kind/text/sentAt", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const canned: A2AEnvelope[] = [
        {
          id: "m1",
          from: "peer@node-1",
          to: "pinky@node-test",
          kind: "message",
          text: "ping",
          sentAt: "2026-08-28T00:00:00.000Z",
        },
      ];
      const messenger = makeFakeMessenger({ canned });
      const ctx = makeCtx(dir, { messenger, agentId: "pinky" });
      const res = await tool.execute({}, ctx);
      expect(res.isError).toBeUndefined();
      const parsed = JSON.parse(res.text) as {
        from: string;
        kind: string;
        text: string;
        sentAt: string;
      }[];
      expect(parsed).toEqual([
        {
          from: "peer@node-1",
          kind: "message",
          text: "ping",
          sentAt: "2026-08-28T00:00:00.000Z",
        },
      ]);
    } finally {
      cleanup();
    }
  });

  test("empty inbox reads '(no messages)'", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const messenger = makeFakeMessenger();
      const ctx = makeCtx(dir, { messenger, agentId: "pinky" });
      const res = await tool.execute({}, ctx);
      expect(res.text).toBe("(no messages)");
    } finally {
      cleanup();
    }
  });

  test("absent messenger is an error", async () => {
    const { dir, cleanup } = makeTmpDir();
    try {
      const res = await tool.execute({}, makeCtx(dir));
      expect(res.isError).toBe(true);
      expect(res.text).toContain("A2A not enabled");
    } finally {
      cleanup();
    }
  });
});
