import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Db } from "@pinky/core";
import type { A2AEnvelope, Messenger, ToolContext } from "@pinky/runtime";

export function makeTmpDir(): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(path.join(tmpdir(), "pinky-tools-"));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

/** Never-called Db stub — no tool here touches the database. */
const fakeDb: Db = {
  query: () => Promise.reject(new Error("db not used in tools tests")),
  queryOne: () => Promise.reject(new Error("db not used in tools tests")),
  tx: () => Promise.reject(new Error("db not used in tools tests")),
  close: () => Promise.resolve(),
};

export interface ContextOverrides {
  messenger?: Messenger | null;
  agentId?: string | null;
}

export function makeCtx(cwd: string, overrides: ContextOverrides = {}): ToolContext {
  const ctx: ToolContext = {
    cwd,
    db: fakeDb,
    thread: { tenantId: "t1", channelId: "c1", threadId: "thread-test" },
    emit: () => Promise.resolve(),
  };
  if (overrides.messenger !== null && overrides.messenger !== undefined) {
    ctx.messenger = overrides.messenger;
  }
  if (overrides.agentId !== null && overrides.agentId !== undefined) {
    ctx.agentId = overrides.agentId;
  }
  return ctx;
}

export interface SentEnvelope extends Omit<A2AEnvelope, "id" | "sentAt"> {
  id: string;
  sentAt: string;
}

export function makeFakeMessenger(opts: {
  nodeId?: string;
  canned?: A2AEnvelope[];
} = {}): Messenger & { sent: SentEnvelope[]; idSeq: number } {
  const sent: SentEnvelope[] = [];
  const received = new Set<string>();
  let idSeq = 0;
  const base: Messenger = {
    nodeId: opts.nodeId ?? "node-test",
    send(env) {
      idSeq++;
      sent.push({ ...env, id: `id-${idSeq}`, sentAt: new Date().toISOString() });
      return Promise.resolve(`id-${idSeq}`);
    },
    inbox() {
      return Promise.resolve(opts.canned ?? []);
    },
    onMessage() {
      return () => {};
    },
    // Inbound relay half of the contract. No tool calls it; the double just
    // has to satisfy Messenger, and dedups on id like a real one would.
    receive(env) {
      const fresh = !received.has(env.id);
      received.add(env.id);
      return Promise.resolve(fresh);
    },
  };
  return Object.assign(base, { sent, get idSeq() { return sent.length; } });
}
