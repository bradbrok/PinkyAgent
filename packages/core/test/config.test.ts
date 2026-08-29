/**
 * EnvConfig bootstrap tests: the two database urls, and the startup guard that
 * refuses to open an A2A-capable process whose HMAC key is empty (an empty key
 * does not disable a signature check — it makes every forged signature verify).
 */
import { describe, expect, it } from "bun:test";
import { DEFAULT_SETTINGS, assertGatewaySecrets, loadEnvConfig, type EnvConfig } from "../src/config";

const base: Record<string, string | undefined> = {
  DATABASE_URL: "postgres://pinky_app:pinky@localhost:5544/pinky",
};

const envConfig = (overrides: Record<string, string | undefined> = {}): EnvConfig =>
  loadEnvConfig({ ...base, ...overrides });

describe("loadEnvConfig database urls", () => {
  it("falls back to DATABASE_URL when DATABASE_ADMIN_URL is unset", () => {
    const cfg = envConfig();
    expect(cfg.databaseAdminUrl).toBe(cfg.databaseUrl);
  });

  it("carries no Slack fields: the Slack surface is gone (JSONL headless is the interface)", () => {
    expect(Object.keys(envConfig())).not.toContain("slack");
    expect(JSON.stringify(envConfig({ SLACK_BOT_TOKEN: "xoxb-1" }))).not.toContain("xoxb-1");
  });

  it("keeps the admin url separate when both are set", () => {
    const cfg = envConfig({ DATABASE_ADMIN_URL: "postgres://postgres:pinky@localhost:5544/pinky" });
    expect(cfg.databaseAdminUrl).toBe("postgres://postgres:pinky@localhost:5544/pinky");
    expect(cfg.databaseUrl).toBe("postgres://pinky_app:pinky@localhost:5544/pinky");
  });
});

describe("assertGatewaySecrets", () => {
  it("passes on a single-node process with no peers and no secret", () => {
    // Nothing to forge: with no peers the A2A relay is disabled outright, so a
    // dev box that never set A2A_SECRET still starts.
    expect(() => assertGatewaySecrets(envConfig())).not.toThrow();
    expect(() => assertGatewaySecrets(envConfig({ A2A_SECRET: "" }))).not.toThrow();
  });

  it("requires A2A_SECRET as soon as PINKY_PEERS names a peer", () => {
    expect(() =>
      assertGatewaySecrets(envConfig({ PINKY_PEERS: "node2=http://node2:3000", A2A_SECRET: "" })),
    ).toThrow(/A2A_SECRET/);
    expect(() =>
      assertGatewaySecrets(envConfig({ PINKY_PEERS: "node2=http://node2:3000", A2A_SECRET: "s" })),
    ).not.toThrow();
  });

  it("refuses a whitespace-only secret, not just an unset one", () => {
    expect(() =>
      assertGatewaySecrets(envConfig({ PINKY_PEERS: "node2=http://node2:3000", A2A_SECRET: "   " })),
    ).toThrow(/A2A_SECRET/);
  });

  it("names every peer in the error, so the operator knows why it fired", () => {
    let message = "";
    try {
      assertGatewaySecrets(
        envConfig({ PINKY_PEERS: "node2=http://node2:3000,node3=http://node3:3000" }),
      );
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toMatch(/node2/);
    expect(message).toMatch(/node3/);
  });
});

describe("DEFAULT_SETTINGS", () => {
  it("ships the documented memory-plane defaults (DESIGN.md §5)", () => {
    expect(DEFAULT_SETTINGS.memory).toEqual({
      embeddingModel: "openai/text-embedding-3-small",
      autoRecall: true,
      recallLimit: 12,
      recallTokenBudget: 5_000,
    });
  });

  it("keeps every behavioral sub-tree in the snapshot, not in EnvConfig", () => {
    // DEFAULT_SETTINGS is the source of truth for "which keys exist"
    // (settings.ts derives its key lists from it), so a new sub-tree that
    // never lands here is a key validateSettings would reject.
    expect(Object.keys(DEFAULT_SETTINGS).sort()).toEqual([
      "context",
      "memory",
      "model",
      "replyGate",
      "selfConfig",
      "tenantId",
    ]);
  });

  it("ships self-configuration off, delegating nothing (DESIGN.md P8, revised)", () => {
    // The safe default is the whole point: enabling it, and naming which keys
    // it covers, are two separate human decisions. Shipping `enabled: true`
    // with an empty allow-list would grant nothing either, but it would put
    // the switch in a state nobody chose.
    expect(DEFAULT_SETTINGS.selfConfig).toEqual({ enabled: false, allowedKeys: [] });
  });

  it("keeps embedder *credentials* out of the settings snapshot", () => {
    // Which embedder to use is behavioral (settings); its API key is
    // bootstrap-only and never reaches a snapshot the runtime can read.
    const cfg = envConfig({ OPENAI_API_KEY: "sk-test" });
    expect(JSON.stringify(cfg)).not.toContain("sk-test");
    expect(JSON.stringify(DEFAULT_SETTINGS)).not.toContain("sk-test");
  });
});
