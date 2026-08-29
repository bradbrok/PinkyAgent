/**
 * EnvConfig bootstrap tests: the two database urls, and the startup guard that
 * refuses to run a gateway whose HMAC keys are empty (an empty key does not
 * disable a signature check — it makes every forged signature verify).
 */
import { describe, expect, it } from "bun:test";
import { assertGatewaySecrets, loadEnvConfig, type EnvConfig } from "../src/config";

const base: Record<string, string | undefined> = {
  DATABASE_URL: "postgres://pinky_app:pinky@localhost:5544/pinky",
  SLACK_BOT_TOKEN: "xoxb-1",
  SLACK_SIGNING_SECRET: "sign-1",
};

const envConfig = (overrides: Record<string, string | undefined> = {}): EnvConfig =>
  loadEnvConfig({ ...base, ...overrides });

describe("loadEnvConfig database urls", () => {
  it("falls back to DATABASE_URL when DATABASE_ADMIN_URL is unset", () => {
    const cfg = envConfig();
    expect(cfg.databaseAdminUrl).toBe(cfg.databaseUrl);
  });

  it("keeps the admin url separate when both are set", () => {
    const cfg = envConfig({ DATABASE_ADMIN_URL: "postgres://postgres:pinky@localhost:5544/pinky" });
    expect(cfg.databaseAdminUrl).toBe("postgres://postgres:pinky@localhost:5544/pinky");
    expect(cfg.databaseUrl).toBe("postgres://pinky_app:pinky@localhost:5544/pinky");
  });
});

describe("assertGatewaySecrets", () => {
  it("passes with both Slack secrets and no peers", () => {
    expect(() => assertGatewaySecrets(envConfig())).not.toThrow();
  });

  it("refuses an empty SLACK_SIGNING_SECRET", () => {
    expect(() => assertGatewaySecrets(envConfig({ SLACK_SIGNING_SECRET: "" }))).toThrow(
      /SLACK_SIGNING_SECRET/,
    );
  });

  it("refuses a whitespace-only secret, not just an unset one", () => {
    expect(() => assertGatewaySecrets(envConfig({ SLACK_SIGNING_SECRET: "   " }))).toThrow(
      /SLACK_SIGNING_SECRET/,
    );
  });

  it("refuses an empty SLACK_BOT_TOKEN", () => {
    expect(() => assertGatewaySecrets(envConfig({ SLACK_BOT_TOKEN: "" }))).toThrow(
      /SLACK_BOT_TOKEN/,
    );
  });

  it("lists every missing secret in one error", () => {
    let message = "";
    try {
      assertGatewaySecrets(envConfig({ SLACK_BOT_TOKEN: "", SLACK_SIGNING_SECRET: "" }));
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toMatch(/SLACK_SIGNING_SECRET/);
    expect(message).toMatch(/SLACK_BOT_TOKEN/);
  });

  it("requires A2A_SECRET only when PINKY_PEERS names a peer", () => {
    expect(() => assertGatewaySecrets(envConfig({ A2A_SECRET: "" }))).not.toThrow();
    expect(() =>
      assertGatewaySecrets(envConfig({ PINKY_PEERS: "node2=http://node2:3000", A2A_SECRET: "" })),
    ).toThrow(/A2A_SECRET/);
    expect(() =>
      assertGatewaySecrets(envConfig({ PINKY_PEERS: "node2=http://node2:3000", A2A_SECRET: "s" })),
    ).not.toThrow();
  });
});
