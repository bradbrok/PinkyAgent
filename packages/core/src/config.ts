/**
 * Configuration is split in two (DESIGN.md: "everything in db"):
 *
 * - EnvConfig — process bootstrap ONLY: db url, secrets, node identity.
 *   Never read by the agent loop directly.
 * - SettingsSnapshot — all *behavioral* config, loaded from the `settings`
 *   table by the human-owned CLI / gateway startup. The agent runtime receives
 *   a snapshot; it has no write path (no self-reconfiguration).
 */

export interface EnvConfig {
  /** Least-privilege connection used by everything at run time (`pinky_app`). */
  databaseUrl: string;
  /**
   * Privileged connection used by `pinky migrate` ONLY: DDL plus CREATE ROLE
   * (migration 0003 creates `pinky_app`). Defaults to databaseUrl so a
   * single-url dev setup still works; in a real deployment DATABASE_URL is the
   * NOBYPASSRLS app role and cannot run migrations at all.
   */
  databaseAdminUrl: string;
  nodeId: string;
  /** peer nodeId -> base URL for cross-machine A2A delivery. */
  peers: Record<string, string>;
  /** HMAC secret shared by all A2A nodes. */
  a2aSecret: string;
  slack: { botToken: string; signingSecret: string };
  port: number;
}

export function loadEnvConfig(env: Record<string, string | undefined> = process.env): EnvConfig {
  const peers: Record<string, string> = {};
  for (const pair of (env.PINKY_PEERS ?? "").split(",")) {
    const trimmed = pair.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    peers[trimmed.slice(0, eq)] = trimmed.slice(eq + 1);
  }
  const databaseUrl = env.DATABASE_URL ?? "postgres://postgres:pinky@localhost:5544/pinky";
  return {
    databaseUrl,
    databaseAdminUrl: env.DATABASE_ADMIN_URL ?? databaseUrl,
    nodeId: env.PINKY_NODE_ID ?? "local",
    peers,
    a2aSecret: env.A2A_SECRET ?? "",
    slack: {
      botToken: env.SLACK_BOT_TOKEN ?? "",
      signingSecret: env.SLACK_SIGNING_SECRET ?? "",
    },
    port: Number(env.PORT ?? 3000),
  };
}

/**
 * Fail fast when a gateway secret is missing.
 *
 * Both signature checks the gateway performs are HMACs, and an HMAC over an
 * EMPTY key is still a perfectly valid HMAC: `verifySlackRequest("", ...)` and
 * the A2A check both accept anything an attacker can compute for themselves.
 * An unset secret is therefore not "auth disabled", it is "auth forged", so a
 * gateway that would listen on a public port refuses to start instead.
 *
 * Rules:
 *  - SLACK_SIGNING_SECRET and SLACK_BOT_TOKEN are always required (the gateway
 *    exists to verify and answer Slack).
 *  - A2A_SECRET is required only when PINKY_PEERS names at least one peer;
 *    with no peers the relay is disabled outright (server.ts answers 503).
 *
 * Exported from core rather than inlined in the CLI so it is unit-testable.
 * Throws one Error listing every problem; returns void when the env is usable.
 */
export function assertGatewaySecrets(env: EnvConfig): void {
  const missing: string[] = [];
  const blank = (v: string): boolean => v.trim() === "";

  if (blank(env.slack.signingSecret)) {
    missing.push("SLACK_SIGNING_SECRET (an empty signing key verifies every forged request)");
  }
  if (blank(env.slack.botToken)) {
    missing.push("SLACK_BOT_TOKEN (needed for auth.test and chat.postMessage)");
  }
  const peers = Object.keys(env.peers);
  if (peers.length > 0 && blank(env.a2aSecret)) {
    missing.push(
      `A2A_SECRET (PINKY_PEERS names ${peers.join(", ")}; an empty HMAC key lets anyone inject A2A messages)`,
    );
  }

  if (missing.length > 0) {
    throw new Error(
      `refusing to start the gateway — missing required secret(s):\n  - ${missing.join("\n  - ")}\n` +
        `Set them in .env (see .env.example) and try again.`,
    );
  }
}

/** Behavioral settings, sourced from the `settings` table (scope overlay:
 *  global < channel:<id> < agent:<id>).
 *
 *  This type plus DEFAULT_SETTINGS *is* the schema: settings.ts derives the
 *  set of legal keys from DEFAULT_SETTINGS and hand-checks each field in
 *  validateSettings(). Adding a field here means adding its rule there. */
export interface SettingsSnapshot {
  tenantId: string;
  /** Provider/model-id, e.g. "openrouter/moonshotai/kimi-k2". */
  model: string;
  context: {
    advisoryFraction: number; // inject pressure notice; 0 < advisory < hard
    hardFraction: number; // force continuity write; advisory < hard <= 1
    approxWindowTokens: number; // positive integer
  };
  replyGate: {
    /** enable the LLM classifier after the deterministic cascade (later slice). */
    classifierEnabled: boolean;
  };
}

/** Base of every snapshot, and the source of truth for which keys exist. */
export const DEFAULT_SETTINGS: SettingsSnapshot = {
  tenantId: "default",
  model: "openrouter/moonshotai/kimi-k2",
  context: { advisoryFraction: 0.7, hardFraction: 0.9, approxWindowTokens: 180_000 },
  replyGate: { classifierEnabled: false },
};
