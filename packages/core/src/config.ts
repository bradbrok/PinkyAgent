/**
 * Configuration is split in two (DESIGN.md: "everything in db"):
 *
 * - EnvConfig — process bootstrap ONLY: db url, secrets, node identity.
 *   Never read by the agent loop directly.
 * - SettingsSnapshot — all *behavioral* config, loaded from the `settings`
 *   table by the human-owned CLI / gateway startup. The agent runtime receives
 *   a snapshot; the only write path it has is the `settings_set` tool, and
 *   only for the keys a human allow-listed under `selfConfig` (DESIGN.md P8 as
 *   revised: "human-granted self-configuration"). Config never lives in a
 *   file, so a malformed value can never stop the process from starting — a
 *   rejected write is a tool error the agent reads and retries.
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
    port: Number(env.PORT ?? 3000),
  };
}

/**
 * Fail fast when a process that can talk to A2A peers has no HMAC key.
 *
 * The A2A signature check is an HMAC, and an HMAC over an EMPTY key is still a
 * perfectly valid HMAC: it accepts anything an attacker can compute for
 * themselves. An unset secret is therefore not "auth disabled", it is "auth
 * forged", so a process that would exchange envelopes with a peer refuses to
 * start instead.
 *
 * One rule, and it only bites when it can: A2A_SECRET is required as soon as
 * PINKY_PEERS names at least one peer. With no peers there is nothing to
 * forge — the relay is disabled outright (server.ts answers 503) — so a
 * single-node dev box needs no secret at all.
 *
 * Called at startup by every process that opens a messenger: the headless
 * service (which is where PINKY_PEERS actually matters), `pinky prompt`, and
 * `pinky smoke` (which zeroes its peers, so it never trips).
 *
 * Exported from core rather than inlined in the CLI so it is unit-testable.
 * Throws one Error listing every problem; returns void when the env is usable.
 */
export function assertGatewaySecrets(env: EnvConfig): void {
  const missing: string[] = [];
  const blank = (v: string): boolean => v.trim() === "";

  const peers = Object.keys(env.peers);
  if (peers.length > 0 && blank(env.a2aSecret)) {
    missing.push(
      `A2A_SECRET (PINKY_PEERS names ${peers.join(", ")}; an empty HMAC key lets anyone inject A2A messages)`,
    );
  }

  if (missing.length > 0) {
    throw new Error(
      `refusing to start — missing required secret(s):\n  - ${missing.join("\n  - ")}\n` +
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
  /** Memory plane (DESIGN.md §5). Behavioral, so it lives here and not in
   *  EnvConfig — the embedder's *credentials* stay in the env, but which
   *  embedder to use is a setting. Agent-writable only through `settings_set`
   *  when a human allow-lists the key (P8). */
  memory: {
    /**
     * Embedder id as "provider/model-id", e.g. "openai/text-embedding-3-small",
     * or the literal "none" for FTS-only recall (no vector voice, and retained
     * rows are stored without an embedding). Same provider-prefix rule as
     * `model`; the runtime's createEmbedder() splits it the same way.
     */
    embeddingModel: string;
    /** Inject the `<memories>` block at context start on every wake (§5.4). */
    autoRecall: boolean;
    /** Candidates auto-recall asks for, before the token-budget cut. */
    recallLimit: number;
    /** Approx. token ceiling for the injected block (§5.4: ~5k). */
    recallTokenBudget: number;
  };
  /**
   * Human-granted self-configuration (DESIGN.md P8, revised).
   *
   * P8 used to read "agents cannot write settings, full stop". It now reads:
   * config lives ONLY in the settings table, the human CLI is the default
   * write path, and an agent may write a setting ONLY through the validated
   * `settings_set` tool, ONLY for keys a human put in `allowedKeys`, and every
   * such write is journaled as a `config` event. Nothing here is editable by
   * a config *file*, which is the point: a bad value is rejected before it
   * lands (SettingsStore.set validates first), so the agent gets a tool error
   * it can read and correct instead of a process that will not boot.
   *
   * This sub-tree is itself never agent-writable — an agent that could widen
   * its own allow-list would have no allow-list. Neither is `tenantId`.
   */
  selfConfig: {
    /** Master switch. Default false: only a human (`pinky config set`) flips it. */
    enabled: boolean;
    /**
     * Keys the agent may write, as exact dotted keys ("model",
     * "context.advisoryFraction"), subtree patterns ("context.*" = every key
     * *under* context but not the whole sub-tree at once), or "*" (everything
     * except the immutables). Default []: enabling self-configuration without
     * naming a key grants nothing.
     */
    allowedKeys: string[];
  };
}

/** Base of every snapshot, and the source of truth for which keys exist. */
export const DEFAULT_SETTINGS: SettingsSnapshot = {
  tenantId: "default",
  model: "openrouter/moonshotai/kimi-k2",
  context: { advisoryFraction: 0.7, hardFraction: 0.9, approxWindowTokens: 180_000 },
  replyGate: { classifierEnabled: false },
  memory: {
    embeddingModel: "openai/text-embedding-3-small",
    autoRecall: true,
    recallLimit: 12,
    recallTokenBudget: 5_000,
  },
  // Off, and empty, until a human says otherwise.
  selfConfig: { enabled: false, allowedKeys: [] },
};
