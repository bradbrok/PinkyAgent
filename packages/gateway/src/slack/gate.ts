/**
 * Deterministic reply gate (DESIGN.md §6). Pure decision over a normalized
 * Slack message: the cheap LLM classifier is a later slice; this rules
 * cascade is the free first pass.
 */

export interface GateInput {
  isBot: boolean;
  isDM: boolean;
  mentioned: boolean;
  isReplyToBot: boolean;
  text: string;
}

export interface GateDecision {
  action: "engage" | "silent";
  reason: string;
}

/** Ordered rules: bot → mention → DM → reply-to-agent → ambient fallback. */
export function gateEvent(ev: GateInput): GateDecision {
  if (ev.isBot) return { action: "silent", reason: "bot message" };
  if (ev.mentioned) return { action: "engage", reason: "mention" };
  if (ev.isDM) return { action: "engage", reason: "dm" };
  if (ev.isReplyToBot) return { action: "engage", reason: "reply to agent" };
  return { action: "silent", reason: "ambient" };
}
