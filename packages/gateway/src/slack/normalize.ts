/**
 * Normalize Slack `event_callback` payloads into the gateway's message shape.
 * Returns null for anything that is not a message event (reactions, joins,
 * url_verification, malformed bodies) so callers can ack-and-ignore.
 */

export interface NormalizedSlackMessage {
  /** Slack event_id — the dedup key. */
  externalId: string;
  channelId: string;
  /** thread_ts if present, else ts (root message starts its own thread). */
  threadId: string;
  author: { platform: "slack"; userId: string };
  /** Message text with the bot's `<@BOTID>` mention stripped. */
  text: string;
  isBot: boolean;
  isDM: boolean;
  mentioned: boolean;
  isReplyToBot: boolean;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * Map a decoded Slack event body to a NormalizedSlackMessage.
 * `botUserId` is needed to detect mentions and replies-to-bot; pass an empty
 * string to disable mention detection gracefully.
 */
export function normalizeSlackEvent(
  body: unknown,
  botUserId: string,
): NormalizedSlackMessage | null {
  const rec = asRecord(body);
  if (!rec) return null;
  if (asString(rec.type) !== "event_callback") return null;

  const externalId = asString(rec.event_id);
  const event = asRecord(rec.event);
  if (!externalId || !event) return null;
  if (asString(event.type) !== "message") return null;

  const channelId = asString(event.channel);
  const ts = asString(event.ts);
  const text = asString(event.text);
  const userId = asString(event.user);
  if (!channelId || !ts || text === null || !userId) return null;

  const mentionToken = `<@${botUserId}>`;
  const mentioned = botUserId.length > 0 && text.includes(mentionToken);
  const stripped = mentioned ? text.split(mentionToken).join(" ").replace(/\s+/g, " ").trim() : text;

  const parentUserId = asString(event.parent_user_id);
  const threadTs = asString(event.thread_ts);

  return {
    externalId,
    channelId,
    threadId: threadTs ?? ts,
    author: { platform: "slack", userId },
    text: stripped,
    isBot: typeof event.bot_id === "string" || asString(event.subtype) === "bot_message",
    isDM: asString(event.channel_type) === "im",
    mentioned,
    isReplyToBot: botUserId.length > 0 && parentUserId === botUserId,
  };
}
