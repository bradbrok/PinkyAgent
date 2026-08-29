import { describe, expect, test } from "bun:test";
import { normalizeSlackEvent } from "../src/slack/normalize";

const BOT = "U0BOT";

function messageEvent(overrides: Record<string, unknown> = {}): unknown {
  return {
    type: "event_callback",
    event_id: "EvABC123",
    event: {
      type: "message",
      channel: "C12345",
      ts: "1700000000.000001",
      user: "U0HUMAN",
      text: "hello world",
      ...overrides,
    },
  };
}

describe("normalizeSlackEvent", () => {
  test("maps a plain message", () => {
    const n = normalizeSlackEvent(messageEvent(), BOT);
    expect(n).toEqual({
      externalId: "EvABC123",
      channelId: "C12345",
      threadId: "1700000000.000001",
      author: { platform: "slack", userId: "U0HUMAN" },
      text: "hello world",
      isBot: false,
      isDM: false,
      mentioned: false,
      isReplyToBot: false,
    });
  });

  test("thread_ts wins over ts for threadId", () => {
    const n = normalizeSlackEvent(messageEvent({ thread_ts: "1699999999.999999" }), BOT);
    expect(n?.threadId).toBe("1699999999.999999");
  });

  test("detects and strips the bot mention", () => {
    const n = normalizeSlackEvent(messageEvent({ text: "hey <@U0BOT> what's up" }), BOT);
    expect(n?.mentioned).toBe(true);
    expect(n?.text).toBe("hey what's up");
  });

  test("mention-only text strips to empty", () => {
    const n = normalizeSlackEvent(messageEvent({ text: "<@U0BOT>" }), BOT);
    expect(n?.mentioned).toBe(true);
    expect(n?.text).toBe("");
  });

  test("other-user mentions are untouched", () => {
    const n = normalizeSlackEvent(messageEvent({ text: "hi <@U0OTHER>" }), BOT);
    expect(n?.mentioned).toBe(false);
    expect(n?.text).toBe("hi <@U0OTHER>");
  });

  test("bot_id marks a bot message", () => {
    expect(normalizeSlackEvent(messageEvent({ bot_id: "B123" }), BOT)?.isBot).toBe(true);
  });

  test("bot_message subtype marks a bot message", () => {
    expect(normalizeSlackEvent(messageEvent({ subtype: "bot_message" }), BOT)?.isBot).toBe(true);
  });

  test("im channel_type marks a DM", () => {
    expect(normalizeSlackEvent(messageEvent({ channel_type: "im" }), BOT)?.isDM).toBe(true);
  });

  test("parent_user_id equal to bot marks reply to agent", () => {
    expect(normalizeSlackEvent(messageEvent({ parent_user_id: "U0BOT" }), BOT)?.isReplyToBot).toBe(
      true,
    );
  });

  test("parent_user_id of another user does not", () => {
    expect(normalizeSlackEvent(messageEvent({ parent_user_id: "U0HUMAN" }), BOT)?.isReplyToBot).toBe(
      false,
    );
  });

  test("non-message events return null", () => {
    expect(
      normalizeSlackEvent(
        { type: "event_callback", event_id: "Ev1", event: { type: "reaction_added" } },
        BOT,
      ),
    ).toBeNull();
  });

  test("url_verification bodies return null", () => {
    expect(normalizeSlackEvent({ type: "url_verification", challenge: "x" }, BOT)).toBeNull();
  });

  test("garbage returns null", () => {
    expect(normalizeSlackEvent(null, BOT)).toBeNull();
    expect(normalizeSlackEvent("string", BOT)).toBeNull();
    expect(normalizeSlackEvent({ type: "event_callback" }, BOT)).toBeNull();
  });
});
