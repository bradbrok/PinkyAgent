import { describe, expect, test } from "bun:test";
import { gateEvent, type GateInput } from "../src/slack/gate";

const base: GateInput = {
  isBot: false,
  isDM: false,
  mentioned: false,
  isReplyToBot: false,
  text: "hello",
};

describe("gateEvent", () => {
  test("bot message is silent", () => {
    expect(gateEvent({ ...base, isBot: true })).toEqual({ action: "silent", reason: "bot message" });
  });

  test("mention engages", () => {
    expect(gateEvent({ ...base, mentioned: true })).toEqual({ action: "engage", reason: "mention" });
  });

  test("DM engages", () => {
    expect(gateEvent({ ...base, isDM: true })).toEqual({ action: "engage", reason: "dm" });
  });

  test("reply to agent engages", () => {
    expect(gateEvent({ ...base, isReplyToBot: true })).toEqual({
      action: "engage",
      reason: "reply to agent",
    });
  });

  test("ambient chatter is silent", () => {
    expect(gateEvent(base)).toEqual({ action: "silent", reason: "ambient" });
  });

  test("bot beats mention (precedence)", () => {
    expect(gateEvent({ ...base, isBot: true, mentioned: true, isDM: true })).toEqual({
      action: "silent",
      reason: "bot message",
    });
  });

  test("mention beats DM/reply (precedence)", () => {
    expect(gateEvent({ ...base, mentioned: true, isDM: true, isReplyToBot: true })).toEqual({
      action: "engage",
      reason: "mention",
    });
  });

  test("DM beats reply-to-agent (precedence)", () => {
    expect(gateEvent({ ...base, isDM: true, isReplyToBot: true })).toEqual({
      action: "engage",
      reason: "dm",
    });
  });
});
