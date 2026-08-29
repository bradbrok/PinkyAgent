import { describe, expect, test } from "bun:test";
import { signSlackRequest, verifySlackRequest } from "../src/slack/verify";

const SECRET = "test-signing-secret";
const BODY = '{"type":"event_callback","event_id":"Ev123"}';

/** Fresh timestamp every call — avoids replay-window flakiness. */
function nowSeconds(): string {
  return String(Math.floor(Date.now() / 1000));
}

describe("verifySlackRequest", () => {
  test("valid signature passes", () => {
    const ts = nowSeconds();
    const sig = signSlackRequest(SECRET, ts, BODY);
    expect(verifySlackRequest(SECRET, ts, BODY, sig)).toBe(true);
  });

  test("tampered body fails", () => {
    const ts = nowSeconds();
    const sig = signSlackRequest(SECRET, ts, BODY);
    expect(verifySlackRequest(SECRET, ts, `${BODY}tampered`, sig)).toBe(false);
  });

  test("stale timestamp fails", () => {
    const stale = String(Math.floor(Date.now() / 1000) - 301);
    const sig = signSlackRequest(SECRET, stale, BODY);
    expect(verifySlackRequest(SECRET, stale, BODY, sig)).toBe(false);
  });

  test("bad signature fails", () => {
    const ts = nowSeconds();
    expect(verifySlackRequest(SECRET, ts, BODY, "v0=deadbeef")).toBe(false);
  });

  test("reversed signature fails", () => {
    const ts = nowSeconds();
    const sig = signSlackRequest(SECRET, ts, BODY);
    expect(verifySlackRequest(SECRET, ts, BODY, sig.split("").reverse().join(""))).toBe(false);
  });

  test("non-numeric timestamp fails", () => {
    const ts = "not-a-number";
    const sig = signSlackRequest(SECRET, ts, BODY);
    expect(verifySlackRequest(SECRET, ts, BODY, sig)).toBe(false);
  });
});
