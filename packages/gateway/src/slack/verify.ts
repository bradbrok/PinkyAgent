/**
 * Slack request signature verification (Slack Events API).
 *
 * Slack signs each request as: base = `v0:${timestamp}:${rawBody}`,
 * signature header = `v0=` + hex(HMAC-SHA256(signingSecret, base)).
 * Requests older than 5 minutes are rejected to prevent replay.
 */
import { createHmac, timingSafeEqual } from "node:crypto";

const MAX_AGE_SECONDS = 300;

/** Compute the Slack signature for a body — exported so tests can sign. */
export function signSlackRequest(secret: string, timestamp: string, rawBody: string): string {
  const base = `v0:${timestamp}:${rawBody}`;
  const digest = createHmac("sha256", secret).update(base).digest("hex");
  return `v0=${digest}`;
}

/** Verify a Slack signature. Rejects stale timestamps and tampered bodies. */
export function verifySlackRequest(
  secret: string,
  timestamp: string,
  rawBody: string,
  signature: string,
): boolean {
  const ts = Number(timestamp);
  const now = Math.floor(Date.now() / 1000);
  if (!Number.isFinite(ts)) return false;
  if (Math.abs(now - ts) > MAX_AGE_SECONDS) return false;

  const expected = signSlackRequest(secret, timestamp, rawBody);
  const a = Buffer.from(expected, "utf8");
  const b = Buffer.from(signature, "utf8");
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}
