/**
 * A2A cross-node delivery endpoint (DESIGN.md §7). A peer node's Messenger
 * POSTs an A2AEnvelope JSON body here; we verify the shared-secret HMAC and
 * hand the envelope to the local messenger (durable mailbox + notify).
 *
 * Signature scheme (sender side lives in the runtime messenger):
 *   X-Pinky-Signature = hex(HMAC-SHA256(secret, `${id}.${sentAt}.${rawBody}`))
 * where id/sentAt are taken from the envelope body itself. The 300s freshness
 * window bounds replay; messenger.receive() makes a replay *harmless* by
 * claiming delivery of the envelope id atomically, so a sender retry is a 200
 * with no second row and no second wakeup.
 */
import { createHmac, timingSafeEqual } from "node:crypto";
import type { A2AEnvelope, Messenger } from "@pinky/runtime";

const MAX_AGE_MS = 300_000;

/** Compute the A2A delivery signature — exported so tests can sign. */
export function signA2ABody(secret: string, id: string, sentAt: string, rawBody: string): string {
  return createHmac("sha256", secret).update(`${id}.${sentAt}.${rawBody}`).digest("hex");
}

function isEnvelope(value: unknown): value is A2AEnvelope {
  if (typeof value !== "object" || value === null) return false;
  const env = value as Record<string, unknown>;
  return (
    typeof env.id === "string" &&
    typeof env.from === "string" &&
    typeof env.to === "string" &&
    (env.kind === "message" || env.kind === "request" || env.kind === "response") &&
    typeof env.text === "string" &&
    typeof env.sentAt === "string"
  );
}

function json(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export interface A2ADeliverOpts {
  secret: string;
  messenger: Messenger;
}

/** POST /a2a/deliver handler. */
export async function handleA2ADeliver(req: Request, opts: A2ADeliverOpts): Promise<Response> {
  if (req.method !== "POST") return json(405, { ok: false, error: "method not allowed" });

  const rawBody = await req.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawBody);
  } catch {
    return json(400, { ok: false, error: "invalid JSON" });
  }
  if (!isEnvelope(parsed)) return json(400, { ok: false, error: "malformed envelope" });

  const signature = req.headers.get("x-pinky-signature") ?? "";
  const freshness = Math.abs(Date.now() - new Date(parsed.sentAt).getTime());
  if (signature.length === 0 || !Number.isFinite(freshness) || freshness > MAX_AGE_MS) {
    return json(401, { ok: false, error: "unauthorized" });
  }

  const expected = signA2ABody(opts.secret, parsed.id, parsed.sentAt, rawBody);
  const a = Buffer.from(expected, "utf8");
  const b = Buffer.from(signature, "utf8");
  if (a.length !== b.length || !timingSafeEqual(a, b)) {
    return json(401, { ok: false, error: "unauthorized" });
  }

  // Idempotent ingest: persist under the sender's ORIGINAL id, then claim
  // delivery for this node — subscribers wake only for the caller that won the
  // claim. `duplicate` is therefore "someone already delivered this", which
  // covers both a sender retry and a shared-database row the peer node has
  // already taken; either way the message IS delivered from the sender's point
  // of view, so it is a 200.
  const claimed = await opts.messenger.receive(parsed);
  return json(200, { ok: true, duplicate: !claimed });
}
