/**
 * Local messenger: durable mailbox + in-process wake + cross-node HTTP delivery.
 *
 * At-least-once, both directions:
 *  - send(): the mailbox row is written before any delivery attempt. A local
 *    recipient is marked delivered and its subscribers fired; a remote one is
 *    POSTed to the peer and marked delivered only on a 2xx, so a failure leaves
 *    the row pending for flushPending().
 *  - receive(): the relay hands us the peer's envelope; we persist it under its
 *    ORIGINAL id (insert ... on conflict do nothing) and then CLAIM delivery
 *    for this node (update ... where delivered_at is null returning id).
 *    The claim, not the insert, is what decides whether subscribers wake, so a
 *    retry is a no-op and a shared database — where the sender already wrote
 *    the row — still wakes the receiver exactly once.
 *
 * The messenger's nodeId is also the mailbox's: addresses without "@", and
 * addresses ending in "@<thisNode>", are the same local mailbox partition.
 *
 * Delivery is not consumption (issue #4). `delivered_at` is the relay's
 * bookkeeping — it makes receive() idempotent and says nothing about whether
 * an agent ever acted. The receipt is `read_at`, stamped by the consumer
 * inside the transaction that journals the work (claimConsumption), so
 * recovery needs no scheduler state at all: redeliverUnconsumed() re-fires
 * everything unread, and a duplicate fire loses the claim and does nothing.
 */
import { createHmac, randomUUID } from "node:crypto";
import { Mailbox, parseA2AAddress } from "@pinky/core";
import type { Db } from "@pinky/core";
import type { A2AEnvelope, Messenger } from "./types";

export interface LocalMessengerOptions {
  /** Local node id; envelopes addressed to agent@thisNode are delivered in-process. */
  nodeId: string;
  /** peer nodeId -> base URL for cross-node delivery. */
  peers?: Record<string, string> | undefined;
  /** HMAC secret shared by all A2A nodes. */
  a2aSecret?: string | undefined;
  /** Injectable for tests. */
  fetchFn?: typeof fetch | undefined;
}

/** HMAC-SHA256 (hex) over "${id}.${ts}.${body}" — the X-Pinky-Signature value. */
export function a2aSignature(secret: string, id: string, ts: string, body: string): string {
  return createHmac("sha256", secret).update(`${id}.${ts}.${body}`).digest("hex");
}

type MessageHandler = (env: A2AEnvelope) => void;

export class LocalMessenger implements Messenger {
  readonly nodeId: string;
  private readonly mailbox: Mailbox;
  private readonly peers: Record<string, string>;
  private readonly a2aSecret: string;
  private readonly fetchFn: typeof fetch;
  private readonly subscribers = new Map<string, Set<MessageHandler>>();

  constructor(db: Db, opts: LocalMessengerOptions) {
    this.nodeId = opts.nodeId;
    this.mailbox = new Mailbox(db, { nodeId: opts.nodeId });
    this.peers = opts.peers ?? {};
    this.a2aSecret = opts.a2aSecret ?? "";
    this.fetchFn = opts.fetchFn ?? fetch;
  }

  async send(env: Omit<A2AEnvelope, "id" | "sentAt">): Promise<string> {
    const envelope: A2AEnvelope = {
      ...env,
      id: randomUUID(),
      sentAt: new Date().toISOString(),
    };
    // Durable first (at-least-once).
    await this.mailbox.put(envelope);

    // Unqualified, "@thisNode", and `broadcast` are all local. Broadcast stays
    // deliberately LOCAL-ONLY: no fan-out to peer nodes (DESIGN.md §7 leaves
    // cross-node broadcast to an explicit per-peer send).
    const { agentId, nodeId } = parseA2AAddress(env.to, this.nodeId);
    if (nodeId === this.nodeId) {
      // Mark this row delivered in the mailbox, then wake live subscribers.
      this.mailbox
        .markDelivered(envelope.id)
        .catch((err) => console.warn("[messenger] markDelivered failed:", err));
      this.fire(agentId, envelope);
      if (agentId === "broadcast") this.fire("*", envelope);
      return envelope.id;
    }

    const base = this.peers[nodeId];
    if (!base) {
      // Unknown peer: row stays undelivered for a later retry sweep.
      console.warn(
        `[messenger] no route to node ${nodeId}; message ${envelope.id} left undelivered`,
      );
      return envelope.id;
    }
    this.deliverRemote(base, envelope).catch((err) => {
      console.warn(
        `[messenger] delivery of ${envelope.id} to ${nodeId} failed; left undelivered for retry:`,
        err instanceof Error ? err.message : err,
      );
    });
    return envelope.id;
  }

  inbox(agentId: string, opts?: { limit?: number }): Promise<A2AEnvelope[]> {
    return this.mailbox.inbox(agentId, opts) as Promise<A2AEnvelope[]>;
  }

  onMessage(agentId: string, handler: MessageHandler): () => void {
    let set = this.subscribers.get(agentId);
    if (!set) {
      set = new Set();
      this.subscribers.set(agentId, set);
    }
    set.add(handler);
    return () => {
      set.delete(handler);
      if (set.size === 0) this.subscribers.delete(agentId);
    };
  }

  /**
   * Called by the HTTP ingress side (gateway /a2a/deliver) when a peer node
   * delivers to us. Returns true iff THIS call is the delivery — the relay
   * reports the inverse as `duplicate`.
   *
   * Two steps, and the order matters:
   *   1. persist the envelope under its ORIGINAL id if we have never seen it
   *      (`on conflict (id) do nothing`). On separate databases this is the
   *      row's first existence; on a SHARED database the sender's own
   *      `Mailbox.put` already wrote it and this is a no-op. Either way the
   *      message is durable before anything else happens.
   *   2. CLAIM it for this node: `update ... set delivered_at = now() where
   *      id = $1 and node_to = <me> and delivered_at is null returning id`.
   *
   * Step 2 is the idempotency point. Using step 1 for that — "the insert found
   * an existing row, therefore this is a duplicate" — is what broke shared-
   * database delivery: the sender's row made every FIRST delivery look like a
   * replay, so receive() returned before firing and no live subscriber was
   * ever woken. The claim is atomic and node-scoped, so it is true exactly
   * once per row per node, in every topology: fresh delivery (insert, then
   * claim), shared database (no insert, still claim), replay (claim finds
   * delivered_at already set → false → the relay answers 200 duplicate:true
   * and nobody is woken twice).
   */
  async receive(env: A2AEnvelope): Promise<boolean> {
    const inserted = await this.mailbox.putIfAbsent(env);
    const { agentId, nodeId } = parseA2AAddress(env.to, this.nodeId);
    if (nodeId !== this.nodeId) {
      // Misrouted: the peer sent us mail for a third node. Nothing to claim —
      // the row is not in our partition — but it is durable and still pending,
      // so flushPending() forwards it if we have a route. `inserted` is the
      // honest answer to "was this call the one that took custody".
      console.warn(`[messenger] received ${env.id} addressed to node ${nodeId}; left pending`);
      return inserted;
    }
    const claimed = await this.mailbox.claimDelivery(env.id);
    if (!claimed) return false; // already delivered here: no second wakeup
    this.fire(agentId, env);
    if (agentId === "broadcast") this.fire("*", env);
    return true;
  }

  /**
   * Recovery for the consumption edge (issue #4): re-fire every message for
   * `agentId` on this node with no receipt yet (`read_at is null`), whatever
   * delivered_at says. Returns how many were fired.
   *
   * `delivered_at` only ever meant "this node took custody" — with nothing
   * subscribed, or a handler that threw, or a process that died between the
   * claim and the turn, the row stays delivered and unconsumed forever and the
   * wake is lost with no way to notice. The receipt (`read_at`, stamped by the
   * consumer in its own transaction) is what says the work happened, so
   * recovery is simply "fire everything unread".
   *
   * Idempotent by construction, not by bookkeeping: a duplicate fire loses the
   * claimConsumption race at the consumer and produces nothing. Broadcast rows
   * fire on the same keys as the live path (`broadcast` and `*`), so a node
   * with no broadcast subscriber leaves them unconsumed — honestly so: nobody
   * ever acted on them.
   */
  async redeliverUnconsumed(agentId: string): Promise<number> {
    const pending = await this.mailbox.unconsumedFor(agentId);
    for (const env of pending) {
      const { agentId: to } = parseA2AAddress(env.to, this.nodeId);
      this.fire(to, env);
      if (to === "broadcast") this.fire("*", env);
    }
    return pending.length;
  }

  /**
   * Stamp the consumption receipt for one message; true only for the caller
   * that stamped it. Pass the consumer's `tx` so the receipt commits with the
   * work — that is what makes redelivery safe and a rolled-back turn
   * recoverable (Mailbox.claimRead).
   */
  claimConsumption(id: string, tx?: Db): Promise<boolean> {
    return tx ? this.mailbox.claimRead(id, tx) : this.mailbox.claimRead(id);
  }

  /** @deprecated Use receive(): it persists and dedups. Kept as a thin shim. */
  receiveRemote(env: A2AEnvelope): void {
    this.receive(env).catch((err) => console.warn("[messenger] receive failed:", err));
  }

  /**
   * Retry sweep for the sender side of at-least-once: every row still pending
   * for a configured peer is re-POSTed. The receiver dedups by envelope id, so
   * re-sending an already-delivered message is harmless. Rows are re-stamped
   * with a fresh sentAt (the id is what identifies the message) so the peer's
   * 300s freshness window does not reject an old pending row.
   */
  async flushPending(): Promise<{ attempted: number; delivered: number }> {
    let attempted = 0;
    let delivered = 0;
    for (const [node, base] of Object.entries(this.peers)) {
      const pending = await this.mailbox.pendingForNode(node);
      for (const env of pending) {
        attempted += 1;
        try {
          await this.deliverRemote(base, { ...env, sentAt: new Date().toISOString() });
          delivered += 1;
        } catch (err) {
          console.warn(
            `[messenger] retry of ${env.id} to ${node} failed; still pending:`,
            err instanceof Error ? err.message : err,
          );
        }
      }
    }
    return { attempted, delivered };
  }

  private fire(agentId: string, env: A2AEnvelope): void {
    for (const handler of this.subscribers.get(agentId) ?? []) handler(env);
  }

  /** POST one envelope to a peer; a 2xx completes delivery (marks the row). */
  private async deliverRemote(base: string, env: A2AEnvelope): Promise<void> {
    const url = `${base.replace(/\/+$/, "")}/a2a/deliver`;
    const body = JSON.stringify(env);
    const res = await this.fetchFn(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "X-Pinky-Signature": a2aSignature(this.a2aSecret, env.id, env.sentAt, body),
        "X-Pinky-Ts": env.sentAt,
      },
      body,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
    await this.mailbox.markDelivered(env.id);
  }
}
