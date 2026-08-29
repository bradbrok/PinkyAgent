/**
 * Durable mailbox (DESIGN.md §7) — the persistence half of the Messenger
 * contract. The live half (onMessage / HTTP relay) lives in packages/runtime.
 *
 * Stored in a2a_messages. Addresses use the form `agentId@nodeId` (or the bare
 * word `broadcast`). A Mailbox is bound to ONE node id (this process's
 * PinkyConfig.nodeId, default "local"): every address is normalized against it,
 * so `beta`, `beta@local` on node "local" and `beta@node2` on node "node2" all
 * land in the same local partition. Rows whose node_to is some *other* node are
 * outbound: they never show up in deliverLocal()/inbox() here, and
 * pendingForNode() hands them to the relay retry sweep.
 */
import type { Db } from "./db";

/** Mirror of runtime.A2AEnvelope (kept in-core to avoid the dependency). */
export interface A2AEnvelope {
  id: string;
  /** sender address, e.g. `weather@local` */
  from: string;
  /** `agentId@nodeId` or the bare word `broadcast` */
  to: string;
  kind: "message" | "request" | "response";
  text: string;
  threadHint?: string;
  /** ISO timestamp of logical send. */
  sentAt: string;
}

interface A2aRow {
  id: string;
  from_agent: string;
  to_agent: string;
  node_from: string;
  node_to: string;
  kind: "message" | "request" | "response";
  text: string;
  thread_hint: string | null;
  delivered_at?: string | Date | null;
  read_at?: string | Date | null;
  /** postgres.js hands back a Date for timestamptz. */
  created_at?: string | Date | null;
}

export interface A2AAddress {
  agentId: string;
  /** Bare `nodeId` — the caller's own node unless the address names another. */
  nodeId: string;
}

/**
 * Single shared address parser (runtime's messenger imports this one too).
 * Rules:
 *  - `broadcast`            -> { agentId: "broadcast", nodeId: defaultNode }
 *  - `agent`  (no "@")      -> { agentId: "agent",     nodeId: defaultNode }
 *  - `agent@node`           -> { agentId: "agent",     nodeId: "node" }
 *  - the LAST "@" separates agent from node, so agent ids may contain "@";
 *    a leading "@" is not a separator ("@x" is an agent id).
 */
export function parseA2AAddress(raw: string, defaultNode: string): A2AAddress {
  const at = raw.lastIndexOf("@");
  if (at <= 0) return { agentId: raw, nodeId: defaultNode };
  return { agentId: raw.slice(0, at), nodeId: raw.slice(at + 1) };
}

const SELECT_COLS = `id, from_agent, to_agent, node_from, node_to, kind, text, thread_hint, delivered_at, read_at, created_at`;

const INSERT_COLS = `insert into a2a_messages (id, from_agent, to_agent, node_from, node_to, kind, text, thread_hint)`;

function isoOrEmpty(value: string | Date | null | undefined): string {
  if (value instanceof Date) return value.toISOString();
  return value ?? "";
}

/** Map a database row to the A2AEnvelope wire shape. */
function toEnvelope(row: A2aRow): A2AEnvelope {
  const env: A2AEnvelope = {
    id: row.id,
    from: `${row.from_agent}@${row.node_from}`,
    to: `${row.to_agent}@${row.node_to}`,
    kind: row.kind,
    text: row.text,
    sentAt: isoOrEmpty(row.created_at),
  };
  if (row.thread_hint !== null) env.threadHint = row.thread_hint;
  return env;
}

export interface MailboxOptions {
  /** This process's node id. Addresses resolve against it; default "local". */
  nodeId?: string;
}

export class Mailbox {
  private db: Db;
  /** The node this mailbox speaks for; the local partition of a2a_messages. */
  readonly nodeId: string;

  constructor(db: Db, opts: MailboxOptions = {}) {
    this.db = db;
    this.nodeId = opts.nodeId ?? "local";
  }

  /** Normalize an address against this mailbox's node id. */
  address(raw: string): A2AAddress {
    return parseA2AAddress(raw, this.nodeId);
  }

  private insertParams(env: A2AEnvelope): unknown[] {
    const from = this.address(env.from);
    // A broadcast has no '@'; the bare word selects to_agent="broadcast" and,
    // because broadcast semantics are LOCAL-ONLY (no fan-out to peer nodes),
    // node_to normalizes to this node.
    const to = this.address(env.to);
    return [
      env.id,
      from.agentId,
      to.agentId,
      from.nodeId,
      to.nodeId,
      env.kind,
      env.text,
      env.threadHint ?? null,
    ];
  }

  /** Persist a logically-sent envelope (first step of at-least-once). */
  async put(env: A2AEnvelope): Promise<void> {
    await this.db.query(
      `${INSERT_COLS}
       values ($1, $2, $3, $4, $5, $6, $7, $8)`,
      this.insertParams(env),
    );
  }

  /**
   * Persist an envelope that already has an id (a relayed message). Returns
   * true when the row was newly inserted, false when the id was already
   * present.
   *
   * NOT the idempotency hinge for receive — {@link Mailbox.claimDelivery} is.
   * When two nodes share one database (`PINKY_NODE_ID=node2 pinky smoke`, and
   * the cross-node integration test) the SENDER has already written this exact
   * row, so a false here means "the sender's row is right there", not "already
   * delivered".
   */
  async putIfAbsent(env: A2AEnvelope): Promise<boolean> {
    const rows = await this.db.query<{ id: string }>(
      `${INSERT_COLS}
       values ($1, $2, $3, $4, $5, $6, $7, $8)
       on conflict (id) do nothing
       returning id`,
      this.insertParams(env),
    );
    return rows.length > 0;
  }

  /** Mark one row delivered (no-op if already delivered). Sender-side: it is
   *  how send()/deliverRemote() record that a message got where it was going,
   *  and it deliberately does NOT filter on node_to (the row may be outbound). */
  async markDelivered(id: string): Promise<void> {
    await this.db.query(
      `update a2a_messages set delivered_at = now() where id = $1 and delivered_at is null`,
      [id],
    );
  }

  /**
   * Receiver-side delivery CLAIM, and the single idempotency point of the
   * ingress path. Flips delivered_at for a row addressed to THIS node, and
   * returns true only for the caller that actually flipped it.
   *
   * Why the claim and not the insert: with one database per node the receiver
   * inserts the row and the insert is a fine idempotency hinge, but two nodes
   * on ONE database share the row the sender already wrote — an insert-based
   * hinge then reports every first delivery as a duplicate and no subscriber
   * is ever woken (the silent bug this replaced). `update ... where
   * delivered_at is null returning id` is atomic in both topologies: exactly
   * one caller wins, replays lose, and a row owned by another node is not
   * ours to claim at all.
   */
  async claimDelivery(id: string): Promise<boolean> {
    const rows = await this.db.query<{ id: string }>(
      `update a2a_messages set delivered_at = now()
       where id = $1 and node_to = $2 and delivered_at is null
       returning id`,
      [id, this.nodeId],
    );
    return rows.length > 0;
  }

  /**
   * Recovery sweep: all undelivered rows for `agentId` in this node's
   * partition; marks delivered_at. The happy path marks a single row via
   * markDelivered(); this drains anything a crash left behind.
   */
  async deliverLocal(agentId: string): Promise<A2AEnvelope[]> {
    return await this.db.tx(async (tx) => {
      const rows = await tx.query<A2aRow>(
        `select ${SELECT_COLS} from a2a_messages
         where to_agent in ($1, 'broadcast') and node_to = $2 and delivered_at is null
         order by created_at`,
        [agentId, this.nodeId],
      );
      if (rows.length > 0) {
        await tx.query(
          `update a2a_messages set delivered_at = now()
           where id in (select id from a2a_messages
                        where to_agent in ($1, 'broadcast') and node_to = $2 and delivered_at is null)`,
          [agentId, this.nodeId],
        );
      }
      return rows.map(toEnvelope);
    });
  }

  /** Delivered-but-unread messages for `agentId` on this node; marks read_at. */
  async inbox(agentId: string, opts?: { limit?: number }): Promise<A2AEnvelope[]> {
    const limit = opts?.limit ?? 50;
    return await this.db.tx(async (tx) => {
      const rows = await tx.query<A2aRow>(
        `select ${SELECT_COLS} from a2a_messages
         where to_agent in ($1, 'broadcast') and node_to = $2
           and delivered_at is not null and read_at is null
         order by created_at limit $3`,
        [agentId, this.nodeId, limit],
      );
      if (rows.length > 0) {
        await tx.query(
          `update a2a_messages set read_at = now()
           where id in (select id from a2a_messages
                        where to_agent in ($1, 'broadcast') and node_to = $2
                          and delivered_at is not null and read_at is null
                        order by created_at limit $3)`,
          [agentId, this.nodeId, limit],
        );
      }
      return rows.map(toEnvelope);
    });
  }

  /** Outbound rows for a peer node (never this node's own) awaiting a relay. */
  async pendingForNode(nodeId: string): Promise<A2AEnvelope[]> {
    const rows = await this.db.query<A2aRow>(
      `select ${SELECT_COLS} from a2a_messages
       where node_to = $1 and node_to != $2 and delivered_at is null
       order by created_at`,
      [nodeId, this.nodeId],
    );
    return rows.map(toEnvelope);
  }

  close(): Promise<void> {
    return this.db.close();
  }
}
