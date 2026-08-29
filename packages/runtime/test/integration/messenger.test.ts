/**
 * Cross-node A2A over a real socket and a real mailbox (DESIGN.md §7).
 *
 * Two LocalMessengers, nodeIds "itA" and "itB", sharing one database — which
 * is a supported topology, not just a test shortcut: `PINKY_NODE_ID=node2
 * pinky smoke` runs a second node against the very same DATABASE_URL. Node B
 * is fronted by an in-process Bun.serve on an ephemeral port whose
 * /a2a/deliver route is the real gateway handler, so every hop is exercised:
 * durable put → HMAC-signed POST → handleA2ADeliver → messenger.receive →
 * mailbox → live subscriber.
 *
 * Skipped unless PINKY_INTEGRATION=1. The connection comes from
 * loadEnvConfig() (DATABASE_URL) — local dev is 5544, CI is 5432.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { createDb, loadEnvConfig, migrate, type Db } from "@pinky/core";
import { handleA2ADeliver } from "@pinky/gateway";
import { LocalMessenger } from "../../src/messenger";
import type { A2AEnvelope } from "../../src/types";

const ENABLED = process.env.PINKY_INTEGRATION === "1";
const suite = ENABLED ? describe : describe.skip;

const DB_URL = loadEnvConfig().databaseUrl;
const SCHEMA_DIR = new URL("../../../core/schema", import.meta.url).pathname;
const REPO_ROOT = new URL("../../../..", import.meta.url).pathname;

const NODE_A = "itA";
const NODE_B = "itB";
const SECRET = "integration-a2a-secret";

/** Agent ids are run-unique so a stale row from an interrupted run cannot be
 *  mistaken for this run's mail. */
const RUN = crypto.randomUUID().slice(0, 8);
const ALPHA = `alpha-${RUN}`;
const BETA = `beta-${RUN}`;

/** Poll until `check` is true — deliverRemote() is fire-and-forget, so send()
 *  returns before the peer has seen anything. */
async function until(check: () => boolean | Promise<boolean>, ms = 4000): Promise<void> {
  const deadline = Date.now() + ms;
  for (;;) {
    if (await check()) return;
    if (Date.now() > deadline) throw new Error("timed out waiting for condition");
    await Bun.sleep(15);
  }
}

interface WireCall {
  url: string;
  body: string;
  headers: Record<string, string>;
}

suite("cross-node A2A (live postgres + live HTTP)", () => {
  let db: Db;
  let a: LocalMessenger;
  let b: LocalMessenger;
  let server: ReturnType<typeof Bun.serve> | undefined;
  let port = 0;
  const wire: WireCall[] = [];
  const bHeard: A2AEnvelope[] = [];
  const aHeard: A2AEnvelope[] = [];

  /** Node B's ingress: the real gateway relay handler, nothing stubbed. */
  function serve(onPort: number): ReturnType<typeof Bun.serve> {
    return Bun.serve({
      port: onPort,
      fetch: async (req) => {
        const url = new URL(req.url);
        if (url.pathname === "/a2a/deliver") {
          return handleA2ADeliver(req, { secret: SECRET, messenger: b });
        }
        return new Response("not found", { status: 404 });
      },
    });
  }

  /** Records every outbound request so "never hit the wire" is checkable, and
   *  so a replay can re-POST the byte-identical body and headers. */
  type FetchInput = Parameters<typeof fetch>[0];
  const spyFetch = (async (input: FetchInput, init?: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries((init?.headers ?? {}) as Record<string, string>)) {
      headers[k] = v;
    }
    wire.push({ url: String(input), body: String(init?.body ?? ""), headers });
    return await fetch(input, init);
  }) as unknown as typeof fetch;

  beforeAll(async () => {
    db = createDb(DB_URL, { max: 8 });
    await migrate(db, SCHEMA_DIR);
    await purge();

    // B first: the server's handler closes over it.
    b = new LocalMessenger(db, { nodeId: NODE_B, peers: {}, a2aSecret: SECRET });
    server = serve(0);
    port = Number(server.port);
    a = new LocalMessenger(db, {
      nodeId: NODE_A,
      peers: { [NODE_B]: `http://127.0.0.1:${port}` },
      a2aSecret: SECRET,
      fetchFn: spyFetch,
    });

    b.onMessage(BETA, (env) => bHeard.push(env));
    a.onMessage(ALPHA, (env) => aHeard.push(env));
  });

  afterAll(async () => {
    server?.stop(true);
    if (db) {
      await purge();
      await db.close();
    }
  });

  /** Scoped delete: only this file's two node partitions. Never a TRUNCATE. */
  async function purge(): Promise<void> {
    await db.query(
      `delete from a2a_messages where node_from in ($1, $2) or node_to in ($1, $2)`,
      [NODE_A, NODE_B],
    );
  }

  function row(id: string): Promise<{ node_to: string; delivered_at: string | null } | null> {
    return db.queryOne<{ node_to: string; delivered_at: string | null }>(
      `select node_to, delivered_at from a2a_messages where id = $1`,
      [id],
    );
  }

  it("A -> agent@itB crosses the wire and the row is marked delivered", async () => {
    const id = await a.send({
      from: `${ALPHA}@${NODE_A}`,
      to: `${BETA}@${NODE_B}`,
      kind: "message",
      text: "hello from itA",
    });

    await until(async () => (await row(id))?.delivered_at != null);

    expect(wire).toHaveLength(1);
    expect(wire[0]!.url).toBe(`http://127.0.0.1:${port}/a2a/deliver`);
    expect(wire[0]!.headers["X-Pinky-Signature"]).toMatch(/^[0-9a-f]{64}$/);

    const stored = await row(id);
    expect(stored?.node_to).toBe(NODE_B);
    expect(stored?.delivered_at).not.toBeNull();
  });

  it("the message shows up in B's inbox exactly once", async () => {
    const inbox = await b.inbox(BETA);
    expect(inbox).toHaveLength(1);
    expect(inbox[0]!.text).toBe("hello from itA");
    expect(inbox[0]!.from).toBe(`${ALPHA}@${NODE_A}`);
    // inbox() marks read, so a second call is empty — no redelivery loop.
    expect(await b.inbox(BETA)).toHaveLength(0);
  });

  it("DEFECT: the delivery fires B's onMessage subscriber exactly once", async () => {
    // Nothing ever reached bHeard when the two nodes shared one database.
    //
    // send() writes the durable row FIRST (Mailbox.put, at-least-once), and on
    // a shared database that row is the same row the receiving side then tries
    // to insert: Mailbox.putIfAbsent does `on conflict (id) do nothing`, sees
    // the sender's row, returns false, and LocalMessenger.receive() took that
    // as "duplicate — already handled" and returned before markDelivered and
    // before fire(). The relay answered 200 {duplicate:true}, the sender's own
    // deliverRemote() then marked the row delivered, so inbox() still yielded
    // the message (previous test) — but no live subscriber was ever woken. A
    // wake that only happens on the next poll is exactly the failure mode the
    // mailbox+notify design exists to avoid, and it was silent.
    //
    // With one database per node (the cross-machine case) putIfAbsent inserts
    // and the subscriber did fire, which is why the fakes-only unit suite and
    // the single-node smoke never saw it.
    //
    // THE FIX: idempotency is keyed on the DELIVERY CLAIM, not on row
    // existence. receive() inserts if the row is unknown, then runs `update
    // a2a_messages set delivered_at = now() where id = $1 and node_to = <me>
    // and delivered_at is null returning id` (Mailbox.claimDelivery) and fires
    // subscribers iff that returned a row. Separate databases, one shared
    // database and replays all take the same path.
    expect(bHeard).toHaveLength(1);
    expect(bHeard[0]!.text).toBe("hello from itA");
  });

  it("a replayed POST of the identical body and headers is a 200 and changes nothing", async () => {
    const replay = wire[0]!;
    const before = await db.query<{ id: string }>(
      `select id from a2a_messages where node_to = $1`,
      [NODE_B],
    );
    const heardBefore = bHeard.length;

    const res = await fetch(replay.url, {
      method: "POST",
      headers: replay.headers,
      body: replay.body,
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, duplicate: true });
    // No second row, and no second wake.
    const after = await db.query<{ id: string }>(
      `select id from a2a_messages where node_to = $1`,
      [NODE_B],
    );
    expect(after.map((r) => r.id).sort()).toEqual(before.map((r) => r.id).sort());
    expect(bHeard).toHaveLength(heardBefore);
  });

  it("a message from A to an agent on A stays local and never touches the wire", async () => {
    const wireBefore = wire.length;
    const id = await a.send({
      from: `${BETA}@${NODE_A}`,
      to: `${ALPHA}@${NODE_A}`,
      kind: "message",
      text: "local only",
    });

    // Local delivery is in-process and synchronous for subscribers.
    expect(aHeard.map((e) => e.id)).toContain(id);
    expect(aHeard.at(-1)!.text).toBe("local only");
    expect(wire).toHaveLength(wireBefore);

    await until(async () => (await row(id))?.delivered_at != null);
    expect((await row(id))?.node_to).toBe(NODE_A);
    // Unqualified addresses resolve to the sender's own node too.
    const bare = await a.send({ from: ALPHA, to: ALPHA, kind: "message", text: "bare" });
    expect((await row(bare))?.node_to).toBe(NODE_A);
    expect(wire).toHaveLength(wireBefore);
  });

  it("with the peer down the row stays pending; flushPending() delivers it after restart", async () => {
    server!.stop(true);
    server = undefined;

    const id = await a.send({
      from: `${ALPHA}@${NODE_A}`,
      to: `${BETA}@${NODE_B}`,
      kind: "request",
      text: "sent while itB was down",
    });

    // The durable row exists and is NOT delivered: at-least-once means the
    // message survives the peer being unreachable.
    await until(() => wire.length > 0 && wire.at(-1)!.body.includes("sent while itB was down"));
    const pending = await row(id);
    expect(pending).not.toBeNull();
    expect(pending?.delivered_at).toBeNull();
    expect((await a.inbox(BETA)).length).toBe(0); // not ours to read; it is outbound

    // Same port, so A's configured peer URL still resolves.
    server = serve(port);
    const swept = await a.flushPending();
    expect(swept.attempted).toBeGreaterThanOrEqual(1);
    expect(swept.delivered).toBeGreaterThanOrEqual(1);

    expect((await row(id))?.delivered_at).not.toBeNull();
    const inbox = await b.inbox(BETA);
    expect(inbox.map((e) => e.text)).toContain("sent while itB was down");
  });

  it("flushPending() on a clean mailbox attempts nothing", async () => {
    expect(await a.flushPending()).toEqual({ attempted: 0, delivered: 0 });
  });

  it(
    "the CLI smoke path runs green on two different node ids",
    async () => {
      // Covers `pinky smoke` itself (migrate + settings + event log + local A2A
      // through the real runtime loop), on the default node and on a second one
      // sharing this database.
      for (const nodeId of [undefined, "node2"]) {
        const proc = Bun.spawn(["bun", "run", "smoke"], {
          cwd: REPO_ROOT,
          env: nodeId === undefined ? process.env : { ...process.env, PINKY_NODE_ID: nodeId },
          stdout: "pipe",
          stderr: "pipe",
        });
        const [code, out, err] = await Promise.all([
          proc.exited,
          new Response(proc.stdout).text(),
          new Response(proc.stderr).text(),
        ]);
        const label = `PINKY_NODE_ID=${nodeId ?? "(default)"}`;
        if (code !== 0) console.error(`${label} smoke failed:\n${out}\n${err}`);
        expect(code).toBe(0);
        expect(out).toContain("smoke: all checks passed");
        expect(out).not.toContain("FAIL");
      }
    },
    60_000,
  );
});

/**
 * The consumption edge (issue #4), against the real table.
 *
 * Its own node partition and its own messenger so the ordered cross-node suite
 * above is untouched. Everything here is about ONE claim: `read_at` is stamped
 * by the consumer, in the consumer's transaction, and nothing else may say a
 * message was acted on.
 */
suite("A2A consumption receipts (live postgres)", () => {
  const NODE_C = `itC-${RUN}`;
  const AGENT = `consumer-${RUN}`;
  let db: Db;
  let m: LocalMessenger;
  const heard: A2AEnvelope[] = [];

  beforeAll(async () => {
    db = createDb(DB_URL, { max: 4 });
    await migrate(db, SCHEMA_DIR);
    await db.query(`delete from a2a_messages where node_to = $1 or node_from = $1`, [NODE_C]);
    m = new LocalMessenger(db, { nodeId: NODE_C, peers: {}, a2aSecret: SECRET });
    m.onMessage(AGENT, (env) => heard.push(env));
  });

  afterAll(async () => {
    if (!db) return;
    await db.query(`delete from a2a_messages where node_to = $1 or node_from = $1`, [NODE_C]);
    await db.close();
  });

  /** Deliver one message to AGENT on this node and return its id. */
  async function arrive(text: string): Promise<string> {
    const id = await m.send({ from: `peer@${NODE_C}`, to: `${AGENT}@${NODE_C}`, kind: "message", text });
    // send() marks the local row delivered fire-and-forget.
    await until(async () => (await marks(id))?.delivered_at != null);
    return id;
  }

  function marks(id: string): Promise<{ delivered_at: Date | null; read_at: Date | null } | null> {
    return db.queryOne<{ delivered_at: Date | null; read_at: Date | null }>(
      `select delivered_at, read_at from a2a_messages where id = $1`,
      [id],
    );
  }

  it("DEFECT: a delivered-but-unconsumed row is redelivered until something consumes it", async () => {
    // The failure this guards, and the reason the issue was filed: the ONLY
    // thing `delivered_at` proves is that this node took custody of the row.
    // Nothing subscribed in production (`pinky headless` never called
    // onMessage), so every arrival was marked delivered and then sat there
    // until the agent happened to poll `a2a_inbox` — and a handler that threw,
    // or a process that died right after the claim, left a row that says
    // "delivered" forever and was never acted on. Zero recovery, and nothing
    // in the data to even notice it.
    //
    // THE FIX: the receipt is `read_at`, stamped by the CONSUMER, and recovery
    // is "re-fire everything unread". Redelivery is safe because the claim is
    // transactional, so a duplicate fire produces no second turn.
    const id = await arrive("work that was never done");
    const row = await marks(id);
    expect(row?.delivered_at).not.toBeNull();
    expect(row?.read_at).toBeNull(); // delivered, unconsumed: the orphan

    const before = heard.length;
    expect(await m.redeliverUnconsumed(AGENT)).toBe(1);
    expect(heard.length).toBe(before + 1);
    expect(heard.at(-1)!.text).toBe("work that was never done");
    // Still unconsumed, so still redelivered — the sweep does not lose it.
    expect(await m.redeliverUnconsumed(AGENT)).toBe(1);
    expect(heard.length).toBe(before + 2);

    // A consumer claims it inside its own transaction, as the wake path does.
    const claimed = await db.tx((tx) => m.claimConsumption(id, tx));
    expect(claimed).toBe(true);
    expect((await marks(id))?.read_at).not.toBeNull();

    // ...and the second delivery is now a no-op at the consumer.
    expect(await m.redeliverUnconsumed(AGENT)).toBe(0);
    expect(heard.length).toBe(before + 2);
    expect(await db.tx((tx) => m.claimConsumption(id, tx))).toBe(false);
  });

  it("a consumer whose transaction rolls back leaves the message unconsumed", async () => {
    // The receipt is atomic with the work: claim it, fail to journal the turn,
    // and the message must come back — otherwise "consumed" would be a lie the
    // log cannot contradict.
    const id = await arrive("rolled back");
    const before = heard.length;

    await expect(
      db.tx(async (tx) => {
        expect(await m.claimConsumption(id, tx)).toBe(true);
        throw new Error("the turn blew up after the claim");
      }),
    ).rejects.toThrow("the turn blew up after the claim");

    expect((await marks(id))?.read_at).toBeNull();
    expect(await m.redeliverUnconsumed(AGENT)).toBe(1);
    expect(heard.length).toBe(before + 1);
    // Consumed for real this time.
    expect(await db.tx((tx) => m.claimConsumption(id, tx))).toBe(true);
    expect(await m.redeliverUnconsumed(AGENT)).toBe(0);
  });

  it("inbox() and claimConsumption stamp the same receipt: one consumes, the other sees nothing", async () => {
    const pollFirst = await arrive("read by the tool");
    const claimFirst = await arrive("read by a woken run");

    // The a2a_inbox tool takes both (it drains what is unread).
    const polled = await m.inbox(AGENT);
    expect(polled.map((e) => e.id).sort()).toEqual([pollFirst, claimFirst].sort());
    // ...so a woken run cannot claim either of them, and nothing is redelivered.
    expect(await db.tx((tx) => m.claimConsumption(claimFirst, tx))).toBe(false);
    expect(await m.redeliverUnconsumed(AGENT)).toBe(0);

    // The other direction: a run consumes first, and the tool sees nothing.
    const consumed = await arrive("claimed before the poll");
    expect(await db.tx((tx) => m.claimConsumption(consumed, tx))).toBe(true);
    expect(await m.inbox(AGENT)).toHaveLength(0);
  });

  it("the receipt is node-scoped: another node's row is not ours to consume", async () => {
    const other = new LocalMessenger(db, { nodeId: `${NODE_C}-x`, peers: {}, a2aSecret: SECRET });
    const id = await arrive("ours");
    // Same row, wrong node: the claim is scoped to node_to, like the delivery
    // claim, so two nodes on one database cannot consume each other's mail.
    expect(await other.claimConsumption(id)).toBe(false);
    expect(await m.claimConsumption(id)).toBe(true);
  });
});
