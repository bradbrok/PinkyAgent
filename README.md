# PinkyAgent

An always-on chat agent whose state is an append-only Postgres event log, not a
chat transcript: every prompt the model sees is a *projection* of that log, so a
thread can be replayed, audited, and cut and restarted from a continuity
document instead of being summarised away. It speaks Slack today, runs the agent
loop against any OpenAI-compatible or Anthropic model, and can pass messages to
other agents on other machines over a signed HTTP mailbox.

The architecture, and the reasoning behind it, is in **[DESIGN.md](./DESIGN.md)**.

## Status

Early. Slice 1 of the [build order](./DESIGN.md#12-build-order-mvp-slices),
parts of slice 3 (the continuity engine), and the deterministic half of slice 4:

| Built | Not built yet |
| --- | --- |
| Event log, projection, per-thread seq | **Memory plane** — the `memories` table and its RLS exist, but nothing writes to or recalls from it, and there are no embeddings |
| Slack gateway: verify → persist → dedup → gate → debounce → one run | **Reply classifier** — only the rule cascade (bot / mention / DM / reply-to-agent / ambient) |
| Agent loop with tools: read, write, edit, glob, grep, bash, a2a | **Subagents** — no spawn, no fan-out, no depth caps |
| A2A mailbox + cross-node HTTP relay, at-least-once both ways | **Sleep-time worker** — no extraction, consolidation, or reflection |
| Settings table + `config` CLI, RLS on `memories`, migrations | **HITL** — the `human_request` event type exists; nothing raises or resumes it |
| Continuity events + `shed_context` | **Sandboxing** — `bash` strips the environment but is not filesystem-confined |

See [Known issues](#known-issues) before trusting `pinky config set`.

## Quickstart

Needs [Bun](https://bun.sh) 1.4 (see [.bun-version](./.bun-version)) and Docker.

```sh
bun install
cp .env.example .env          # defaults work as-is for local dev
bun run db:up                 # postgres 16 on localhost:5544
bun run migrate               # applies packages/core/schema/*.sql
bun run smoke                 # end-to-end check, no API key needed
```

`smoke` runs the real agent loop against a scripted fake provider and prints a
PASS line per check. To talk to a real model, set a key in `.env` and pick a
model:

```sh
bun run packages/cli/src/index.ts config set model '"openrouter/moonshotai/kimi-k2"'
bun run packages/cli/src/index.ts prompt "summarise DESIGN.md section 4"
```

To run the Slack gateway you need a Slack app's `SLACK_BOT_TOKEN` and
`SLACK_SIGNING_SECRET` in `.env`; it refuses to start without them.

```sh
bun run gateway               # listens on $PORT (default 3000)
```

> `gateway` and `prompt` are the two commands not exercised in this repo's
> checks: one needs Slack credentials, the other an LLM API key.

## CLI

There is no installed `pinky` binary yet — run the entry point directly.
`bun run migrate|gateway|smoke` are shortcuts for the three most common.

```sh
bun run packages/cli/src/index.ts <command>
```

| Command | What it does |
| --- | --- |
| `migrate` | Apply pending schema migrations (uses `DATABASE_ADMIN_URL`) |
| `config get [key] [--scope S]` | Print the effective settings snapshot, or one dotted key |
| `config set <key> <json> [--scope S]` | Write one setting. The value is parsed as JSON, so strings need quotes |
| `gateway` | Run the Slack + A2A HTTP server |
| `smoke` | In-process end-to-end check: migrate, agent loop, local A2A, event log |
| `prompt "<text>"` | One agent turn on the local `cli:local/main` thread |

`gateway`, `smoke` and `prompt` auto-migrate at startup on a short-lived
privileged connection.

## Configuration

Two layers, deliberately (DESIGN.md P8 — *the DB is the only config; agents
cannot self-lobotomize*):

**Environment** bootstraps the process only. Copy [.env.example](./.env.example),
which documents every variable:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | The connection everything runs on. Should be the unprivileged `pinky_app` role in production |
| `DATABASE_ADMIN_URL` | Superuser, migrations only (DDL + `CREATE ROLE`). Defaults to `DATABASE_URL` |
| `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Model credentials, plus optional `*_BASE_URL` overrides |
| `PINKY_LLM_MAX_RETRIES` / `PINKY_LLM_TIMEOUT_MS` / `PINKY_LLM_INCLUDE_USAGE` | LLM transport hardening (not agent behavior, hence env) |
| `PINKY_NODE_ID` / `PINKY_PEERS` / `A2A_SECRET` | A2A identity, peer routing table, shared HMAC key |
| `SLACK_BOT_TOKEN` / `SLACK_SIGNING_SECRET` / `PORT` | Slack gateway |
| `PINKY_INTEGRATION` | Set to `1` to un-skip the integration tests |

**Settings** hold all behavior — `tenantId`, `model`, context thresholds, the
reply gate — in the `settings` table, written only by the human-run CLI. The
runtime reads a validated snapshot per wake and has no write path.

Scopes overlay `global < channel:<id> < agent:<id>`, and only the scopes the
caller asks for are read, so one channel's row never leaks into another's. Keys
are field names or dotted sub-paths; a whole sub-tree replaces, a dotted key
refines:

```sh
bun run packages/cli/src/index.ts config set context.hardFraction 0.85
bun run packages/cli/src/index.ts config set model '"anthropic/claude-sonnet-4-5"' --scope agent:pinky
bun run packages/cli/src/index.ts config get --scope agent:pinky
```

Bad values are rejected before the write, with the offending key named.

## Testing

```sh
bun test                 # unit suite — no database, no network, ~1.3s
bun run typecheck        # tsc --noEmit over src and test
bun run db:up            # required for the next one
bun run test:integration # live Postgres, live HTTP, live subprocess
```

The unit suite is entirely fakes. That is fast and it is also how a real
cross-node mailbox bug shipped green (see *Fixed defects* below), so the SQL,
the jsonb round-trip, the wire format and the CLI path have their own suite
under
`packages/*/test/integration/`, gated on `PINKY_INTEGRATION=1`:

| File | Covers |
| --- | --- |
| `core/test/integration/rls.test.ts` | Tenant isolation actually enforced by Postgres, as `pinky_app` |
| `core/test/integration/event-store.test.ts` | Concurrent appends, the continuity boundary, the context cap, dedup |
| `core/test/integration/settings.test.ts` | The jsonb round-trip and scope overlay |
| `core/test/integration/migrate.test.ts` | Migrations into a throwaway database, then re-run as a no-op |
| `runtime/test/integration/messenger.test.ts` | Two nodes, one HMAC-signed socket, replay, offline retry, and `bun run smoke` |
| `gateway/test/integration/gateway.test.ts` | Signed Slack event → events table → one debounced agent run |

`bun test` on its own reports these as *skipped*, so the hermetic suite stays
hermetic. `bun run smoke` is the fastest single check that the whole stack is
wired up.

CI ([.github/workflows/ci.yml](./.github/workflows/ci.yml)) runs typecheck and
the unit suite on every push, then the integration suite against a
`pgvector/pgvector:pg16` service.

## Fixed defects (now regression-tested)

Each of these was found by the integration suite, shipped green under the unit
suite, and now has a live test named `DEFECT:` guarding it. Nothing here is
outstanding.

- **jsonb values were encoded twice on write.** Callers bound
  `JSON.stringify(value)` and postgres.js applied its own jsonb serializer on
  top, so documents landed as jsonb *strings*: `jsonb_typeof(events.data)` was
  `string` and `data->>'type'` `NULL` for every row, and
  `config set model '"openrouter/x/y"'` read back as `"\"openrouter/x/y\""`.
  jsonb params now take the **plain value** (the contract is documented at the
  top of `core/src/pg.ts`); `jsonbParam()` rewraps only the JS types postgres.js
  would tag with a non-jsonb wire type — a bare boolean, which Postgres refuses
  to coerce to jsonb at all. Rows written by the old code are repaired in place
  by the re-runnable `core/schema/0004_jsonb_repair.rerun.sql`. See the
  `DEFECT:` tests in `settings.test.ts` and `event-store.test.ts`.
- **`appendBatch` assigned non-contiguous seqs** (1, 11, 111). Postgres returns
  `bigint` as a JS string, so `nextSeq += 1` concatenated, and the same
  string-typed `seq` made `buildContext`'s `e.seq >= boundarySeq` filter
  lexicographic once a thread passed ten events. `toSeq()` now coerces at the
  boundary. See `event-store.test.ts`.
- **Cross-node delivery never woke a live subscriber when both nodes shared one
  database** (a supported topology — `PINKY_NODE_ID=node2` against the same
  `DATABASE_URL`). The sender's durable row was the row the receiver tried to
  insert, so `putIfAbsent` reported a duplicate and `receive()` returned before
  firing; the message still reached `inbox()`, so it looked fine until you
  relied on the wake. Idempotency now hinges on a delivery **claim**
  (`Mailbox.claimDelivery`: `update ... where id = $1 and node_to = <me> and
  delivered_at is null returning id`) rather than on row existence, which
  behaves identically for separate databases, a shared database, and replays.
  See `messenger.test.ts`.


## Security notes

- **`bash` is opt-in and environment-stripped.** `createTools()` omits it
  unless `{ shell: true }`; the Slack-reachable gateway does not opt in, the
  local `prompt` surface does. Even then the child gets an explicit minimal env
  (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`, `TMPDIR`), so `DATABASE_URL` and
  the API keys are never one tool call away from the agent. It is not
  filesystem-sandboxed — that is slice 8.
- **The gateway refuses to start without its secrets.** An HMAC over an empty
  key is still a valid HMAC, so a missing `SLACK_SIGNING_SECRET` is not "auth
  off", it is "auth forged". `A2A_SECRET` becomes mandatory as soon as
  `PINKY_PEERS` names a peer.
- **RLS covers `memories` only, and only under `pinky_app`.** The policy is
  `FORCE`d and keys off the transaction-local `pinky.tenant_id` GUC that
  `withTenant()` sets; superusers bypass RLS unconditionally, so the default
  single-url dev setup enforces nothing. `events`, `threads`, `a2a_messages`
  and `ingress_dedup` carry `tenant_id` but are app-filtered only.
- **A2A is HMAC-signed and replay-windowed.** `X-Pinky-Signature` is
  `HMAC-SHA256(secret, "<id>.<sentAt>.<body>")`, rejected outside a 300s
  freshness window, and delivery is idempotent on the envelope id so a sender
  retry is a 200 with no second row.

## Layout

```
packages/core      event log, mailbox, settings, migrations, db + tenant scoping
packages/runtime   agent loop, providers, continuity, A2A messenger
packages/tools     read / write / edit / glob / grep / bash / a2a
packages/gateway   Slack ingress, reply gate, lane debounce, A2A relay
packages/cli       pinky — the human-owned control surface
```
