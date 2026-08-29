# PinkyAgent

An always-on agent whose state is an append-only Postgres event log, not a
chat transcript: every prompt the model sees is a *projection* of that log, so a
thread can be replayed, audited, and cut and restarted from a continuity
document instead of being summarised away. It runs as a headless JSONL service
— one command object per stdin line, one event object per stdout line — against
any OpenAI-compatible or Anthropic model, remembers across context restarts
through a scoped memory plane, and can pass messages to other agents on other
machines over a signed HTTP mailbox.

The architecture, and the reasoning behind it, is in **[DESIGN.md](./DESIGN.md)**.

## Status

Early. Slices 1 and 2 of the [build order](./DESIGN.md#12-build-order-mvp-slices),
parts of slice 3 (the continuity engine), and the revised P8 (human-granted
self-configuration):

| Built | Not built yet |
| --- | --- |
| Event log, projection, per-thread seq | **Multi-party ingress** — the stdio pipe is the only front door; no Slack, Discord or webhook gateway |
| JSONL headless service: ingest + dedup in one transaction → one serialized run per thread → every event streamed | **Reply gating** — nothing to gate yet: every prompt on the pipe is addressed to the agent, so the rule cascade and classifier are unbuilt |
| Memory plane v1: scopes, hybrid FTS + pgvector recall, auto-recall at context start, invalidate-never-delete | **Subagents** — no spawn, no fan-out, no depth caps |
| Agent loop with tools: read, write, edit, glob, grep, bash, a2a, recall/retain/memory_edit, settings_get/set | **Sleep-time worker** — no extraction, consolidation, or reflection |
| A2A mailbox + cross-node HTTP relay, at-least-once both ways | **HITL** — the `human_request` event type exists; nothing raises or resumes it |
| Settings table, `config` + `memory` CLI, allow-listed self-configuration, RLS on `memories`, migrations | **Cross-tenant `global` memories** — `global` visibility is still fenced by `tenant_id` in v1 |
| Continuity events + `shed_context` | **Sandboxing** — `bash` strips the environment but is not filesystem-confined |

Bugs the unit suite missed and the integration suite caught are recorded under
[Fixed defects](#fixed-defects-now-regression-tested).

## Quickstart

Needs [Bun](https://bun.sh) 1.4 (see [.bun-version](./.bun-version)) and Docker.

```sh
bun install
cp .env.example .env          # defaults work as-is for local dev
bun run db:up                 # postgres 16 on localhost:5544
bun run migrate               # applies packages/core/schema/*.sql
bun run smoke                 # end-to-end check, no API key needed
```

`smoke` runs the real agent loop against a scripted fake provider — A2A round
trip, retain → recall, auto-recall on a fresh thread — and prints a PASS line
per check.

The service itself is `bun run headless`: it reads commands from stdin and
writes events to stdout for as long as the pipe is open. There is a keyless
model route (`fake/echo`, which replies with the text you sent), so you can
watch the whole stack run before you own an API key:

```sh
bun run packages/cli/src/index.ts config set model '"fake/echo"'
echo '{"type":"prompt","text":"hi"}' | bun run headless
```

```jsonc
{"type":"ready","nodeId":"local","agentId":"pinky","tenantId":"default","defaultModel":"fake/echo"}
{"type":"run_started","threadId":"main","channelId":"jsonl:local"}
{"type":"event","threadId":"main","channelId":"jsonl:local","event":{"id":"…","seq":2,"data":{"type":"message","role":"assistant","text":"echo: [jsonl local]: hi",…}}}
{"type":"reply","threadId":"main","channelId":"jsonl:local","text":"echo: [jsonl local]: hi"}
{"type":"event","threadId":"main","channelId":"jsonl:local","event":{"id":"…","seq":3,"data":{"type":"egress",…}}}
{"type":"run_finished","threadId":"main","channelId":"jsonl:local","stopReason":"completed","turns":1}
{"type":"exiting"}
```

To talk to a real model, set a key in `.env` and pick one:

```sh
bun run packages/cli/src/index.ts config set model '"openrouter/moonshotai/kimi-k2"'
bun run packages/cli/src/index.ts prompt "summarise DESIGN.md section 4"
```

## JSONL protocol

`pinky headless` speaks JSON Lines: one command per stdin line, one event per
stdout line. **stdout is protocol-only** — every human-facing line (warnings,
Postgres notices, A2A sweep logs) goes to stderr, so a client can pipe stdout
straight into a parser.

**Commands** (stdin → service). Every field except `type` and `prompt.text` is
optional; the defaults are shown.

```jsonc
{"type":"prompt","text":"…","threadId":"main","channelId":"jsonl:local",
 "id":"<dedup key>","author":{"userId":"local","displayName":"Brad"}}
{"type":"abort","threadId":"main"}      // cancels the run in flight on that threadId AND the ones queued behind it
{"type":"exit","abort":false}           // finish in-flight and queued runs (true: cancel them all), then close
```

`id` is the ingest dedup key; omit it and one is generated (`jsonl:<uuid>`).
Re-sending an id depends on what that thread is doing: if a run is in flight or
queued for it, the resend is refused with an `error` line (answering twice
would double the reply, and a duplicate there is usually a client bug); if
nothing is running it, the prompt is *replayed* — its `run_started` carries
`"replay":true` — because an ingress that was persisted but never answered (a
process that died between ingest and run) would otherwise be unrecoverable.
A malformed line answers `{"type":"error","message":…,"line":"<first 200 chars>"}`
and the session continues; a line over 1 MiB is dropped whole
(`{"type":"error","message":"line exceeds 1048576 bytes; dropped"}`) and the
framer resumes at the next newline. Stdin EOF is an exit, and so is losing
stdout: a client that closes the pipe ends the session like `exit --abort`
(in-flight runs cancelled, queued runs dropped) and the process exits 0.

**Events** (service → stdout):

```jsonc
{"type":"ready","nodeId":"local","agentId":"pinky","tenantId":"default","defaultModel":"…"}
{"type":"run_started","threadId":"main","channelId":"jsonl:local"}
{"type":"event","threadId":"main","channelId":"jsonl:local","event":{…ThreadEvent as stored}}
{"type":"reply","threadId":"main","channelId":"jsonl:local","text":"…"}
{"type":"run_finished","threadId":"main","channelId":"jsonl:local","stopReason":"completed","turns":3}
{"type":"error","threadId":"main","channelId":"jsonl:local","run":"failed","message":"…"}
{"type":"exiting"}
```

`ready.defaultModel` is the bootstrap snapshot and nothing more: every run
re-resolves the model from `channel:<id>` + `agent:pinky` settings, so a
particular run may use another one — read the run's `event` lines if you need
the model that actually answered.

Ordering is guaranteed **per thread**, not globally: `run_started → (event |
reply)* → run_finished`. Runs on one `(channelId, threadId)` are serialized in
arrival order; different threads run concurrently and their lines interleave,
which is why every run line carries both ids. The `ingress` event is written by
`ingest` before the run starts, so it is not among the streamed `event` lines —
everything the loop itself appends is.

Every enqueued prompt produces exactly one closing line, so a client can
balance its own accounting: `run_finished`, or an `error` carrying
`"run":"failed"` when the run threw (the thread stays usable either way).
`error` lines *without* that tag close no run — a bad command line, a failed
ingest, a refused duplicate, an oversized line. A run cancelled before it ever
reached the agent (an `abort` while it was queued, or `exit --abort`) still
reports `run_started` then `run_finished` with `"stopReason":"aborted"` and
`"turns":0`.

## CLI

There is no installed `pinky` binary yet — run the entry point directly.
`bun run migrate|headless|smoke` are shortcuts for the three most common.

```sh
bun run packages/cli/src/index.ts <command>
```

| Command | What it does |
| --- | --- |
| `migrate` | Apply pending schema migrations (uses `DATABASE_ADMIN_URL`) |
| `config get [key] [--scope S]` | Print the effective settings snapshot, or one dotted key |
| `config set <key> <json> [--scope S]` | Write one setting. The value is parsed as JSON, so strings need quotes |
| `config unset <key> [--scope S]` | Delete that scope's row for the key. Prints `nothing to unset` when there was none |
| `memory list [--scope-channel <id>] [--limit N] [--all]` | Newest-first listing of what this agent remembers (`--all` includes invalidated rows) |
| `memory search "<q>" [--limit N]` | Hybrid recall (FTS, plus the vector voice where pgvector and a key exist) |
| `memory show <id-or-prefix>` | One memory in full: scope, importance, validity, embedder, meta |
| `memory forget <id-or-prefix> [--reason "…"]` | Invalidate it — memories are retired, never deleted |
| `headless [--shell] [--a2a] [--shared]` | The JSONL service on stdin/stdout. `--shell` grants `bash`; `--a2a` also opens the relay port; `--shared` drops the trusted-local recall scope |
| `smoke` | In-process end-to-end check: migrate, agent loop, local A2A, memory round trip, event log |
| `prompt "<text>"` | One agent turn on the local `cli:local/main` thread |

`memory`, `smoke`, `prompt` and `headless` auto-migrate at startup on a
short-lived privileged connection; `config` does not.

`headless` runs as a **trusted local surface** by default: `user`- and
`private`-visibility memories are recallable, and the subject user is whatever
the client puts in `author`. That is right for a program you wrote driving your
own agent. Pass `--shared` when the driving program bridges several people —
without it, one party's claimed identity would recall another party's `user`
memories (and everyone's prompt the agent's `private` ones) into a shared
thread, which [DESIGN.md §5.1](./DESIGN.md#51-scopes-and-isolation) says never
happens.

## Configuration

Two layers, deliberately (DESIGN.md P8 — *the DB is the only config*):

**Environment** bootstraps the process only. Copy [.env.example](./.env.example),
which documents every variable:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | The connection everything runs on. Should be the unprivileged `pinky_app` role in production |
| `DATABASE_ADMIN_URL` | Superuser, migrations only (DDL + `CREATE ROLE`). Defaults to `DATABASE_URL` |
| `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Model credentials, plus optional `*_BASE_URL` overrides. The OpenAI and OpenRouter keys also pay for **embeddings** |
| `PINKY_LLM_MAX_RETRIES` / `PINKY_LLM_TIMEOUT_MS` / `PINKY_LLM_INCLUDE_USAGE` | LLM transport hardening (not agent behavior, hence env) |
| `PINKY_EMBED_TIMEOUT_MS` / `PINKY_EMBED_MAX_RETRIES` / `PINKY_EMBED_DIMENSIONS` | Embedding transport: a short budget (15s × 2) so a hung endpoint degrades recall to FTS-only instead of stalling wakes; dimensions pinned to the `vector(1536)` column (0 = omit the field) |
| `PINKY_NODE_ID` / `PINKY_PEERS` / `A2A_SECRET` / `PORT` | A2A identity, peer routing table, shared HMAC key, relay port |
| `PINKY_INTEGRATION` | Set to `1` to un-skip the integration tests |

**Settings** hold all behavior — `tenantId`, `model`, context thresholds, the
memory plane — in the `settings` table. Config never lives in a file, and that
is the point: every value is validated before it is written, so nothing that
lands there can stop the process from starting.

Scopes overlay `global < channel:<id> < agent:<id>`, and only the scopes the
caller asks for are read, so one channel's row never leaks into another's. Keys
are field names or dotted sub-paths; a whole sub-tree replaces, a dotted key
refines:

```sh
bun run packages/cli/src/index.ts config set context.hardFraction 0.85
bun run packages/cli/src/index.ts config set model '"anthropic/claude-sonnet-4-5"' --scope agent:pinky
bun run packages/cli/src/index.ts config get --scope agent:pinky
```

Bad values are rejected before the write, with the offending key named. And in
the other direction, a row that is already bad — a key that no longer exists, a
value an older build wrote — is pruned by `load()` with a warning on stderr
rather than throwing, so a single damaged row can never brick a wake; `config
unset <key> --scope S` is how you then remove it for good.

**Memory keys** ([DESIGN.md §5](./DESIGN.md#5-the-memory-plane)):

| Key | Default | Meaning |
| --- | --- | --- |
| `memory.embeddingModel` | `"openai/text-embedding-3-small"` | Embedder as `provider/model-id`, or `"none"` for FTS-only recall (and no embedding on retain) |
| `memory.autoRecall` | `true` | Inject the token-budgeted `<memories>` block at context start and after each restart |
| `memory.recallLimit` | `12` | Candidates auto-recall asks for, before the budget cut |
| `memory.recallTokenBudget` | `5000` | Approximate token ceiling for that block |

Embeddings are optional everywhere. With no matching API key the process warns
once on stderr and recall runs on the lexical voice alone; on a Postgres
without pgvector the `embedding` column never exists and the same thing
happens. Both are supported modes, not failures.

**Self-configuration keys** — P8 as revised. The human CLI is still the default
write path, and config is still never a file. What changed is that an agent may
now adjust *allow-listed* keys through the validated `settings_set` tool: the
value is checked before the write, so a mistake is a tool error the agent reads
and retries rather than a table that breaks the next boot, and every write is
journaled as an audit-only `config` event. Off by default:

| Key | Default | Meaning |
| --- | --- | --- |
| `selfConfig.enabled` | `false` | Master switch. Only a human can flip it |
| `selfConfig.allowedKeys` | `[]` | Exact keys (`"model"`), sub-tree patterns (`"context.*"`), or `"*"`. Empty grants nothing |

```sh
bun run packages/cli/src/index.ts config set selfConfig.enabled true
bun run packages/cli/src/index.ts config set selfConfig.allowedKeys '["model","context.*"]'
# or delegate to one agent only:
bun run packages/cli/src/index.ts config set selfConfig.enabled true --scope agent:pinky
```

Delegating `model` does not delegate *every* model string: `settings_set`
refuses a provider this build cannot route (naming the ones it can) and refuses
`fake/*` outright, since the scripted test route would validate cleanly and
turn the agent into an echo bot on its next wake. A human can still set either
with `config set`. The write is also validated against the whole overlay the
run reads (channel **and** agent), not just the scope it lands in, so a value
that is only valid in isolation cannot assemble into a broken snapshot.

`tenantId` and `selfConfig` itself are never agent-writable, even under `"*"`
— an agent that could widen its own allow-list would not have one — and the
tool can only write `agent:<self>` or `channel:<current>`, never `global`.
Writes land in the table, not in the running snapshot: settings are re-read per
run, so a change takes effect on the next one.

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
under `packages/{core,runtime,cli}/test/integration/`, gated on
`PINKY_INTEGRATION=1`:

| File | Covers |
| --- | --- |
| `core/test/integration/rls.test.ts` | Tenant isolation actually enforced by Postgres, as `pinky_app` |
| `core/test/integration/event-store.test.ts` | Concurrent appends, the continuity boundary, the context cap, dedup |
| `core/test/integration/settings.test.ts` | The jsonb round-trip and scope overlay |
| `core/test/integration/migrate.test.ts` | Migrations into a throwaway database, then re-run as a no-op |
| `core/test/integration/memory.test.ts` | Scope predicate, FTS, one-transaction `update()`, RLS, and the `::vector` voice |
| `runtime/test/integration/messenger.test.ts` | Two nodes, one HMAC-signed socket, replay, offline retry, and `bun run smoke` |
| `cli/test/integration/headless.test.ts` | `pinky headless` as a real child process: the JSONL contract, and that *nothing* but the protocol reached stdout |

The vector half of recall needs pgvector, which the default alpine server does
not have, so it runs against a second server on 5545:

```sh
bun run db:up:vector
DATABASE_URL=postgres://postgres:pinky@localhost:5545/pinky \
DATABASE_ADMIN_URL=postgres://postgres:pinky@localhost:5545/pinky bun run migrate
bun run test:integration:vector
```

Neither branch is reachable on the other image — 5544 exercises "no pgvector:
retain drops the embedding, search falls back to FTS", 5545 exercises both
voices — so a full local check is one run per server.

`bun test` on its own reports the integration files as *skipped*, so the
hermetic suite stays hermetic. `bun run smoke` is the fastest single check that
the whole stack is wired up.

CI ([.github/workflows/ci.yml](./.github/workflows/ci.yml)) runs typecheck and
the unit suite on every push, then the integration suite against a
`pgvector/pgvector:pg16` service — which is where the vector voice and
`0002_embeddings.rerun.sql` are executed on every push.

## Fixed defects (now regression-tested)

Each of these was found by the integration suite, shipped green under the unit
suite, and now has a live test guarding it — the first three named `DEFECT:`.
Nothing here is outstanding.

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
- **Postgres NOTICEs printed to stdout, which is the JSONL stream.**
  `createDb`'s `onnotice` handler used `console.log`, and every runtime command
  auto-migrates at startup — so the first `pinky headless` run on a fresh
  database emitted `notice: pgvector unavailable …` as an unparseable line in
  the middle of the protocol. Invisible to `gateway/test/headless.test.ts`,
  which injects its own `write`; caught by the e2e test that spawns the real
  CLI and asserts every stdout line parses as JSON. Notices now go to stderr
  with the rest of the human output; that whole-file assertion in
  `cli/test/integration/headless.test.ts` is the guard.

## Security notes

- **`bash` is opt-in and environment-stripped.** `createTools()` omits it
  unless `{ shell: true }`. The local `prompt` surface opts in, because that is
  a human at their own terminal; `pinky headless` does not unless `--shell`,
  because it is driven by another program. Even then the child gets an explicit
  minimal env (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`, `TMPDIR`), so
  `DATABASE_URL` and the API keys are never one tool call away from the agent —
  which is also what keeps the settings table reachable only through
  `settings_set` and its allow-list, rather than through a `psql` the agent
  shelled out to. It is not filesystem-sandboxed — that is slice 8.
- **Agent self-configuration is off by default and allow-listed.**
  `selfConfig.enabled` is `false` and `allowedKeys` is empty until a human says
  otherwise; `tenantId`, `selfConfig` itself and the `global` scope are refused
  regardless. Values are validated *before* the write, so a rejected one never
  lands, and every accepted one is journaled as a `config` event.
- **A2A refuses to run unauthenticated.** An HMAC over an empty key is still a
  valid HMAC, so a missing secret is not "auth off", it is "auth forged":
  `A2A_SECRET` becomes mandatory as soon as `PINKY_PEERS` names a peer, and
  with no secret at all `POST /a2a/deliver` answers 503 rather than accepting
  every unsigned envelope. `pinky headless --a2a` opens no listener when the
  secret is blank.
- **RLS covers `memories` only, and only under `pinky_app`.** The policy is
  `FORCE`d and keys off the transaction-local `pinky.tenant_id` GUC that
  `withTenant()` sets; superusers bypass RLS unconditionally, so the default
  single-url dev setup enforces nothing — which is why `MemoryStore` also
  writes `tenant_id` explicitly and repeats it in every `WHERE`. `events`,
  `threads`, `a2a_messages` and `ingress_dedup` carry `tenant_id` but are
  app-filtered only.
- **A2A is HMAC-signed and replay-windowed.** `X-Pinky-Signature` is
  `HMAC-SHA256(secret, "<id>.<sentAt>.<body>")`, rejected outside a 300s
  freshness window, and delivery is idempotent on the envelope id so a sender
  retry is a 200 with no second row.

## Layout

```
packages/core      event log, memory plane, mailbox, settings, migrations, db + tenant scoping
packages/runtime   agent loop, providers, embedders, continuity, auto-recall, A2A messenger
packages/tools     read / write / edit / glob / grep / bash / a2a / recall / retain / memory_edit / settings
packages/gateway   JSONL headless protocol, A2A relay
packages/cli       pinky — the human-owned control surface
```
