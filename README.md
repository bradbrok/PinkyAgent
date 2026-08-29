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
| A2A mailbox + cross-node HTTP relay, at-least-once both ways, wake-on-message with consumption receipts | **HITL** — the `human_request` event type exists; nothing raises or resumes it |
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

The `event` lines carry a stored `ThreadEvent`, so their `event.data.type` is
the log's own vocabulary — `message`, `tool_result`, `egress`, `memory`,
`continuity`, `restart`, and **`notice`**, the harness's own turn: the
context-pressure notes the loop injects (`[harness notice] …`) are journaled
before they are pushed into the prompt, so a client sees them on the stream in
the same order the model did.

```jsonc
{"type":"event","threadId":"main","channelId":"jsonl:local","event":{"id":"…","seq":42,"data":{"type":"notice","text":"[harness notice] context pressure: …"}}}
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

### Wake-on-message

stdin is not the only way to reach the agent. While `pinky headless` runs, a
message from a peer (`a2a_send`, or a signed `POST /a2a/deliver` from another
node) wakes a run by itself — no prompt required. It arrives on the same lanes
and reports the same lines, with `"cause":"a2a"` on `run_started`:

```jsonc
{"type":"run_started","threadId":"main","channelId":"a2a:weather@node2","cause":"a2a"}
```

The thread is the *sender's*: `a2a:<agentId@nodeId>` as the channel — so a peer
can be given its own model or budget with `config set --scope
channel:a2a:weather@node2` — and the sender's thread hint (else `main`) as the
thread. The message is journaled as an `a2a` event and projected as a user turn
(`[a2a request from weather@node2]: …`), so the woken run sees what woke it.

**Delivery is not consumption.** `delivered_at` only records that this node
took custody of a message; the receipt is `read_at`, and the consumer stamps it
in the *same transaction* that journals the `a2a` event — so a message counts
as handled only when the work is in the log, and a turn that fails to commit is
handled again rather than lost. Recovery needs no scheduler state: everything
unread is re-fired at startup (before the first stdin line is read) and every
30s afterwards, which is what makes a message that arrived while the process
was down, or during a crash right after delivery, still get answered. Re-firing
is safe because the receipt is the idempotency hinge: a message already
consumed produces no second run. The `a2a_inbox` tool stamps the same receipt,
so polling and waking never double-handle one message.

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
| `stats restarts [--channel <id>] [--limit N]` | What context restarts cost: `tokensBefore → tokensAfter` per boundary, the successor's first-turn cache split, and the total rebuild bill |
| `stats cache [--channel <id>] [--thread <id>] [--limit N]` | The steady-state prompt-cache hit rate over the newest N turns (default 50): per turn `prompt = read + write + uncached` with a `hit` share, `⊘ cold` on a warm→cold transition, then the mean and totals over the measured turns and the "prefix rewritten" count |
| `headless [--shell] [--a2a] [--shared]` | The JSONL service on stdin/stdout, plus [wake-on-message](#wake-on-message) from the A2A mailbox. `--shell` grants `bash`; `--a2a` also opens the relay port (inbound from another *node*); `--shared` drops the trusted-local recall scope |
| `smoke` | In-process end-to-end check: migrate, agent loop, local A2A, memory round trip, event log |
| `prompt "<text>"` | One agent turn on the local `cli:local/main` thread |

`memory`, `smoke`, `prompt` and `headless` auto-migrate at startup on a
short-lived privileged connection; `config` and `stats` do not.

In `stats cache`, `hit n/a` means the route reported no cache counters at all —
not a 0% hit rate — so those turns are counted separately and never averaged in
as zeroes. There is one denominator: the turns with a computable hit share
(`with cache counters N`) carry the mean *and* the token totals, printed as
`tokens over the N measured turns …`. `⊘ cold` is read-counter-only (the
previous turn in that thread read a real cached prefix and this one read
nothing from a prompt big enough to have been cached), so it also fires on
routes that report hits and no writes, like OpenAI and DeepSeek; "prefix
rewritten" needs the write counter and prints `n/a (no turn reported a
cache-write counter)` where nobody counted. `--limit` samples the newest N
turns and transitions are found inside that sample, so a thread's first sampled
turn is never marked cold — widen `--limit`, or pin `--thread`, to see further
back.

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
| `PINKY_ANTHROPIC_CACHE_TTL` | `5m` (default) or `1h`: the lifetime stamped on every `cache_control` breakpoint. `1h` writes cost ~2× an input token instead of ~1.25×, so it pays off exactly when start-to-start wake gaps land in the 5–60 minute band — often enough to reuse the prefix, too rare for the 5m entry to survive |
| `PINKY_LLM_PROMPT_CACHE_KEY` | Boolean: send `prompt_cache_key` (the thread's identity) on the OpenAI-compatible routes so a thread keeps landing on one cache shard. The default is route-dependent, because the field is OpenAI-*native*: on for `openai/` against real api.openai.com, off for `openrouter/` and for any `openai/` with an `OPENAI_BASE_URL` override (vLLM, llama.cpp, LM Studio, Azure — strict servers reject an unknown top-level field). Set it and it wins on every route; a non-boolean throws at startup. It is a routing hint, so losing it costs hit rate, never correctness |
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
| `runtime/test/integration/providers-cache.test.ts` | That prompt caching actually *works* against the live Anthropic API: two byte-identical requests, the second must read the prefix the first wrote |
| `cli/test/integration/headless.test.ts` | `pinky headless` as a real child process: the JSONL contract, and that *nothing* but the protocol reached stdout |
| `cli/test/integration/stats.test.ts` | `pinky stats restarts` and `stats cache` as real child processes: the lateral join from each restart to the turn that paid for it, and the per-turn hit shares, cold transitions and uncounted-turn handling |

One of those is gated twice over: `providers-cache.test.ts` runs only when
`ANTHROPIC_API_KEY` is set as well as `PINKY_INTEGRATION=1`, because it spends
real money — about **$0.02 a run**: two calls on the cheapest model
(`claude-haiku-4-5`, override with `PINKY_CACHE_TEST_MODEL`) that answer in 16
tokens but must send a system prompt padded to ≈13–15k tokens, because a prefix
under the model's minimum cacheable size (4096 on Haiku 4.5) simply never
caches and the failure would look like a bug in the provider. No key is a clean
skip, never a red suite. It is the only check that a cache *hit* happened rather than that the
request looked right, which is what catches a byte that moved in the prefix or
a proxy quietly dropping `cache_control`.

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

### Restart economics

[DESIGN.md §13](./DESIGN.md#13-open-questions) leaves the cost model open —
*restarts discard cache warmth; measure $/task against a compaction baseline* —
so the runtime measures it instead of deferring it. Every rebuild from a
continuity boundary journals an audit-only `restart` event (`boundarySeq`,
`tokensBefore` → `tokensAfter`, `recallTokens`, `messages`), written once per
boundary: by the loop right after a shed, or backfilled by the successor wake
when the shedding run stopped before it could. The bill itself lands on the
next turn, whose `message` event already carries the provider's `usage` — and
on a fresh window that input is nearly all cache *creation*, at ~1.25x an
ordinary input token.

`pinky stats restarts` is the §13 eval as a query rather than a study: one row
per restart with the window it replaced, the recall budget it spent, and the
successor's `input / cacheRead / cacheCreation / output` split, then a summary
line — restarts, mean `tokensAfter`, mean cache-write share of first-turn
input, and Σ `tokensAfter` as the estimated rebuild cost. If the fresh window
starts creeping up, or the stable prefix stops re-warming cheaply, the time
series says so before the invoice does.

```
thread                                      bnd    before      after           change   recall  msgs  first turn
cli:local/main                                3      1092 ->     425      -667 (-61%)        0     1  in 180 read 0 write 1240 out 64

restarts 1  mean tokensAfter 425  mean cache-write share 87% (1/1 turns reported cache usage)  est. rebuild cost 425 tokens
```

That is half the instrument: it prices the *first* turn of a window. What
decides $/task over a whole thread is the steady state between restarts, so
`pinky stats cache` reads the same journaled `usage` across every assistant
turn — `prompt = read + write + uncached` per turn with its `hit` share, `⊘
cold` where a thread that was reading a real cached prefix suddenly read
nothing at all, and a "prefix rewritten" count for turns whose write was ≥80%
of the prompt. The cold marker reads the hit counter alone, so it works on
routes that never report writes; the rewrite count is scored only over the
turns that do, and says `n/a` where none did. A cold transition or a rewrite is
a bug in how the request was
assembled (a tool definition changed, the system prefix moved, `messages[0]`
was re-rendered), not a fact about the workload — which is why both are called
out by name rather than averaged away.

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
