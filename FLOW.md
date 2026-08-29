# PinkyAgent — agent flow

How one prompt travels from a client program to a reply, and what the agent
loop does in between. Diagrams follow the code as of the prompt-cache alignment
work; the file paths in the notes are where each box lives. DESIGN.md is still
the spec — this is the map of what is built.

## 1. End to end: JSONL client → headless service → agent loop → reply

```mermaid
sequenceDiagram
    autonumber
    participant C as Client program<br/>(stdin / stdout)
    participant H as pinky headless<br/>gateway/headless.ts
    participant ES as EventStore<br/>core/event-store.ts
    participant L as runAgentLoop<br/>runtime/loop.ts
    participant M as Memory plane<br/>core/memory.ts
    participant P as LLM provider<br/>runtime/providers/*
    participant T as Tools<br/>packages/tools

    H-->>C: {"type":"ready", defaultModel, nodeId, ...}
    C->>H: {"type":"prompt","text":..., "threadId"?, "channelId"?, "id"?}
    H->>H: parseCommand (bad line → {"type":"error"} and continue)
    H->>ES: ingest(thread, id, [ingress]) — dedup claim + ingress event in ONE tx
    alt id already seen and a run is pending
        H-->>C: {"type":"error", "message":"duplicate id ..."}
    else new (or replay of an idle thread)
        H->>H: enqueue on the (channelId, threadId) lane — serialized per thread
        H-->>C: {"type":"run_started"}
        H->>L: runAgent(thread, batch, {signal, onEvent, deliver})
        Note over L: settings reloaded for this run:<br/>global, then channel:id, then agent:pinky
        L->>ES: contextEvents(thread) — from the latest continuity boundary
        L->>L: buildContext → projection (ingress/a2a/notice/message/tool_result/continuity,<br/>tool args canonicalized, plus the journaled recall block hoisted to index 0)
        opt memory.autoRecall, and no recall event in this window carries a block key
            L->>M: search(query from last user texts + continuity memoryHints)
            M-->>L: hits (FTS + vector, RRF-fused, recency/importance rescored)
            L->>L: memories block inserted as a user message at index 0 (never the system prompt)<br/>journaled with its scope on the memory recall event, so every later wake replays it verbatim
        end
        loop each turn (≤ maxTurns)
            L->>L: pressure ladder: advisory once per window / forced shed_context — each notice journaled, then pushed
            L->>P: complete(system, messages, full tool list, cacheKey — tool_choice only on a forced retry)
            P-->>L: AssistantTurn {text, toolCalls, usage}
            L->>ES: append message event (+usage)
            ES-->>H: onEvent → {"type":"event", event}
            H-->>C: streamed
            alt no tool calls
                L->>H: deliver(text)
                H-->>C: {"type":"reply","text":...}
                L->>ES: append egress event
            else tool calls (shed_context always runs LAST)
                L->>T: execute(args, ctx{db, thread, memory, settings, messenger})
                T-->>L: ToolResult
                L->>ES: append tool_result event
            end
        end
        L-->>H: {turns, stopReason}
        H-->>C: {"type":"run_finished","stopReason","turns"}
    end
    C->>H: {"type":"exit"}  (or EOF)
    H-->>C: {"type":"exiting"}
```

Every event the loop appends is streamed live through `onEvent`; the `ingress`
written by `ingest` predates the run and is not re-streamed. Ordering is
guaranteed per thread: `run_started → (event | reply)* → run_finished`.

## 2. Inside one run: the turn cycle and the continuity restart

```mermaid
flowchart TD
    start([run starts]) --> load[loadContext: events since the latest continuity event<br/>buildContext → projection<br/>non-empty journaled recall block hoisted to index 0, notices in seq order<br/>tool args canonicalized on both sides of the log]
    load --> journaled{"windowRecall: does this window already<br/>carry a recall event with a block key?"}
    journaled -- "yes, memory on, scope not narrower" --> opened
    journaled -- "no key, or memory off, or narrower scope than journaled" --> recall{memory.autoRecall<br/>and memory context?}
    recall -- yes --> block["autoRecall: FTS + vector search, budgeted<br/>&lt;memories&gt; block → user message at index 0<br/>journaled with block + scope on the memory recall event<br/>found nothing → empty block, still claims the window<br/>store failed → nothing journaled, retried next wake"]
    recall -- no --> opened
    block --> opened{window opens on a<br/>continuity boundary?}
    opened -- yes --> restart0[emit restart event<br/>tokensBefore / tokensAfter / recallTokens]
    opened -- no --> ladder
    restart0 --> ladder

    ladder{estimateTokens vs<br/>context.* thresholds}
    ladder -- "≥ hardFraction" --> forcing{"first forced attempt, or the retry?<br/>full tool list kept either way"}
    ladder -- "context cap truncated the window" --> forcing
    forcing -- first --> hard1["append notice event, then push HARD notice as user msg<br/>NO tool_choice — appending keeps the messages cache warm;<br/>the note plus the harness guard hold the boundary"]
    forcing -- "retry — last attempt" --> hard2["append notice event, then push HARD RETRY notice as user msg<br/>tool_choice: shed_context — the guarantee, paid for with<br/>one uncached re-read of the transcript"]
    ladder -- "≥ advisoryFraction — once per window, armed from the log" --> adv[append notice event, then push ADVISORY notice as user msg]
    ladder -- below --> llm
    hard1 --> llm
    hard2 --> llm
    adv --> llm

    llm[provider.complete<br/>system prompt = stable cached prefix, never rewritten] --> journal[append message event<br/>with usage: input / output / cacheRead / cacheCreation]
    journal --> calls{tool calls?}

    calls -- none --> deliver[deliver text → reply line<br/>append egress event]
    deliver --> forced{forced shed turn?}
    forced -- no --> done([completed])
    forced -- yes --> miss[count forced-shed miss]
    miss --> giveup{2 misses?}
    giveup -- yes --> failed([shed_failed])
    giveup -- no --> ladder

    calls -- some --> order[sort: shed_context runs LAST<br/>cut-point safety]
    order --> exec[execute each tool<br/>append tool_result per call]
    exec --> tools

    subgraph tools [tools available to the agent]
        direction LR
        fs[read / write / edit / glob / grep<br/>sandboxed under cwd]
        bash[bash — opt-in only<br/>prompt, or headless --shell]
        mem[recall / retain / memory_edit<br/>→ memory plane, memory events]
        cfg[settings_get / settings_set<br/>human allow-listed, validated before write,<br/>config event]
        a2a[a2a_send / a2a_inbox<br/>→ mailbox]
        shed[shed_context<br/>validates ContinuityDoc → continuity event]
    end

    tools --> shedq{shed_context<br/>succeeded?}
    shedq -- no --> more{turns left?}
    more -- yes --> ladder
    more -- no --> maxed([max_turns])
    shedq -- yes --> rebuild[NEW BOUNDARY: reload projection from the continuity event<br/>re-run autoRecall with the memoryHints in the query<br/>emit restart event]
    rebuild --> more2{turns left?}
    more2 -- yes --> ladder
    more2 -- no --> paused([shed — resumes on next wake])
```

Nothing in the log is ever rewritten: a restart moves the projection boundary
forward, and everything before it stays for audit, replay, memory extraction
and `pinky stats restarts`.

Everything the loop pushes into `messages` it journals first — the pressure
notices as `notice` events, the recalled block on its `memory` recall event —
so the next wake's projection reproduces the same bytes in the same slots and
the request stays a prefix-extension of the one the provider already cached.
The same reason the loop and the projection both canonicalize tool-call
arguments: jsonb re-sorts an object's keys, and a replayed `tool_use` that
differs by one byte breaks the match from there to the end of the transcript.
`pinky stats cache` is where you check that it held.

## 3. Wake-on-message: an A2A envelope becomes a run

```mermaid
sequenceDiagram
    autonumber
    participant S as Sender agent<br/>(a2a_send tool, any node)
    participant MB as Mailbox<br/>a2a_messages
    participant R as Relay<br/>POST /a2a/deliver (peer nodes)
    participant MS as LocalMessenger<br/>runtime/messenger.ts
    participant H as pinky headless<br/>wake source (cli)
    participant ES as EventStore
    participant L as runAgentLoop

    S->>MB: put(envelope) — durable first, at-least-once
    alt recipient on another node
        S->>R: HMAC-signed envelope (retried by the 30s sweep until accepted)
        R->>MS: receive(env)
    end
    MS->>MB: claimDelivery — delivered_at = now() where delivered_at is null
    Note over MB: delivered_at = "this node accepted it"<br/>relay idempotency, NOT proof of consumption
    MS->>H: fire subscriber (onMessage)
    H->>ES: ONE tx: claimRead (read_at = now() where read_at is null)<br/>+ append a2a event to thread a2a:sender / threadHint
    Note over ES: read_at is the receipt — stamped with the work,<br/>so a crash before commit leaves it unread
    alt claim won
        H->>L: enqueue run (run_started cause:"a2a")
        L->>L: projection renders the a2a event as a user message
    else already consumed
        H->>H: no-op (a redelivery landed twice)
    end

    Note over H,MB: Recovery: at startup and every 30s,<br/>redeliverUnconsumed(agent) re-fires every row with read_at null —<br/>regardless of delivered_at. Same rule for any future timer:<br/>emit an event, receipt the consumption, never mark "fired" scheduler-side.
```

## Where the pieces live

| Box | File |
| --- | --- |
| JSONL protocol, lanes, wake seam | `packages/gateway/src/headless.ts` |
| Bootstrap, per-run settings reload, wake wiring, `pinky` commands | `packages/cli/src/index.ts` |
| Ingest (dedup + append in one tx), projection window | `packages/core/src/event-store.ts` |
| Projection rules (what the model sees) | `packages/core/src/projection.ts` |
| Turn cycle, pressure ladder, restart | `packages/runtime/src/loop.ts` |
| Auto-recall block | `packages/runtime/src/memory-recall.ts` |
| `shed_context` + ContinuityDoc validation | `packages/runtime/src/continuity.ts` |
| Memory store (scopes, FTS + vector, fusion) | `packages/core/src/memory.ts` |
| Mailbox receipts, messenger delivery/redelivery | `packages/core/src/mailbox.ts`, `packages/runtime/src/messenger.ts` |
| Tools | `packages/tools/src/*.ts` |
