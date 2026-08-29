-- PinkyAgent schema v1 (DESIGN.md §3, §5, §7)
-- No explicit begin/commit: the migrate runner wraps each file in a tx.

-- pgvector is required for slice-2 embeddings. On images without it (plain
-- postgres:16-alpine) this is a no-op so slice-1 (events/mailbox/settings)
-- still migrates; the memories.embedding column is added by the re-runnable
-- 0002_embeddings.rerun.sql, which is re-attempted on every `pinky migrate`
-- and so materialises the moment the image gains pgvector.
do $$
begin
  create extension if not exists vector;
exception when undefined_file or feature_not_supported then
  raise notice 'pgvector not available; embedding column deferred to 0002_embeddings.rerun.sql';
end $$;

create table if not exists schema_migrations (
  version integer primary key,
  applied_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Threads: one per (tenant, channel, thread) conversation.
-- ---------------------------------------------------------------------------
create table if not exists threads (
  tenant_id  text not null,
  channel_id text not null,
  thread_id  text not null,
  created_at timestamptz not null default now(),
  meta       jsonb not null default '{}',
  primary key (tenant_id, channel_id, thread_id)
);

-- ---------------------------------------------------------------------------
-- Append-only event log. seq is per-thread, assigned under the thread lock.
-- ---------------------------------------------------------------------------
create table if not exists events (
  id         text primary key,
  tenant_id  text not null,
  channel_id text not null,
  thread_id  text not null,
  seq        bigint not null,
  ts         timestamptz not null default now(),
  type       text not null,
  data       jsonb not null,
  foreign key (tenant_id, channel_id, thread_id)
    references threads (tenant_id, channel_id, thread_id),
  unique (tenant_id, channel_id, thread_id, seq)
);
create index if not exists events_thread_idx
  on events (tenant_id, channel_id, thread_id, seq);

-- Ingress dedup: stable external ids (Slack event_id etc.), one per thread.
create table if not exists ingress_dedup (
  tenant_id   text not null,
  external_id text not null,
  first_seen  timestamptz not null default now(),
  primary key (tenant_id, external_id)
);

-- ---------------------------------------------------------------------------
-- A2A mailbox: durable agent-to-agent messages, local or cross-machine.
-- ---------------------------------------------------------------------------
create table if not exists a2a_messages (
  id          text primary key,
  from_agent  text not null,  -- agentId
  to_agent    text not null,  -- agentId or "broadcast"
  node_from   text not null,
  node_to     text not null,
  kind        text not null check (kind in ('message', 'request', 'response')),
  text        text not null,
  thread_hint text,
  delivered_at timestamptz,
  read_at     timestamptz,
  created_at  timestamptz not null default now()
);
create index if not exists a2a_inbox_idx
  on a2a_messages (to_agent, read_at, created_at);

-- ---------------------------------------------------------------------------
-- Memory plane (v1: semantic + episodic; procedural later if needed).
-- RLS on tenant_id (DESIGN.md §5.1): a missing WHERE cannot leak across
-- tenants. The policy below is only half the mechanism -- it is inert until
-- (a) something sets the `pinky.tenant_id` GUC and (b) the connection is a
-- non-superuser role. 0003_rls.sql supplies both (FORCE RLS + the pinky_app
-- role); packages/core/src/tenant.ts supplies the GUC.
-- ---------------------------------------------------------------------------
create table if not exists memories (
  id          text primary key,
  tenant_id   text not null,
  agent_id    text not null,
  visibility  text not null check (visibility in ('private', 'user', 'channel', 'tenant', 'global')),
  user_id     text,
  channel_id  text,
  kind        text not null check (kind in ('semantic', 'episodic', 'procedural')),
  text        text not null,
  importance  smallint not null default 5 check (importance between 1 and 10),
  valid_from  timestamptz not null default now(),
  valid_to    timestamptz,               -- set => invalidated, never deleted
  recorded_at timestamptz not null default now(),
  meta        jsonb not null default '{}'
);
create index if not exists memories_scope_idx
  on memories (tenant_id, agent_id, visibility, valid_to);
-- embedding column + HNSW index arrive via 0002_embeddings.rerun.sql once
-- pgvector is present.

-- NOTE: enable-only. A table owner and any superuser still bypass this;
-- 0003_rls.sql adds `force row level security`, a WITH CHECK for writes, and
-- the unprivileged pinky_app role the app is meant to connect as.
alter table memories enable row level security;
create policy memories_tenant_isolation on memories
  using (tenant_id = current_setting('pinky.tenant_id', true));

comment on table memories is 'DESIGN.md §5: invalidation-not-deletion; current truth = valid_to is null';

-- ---------------------------------------------------------------------------
-- Settings: ALL mutable config lives here (DESIGN.md: "everything in db").
-- env only bootstraps: DATABASE_URL, API secrets, node identity.
-- Written ONLY by the human-run CLI. The agent runtime reads a snapshot per
-- wake and has no write path — agents cannot reconfigure themselves.
-- Scope: 'global', or 'agent:<id>', or 'channel:<id>' (overlay, agent < channel < global? No: global < channel < agent).
-- ---------------------------------------------------------------------------
create table if not exists settings (
  scope       text not null,           -- 'global' | 'channel:<id>' | 'agent:<id>'
  key         text not null,           -- dotted, e.g. 'model', 'context.advisoryFraction'
  value       jsonb not null,
  updated_at  timestamptz not null default now(),
  updated_by  text not null default 'cli',
  primary key (scope, key)
);

comment on table settings is 'Human-owned configuration. Runtime reads only; no agent-writable path exists.';
