-- PinkyAgent schema v6: the deferred-tool catalog (slice 9, MCP + deferred tools).
--
-- WHY A TABLE AT ALL. Tool schemas render at prefix position 0 of every
-- request (tools -> system -> messages). An MCP server can publish hundreds of
-- them, and the set changes whenever a server is added, restarted or updated,
-- so keeping every tool in the header would (a) spend thousands of tokens per
-- turn on schemas the model will never call and (b) invalidate every provider
-- cache tier the moment the list moves. The rule of this slice is therefore:
-- LOADING A TOOL NEVER REWRITES THE HEADER. The header carries the always-on
-- tools plus three fixed meta-tools (tool_search / tool_describe / tool_call);
-- everything else lives here and reaches the model as an ordinary tool result,
-- APPENDED to the conversation like any other message.
--
-- That makes this table the model's index of what it could call. It is a
-- CACHE OF TOOL SCHEMAS, not user data: names, descriptions and JSON Schemas
-- published by MCP servers and by our own built-ins.
--
-- INVALIDATE, NEVER DELETE (same rule as the memory plane, DESIGN.md §5.2).
-- A tool that disappears from a server's tools/list is stamped with
-- `removed_at`; it is never deleted. Current truth is `removed_at is null`.
-- Two reasons, both practical: a name that vanished is still worth resolving
-- when it turns up in an old continuity document or a replayed event log
-- ("that tool used to exist, and here is the schema it had"), and a server
-- that comes back after an outage clears the stamp instead of re-inserting a
-- row, so ids and history survive a flap.
--
-- PRIMARY KEY (tenant_id, name). `name` is the FINAL, namespaced name the
-- model sees -- `mcp__<server>__<raw>` for MCP tools, the plain name for
-- built-ins -- so it is the only key any caller has. `raw_name` keeps the
-- server's own spelling, which is what a callTool RPC must send back.
--
-- SEARCH IS FTS OVER THREE FIELDS. `tsv` is a stored generated column over
-- `name || ' ' || description || ' ' || arg_text`; `arg_text` is computed in
-- the application (packages/core/src/tool-catalog.ts, argText()) by flattening
-- the JSON Schema's property names and their descriptions, because that is
-- where the vocabulary a model searches for actually lives ("repository",
-- "issue number", "sha") -- a one-line tool description rarely mentions it.
-- Generated rather than trigger-maintained for the same reason as
-- memories.tsv (0005): the two-argument to_tsvector is IMMUTABLE, so Postgres
-- keeps the column in lockstep for free. Every source column is NOT NULL with
-- a default, because `x || null` is null and would silently produce an
-- unsearchable row.
--
-- 'english' IS HARD-CODED AND MUST MATCH THE QUERY SIDE
-- (websearch_to_tsquery('english', $1) in tool-catalog.ts). A tsvector built
-- with one configuration and probed with another matches nothing at all.
-- Changing it means rebuilding the column, i.e. a new migration, not an edit
-- here.
--
-- NO ROW-LEVEL SECURITY IN THIS SLICE -- DELIBERATE, AND A FOLLOW-UP.
-- `memories` is the only table with a policy (0003_rls.sql); events, threads,
-- a2a_messages and ingress_dedup are app-layer-fenced only, and this table
-- joins them. It holds no user data: only tool schemas, which are public
-- artefacts of the servers configured for a deployment. Every read and write
-- below still states `tenant_id` explicitly and repeats it in every WHERE (the
-- store also takes a withTenant()-wrapped Db), so the app-layer fence is real;
-- what is missing is the database-side backstop. FOLLOW-UP: add
-- `tool_catalog_tenant_isolation` alongside the events/threads policies when
-- RLS is extended past `memories` (CLAUDE.md slice 8, hardening) -- the GUC
-- and the wrapper it keys on are already in place.
--
-- ONE-SHOT, DELIBERATELY UNGUARDED (like 0005). One-shot files are recorded in
-- schema_migrations the first time they succeed and `pinky migrate` runs on
-- DATABASE_ADMIN_URL. DDL here is unconditional so a migrate that lacks the
-- privilege FAILS LOUDLY and is retried with admin credentials, rather than
-- recording itself as applied while the table is missing and every tool_search
-- degrades to "no tools found" with nothing to see.

create table if not exists tool_catalog (
  tenant_id   text not null,
  -- Final (namespaced) name as the model sees it: mcp__<server>__<raw>, or the
  -- plain name for a built-in.
  name        text not null,
  source      text not null check (source in ('builtin','mcp')),
  -- The settings key of the MCP server that published this tool; null for a
  -- built-in.
  server      text,
  -- The server's own spelling of the name, which callTool must send back;
  -- null for a built-in (its name is already raw).
  raw_name    text,
  description text not null default '',
  -- JSON Schema for the arguments (MCP inputSchema / Tool.parameters). Written
  -- as a PLAIN jsonb param, never JSON.stringify -- see the JSONB CONTRACT in
  -- packages/core/src/pg.ts and 0004_jsonb_repair.rerun.sql for what
  -- double-encoding costs.
  parameters  jsonb not null default '{}'::jsonb,
  -- Flattened property names + their descriptions (argText()), fed to the tsv.
  arg_text    text not null default '',
  -- sha256 of the server's McpServerConfig with ${ENV} placeholders left
  -- unresolved; null for a built-in. McpManager trusts the catalog on start
  -- when this still matches the configured hash, so request 1 can already see
  -- the server's tools -- before the server itself has answered.
  config_hash text,
  updated_at  timestamptz not null default now(),
  -- Non-null => the tool is gone from its source; the row is history, not
  -- current truth. Never deleted.
  removed_at  timestamptz,
  tsv tsvector generated always as (
    to_tsvector('english', name || ' ' || description || ' ' || arg_text)
  ) stored,
  primary key (tenant_id, name)
);

-- The lexical voice of tool_search.
create index if not exists tool_catalog_tsv_idx on tool_catalog using gin (tsv);

-- replaceServer()'s generational update and serverState()'s trust probe both
-- key on (tenant_id, server).
create index if not exists tool_catalog_server_idx on tool_catalog (tenant_id, server);

-- 0003 set default privileges for future tables, so this grant is usually
-- redundant -- stated anyway so the migration is self-contained and a database
-- whose default privileges were altered cannot silently lock the app role out
-- of the catalog. `pinky_app` exists unconditionally: 0003 creates it.
grant select, insert, update, delete on tool_catalog to pinky_app;

comment on table tool_catalog is
  'Slice 9 deferred-tool catalog: tool schemas the model reaches via tool_search/'
  'tool_describe/tool_call instead of the request header. Invalidate, never delete '
  '(current truth = removed_at is null). NO RLS policy yet -- holds tool schemas, '
  'not user data; app-layer tenant fence only (see the header of this migration).';
comment on column tool_catalog.name is
  'Final namespaced name the model calls: mcp__<server>__<raw>, or a plain built-in name.';
comment on column tool_catalog.arg_text is
  'Flattened JSON Schema property names + descriptions (tool-catalog.ts argText); feeds tsv.';
comment on column tool_catalog.tsv is
  'FTS over name || description || arg_text with the english config; query with the same config.';
comment on column tool_catalog.removed_at is
  'Non-null = withdrawn by its source. Cleared if the tool comes back (server flap).';
