-- PinkyAgent schema: embeddings (requires pgvector).  *** RE-RUNNABLE ***
--
-- The `.rerun.sql` suffix is a migrate-runner convention (packages/core/src/
-- migrate.ts): such files are NOT recorded in schema_migrations and are
-- executed on EVERY `pinky migrate`. They must therefore be idempotent —
-- every statement here is guarded by `if not exists` or a DO-block probe.
--
-- Why: on a vector-less image (postgres:16-alpine) the embedding column
-- cannot be created. A one-shot migration would be marked applied and never
-- reconsidered, so the column would never appear. Being re-runnable, this
-- file is re-attempted on every migrate: an operator who switches the
-- container to pgvector/pgvector:pg16 just runs `pinky migrate` again and the
-- column + HNSW index materialise then. No new numbered file needed.
--
-- (A `2` row may linger in schema_migrations on databases migrated before
-- this file was renamed. It is inert: rerun files are keyed by name, not
-- version, and the number here only fixes ordering relative to 0001/0003.)
-- insufficient_privilege is tolerated on purpose. `pinky migrate` is meant to
-- run as the superuser (DATABASE_ADMIN_URL), but the gateway/smoke paths also
-- call migrate() at startup with the unprivileged pinky_app connection. That
-- role may neither CREATE EXTENSION nor ALTER a table it does not own, so on a
-- pgvector image it must skip quietly rather than crash the process -- the
-- work then happens on the next admin-credentialed migrate.
do $$
begin
  create extension if not exists vector;
exception when undefined_file or feature_not_supported or insufficient_privilege then
  raise notice 'pgvector unavailable to this role/image; embedding column deferred';
end $$;

do $$
begin
  if exists (select 1 from pg_extension where extname = 'vector') then
    alter table memories add column if not exists embedding vector(1536);
    create index if not exists memories_embedding_idx
      on memories using hnsw (embedding vector_cosine_ops);
  end if;
exception when insufficient_privilege then
  raise notice 'not the owner of memories; embedding column deferred to an admin migrate';
end $$;
