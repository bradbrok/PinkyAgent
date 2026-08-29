-- PinkyAgent schema v5: the lexical half of hybrid recall (DESIGN.md §5.4).
--
-- §5.4 fuses three signals: vector cosine, BM25/FTS, and recency/importance.
-- 0002_embeddings.rerun.sql supplies the vector voice (an `embedding
-- vector(1536)` column + HNSW index) but ONLY where pgvector exists; on plain
-- postgres:16-alpine it is a documented no-op, which would leave recall with
-- no relevance signal at all. This file supplies the voice that needs nothing
-- but stock Postgres: a stored generated tsvector plus a GIN index, so
-- `where tsv @@ websearch_to_tsquery('english', $1)` ordered by ts_rank_cd is
-- available on every image the project runs on.
--
-- Generated, not trigger-maintained: `to_tsvector(regconfig, text)` is
-- IMMUTABLE (the one-argument form, which reads default_text_search_config,
-- is only STABLE and cannot be used here), so Postgres can keep the column in
-- lockstep with `text` itself. Nothing in the app writes `tsv`, and a memory
-- edited through the invalidate-then-insert path (§5.2) gets a correct tsv for
-- free.
--
-- 'english' is hard-coded on purpose and must match the query side
-- (packages/core/src/memory.ts): a tsvector built with one configuration and
-- probed with another silently matches nothing. Changing it means rebuilding
-- the column, i.e. a new migration, not an edit here.
--
-- ONE-SHOT, DELIBERATELY UNGUARDED (contrast 0002/0004, which are `.rerun.sql`
-- and swallow insufficient_privilege). One-shot files are recorded in
-- schema_migrations the first time they succeed, and `pinky migrate` is
-- specified to run on DATABASE_ADMIN_URL -- the same assumption 0001 (CREATE
-- EXTENSION) and 0003 (CREATE ROLE) already make. DDL here is unconditional so
-- a migrate that lacks the privilege FAILS LOUDLY and is retried with admin
-- credentials, rather than recording itself as applied while the column is
-- missing and recall degrades to "no rows ever match" with nothing to see.
-- (Consequence, stated plainly: memory/smoke/prompt/headless auto-migrate at startup,
-- so a deployment whose DATABASE_ADMIN_URL points at the unprivileged
-- pinky_app role will not start until an admin migrate has run.)

-- The tsvector itself. `text` here is the memories column, not the type.
alter table memories
  add column if not exists tsv tsvector
  generated always as (to_tsvector('english', text)) stored;

create index if not exists memories_tsv_idx on memories using gin (tsv);

-- Which embedder produced `embedding`; NULL = no vector for this row (the
-- normal state on a vector-less image, and for rows retained while the
-- embedder was unavailable). Recorded so a later re-embed can find the rows
-- written under a different model instead of guessing.
alter table memories add column if not exists embedding_model text;

comment on column memories.tsv is
  'DESIGN.md §5.4 FTS voice: generated from text with the english config; query with the same config.';
comment on column memories.embedding_model is
  'DESIGN.md §5.5: "provider/model-id" that produced embedding; null = row has no embedding.';
