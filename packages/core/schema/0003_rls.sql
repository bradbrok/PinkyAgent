-- PinkyAgent schema v3: make row-level security real (DESIGN.md §5.1).
--
-- 0001 enabled RLS on `memories` and wrote a policy, but the mechanism was
-- decorative: nothing set the `pinky.tenant_id` GUC, and the app connected as
-- the `postgres` superuser, which bypasses RLS unconditionally. This file
-- supplies the two missing halves on the database side; the GUC itself is set
-- per transaction by withTenant() in packages/core/src/tenant.ts.
--
-- WHAT THIS DOES AND DOES NOT BUY YOU
--   * FORCE RLS => even the table *owner* is subject to the policy.
--   * A superuser (postgres) STILL bypasses RLS entirely. That is Postgres,
--     not a bug here. Isolation only holds for connections made as a role
--     with NOSUPERUSER + NOBYPASSRLS -- i.e. `pinky_app`.
--   * Only `memories` is covered. events / threads / a2a_messages /
--     ingress_dedup carry tenant_id but have NO policy yet; they are still
--     app-layer-filtered only.

-- ---------------------------------------------------------------------------
-- Application role. The app (gateway, runtime, `pinky config`) connects as
-- this; only `pinky migrate` connects as the superuser.
--
-- The dev-default password below matches docker-compose.yml's POSTGRES_PASSWORD
-- so a local checkout works with no extra steps. IT IS NOT A SECRET.
-- PRODUCTION MUST ROTATE IT:
--     alter role pinky_app password '<generated>';
-- and set DATABASE_URL accordingly. This migration never overwrites the
-- password of an existing role, so rotation survives re-migration.
-- ---------------------------------------------------------------------------
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'pinky_app') then
    create role pinky_app login password 'pinky' nosuperuser nobypassrls;
  end if;
end $$;

-- Idempotent and safe on an existing role: re-asserts the two attributes RLS
-- depends on without touching the password.
alter role pinky_app nosuperuser nobypassrls;

grant usage on schema public to pinky_app;
grant select, insert, update, delete on all tables in schema public to pinky_app;
grant usage, select on all sequences in schema public to pinky_app;

-- Future tables/sequences created by the migrating role are covered too, so a
-- new migration does not silently lock the app out.
alter default privileges in schema public
  grant select, insert, update, delete on tables to pinky_app;
alter default privileges in schema public
  grant usage, select on sequences to pinky_app;

-- ---------------------------------------------------------------------------
-- memories: FORCE the policy, and check writes as well as reads.
-- ---------------------------------------------------------------------------
alter table memories force row level security;

-- 0001's policy was USING-only. Recreated here with an explicit WITH CHECK so
-- INSERT/UPDATE cannot write a row belonging to another tenant.
--
-- Note the nullif(). current_setting(..., true) returns NULL only while the
-- GUC has never been touched on that backend; once withTenant() has run one
-- transaction, the transaction-local value is discarded at COMMIT and the
-- setting reverts to the EMPTY STRING, not to undefined. On a pooled
-- connection that is the normal steady state, so `tenant_id = ''` would be
-- the predicate an un-scoped query runs -- and a row that happened to carry
-- tenant_id = '' would leak to everyone. nullif maps both spellings of
-- "unset" to NULL, and `tenant_id = NULL` is NULL (not true): zero rows
-- visible, zero rows writable. Fail-closed by construction.
drop policy if exists memories_tenant_isolation on memories;
create policy memories_tenant_isolation on memories
  for all
  using (tenant_id = nullif(current_setting('pinky.tenant_id', true), ''))
  with check (tenant_id = nullif(current_setting('pinky.tenant_id', true), ''));

-- Explicit INSERT policy. Redundant with the WITH CHECK above (both are
-- PERMISSIVE, so they OR, and the predicate is identical) -- kept because
-- INSERT is the direction people forget, and a future edit that relaxes the
-- ALL policy should still trip over this one in review.
drop policy if exists memories_tenant_insert on memories;
create policy memories_tenant_insert on memories
  for insert
  with check (tenant_id = nullif(current_setting('pinky.tenant_id', true), ''));

comment on table memories is
  'DESIGN.md §5: invalidation-not-deletion; current truth = valid_to is null. '
  'RLS FORCED on tenant_id via the pinky.tenant_id GUC (see src/tenant.ts); '
  'superuser connections still bypass it.';
