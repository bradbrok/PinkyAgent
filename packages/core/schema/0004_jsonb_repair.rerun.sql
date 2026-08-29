-- PinkyAgent schema: repair doubly-encoded jsonb.  *** RE-RUNNABLE ***
--
-- WHAT WENT WRONG
-- Every jsonb write in this repo used to bind `JSON.stringify(value)`.
-- postgres.js learns a parameter's type from the server's Describe (jsonb) and
-- then applies its OWN serializer -- JSON.stringify -- so the text was encoded
-- a SECOND time and the row landed as a jsonb *string*:
--
--   jsonb_typeof(events.data)   = 'string'   and data->>'type'   is NULL
--   jsonb_typeof(settings.value)= 'string'   and `pinky config get model`
--                                             printed "\"openrouter/...\""
--
-- src/pg.ts now documents the contract (jsonb params take PLAIN values; never
-- pre-stringify) and src/event-store.ts / src/settings.ts obey it, so no NEW
-- row can be damaged. This file repairs the rows already written.
--
-- WHY RE-RUNNABLE (see src/migrate.ts): `.rerun.sql` files are not recorded in
-- schema_migrations and run on EVERY `pinky migrate`, so a database restored
-- from an old dump, or a second node whose rows were written by an unpatched
-- build, is fixed the next time anything migrates. It must therefore be a
-- no-op once clean -- it is: a repaired value is no longer a jsonb string (or,
-- for a genuine string setting, no longer matches the JSON-looking pattern),
-- so the second run selects zero rows.
--
-- THE RULE, and the one judgement call in it
-- A damaged row is a jsonb STRING whose text is itself JSON. Reparsing it
-- (`value #>> '{}'`)::jsonb undoes exactly one encoding.
--   * events.data, threads.meta and memories.meta are ALWAYS written as JSON
--     objects, so the test there is the strict one: the inner text must start
--     with '{' or '['. Nothing legitimate can match.
--   * settings.value may legitimately BE a jsonb string ("model" is one), so
--     the test additionally accepts a leading '"' and the true/false/null and
--     number literals -- that is what a double-encoded scalar looks like.
--     Residual ambiguity, stated plainly: a CORRECTLY stored string setting
--     whose text happens to be valid JSON of another type (tenantId = '42',
--     say) is indistinguishable from a double-encoded number and would be
--     converted. No key in DEFAULT_SETTINGS can take such a value today
--     (tenantId/model are free-form strings, but a bare number or a quoted
--     string is not a plausible value for either), so the window is theoretical.
--     A value that is not valid JSON after all is caught per row and left alone.
--
-- a2a_messages is deliberately absent: it has no jsonb column (text/timestamptz
-- only), so nothing there can be double-encoded.
--
-- Privileges: `pinky migrate` runs as the superuser, but gateway/smoke/prompt
-- also call migrate() at startup on the unprivileged pinky_app connection.
-- That role holds UPDATE on every table (0003), so this normally just works;
-- the insufficient_privilege guard keeps a locked-down role from crashing the
-- process instead of deferring the repair to the next admin migrate. On
-- `memories` pinky_app additionally sees no rows at all (FORCE RLS, no
-- pinky.tenant_id set), which makes its pass a silent no-op by construction.

-- ---------------------------------------------------------------------------
-- events.data — always an object.
-- ---------------------------------------------------------------------------
do $$
declare
  r record;
  fixed integer := 0;
begin
  for r in
    select id, data #>> '{}' as inner_text from events
     where jsonb_typeof(data) = 'string' and (data #>> '{}') ~ '^\s*[\[{]'
  loop
    begin
      update events set data = r.inner_text::jsonb where id = r.id;
      fixed := fixed + 1;
    exception when invalid_text_representation then
      raise notice '0004: events % holds a jsonb string that is not JSON; left as-is', r.id;
    end;
  end loop;
  if fixed > 0 then
    raise notice '0004: repaired % double-encoded events.data row(s)', fixed;
  end if;
exception when insufficient_privilege then
  raise notice '0004: no privilege to repair events.data; deferred to an admin migrate';
end $$;

-- ---------------------------------------------------------------------------
-- settings.value — object OR scalar, so the wider (documented) rule.
-- ---------------------------------------------------------------------------
do $$
declare
  r record;
  fixed integer := 0;
begin
  for r in
    select scope, key, value #>> '{}' as inner_text from settings
     where jsonb_typeof(value) = 'string'
       and (value #>> '{}') ~ '^\s*([\[{"]|true\s*$|false\s*$|null\s*$|-?[0-9])'
  loop
    begin
      update settings set value = r.inner_text::jsonb
       where scope = r.scope and key = r.key;
      fixed := fixed + 1;
    exception when invalid_text_representation then
      raise notice '0004: settings %/% holds a jsonb string that is not JSON; left as-is',
        r.scope, r.key;
    end;
  end loop;
  if fixed > 0 then
    raise notice '0004: repaired % double-encoded settings.value row(s)', fixed;
  end if;
exception when insufficient_privilege then
  raise notice '0004: no privilege to repair settings.value; deferred to an admin migrate';
end $$;

-- ---------------------------------------------------------------------------
-- threads.meta / memories.meta — always objects; both default to '{}', so on a
-- healthy database these loops select nothing at all.
-- ---------------------------------------------------------------------------
do $$
declare
  r record;
  fixed integer := 0;
begin
  for r in
    select tenant_id, channel_id, thread_id, meta #>> '{}' as inner_text from threads
     where jsonb_typeof(meta) = 'string' and (meta #>> '{}') ~ '^\s*[\[{]'
  loop
    begin
      update threads set meta = r.inner_text::jsonb
       where (tenant_id, channel_id, thread_id) = (r.tenant_id, r.channel_id, r.thread_id);
      fixed := fixed + 1;
    exception when invalid_text_representation then
      raise notice '0004: threads %/%/% meta is not JSON; left as-is',
        r.tenant_id, r.channel_id, r.thread_id;
    end;
  end loop;
  if fixed > 0 then
    raise notice '0004: repaired % double-encoded threads.meta row(s)', fixed;
  end if;
exception when insufficient_privilege then
  raise notice '0004: no privilege to repair threads.meta; deferred to an admin migrate';
end $$;

do $$
declare
  r record;
  fixed integer := 0;
begin
  for r in
    select id, meta #>> '{}' as inner_text from memories
     where jsonb_typeof(meta) = 'string' and (meta #>> '{}') ~ '^\s*[\[{]'
  loop
    begin
      update memories set meta = r.inner_text::jsonb where id = r.id;
      fixed := fixed + 1;
    exception when invalid_text_representation then
      raise notice '0004: memories % meta is not JSON; left as-is', r.id;
    end;
  end loop;
  if fixed > 0 then
    raise notice '0004: repaired % double-encoded memories.meta row(s)', fixed;
  end if;
exception when insufficient_privilege then
  raise notice '0004: no privilege to repair memories.meta; deferred to an admin migrate';
end $$;
