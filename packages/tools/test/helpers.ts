import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type {
  Db,
  MemoryHit,
  MemoryKind,
  MemoryRow,
  MemoryStore,
  MemoryVisibility,
  RecallScope,
  RetainInput,
  SearchInput,
  SettingsSnapshot,
} from "@pinky/core";
import { FakeEmbedder } from "@pinky/runtime";
import type {
  A2AEnvelope,
  Embedder,
  MemoryContext,
  Messenger,
  ToolContext,
} from "@pinky/runtime";
import { visibleInScope } from "../src/memory";

export function makeTmpDir(): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(path.join(tmpdir(), "pinky-tools-"));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

/** Never-called Db stub — no tool here touches the database. */
const fakeDb: Db = {
  query: () => Promise.reject(new Error("db not used in tools tests")),
  queryOne: () => Promise.reject(new Error("db not used in tools tests")),
  tx: () => Promise.reject(new Error("db not used in tools tests")),
  close: () => Promise.resolve(),
};

export interface ContextOverrides {
  messenger?: Messenger | null;
  agentId?: string | null;
  memory?: MemoryContext | null;
  /** Read-only settings snapshot the run started with (DESIGN.md P8, revised).
   *  `null` means "the runtime passed none", which the settings tools must
   *  survive — hence the explicit null rather than just omitting it. */
  settings?: SettingsSnapshot | null;
  /** Swap the never-called stub for a db double (the settings tools use it). */
  db?: Db | null;
}

export function makeCtx(cwd: string, overrides: ContextOverrides = {}): ToolContext {
  const ctx: ToolContext = {
    cwd,
    db: overrides.db ?? fakeDb,
    thread: { tenantId: "t1", channelId: "c1", threadId: "thread-test" },
    emit: () => Promise.resolve(),
  };
  if (overrides.messenger !== null && overrides.messenger !== undefined) {
    ctx.messenger = overrides.messenger;
  }
  if (overrides.agentId !== null && overrides.agentId !== undefined) {
    ctx.agentId = overrides.agentId;
  }
  if (overrides.memory !== null && overrides.memory !== undefined) {
    ctx.memory = overrides.memory;
  }
  if (overrides.settings !== null && overrides.settings !== undefined) {
    ctx.settings = overrides.settings;
  }
  return ctx;
}

export interface SentEnvelope extends Omit<A2AEnvelope, "id" | "sentAt"> {
  id: string;
  sentAt: string;
}

export function makeFakeMessenger(opts: {
  nodeId?: string;
  canned?: A2AEnvelope[];
} = {}): Messenger & { sent: SentEnvelope[]; idSeq: number } {
  const sent: SentEnvelope[] = [];
  const received = new Set<string>();
  let idSeq = 0;
  const base: Messenger = {
    nodeId: opts.nodeId ?? "node-test",
    send(env) {
      idSeq++;
      sent.push({ ...env, id: `id-${idSeq}`, sentAt: new Date().toISOString() });
      return Promise.resolve(`id-${idSeq}`);
    },
    inbox() {
      return Promise.resolve(opts.canned ?? []);
    },
    onMessage() {
      return () => {};
    },
    // Inbound relay half of the contract. No tool calls it; the double just
    // has to satisfy Messenger, and dedups on id like a real one would.
    receive(env) {
      const fresh = !received.has(env.id);
      received.add(env.id);
      return Promise.resolve(fresh);
    },
    // Consumption edge (issue #4): the surface consumes, not the tools, so the
    // double only has to satisfy Messenger.
    redeliverUnconsumed() {
      return Promise.resolve(0);
    },
    claimConsumption() {
      return Promise.resolve(false);
    },
  };
  return Object.assign(base, { sent, get idSeq() { return sent.length; } });
}

// ---------------------------------------------------------------------------
// Memory plane doubles (DESIGN.md §5)
// ---------------------------------------------------------------------------

export const DEFAULT_SCOPE: RecallScope = {
  agentId: "pinky",
  channelId: "c1",
  userId: "u1",
  includeUser: false,
  includePrivate: false,
};

let rowSeq = 0;

/** A MemoryRow with sane defaults; override whatever the test cares about. */
export function makeMemoryRow(over: Partial<MemoryRow> = {}): MemoryRow {
  rowSeq++;
  // Distinct in the FIRST 8 chars: that is the slice `recall` prints, and
  // memory_edit resolves ids by that prefix.
  const id = `${String(rowSeq).padStart(8, "0")}-0000-4000-8000-000000000000`;
  return {
    id,
    tenantId: "t1",
    agentId: "pinky",
    visibility: "channel" as MemoryVisibility,
    userId: null,
    channelId: "c1",
    kind: "semantic" as MemoryKind,
    text: `memory ${rowSeq}`,
    importance: 5,
    validFrom: "2026-08-01T00:00:00.000Z",
    validTo: null,
    recordedAt: "2026-08-01T00:00:00.000Z",
    embeddingModel: null,
    meta: {},
    ...over,
  };
}

export function makeMemoryHit(over: Partial<MemoryHit> = {}): MemoryHit {
  const { score, voices, ...rowOver } = over;
  return {
    ...makeMemoryRow(rowOver),
    score: score ?? 0.5,
    voices: voices ?? { fts: 1 },
  };
}

/**
 * RecallScope overrides for tests. `channelId`/`userId` accept an explicit
 * `undefined` to mean "absent from the scope" — which `Partial<RecallScope>`
 * cannot express under exactOptionalPropertyTypes.
 */
export interface ScopeOverride {
  agentId?: string;
  channelId?: string | undefined;
  userId?: string | undefined;
  includeUser?: boolean;
  includePrivate?: boolean;
}

export function makeScope(over: ScopeOverride = {}): RecallScope {
  const channelId = "channelId" in over ? over.channelId : DEFAULT_SCOPE.channelId;
  const userId = "userId" in over ? over.userId : DEFAULT_SCOPE.userId;
  return {
    agentId: over.agentId ?? DEFAULT_SCOPE.agentId,
    includeUser: over.includeUser ?? DEFAULT_SCOPE.includeUser,
    includePrivate: over.includePrivate ?? DEFAULT_SCOPE.includePrivate,
    ...(channelId !== undefined ? { channelId } : {}),
    ...(userId !== undefined ? { userId } : {}),
  };
}

export interface FakeMemoryOptions {
  scope?: ScopeOverride;
  /** What `search` returns, regardless of the query. */
  hits?: MemoryHit[];
  /** Rows `get`/`list` can see (also seeded by retain/update). */
  rows?: MemoryRow[];
  /** Whether the store claims the pgvector column exists. Default true. */
  supportsVectors?: boolean;
  embedder?: Embedder;
}

export interface FakeListArgs {
  scope: RecallScope;
  limit?: number;
  includeInvalid?: boolean;
  kinds?: MemoryKind[];
}

export interface FakePrefixArgs {
  prefix: string;
  scope: RecallScope;
  limit?: number;
  includeInvalid?: boolean;
}

export interface FakeMemory {
  context: MemoryContext;
  scope: RecallScope;
  rows: Map<string, MemoryRow>;
  searches: SearchInput[];
  retains: RetainInput[];
  updates: { id: string; replacement: Record<string, unknown> }[];
  invalidations: { id: string; reason?: string }[];
  gets: string[];
  lists: FakeListArgs[];
  prefixes: FakePrefixArgs[];
  supportsVectorsCalls: number;
}

/**
 * In-memory MemoryStore double. Records every write and answers reads from a
 * row map, so a test can assert on what the tool asked the store to do without
 * a database. Typed through `as unknown as MemoryStore` because MemoryStore is
 * a class (nominal-ish surface, but the tools only ever call its methods).
 */
export function makeFakeMemory(opts: FakeMemoryOptions = {}): FakeMemory {
  const scope = makeScope(opts.scope);
  const rows = new Map<string, MemoryRow>();
  for (const row of opts.rows ?? []) rows.set(row.id, row);

  const searches: SearchInput[] = [];
  const retains: RetainInput[] = [];
  const updates: { id: string; replacement: Record<string, unknown> }[] = [];
  const invalidations: { id: string; reason?: string }[] = [];
  const gets: string[] = [];
  const lists: FakeListArgs[] = [];
  const prefixes: FakePrefixArgs[] = [];
  const state = { supportsVectorsCalls: 0 };

  /**
   * `list`/`findByIdPrefix` are scope-fenced in SQL (core's `scopePredicate`),
   * so a double that answers from every row it holds is not a double of them:
   * it makes an out-of-scope row look reachable and hides exactly the leak the
   * real predicate exists to stop. `visibleInScope` is the in-TS twin of that
   * predicate, and `validTo` is nulled so `includeInvalid` stays a separate
   * decision here just as it is in the store.
   */
  const inScope = (row: MemoryRow, includeInvalid: boolean): boolean =>
    visibleInScope({ ...row, validTo: null }, scope) &&
    (includeInvalid || row.validTo === null);

  /** Same order as the store: recorded_at desc, id desc. */
  const newestFirst = (): MemoryRow[] =>
    [...rows.values()].sort((a, b) =>
      a.recordedAt === b.recordedAt
        ? b.id.localeCompare(a.id)
        : b.recordedAt.localeCompare(a.recordedAt),
    );

  const rowFromInput = (input: RetainInput): MemoryRow =>
    makeMemoryRow({
      agentId: input.agentId,
      visibility: input.visibility,
      userId: input.userId ?? null,
      channelId: input.channelId ?? null,
      kind: input.kind,
      text: input.text,
      importance: input.importance ?? 5,
      embeddingModel: input.embeddingModel ?? null,
      meta: input.meta ?? {},
    });

  const store = {
    supportsVectors(): Promise<boolean> {
      state.supportsVectorsCalls++;
      return Promise.resolve(opts.supportsVectors ?? true);
    },
    search(input: SearchInput): Promise<MemoryHit[]> {
      searches.push(input);
      return Promise.resolve(opts.hits ?? []);
    },
    retain(input: RetainInput): Promise<MemoryRow> {
      retains.push(input);
      const row = rowFromInput(input);
      rows.set(row.id, row);
      return Promise.resolve(row);
    },
    get(id: string): Promise<MemoryRow | null> {
      gets.push(id);
      return Promise.resolve(rows.get(id) ?? null);
    },
    invalidate(id: string, o?: { reason?: string }): Promise<boolean> {
      invalidations.push({ id, ...(o?.reason ? { reason: o.reason } : {}) });
      const row = rows.get(id);
      if (!row || row.validTo !== null) return Promise.resolve(false);
      rows.set(id, { ...row, validTo: "2026-08-28T00:00:00.000Z" });
      return Promise.resolve(true);
    },
    update(id: string, replacement: Record<string, unknown>): Promise<MemoryRow> {
      updates.push({ id, replacement });
      const old = rows.get(id);
      if (!old) return Promise.reject(new Error(`unknown memory ${id}`));
      rows.set(id, { ...old, validTo: "2026-08-28T00:00:00.000Z" });
      const next = rowFromInput({
        agentId: old.agentId,
        ...replacement,
      } as unknown as RetainInput);
      const superseded: MemoryRow = { ...next, meta: { ...next.meta, supersedes: id } };
      rows.set(superseded.id, superseded);
      return Promise.resolve(superseded);
    },
    list(o: FakeListArgs): Promise<MemoryRow[]> {
      lists.push(o);
      const matches = newestFirst().filter(
        (r) =>
          inScope(r, o.includeInvalid === true) &&
          (o.kinds === undefined || o.kinds.includes(r.kind)),
      );
      return Promise.resolve(matches.slice(0, o.limit ?? 20));
    },
    /**
     * Prefix match in the store, not a page of `list` filtered afterwards —
     * that is the whole point of the method (a relevant memory from further
     * back is otherwise unreachable by the id `recall` printed for it). The
     * real one rejects a prefix that is not uuid-shaped rather than running a
     * `like` against arbitrary text.
     */
    findByIdPrefix(prefix: string, o: Omit<FakePrefixArgs, "prefix">): Promise<MemoryRow[]> {
      prefixes.push({ prefix, ...o });
      // Same guard as the store: the prefix is concatenated into a LIKE
      // pattern, so the character class IS the escaping (core/memory.ts).
      if (!/^[0-9a-f-]{4,36}$/.test(prefix)) {
        return Promise.reject(
          new Error(
            `memory: id prefix must be 4..36 characters of lowercase hex or "-", got ${JSON.stringify(prefix)}`,
          ),
        );
      }
      const matches = newestFirst().filter(
        (r) => r.id.startsWith(prefix) && inScope(r, o.includeInvalid === true),
      );
      return Promise.resolve(matches.slice(0, o.limit ?? 2));
    },
  };

  const context: MemoryContext = {
    store: store as unknown as MemoryStore,
    scope,
    ...(opts.embedder ? { embedder: opts.embedder } : {}),
  };

  return {
    context,
    scope,
    rows,
    searches,
    retains,
    updates,
    invalidations,
    gets,
    lists,
    prefixes,
    get supportsVectorsCalls() {
      return state.supportsVectorsCalls;
    },
  };
}

export interface TestEmbedder extends Embedder {
  calls: string[][];
}

/**
 * Embedder double. The happy path is runtime's own `FakeEmbedder`, so these
 * tests exercise the same double the smoke run and the integration suite use.
 * `fail` swaps in a stub that rejects — the case FakeEmbedder cannot express,
 * and half of what these tests are about: a blipping embedding API must
 * degrade recall to FTS and must never lose a retain.
 */
export function makeFakeEmbedder(
  opts: { model?: string; dimensions?: number; fail?: string | Error } = {},
): TestEmbedder {
  if (opts.fail === undefined) {
    return new FakeEmbedder({
      ...(opts.model !== undefined ? { model: opts.model } : {}),
      ...(opts.dimensions !== undefined ? { dimensions: opts.dimensions } : {}),
    });
  }
  const fail = opts.fail;
  const calls: string[][] = [];
  return {
    model: opts.model ?? "fake/failing",
    dimensions: opts.dimensions ?? 8,
    calls,
    embed(texts: string[]): Promise<number[][]> {
      calls.push([...texts]);
      return Promise.reject(fail instanceof Error ? fail : new Error(fail));
    },
  };
}
