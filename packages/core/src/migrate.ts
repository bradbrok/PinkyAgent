/**
 * Migration runner for packages/core/schema.
 *
 * Two file conventions, both ordered by their numeric prefix:
 *
 *   NNNN_name.sql        one-shot. Applied once, ever; its version number is
 *                        recorded in `schema_migrations` and it is skipped
 *                        from then on.
 *   NNNN_name.rerun.sql  re-runnable. Executed on EVERY migrate and never
 *                        recorded. MUST be idempotent (`if not exists`,
 *                        DO-block probes). This is how conditional DDL that
 *                        depends on the server's capabilities gets a second
 *                        chance -- e.g. the pgvector embedding column, which
 *                        cannot be created on postgres:16-alpine but should
 *                        appear the first time `pinky migrate` runs after the
 *                        image is switched to pgvector.
 *
 * Both kinds run inside a transaction (one per file).
 */
import type { Db } from "./db";

/** `undefined_table` — the only error tolerated when probing schema_migrations. */
const UNDEFINED_TABLE = "42P01";

const RERUN_FILE = /^(\d+)_(.+)\.rerun\.sql$/;
const ONESHOT_FILE = /^(\d+)_(.+)\.sql$/;

export async function migrate(db: Db, schemaDir: string): Promise<void> {
  // Lexicographic sort keeps numeric prefixes (0001, 0002, ...) in order, and
  // interleaves rerun files at their own number.
  const entries = Array.from((await listDir(schemaDir)).sort());
  const sqlFiles = entries.filter((e) => e.endsWith(".sql"));

  const applied = await appliedVersions(db);

  for (const file of sqlFiles) {
    if (RERUN_FILE.test(file)) {
      const text = await readText(join(schemaDir, file));
      await db.tx(async (tx) => {
        await tx.query(text);
      });
      continue;
    }

    const match = ONESHOT_FILE.exec(file);
    if (!match || !match[1]) continue;
    const version = Number(match[1]);
    if (applied.has(version)) continue;
    const text = await readText(join(schemaDir, file));
    await db.tx(async (tx) => {
      // Multi-statement in one query: caller uses query (no params) so
      // postgres.js simple protocol accepts it.
      await tx.query(text);
      await tx.query(`insert into schema_migrations (version) values ($1)`, [version]);
    });
  }
}

/**
 * Read the applied set. On a virgin database `schema_migrations` does not
 * exist yet (0001 creates it), so 42P01 means "nothing applied". Every other
 * failure -- connection refused, auth rejected, permission denied -- must
 * surface: swallowing those would silently re-apply every migration against a
 * database we cannot actually read.
 */
async function appliedVersions(db: Db): Promise<Set<number>> {
  try {
    const rows = await db.query<{ version: number }>(`select version from schema_migrations`);
    return new Set(rows.map((r) => r.version));
  } catch (err) {
    if (pgErrorCode(err) === UNDEFINED_TABLE) return new Set();
    const detail = err instanceof Error ? err.message : String(err);
    const code = pgErrorCode(err);
    throw new Error(
      `migrate: cannot read schema_migrations${code ? ` (SQLSTATE ${code})` : ""}: ${detail}`,
      { cause: err },
    );
  }
}

/** SQLSTATE from a postgres.js error, if the thrown value carries one. */
function pgErrorCode(err: unknown): string | undefined {
  if (typeof err !== "object" || err === null || !("code" in err)) return undefined;
  const code = (err as { code?: unknown }).code;
  return typeof code === "string" ? code : undefined;
}

// Minimal file/dir helpers local to this module — Bun APIs used as required.
import { readdir } from "node:fs/promises";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

async function listDir(dir: string): Promise<string[]> {
  return readdir(dir);
}

async function readText(path: string): Promise<string> {
  return readFile(path, "utf8");
}
