/**
 * Lexical sandbox containment for path-taking tools. Symlink-naive on purpose:
 * resolve the request against the cwd root and require the resolved path to be
 * the root itself or live under `root + path.sep`.
 */
import path from "node:path";

export type SandboxResult = { ok: true; abs: string } | { ok: false; error: string };

export function sandboxResolve(cwd: string, requested?: string): SandboxResult {
  const root = path.resolve(cwd);
  if (requested === undefined || requested.length === 0) {
    return { ok: true, abs: root };
  }
  const resolved = path.resolve(root, requested);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    return { ok: false, error: `path ${JSON.stringify(requested)} escapes sandbox root ${root}` };
  }
  return { ok: true, abs: resolved };
}
