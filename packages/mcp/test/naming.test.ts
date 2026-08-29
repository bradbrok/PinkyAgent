/**
 * The pure half of the MCP plane. Everything here is a deterministic transform
 * two different processes have to agree on without talking, so the tests are
 * mostly "same input, same output, on any machine".
 */
import { describe, expect, it } from "bun:test";
import { createHash } from "node:crypto";
import type { McpServerConfig } from "@pinky/core";
import {
  MAX_TOOL_NAME_LENGTH,
  canonicalJson,
  compareNames,
  hashServerConfig,
  isValidServerKey,
  hashedMcpToolName,
  mcpToolName,
  mcpToolNames,
  resolveEnvPlaceholders,
  sanitizeRawToolName,
  sortObjectKeysDeep,
  splitMcpToolName,
} from "../src/naming";

describe("sanitizeRawToolName", () => {
  it("keeps the provider-legal alphabet verbatim", () => {
    expect(sanitizeRawToolName("read_file-2")).toBe("read_file-2");
  });

  it("maps every other character to a single underscore", () => {
    expect(sanitizeRawToolName("git.status")).toBe("git_status");
    expect(sanitizeRawToolName("a b/c:d")).toBe("a_b_c_d");
    // Length-preserving: one replacement per character, so the sanitized name
    // is still eyeball-comparable with the server's own spelling.
    expect(sanitizeRawToolName("héllo")).toHaveLength("héllo".length);
  });

  it("never yields the empty string", () => {
    expect(sanitizeRawToolName("")).toBe("_");
    expect(sanitizeRawToolName("···")).toBe("___");
  });
});

describe("mcpToolName", () => {
  it("namespaces with the server key", () => {
    expect(mcpToolName("github", "create_issue")).toBe("mcp__github__create_issue");
  });

  it("sanitizes the raw name in place", () => {
    expect(mcpToolName("fs", "read.file")).toBe("mcp__fs__read_file");
  });

  it("caps at the provider limit with a stable hash suffix", () => {
    const raw = "a".repeat(200);
    const name = mcpToolName("srv", raw);
    expect(name).toHaveLength(MAX_TOOL_NAME_LENGTH);
    expect(name.startsWith("mcp__srv__")).toBe(true);
    const digest = createHash("sha256").update(`srv\0${raw}`).digest("hex").slice(0, 8);
    expect(name.endsWith(`_${digest}`)).toBe(true);
    expect(mcpToolName("srv", raw)).toBe(name); // deterministic
  });

  it("hashes the ORIGINAL raw name, so two tools that sanitize alike stay distinct", () => {
    const a = mcpToolName("srv", `${"x".repeat(70)}.a`);
    const b = mcpToolName("srv", `${"x".repeat(70)}_a`);
    // Same sanitized spelling, same truncation point — only the suffix differs.
    expect(a).not.toBe(b);
    expect(a.slice(0, MAX_TOOL_NAME_LENGTH - 9)).toBe(b.slice(0, MAX_TOOL_NAME_LENGTH - 9));
  });

  it("leaves a name that is exactly at the limit alone", () => {
    const raw = "y".repeat(MAX_TOOL_NAME_LENGTH - "mcp__srv__".length);
    const name = mcpToolName("srv", raw);
    expect(name).toHaveLength(MAX_TOOL_NAME_LENGTH);
    expect(name).toBe(`mcp__srv__${raw}`);
  });
});

describe("splitMcpToolName", () => {
  it("round-trips a name that needed no sanitizing or truncation", () => {
    expect(splitMcpToolName("mcp__github__create_issue")).toEqual({
      server: "github",
      raw: "create_issue",
    });
  });

  it("rejects anything that is not an MCP name", () => {
    expect(splitMcpToolName("read")).toBeNull();
    expect(splitMcpToolName("mcp__")).toBeNull();
    expect(splitMcpToolName("mcp____tool")).toBeNull();
    expect(splitMcpToolName("mcp__srv__")).toBeNull();
  });

  it("still recovers the SERVER from a truncated name (the manager only needs that)", () => {
    const name = mcpToolName("github", "z".repeat(200));
    expect(splitMcpToolName(name)?.server).toBe("github");
  });
});

describe("compareNames", () => {
  it("is a code-unit compare, not a locale compare", () => {
    // en_US collation ignores the hyphen at the first level and would order
    // these the other way round; C/code-unit order puts '-' (0x2D) first.
    const sorted = ["a-b", "ab"].sort(compareNames);
    expect(sorted).toEqual(["a-b", "ab"]);
    expect(["B", "a"].sort(compareNames)).toEqual(["B", "a"]);
  });
});

describe("isValidServerKey", () => {
  it("accepts the settings key alphabet and rejects the rest", () => {
    expect(isValidServerKey("github")).toBe(true);
    expect(isValidServerKey("fs-2_x")).toBe(true);
    expect(isValidServerKey("A")).toBe(false);
    expect(isValidServerKey("-x")).toBe(false);
    expect(isValidServerKey("")).toBe(false);
    expect(isValidServerKey("x".repeat(33))).toBe(false);
  });

  it("rejects a key containing the field separator, which would make the split ambiguous", () => {
    // `mcp__a__b__tool` parses two ways; splitMcpToolName is how call() picks
    // WHICH SERVER to dispatch to, so an ambiguous key is a mis-route.
    expect(isValidServerKey("a__b")).toBe(false);
    expect(isValidServerKey("a_b")).toBe(true);
    expect(() => mcpToolName("a__b", "t")).toThrow(/Invalid MCP server key/);
    expect(() => mcpToolNames("a__b", ["t"])).toThrow(/Invalid MCP server key/);
  });
});

describe("canonicalJson", () => {
  it("sorts object keys and drops undefined members", () => {
    expect(canonicalJson({ b: 1, a: 2, c: undefined })).toBe('{"a":2,"b":1}');
  });

  it("preserves array order (an array's order is data)", () => {
    expect(canonicalJson({ args: ["b", "a"] })).toBe('{"args":["b","a"]}');
  });

  it("recurses", () => {
    expect(canonicalJson({ z: { y: 1, x: 2 } })).toBe('{"z":{"x":2,"y":1}}');
  });
});

describe("hashServerConfig", () => {
  const base: McpServerConfig = {
    transport: "stdio",
    command: "bun",
    args: ["run", "server.ts"],
    env: { TOKEN: "${GITHUB_TOKEN}" },
  };

  it("is stable across key order", () => {
    const reordered = { env: { TOKEN: "${GITHUB_TOKEN}" }, args: ["run", "server.ts"], command: "bun", transport: "stdio" } as McpServerConfig;
    expect(hashServerConfig(reordered)).toBe(hashServerConfig(base));
  });

  it("changes when the command, args or a placeholder NAME changes", () => {
    expect(hashServerConfig({ ...base, command: "node" })).not.toBe(hashServerConfig(base));
    expect(hashServerConfig({ ...base, args: ["run", "other.ts"] })).not.toBe(hashServerConfig(base));
    expect(hashServerConfig({ ...base, env: { TOKEN: "${OTHER_TOKEN}" } })).not.toBe(
      hashServerConfig(base),
    );
  });

  it("hashes the placeholder, NOT its resolved value", () => {
    // The whole point: rotating GITHUB_TOKEN must not invalidate the catalog,
    // and no secret ever reaches the database through this field.
    const withSecret: McpServerConfig = { ...base, env: { TOKEN: "ghp_realsecret" } };
    expect(hashServerConfig(withSecret)).not.toBe(hashServerConfig(base));
    // ... and the hash of the placeholder form is independent of the env.
    process.env.PINKY_TEST_TOKEN = "one";
    const first = hashServerConfig({ ...base, env: { TOKEN: "${PINKY_TEST_TOKEN}" } });
    process.env.PINKY_TEST_TOKEN = "two";
    const second = hashServerConfig({ ...base, env: { TOKEN: "${PINKY_TEST_TOKEN}" } });
    delete process.env.PINKY_TEST_TOKEN;
    expect(second).toBe(first);
  });

  it("separates the stdio and http shapes", () => {
    const http: McpServerConfig = { transport: "http", url: "https://example.com/mcp" };
    expect(hashServerConfig(http)).not.toBe(hashServerConfig(base));
    expect(hashServerConfig(http)).toBe(hashServerConfig({ url: "https://example.com/mcp", transport: "http" } as McpServerConfig));
  });
});

describe("resolveEnvPlaceholders", () => {
  const env = { PRESENT: "value", EMPTY: "" };

  it("substitutes a whole-value placeholder", () => {
    expect(resolveEnvPlaceholders({ A: "${PRESENT}" }, env)).toEqual({ A: "value" });
  });

  it("passes non-placeholder values through verbatim", () => {
    expect(resolveEnvPlaceholders({ A: "literal", B: "pre${PRESENT}post" }, env)).toEqual({
      A: "literal",
      B: "pre${PRESENT}post",
    });
  });

  it("resolves a missing variable to the empty string, never a literal ${NAME}", () => {
    expect(resolveEnvPlaceholders({ A: "${MISSING}" }, env)).toEqual({ A: "" });
  });

  it("handles an absent record", () => {
    expect(resolveEnvPlaceholders(undefined, env)).toEqual({});
  });
});

describe("mcpToolNames", () => {
  it("disambiguates a sanitization collision instead of dropping one", () => {
    const names = mcpToolNames("s", ["a/b", "a.b"]);
    expect(names).toHaveLength(2);
    const byRaw = new Map(names.map((n) => [n.raw, n.name]));
    // Raw names are sorted first, so "a.b" (0x2E) keeps the plain spelling
    // whatever order the server listed them in.
    expect(byRaw.get("a.b")).toBe("mcp__s__a_b");
    expect(byRaw.get("a/b")).toBe(hashedMcpToolName("s", "a/b"));
    expect(byRaw.get("a/b")).toMatch(/^mcp__s__a_b_[0-9a-f]{8}$/);
  });

  it("is order-independent (the header must not churn when a server reorders its list)", () => {
    const one = mcpToolNames("s", ["a/b", "a.b", "z"]);
    const two = mcpToolNames("s", ["z", "a.b", "a/b"]);
    expect(JSON.stringify(one)).toBe(JSON.stringify(two));
  });

  it("leaves non-colliding names alone and collapses truly identical raw names", () => {
    expect(mcpToolNames("s", ["read", "write"]).map((n) => n.name)).toEqual([
      "mcp__s__read",
      "mcp__s__write",
    ]);
    expect(mcpToolNames("s", ["dup", "dup"])).toHaveLength(1);
  });
});

describe("sortObjectKeysDeep", () => {
  it("rebuilds objects in code-unit key order, recursively", () => {
    const out = sortObjectKeysDeep({ b: 1, a: { z: 1, y: { q: 1, p: 2 } } });
    expect(JSON.stringify(out)).toBe('{"a":{"y":{"p":2,"q":1},"z":1},"b":1}');
  });

  it("leaves arrays and primitives alone (an array's order is data)", () => {
    expect(JSON.stringify(sortObjectKeysDeep({ e: ["b", "a"], n: 1, s: null }))).toBe(
      '{"e":["b","a"],"n":1,"s":null}',
    );
    expect(sortObjectKeysDeep(5)).toBe(5);
  });

  it("is a fixed point: two different key orders converge on the same bytes", () => {
    const a = { type: "object", properties: { x: { type: "string" } }, required: ["x"] };
    const b = { required: ["x"], properties: { x: { type: "string" } }, type: "object" };
    expect(JSON.stringify(sortObjectKeysDeep(a))).toBe(JSON.stringify(sortObjectKeysDeep(b)));
  });
});
