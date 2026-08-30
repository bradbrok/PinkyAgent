/**
 * The worker's tool schemas and their validators (slice 6, contract §3.4).
 *
 * Every rejection path is tested with its MESSAGE, because that message is
 * what lands in the `error` event a failed pass journals — "invalid arguments"
 * would tell whoever reads that event nothing about which field to look at.
 *
 * The two rules worth the most here are not JSON Schema rules at all: exactly
 * one decision per candidate, and an UPDATE/DELETE target drawn from THAT
 * candidate's own neighbours. Without the second, one hallucinated id
 * invalidates an unrelated memory (DESIGN.md §5.2).
 */
import { describe, expect, test } from "bun:test";
import {
  DECIDE_TOOL,
  DECIDE_TOOL_NAME,
  EXTRACT_TOOL,
  EXTRACT_TOOL_NAME,
  MAX_CANDIDATES,
  MAX_INSIGHTS,
  MAX_MEMORY_CHARS,
  REFLECT_TOOL,
  REFLECT_TOOL_NAME,
  parseDecide,
  parseExtract,
  parseReflect,
} from "../src/schemas";
import type { Candidate } from "../src/schemas";

/** The parse error text, or a marker naming what came back instead. */
function err(result: unknown): string {
  return result && typeof result === "object" && "error" in result
    ? String((result as { error: unknown }).error)
    : `expected an error, got ${JSON.stringify(result)}`;
}

const candidate = (over: Partial<Candidate> = {}): Candidate => ({
  text: "Brad prefers terse answers",
  kind: "semantic",
  importance: 7,
  visibility: "channel",
  ...over,
});

describe("tool schemas", () => {
  test("the three names and shapes are what the prompts promise", () => {
    expect(EXTRACT_TOOL.name).toBe(EXTRACT_TOOL_NAME);
    expect(DECIDE_TOOL.name).toBe(DECIDE_TOOL_NAME);
    expect(REFLECT_TOOL.name).toBe(REFLECT_TOOL_NAME);
    for (const spec of [EXTRACT_TOOL, DECIDE_TOOL, REFLECT_TOOL]) {
      expect(spec.parameters["type"]).toBe("object");
      expect(spec.parameters["additionalProperties"]).toBe(false);
      expect(spec.description.length).toBeGreaterThan(0);
    }
    expect(EXTRACT_TOOL.parameters["required"]).toEqual(["candidates"]);
    expect(DECIDE_TOOL.parameters["required"]).toEqual(["decisions"]);
    expect(REFLECT_TOOL.parameters["required"]).toEqual(["insights"]);
  });

  test("procedural memories are NOT offered (DESIGN.md §13: human-approved only)", () => {
    const items = (EXTRACT_TOOL.parameters["properties"] as Record<string, Record<string, unknown>>)[
      "candidates"
    ]?.["items"] as Record<string, Record<string, Record<string, unknown>>>;
    expect(items["properties"]?.["kind"]?.["enum"]).toEqual(["semantic", "episodic"]);
    // Nor `private`: the worker never writes the agent's own scratch (§5.1).
    expect(items["properties"]?.["visibility"]?.["enum"]).toEqual(["channel", "tenant", "user"]);
  });

  test("parameters keys are in canonical (code-unit) order", () => {
    for (const spec of [EXTRACT_TOOL, DECIDE_TOOL, REFLECT_TOOL]) {
      const keys = Object.keys(spec.parameters);
      expect(keys).toEqual([...keys].sort());
    }
  });
});

describe("parseExtract", () => {
  test("accepts a well-formed list and trims the text", () => {
    const parsed = parseExtract({
      candidates: [{ text: "  a fact  ", kind: "episodic", importance: 3, visibility: "tenant" }],
    });
    expect(parsed).toEqual({
      candidates: [{ text: "a fact", kind: "episodic", importance: 3, visibility: "tenant" }],
    });
  });

  test("an EMPTY list is a success — the pass still journals a receipt", () => {
    expect(parseExtract({ candidates: [] })).toEqual({ candidates: [] });
  });

  test("keeps userId only when it was given", () => {
    const parsed = parseExtract({
      candidates: [{ text: "t", kind: "semantic", importance: 5, visibility: "user", userId: "brad" }],
    });
    expect(parsed).toEqual({
      candidates: [{ text: "t", kind: "semantic", importance: 5, visibility: "user", userId: "brad" }],
    });
  });

  test("rejects non-object arguments", () => {
    expect(err(parseExtract("nope"))).toContain("arguments must be an object");
  });

  test("rejects an unexpected top-level property", () => {
    expect(err(parseExtract({ candidates: [], notes: "hi" }))).toContain('unexpected property "notes"');
  });

  test("rejects a missing or non-array candidates", () => {
    expect(err(parseExtract({}))).toContain('"candidates" must be an array');
    expect(err(parseExtract({ candidates: {} }))).toContain('"candidates" must be an array');
  });

  test("rejects more than MAX_CANDIDATES", () => {
    const many = Array.from({ length: MAX_CANDIDATES + 1 }, () => ({
      text: "t",
      kind: "semantic",
      importance: 5,
      visibility: "channel",
    }));
    expect(err(parseExtract({ candidates: many }))).toContain(
      `at most ${MAX_CANDIDATES} are allowed`,
    );
  });

  test("names the failing candidate and field", () => {
    const good = { text: "t", kind: "semantic", importance: 5, visibility: "channel" };
    expect(err(parseExtract({ candidates: [good, "x"] }))).toContain("candidates[1] must be an object");
    expect(err(parseExtract({ candidates: [good, { ...good, oops: 1 }] }))).toContain(
      'candidates[1]: unexpected property "oops"',
    );
    expect(err(parseExtract({ candidates: [{ ...good, text: "   " }] }))).toContain(
      "candidates[0].text must be a non-empty string",
    );
    expect(err(parseExtract({ candidates: [{ ...good, text: 7 }] }))).toContain(
      "candidates[0].text must be a string",
    );
    expect(err(parseExtract({ candidates: [{ ...good, text: "x".repeat(MAX_MEMORY_CHARS + 1) }] }))).toContain(
      `at most ${MAX_MEMORY_CHARS} are allowed`,
    );
    expect(err(parseExtract({ candidates: [{ ...good, kind: "procedural" }] }))).toContain(
      "candidates[0].kind must be one of semantic, episodic",
    );
    expect(err(parseExtract({ candidates: [{ ...good, importance: 0 }] }))).toContain(
      "candidates[0].importance must be an integer 1..10",
    );
    expect(err(parseExtract({ candidates: [{ ...good, importance: 5.5 }] }))).toContain(
      "candidates[0].importance must be an integer 1..10",
    );
    expect(err(parseExtract({ candidates: [{ ...good, importance: 11 }] }))).toContain(
      "candidates[0].importance must be an integer 1..10",
    );
    expect(err(parseExtract({ candidates: [{ ...good, visibility: "private" }] }))).toContain(
      "candidates[0].visibility must be one of channel, tenant, user",
    );
    expect(err(parseExtract({ candidates: [{ ...good, userId: "" }] }))).toContain(
      "candidates[0].userId must be a non-empty string",
    );
  });
});

describe("parseDecide", () => {
  const candidates = [candidate({ text: "one" }), candidate({ text: "two" })];
  const neighbors = [["n1", "n2"], ["n3"]];

  test("accepts one decision per candidate and returns them in candidate order", () => {
    const parsed = parseDecide(
      {
        decisions: [
          { candidate: 1, action: "DELETE", target: "n3", reason: "contradicted" },
          { candidate: 0, action: "ADD" },
        ],
      },
      candidates,
      neighbors,
    );
    expect(parsed).toEqual({
      decisions: [
        { candidate: 0, action: "ADD" },
        { candidate: 1, action: "DELETE", target: "n3", reason: "contradicted" },
      ],
    });
  });

  test("an UPDATE with no text inherits the candidate's own wording", () => {
    const parsed = parseDecide(
      { decisions: [{ candidate: 0, action: "UPDATE", target: "n1" }, { candidate: 1, action: "NOOP" }] },
      candidates,
      neighbors,
    );
    expect(parsed).toEqual({
      decisions: [
        { candidate: 0, action: "UPDATE", target: "n1", text: "one" },
        { candidate: 1, action: "NOOP" },
      ],
    });
  });

  test("an explicit UPDATE text wins", () => {
    const parsed = parseDecide(
      {
        decisions: [
          { candidate: 0, action: "UPDATE", target: "n2", text: "  merged  " },
          { candidate: 1, action: "NOOP" },
        ],
      },
      candidates,
      neighbors,
    );
    expect(parsed).toEqual({
      decisions: [
        { candidate: 0, action: "UPDATE", target: "n2", text: "merged" },
        { candidate: 1, action: "NOOP" },
      ],
    });
  });

  test("rejects a missing decision — a dropped candidate is silent data loss", () => {
    expect(err(parseDecide({ decisions: [{ candidate: 0, action: "ADD" }] }, candidates, neighbors))).toBe(
      "decisions: candidate 1 has no decision",
    );
  });

  test("rejects a duplicate decision", () => {
    const args = {
      decisions: [
        { candidate: 0, action: "ADD" },
        { candidate: 0, action: "NOOP" },
        { candidate: 1, action: "NOOP" },
      ],
    };
    expect(err(parseDecide(args, candidates, neighbors))).toBe(
      "decisions[1].candidate 0 already has a decision",
    );
  });

  test("rejects an out-of-range or non-integer index", () => {
    expect(err(parseDecide({ decisions: [{ candidate: 5, action: "ADD" }] }, candidates, neighbors))).toContain(
      "decisions[0].candidate must be an integer 0..1",
    );
    expect(err(parseDecide({ decisions: [{ candidate: -1, action: "ADD" }] }, candidates, neighbors))).toContain(
      "decisions[0].candidate must be an integer 0..1",
    );
  });

  test("rejects an unknown action", () => {
    expect(
      err(parseDecide({ decisions: [{ candidate: 0, action: "MERGE" }] }, candidates, neighbors)),
    ).toContain("decisions[0].action must be one of ADD, UPDATE, DELETE, NOOP");
  });

  test("UPDATE and DELETE require a target from THAT candidate's neighbours", () => {
    // n3 belongs to candidate 1: accepting it for candidate 0 would retire a
    // row this candidate was never compared against.
    expect(
      err(
        parseDecide(
          { decisions: [{ candidate: 0, action: "UPDATE", target: "n3" }, { candidate: 1, action: "NOOP" }] },
          candidates,
          neighbors,
        ),
      ),
    ).toContain("decisions[0].target must be one of candidate 0's neighbour ids (n1, n2)");

    expect(
      err(
        parseDecide(
          { decisions: [{ candidate: 0, action: "DELETE" }, { candidate: 1, action: "NOOP" }] },
          candidates,
          neighbors,
        ),
      ),
    ).toContain("decisions[0].target must be one of candidate 0's neighbour ids");
  });

  test("a candidate with no neighbours is told only ADD or NOOP are possible", () => {
    expect(
      err(
        parseDecide(
          { decisions: [{ candidate: 0, action: "UPDATE", target: "made-up" }] },
          [candidates[0] as Candidate],
          [[]],
        ),
      ),
    ).toContain("it has none, so only ADD or NOOP are possible");
  });

  test("rejects malformed shapes and over-long fields", () => {
    expect(err(parseDecide(null, candidates, neighbors))).toContain("arguments must be an object");
    expect(err(parseDecide({ decisions: {} }, candidates, neighbors))).toContain(
      '"decisions" must be an array',
    );
    expect(err(parseDecide({ decisions: [], extra: 1 }, candidates, neighbors))).toContain(
      'unexpected property "extra"',
    );
    expect(err(parseDecide({ decisions: [{ candidate: 0, action: "ADD", oops: 1 }] }, candidates, neighbors))).toContain(
      'decisions[0]: unexpected property "oops"',
    );
    expect(
      err(
        parseDecide(
          { decisions: [{ candidate: 0, action: "ADD", reason: "r".repeat(301) }] },
          candidates,
          neighbors,
        ),
      ),
    ).toContain("decisions[0].reason is 301 characters");
  });

  test("zero candidates and zero decisions is a success", () => {
    expect(parseDecide({ decisions: [] }, [], [])).toEqual({ decisions: [] });
  });
});

describe("parseReflect", () => {
  const batch = ["a", "b", "c"];

  test("accepts up to MAX_INSIGHTS with sources drawn from the batch", () => {
    const parsed = parseReflect(
      { insights: [{ text: "they all prefer terse answers", importance: 6, sources: ["a", "b"] }] },
      batch,
    );
    expect(parsed).toEqual({
      insights: [{ text: "they all prefer terse answers", importance: 6, sources: ["a", "b"] }],
    });
  });

  test("zero insights is a success", () => {
    expect(parseReflect({ insights: [] }, batch)).toEqual({ insights: [] });
  });

  test("rejects more than MAX_INSIGHTS", () => {
    const many = Array.from({ length: MAX_INSIGHTS + 1 }, () => ({
      text: "t",
      importance: 5,
      sources: ["a"],
    }));
    expect(err(parseReflect({ insights: many }, batch))).toContain(`at most ${MAX_INSIGHTS} are allowed`);
  });

  test("rejects a source that is not in the batch", () => {
    expect(
      err(parseReflect({ insights: [{ text: "t", importance: 5, sources: ["a", "zz"] }] }, batch)),
    ).toContain('insights[0].sources contains "zz", which is not in this batch');
  });

  test("rejects an empty sources list", () => {
    expect(err(parseReflect({ insights: [{ text: "t", importance: 5, sources: [] }] }, batch))).toContain(
      "insights[0].sources must name at least one memory from the batch",
    );
  });

  test("supersedes must be a SUBSET of that insight's own sources", () => {
    // Retiring a row the insight never cited is a destructive edit with no
    // stated evidence (DESIGN.md §9, Mem0's quality regression).
    expect(
      err(
        parseReflect(
          { insights: [{ text: "t", importance: 5, sources: ["a"], supersedes: ["b"] }] },
          batch,
        ),
      ),
    ).toContain('insights[0].supersedes contains "b", which is not one of its own sources');
  });

  test("supersedes equal to sources is allowed", () => {
    expect(
      parseReflect({ insights: [{ text: "t", importance: 5, sources: ["a"], supersedes: ["a"] }] }, batch),
    ).toEqual({ insights: [{ text: "t", importance: 5, sources: ["a"], supersedes: ["a"] }] });
  });

  test("names malformed fields", () => {
    expect(err(parseReflect(7, batch))).toContain("arguments must be an object");
    expect(err(parseReflect({ insights: "x" }, batch))).toContain('"insights" must be an array');
    expect(err(parseReflect({ insights: [], extra: true }, batch))).toContain('unexpected property "extra"');
    expect(err(parseReflect({ insights: [{ text: "", importance: 5, sources: ["a"] }] }, batch))).toContain(
      "insights[0].text must be a non-empty string",
    );
    expect(err(parseReflect({ insights: [{ text: "t", importance: 99, sources: ["a"] }] }, batch))).toContain(
      "insights[0].importance must be an integer 1..10",
    );
    expect(err(parseReflect({ insights: [{ text: "t", importance: 5, sources: "a" }] }, batch))).toContain(
      "insights[0].sources must be an array",
    );
    expect(err(parseReflect({ insights: [{ text: "t", importance: 5, sources: [1] }] }, batch))).toContain(
      "insights[0].sources[0] must be a non-empty string",
    );
    expect(
      err(parseReflect({ insights: [{ text: "t", importance: 5, sources: ["a"], nope: 1 }] }, batch)),
    ).toContain('insights[0]: unexpected property "nope"');
  });
});
