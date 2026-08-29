import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createTools } from "../src/index";
import { makeCtx, makeTmpDir } from "./helpers";

const BASE_TOOLS = [
  "read",
  "write",
  "edit",
  "glob",
  "grep",
  "a2a_send",
  "a2a_inbox",
  "recall",
  "retain",
  "memory_edit",
  // Always registered, like the memory tools: without a settings snapshot in
  // context they answer with a clean error, and `settings_set` is inert until
  // a human turns selfConfig on (DESIGN.md P8, revised).
  "settings_get",
  "settings_set",
  // The catalog door (slice 9). Unconditional on purpose: the tool list is the
  // cached prefix, so a header that gained or lost these three depending on
  // whether an MCP server was reachable at boot would cost a cold cache per
  // wake. Without `ctx.deferred` they answer "no deferred tools on this
  // surface" (DESIGN.md §4.5/§9).
  "tool_search",
  "tool_describe",
  "tool_call",
];

describe("createTools", () => {
  test("omits bash by default (Slack-reachable surfaces get no shell)", () => {
    const names = createTools().map((t) => t.name);
    expect(names).not.toContain("bash");
    expect(names).toEqual(BASE_TOOLS);
  });

  test("shell: true opts the bash tool in", () => {
    const names = createTools({ shell: true }).map((t) => t.name);
    expect(names).toContain("bash");
    expect(names).toEqual(["bash", ...BASE_TOOLS]);
  });

  test("shell: false is the same as omitting it", () => {
    expect(createTools({ shell: false }).map((t) => t.name)).toEqual(BASE_TOOLS);
  });

  describe("shellEnv", () => {
    beforeAll(() => {
      process.env.PINKY_TEST_SECRET_TOOLSET = "leak-toolset";
    });

    afterAll(() => {
      delete process.env.PINKY_TEST_SECRET_TOOLSET;
    });

    test("shellEnv reaches the bash tool without leaking host env", async () => {
      const bash = createTools({ shell: true, shellEnv: { PINKY_ALLOWED: "from-opts" } }).find(
        (t) => t.name === "bash",
      );
      expect(bash).toBeDefined();
      const { dir, cleanup } = makeTmpDir();
      try {
        const res = await bash!.execute(
          {
            command:
              'echo "allowed=[$PINKY_ALLOWED]"; echo "secret=[$PINKY_TEST_SECRET_TOOLSET]"',
          },
          makeCtx(dir),
        );
        expect(res.isError).toBeUndefined();
        expect(res.text).toContain("allowed=[from-opts]");
        expect(res.text).toContain("secret=[]");
        expect(res.text).not.toContain("leak-toolset");
      } finally {
        cleanup();
      }
    });
  });
});
