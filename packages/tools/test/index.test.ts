import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createTools } from "../src/index";
import { makeCtx, makeTmpDir } from "./helpers";

const BASE_TOOLS = ["read", "write", "edit", "glob", "grep", "a2a_send", "a2a_inbox"];

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
