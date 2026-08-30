/**
 * @pinky/sleep — the sleep-time worker (DESIGN.md §5.3 item 3, slice 6).
 *
 * Extraction (per thread) and reflection (cross-thread) from the event log
 * into the memory plane while the agent is idle. This is the ONLY path allowed
 * to consolidate memories (DESIGN.md §9, "sleep-worker-only consolidation") —
 * the hot tools append and annotate, they never synthesize across rows.
 *
 * Imports core + runtime; imported only by the CLI, which owns every surface
 * that can start it. Never imports @pinky/tools: the worker is not an agent
 * and has no tool surface of its own — it forces three fixed schemas
 * (schemas.ts) and reads the answers.
 */
export * from "./types";
export * from "./transcript";
export * from "./schemas";
export * from "./prompts";
export * from "./extract";
export * from "./reflect";
export * from "./discovery";
export * from "./worker";
