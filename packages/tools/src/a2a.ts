/**
 * A2A tools: a2a_send sends an envelope through ctx.messenger, a2a_inbox lists
 * unread envelopes. Both fail cleanly when A2A is not enabled.
 */
import type { Tool, ToolContext, ToolResult } from "@pinky/runtime";

type Kind = "message" | "request" | "response";
const KINDS: Kind[] = ["message", "request", "response"];

export class A2ASendTool implements Tool {
  readonly name = "a2a_send";
  readonly description = "Send an A2A envelope to an agent (agentId@nodeId) or broadcast.";
  readonly parameters = {
    type: "object",
    properties: {
      to: { type: "string", description: "agentId@nodeId or 'broadcast'" },
      text: { type: "string", description: "Message body" },
      kind: { type: "string", enum: KINDS, description: "Envelope kind (default 'message')" },
    },
    required: ["to", "text"],
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (!ctx.messenger) {
      return { text: `${this.name}: A2A not enabled (no messenger in context)`, isError: true };
    }
    if (!ctx.agentId) {
      return { text: "a2a_send: ctx.agentId is required to send", isError: true };
    }
    if (typeof args.to !== "string" || args.to.length === 0) {
      return { text: "a2a_send: 'to' must be a non-empty string", isError: true };
    }
    if (typeof args.text !== "string") {
      return { text: "a2a_send: 'text' must be a string", isError: true };
    }
    let kind: Kind = "message";
    if (args.kind !== undefined) {
      if (typeof args.kind !== "string" || !KINDS.includes(args.kind as Kind)) {
        return { text: `a2a_send: 'kind' must be one of ${KINDS.join(", ")}`, isError: true };
      }
      kind = args.kind as Kind;
    }

    const from = `${ctx.agentId}@${ctx.messenger.nodeId}`;
    const id = await ctx.messenger.send({
      from,
      to: args.to,
      kind,
      text: args.text,
      threadHint: ctx.thread.threadId,
    });
    return { text: `a2a_send: sent, id=${id}` };
  }
}

export class A2AInboxTool implements Tool {
  readonly name = "a2a_inbox";
  readonly description = "List unread A2A envelopes for this agent (marks them read).";
  readonly parameters = {
    type: "object",
    properties: {
      limit: { type: "number", description: "Max envelopes to return" },
    },
  };

  async execute(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult> {
    if (!ctx.messenger) {
      return { text: `${this.name}: A2A not enabled (no messenger in context)`, isError: true };
    }
    if (!ctx.agentId) {
      return { text: "a2a_inbox: ctx.agentId is required to read the inbox", isError: true };
    }
    let limit: number | undefined;
    if (args.limit !== undefined) {
      const n = Number(args.limit);
      if (!Number.isInteger(n) || n < 1) {
        return { text: "a2a_inbox: 'limit' must be a positive integer", isError: true };
      }
      limit = n;
    }

    const envelopes = await ctx.messenger.inbox(
      ctx.agentId,
      limit === undefined ? undefined : { limit },
    );
    if (envelopes.length === 0) {
      return { text: "(no messages)" };
    }
    const summary = envelopes.map(({ from, kind, text, sentAt }) => ({ from, kind, text, sentAt }));
    return { text: JSON.stringify(summary, null, 2) };
  }
}
