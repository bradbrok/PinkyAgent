/**
 * Scripted provider for tests and the CLI smoke command.
 * Pops turns from a script (or delegates to a function) and records every
 * CompleteOptions it receives for assertions.
 */
import type { AssistantTurn, CompleteOptions, LlmMessage, Provider } from "../types";

export type FakeScript = AssistantTurn[] | ((messages: LlmMessage[]) => AssistantTurn);

export class FakeProvider implements Provider {
  readonly name = "fake";
  private readonly script: FakeScript;
  /** Every CompleteOptions passed to complete(), in order. */
  readonly received: CompleteOptions[] = [];

  constructor(script: FakeScript) {
    this.script = Array.isArray(script) ? [...script] : script;
  }

  complete(opts: CompleteOptions): Promise<AssistantTurn> {
    this.received.push(opts);
    if (typeof this.script === "function") {
      return Promise.resolve(this.script(opts.messages));
    }
    const next = this.script.shift();
    if (!next) {
      return Promise.reject(
        new Error(`FakeProvider: script exhausted after ${this.received.length} call(s)`),
      );
    }
    return Promise.resolve(next);
  }
}
