/**
 * Minimal Slack Web API client. Only what the gateway needs: postMessage and
 * auth.test (bot identity resolution). fetch-based; no SDK dependency.
 */

export class SlackApiError extends Error {
  override name = "SlackApiError";
  /** Slack error code when the API replied ok:false, else undefined. */
  readonly slackError: string | undefined;

  constructor(message: string, slackError?: string) {
    super(message);
    this.slackError = slackError;
  }
}

interface SlackApiResponse {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

export class SlackClient {
  private readonly token: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: { token: string; baseUrl?: string; fetchImpl?: typeof fetch }) {
    this.token = opts.token;
    this.baseUrl = opts.baseUrl ?? "https://slack.com/api";
    this.fetchImpl = opts.fetchImpl ?? fetch;
  }

  /** chat.postMessage — post to a channel, optionally into a thread. */
  async postMessage(opts: { channel: string; text: string; thread_ts?: string }): Promise<{ ts: string }> {
    const res = await this.call("chat.postMessage", opts);
    return { ts: res.ts as string };
  }

  /** auth.test — resolve this bot token's user id (for mention detection). */
  async authTest(): Promise<{ userId: string }> {
    const res = await this.call("auth.test", {});
    return { userId: res.user_id as string };
  }

  private async call(method: string, body: Record<string, unknown>): Promise<SlackApiResponse> {
    const res = await this.fetchImpl(`${this.baseUrl}/${method}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json; charset=utf-8",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new SlackApiError(`Slack ${method} HTTP ${res.status}`);
    }
    const json = (await res.json()) as SlackApiResponse;
    if (!json.ok) {
      throw new SlackApiError(`Slack ${method} failed: ${json.error ?? "unknown_error"}`, json.error);
    }
    return json;
  }
}
