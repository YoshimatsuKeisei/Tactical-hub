import type { EncodedLegalActions } from "./rlActionEncoder";
import type { RlFeatureSpec } from "./rlFeatureSpec";
import type { EncodedObservation } from "./rlObservationEncoder";

export type BrowserBcInferenceRequest = {
  decisionKey: string;
  featureSpec: RlFeatureSpec;
  observation: EncodedObservation;
  legalActions: EncodedLegalActions;
};

export type BrowserBcInferenceResult = { actionKey: string };

export interface BrowserBcInferenceClient {
  infer(request: BrowserBcInferenceRequest): Promise<BrowserBcInferenceResult>;
  health(): Promise<boolean>;
}

export class HttpBrowserBcInferenceClient implements BrowserBcInferenceClient {
  private readonly pending = new Map<string, Promise<BrowserBcInferenceResult>>();

  constructor(private readonly baseUrl = "http://127.0.0.1:8765") {}

  async health() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok && Boolean((await response.json() as { ready?: boolean }).ready);
    } catch { return false; }
  }

  infer(request: BrowserBcInferenceRequest) {
    const existing = this.pending.get(request.decisionKey);
    if (existing) return existing;
    const promise = this.request(request).finally(() => this.pending.delete(request.decisionKey));
    this.pending.set(request.decisionKey, promise);
    return promise;
  }

  private async request(request: BrowserBcInferenceRequest) {
    const controller = new AbortController();
    const timeout = globalThis.setTimeout(() => controller.abort(), 15_000);
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/infer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          decisionKey: request.decisionKey,
          featureSpec: request.featureSpec,
          observation: request.observation,
          actions: request.legalActions.actions,
          actionKeys: request.legalActions.actionKeys,
        }),
      });
    } finally {
      globalThis.clearTimeout(timeout);
    }
    const payload = await response.json().catch(() => ({})) as { decisionKey?: string; actionKey?: string; error?: string };
    if (!response.ok) throw new Error(payload.error ?? `BC inference HTTP ${response.status}`);
    if (payload.decisionKey !== request.decisionKey) throw new Error("BC inference returned a stale decisionKey");
    if (typeof payload.actionKey !== "string") throw new Error("BC inference response is missing actionKey");
    return { actionKey: payload.actionKey };
  }
}
