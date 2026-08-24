import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type { EncodedLegalActionsV2 } from "./rlActionEncoder";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { EncodedObservation } from "./rlObservationEncoder";
import type { RlSelectedTorchDevice, RlTorchDevice } from "./rlTorchDevice";
import { packPpoActInput, packPpoEncodedSamples, type PpoEncodedSample } from "./rlPpoPackedBatch";
import type { PackedBcBatch } from "./rlBcPackedBatch";

export type PpoHyperparameters = {
  learningRate: number;
  gamma: number;
  gaeLambda: number;
  clipEpsilon: number;
  valueCoefficient: number;
  entropyCoefficient: number;
  maxGradientNorm: number;
};

export type PpoTrainingSample = PpoEncodedSample;

type Response =
  | { type: "ready"; selectedDevice: RlSelectedTorchDevice; updateCount: number; episodeCount: number }
  | { type: "action"; requestId: number; actionIndex: number; logProbability: number; value: number }
  | { type: "updateBegun"; requestId: number; totalSamples: number }
  | { type: "updateChunkAccepted"; requestId: number; acceptedSamples: number; accumulatedSamples: number }
  | { type: "updateResult"; requestId: number; sampleCount: number; loss: number; policyLoss: number; valueLoss: number; entropy: number; gradientNorm: number; policyParametersChanged: boolean; valueParametersChanged: boolean; updateCount: number; episodeCount: number }
  | { type: "saved"; requestId: number; path: string; updateCount: number; episodeCount: number }
  | { type: "closed" }
  | { type: "error"; requestId?: number; message: string };

export class PythonPpoClient {
  private process?: ChildProcessWithoutNullStreams;
  private lines?: Interface;
  private nextRequestId = 1;
  private readonly waiting: Array<{ resolve: (response: Response) => void; reject: (error: Error) => void }> = [];
  private stderr = "";
  private featureSpec?: RlFeatureSpecV2;

  constructor(private readonly options: { command?: string; args?: string[]; cwd?: string; device?: RlTorchDevice } = {}) {}

  private wait() { return new Promise<Response>((resolve, reject) => this.waiting.push({ resolve, reject })); }
  private send(payload: unknown) {
    if (!this.process?.stdin.writable) throw new Error("Python PPO process is not running");
    this.process.stdin.write(`${JSON.stringify(payload)}\n`);
  }
  private sendPreparedPacked(value: Record<string, unknown>, packed: PackedBcBatch) {
    if (!this.process?.stdin.writable) throw new Error("Python PPO process is not running");
    this.process.stdin.write(`${JSON.stringify({
      ...value,
      encoding: "packed-v1",
      byteLength: packed.payload.byteLength,
      batchSize: packed.batchSize,
      tensors: packed.tensors,
    })}\n`);
    this.process.stdin.write(packed.payload);
  }
  private async request(payload: Record<string, unknown>) {
    const responsePromise = this.wait();
    this.send(payload);
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    return response;
  }

  async start(input: { seed: number; featureSpec: RlFeatureSpecV2; hyperparameters: PpoHyperparameters; initialCheckpoint: string; resume?: string }) {
    if (this.process) throw new Error("Python PPO process is already running");
    this.process = spawn(this.options.command ?? "python", this.options.args ?? ["-u", "-m", "rl.ppo_server"], {
      cwd: this.options.cwd ?? process.cwd(), stdio: ["pipe", "pipe", "pipe"],
    });
    this.process.stderr.on("data", (chunk) => { this.stderr += String(chunk); process.stderr.write(chunk); });
    this.lines = createInterface({ input: this.process.stdout });
    this.lines.on("line", (line) => {
      const pending = this.waiting.shift();
      if (!pending) return;
      try { pending.resolve(JSON.parse(line) as Response); } catch { pending.reject(new Error(`Invalid JSON from PPO server: ${line}`)); }
    });
    const rejectAll = (error: Error) => { while (this.waiting.length) this.waiting.shift()!.reject(error); };
    this.process.on("error", rejectAll);
    this.process.on("exit", (code) => rejectAll(new Error(`Python PPO process exited with code ${code}${this.stderr ? `: ${this.stderr.trim()}` : ""}`)));
    const response = await this.request({ type: "init", ...input, device: this.options.device ?? "auto" });
    if (response.type !== "ready") throw new Error(`Unexpected PPO initialization response: ${response.type}`);
    this.featureSpec = input.featureSpec;
    return response;
  }

  async act(observation: EncodedObservation, legalActions: EncodedLegalActionsV2) {
    if (!this.featureSpec) throw new Error("Python PPO Feature Spec is not initialized");
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.sendPreparedPacked(
      { type: "packedAct", requestId },
      packPpoActInput(observation, legalActions.actions, this.featureSpec),
    );
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "action" || response.requestId !== requestId) throw new Error("Unexpected PPO action response");
    if (!Number.isInteger(response.actionIndex) || response.actionIndex < 0 || response.actionIndex >= legalActions.actionKeys.length) throw new Error("PPO returned an illegal action index");
    if (![response.logProbability, response.value].every(Number.isFinite)) throw new Error("PPO returned NaN or Inf");
    return { ...response, actionKey: legalActions.actionKeys[response.actionIndex] };
  }

  async beginUpdate(totalSamples: number) {
    if (!Number.isInteger(totalSamples) || totalSamples <= 0) throw new Error("PPO totalSamples must be a positive integer");
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "beginUpdate", requestId, totalSamples });
    if (response.type !== "updateBegun" || response.requestId !== requestId || response.totalSamples !== totalSamples) {
      throw new Error("Unexpected PPO begin-update response");
    }
    return response;
  }

  async accumulatePacked(samples: PpoEncodedSample[]) {
    if (!this.featureSpec) throw new Error("Python PPO Feature Spec is not initialized");
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.sendPreparedPacked(
      { type: "packedUpdateChunk", requestId },
      packPpoEncodedSamples(samples, this.featureSpec),
    );
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "updateChunkAccepted" || response.requestId !== requestId || response.acceptedSamples !== samples.length) {
      throw new Error("Unexpected PPO update-chunk response");
    }
    return response;
  }

  async finishUpdate(completedEpisodes: number) {
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "finishUpdate", requestId, completedEpisodes });
    if (response.type !== "updateResult" || response.requestId !== requestId) throw new Error("Unexpected PPO finish-update response");
    if (![response.loss, response.policyLoss, response.valueLoss, response.entropy, response.gradientNorm].every(Number.isFinite)) throw new Error("PPO update returned NaN or Inf");
    return response;
  }

  async update(samples: PpoTrainingSample[], completedEpisodes: number) {
    const requestId = this.nextRequestId++;
    const response = await this.request({
      type: "update",
      requestId,
      samples: samples.map(({ targetIndex, ...sample }) => ({ ...sample, selectedActionIndex: targetIndex })),
      completedEpisodes,
    });
    if (response.type !== "updateResult" || response.requestId !== requestId) throw new Error("Unexpected PPO update response");
    if (![response.loss, response.policyLoss, response.valueLoss, response.entropy, response.gradientNorm].every(Number.isFinite)) throw new Error("PPO update returned NaN or Inf");
    return response;
  }

  async save(path: string, metadata: Record<string, unknown> = {}) {
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "save", requestId, path, metadata });
    if (response.type !== "saved" || response.requestId !== requestId) throw new Error("Unexpected PPO save response");
    return response;
  }

  async close() {
    if (!this.process) return;
    if (this.process.exitCode === null && this.process.stdin.writable) {
      const response = this.wait(); this.send({ type: "close" }); await response.catch(() => undefined);
    }
    this.lines?.close(); this.process.kill(); this.process = undefined;
    this.featureSpec = undefined;
  }
}
