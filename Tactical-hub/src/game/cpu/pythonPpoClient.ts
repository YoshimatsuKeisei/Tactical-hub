import { performance } from "node:perf_hooks";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { deflateRawSync } from "node:zlib";
import { createInterface, type Interface } from "node:readline";
import type {
  EncodedLegalActionsSparseV2,
  EncodedLegalActionsV2,
} from "./rlActionEncoder";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { EncodedObservation } from "./rlObservationEncoder";
import type { RlSelectedTorchDevice, RlTorchDevice } from "./rlTorchDevice";
import {
  appendPpoUpdateScalars,
  packPpoActBatchInput,
  packPpoActInput,
  packPpoEncodedSamples,
  type PpoEncodedSample,
  type PpoUpdateScalarSample,
} from "./rlPpoPackedBatch";
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

export type PpoRetentionChunkStats = {
  retentionId: string;
  batchSize: number;
  rawBytes: number;
  compressedBytes: number;
  rawSha256: string;
  tensorSignature: string;
};

export type PpoRetentionStats = {
  currentChunks: number;
  pendingChunks?: number;
  currentRetainedBytes: number;
  peakRetainedBytes: number;
  pendingRawBytes?: number;
  peakPendingRawBytes?: number;
  storedChunks: number;
  storedSamples: number;
  rawBytes: number;
  compressedBytes: number;
};

type Response =
  | { type: "ready"; selectedDevice: RlSelectedTorchDevice; updateCount: number; episodeCount: number }
  | { type: "action"; requestId: number; actionIndex: number; logProbability: number; value: number }
  | { type: "actions"; requestId: number; actionIndices: number[]; logProbabilities: number[]; values: number[] }
  | { type: "updateBegun"; requestId: number; totalSamples: number }
  | { type: "updateChunkAccepted"; requestId: number; acceptedSamples: number; accumulatedSamples: number; featureAudit?: { batchSize: number; rawBytes: number; rawSha256: string; tensorSignature: string } }
  | { type: "retainedChunkStored"; requestId: number; retentionId: string; batchSize: number; rawBytes: number; compressedBytes: number; rawSha256: string }
  | { type: "retentionDiscarded"; requestId: number; discardedCount: number }
  | { type: "retentionStats"; requestId: number; currentChunks: number; pendingChunks?: number; currentRetainedBytes: number; peakRetainedBytes: number; pendingRawBytes?: number; peakPendingRawBytes?: number; storedChunks: number; storedSamples: number; rawBytes: number; compressedBytes: number }
  | { type: "diagnostics"; requestId: number; parameterHash: string; optimizerHash: string; rngHash: string; gradientHash: string }
  | { type: "updateResult"; requestId: number; sampleCount: number; loss: number; policyLoss: number; valueLoss: number; entropy: number; gradientNorm: number; policyParametersChanged: boolean; valueParametersChanged: boolean; updateCount: number; episodeCount: number; diagnostics?: { gradientHash: string; parameterHash: string; optimizerHash: string; rngHash: string } }
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
  private readonly actPathTimings = new Map<string, { count: number; totalMs: number }>();

  private actPathProfileEnabled() {
    return (
      this.options.env?.PPO_ACT_PATH_PROFILE === "1"
      || process.env.PPO_ACT_PATH_PROFILE === "1"
    );
  }

  private recordActPath(stage: string, elapsedMs: number) {
    if (!this.actPathProfileEnabled()) return;
    const current = this.actPathTimings.get(stage) ?? { count: 0, totalMs: 0 };
    current.count += 1;
    current.totalMs += elapsedMs;
    this.actPathTimings.set(stage, current);
  }

  private emitActPathProfile() {
    if (!this.actPathProfileEnabled() || !this.actPathTimings.size) return;
    const stages = Object.fromEntries(
      Array.from(this.actPathTimings.entries()).map(([name, item]) => [
        name,
        {
          count: item.count,
          totalMs: Number(item.totalMs.toFixed(3)),
          avgMs: Number((item.totalMs / item.count).toFixed(6)),
        },
      ]),
    );
    process.stderr.write(
      "[PPO act path node] " + JSON.stringify({ stages }) + "\n",
    );
  }

  constructor(private readonly options: {
    command?: string;
    args?: string[];
    cwd?: string;
    device?: RlTorchDevice;
    env?: NodeJS.ProcessEnv;
    compactPaddedRows?: boolean;
  } = {}) {}

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
      ...(packed.rowCompaction
        ? { rowCompaction: packed.rowCompaction }
        : {}),
      ...(packed.actionSparseShape
        ? { actionSparseShape: packed.actionSparseShape }
        : {}),
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
      env: this.options.env ? { ...process.env, ...this.options.env } : process.env,
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

  async act(
    observation: EncodedObservation,
    legalActions: EncodedLegalActionsV2,
    options: { retentionId?: string } = {},
  ) {
    if (!this.featureSpec) throw new Error("Python PPO Feature Spec is not initialized");
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.sendPreparedPacked(
      { type: "packedAct", requestId, ...(options.retentionId ? { retentionId: options.retentionId } : {}) },
      packPpoActInput(
        observation,
        legalActions.actions,
        this.featureSpec,
        {
          compactMaskedPrefixes:
            this.options.compactPaddedRows ?? false,
        },
      ),
    );
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "action" || response.requestId !== requestId) throw new Error("Unexpected PPO action response");
    if (!Number.isInteger(response.actionIndex) || response.actionIndex < 0 || response.actionIndex >= legalActions.actionKeys.length) throw new Error("PPO returned an illegal action index");
    if (![response.logProbability, response.value].every(Number.isFinite)) throw new Error("PPO returned NaN or Inf");
    return { ...response, actionKey: legalActions.actionKeys[response.actionIndex] };
  }

  async actBatch(
    samples: Array<{
      observation: EncodedObservation;
      legalActions: EncodedLegalActionsV2 | EncodedLegalActionsSparseV2;
    }>,
    options: { retentionIds?: string[]; retentionBatchId?: string } = {},
  ) {
    if (!this.featureSpec) throw new Error("Python PPO Feature Spec is not initialized");
    if (!samples.length) throw new Error("PPO action batch cannot be empty");
    const retentionIds = options.retentionIds;
    const retentionBatchId = options.retentionBatchId;
    if (retentionIds && retentionBatchId) {
      throw new Error(
        "PPO batched action cannot use retentionIds and retentionBatchId together",
      );
    }
    if (retentionBatchId !== undefined && !retentionBatchId) {
      throw new Error("PPO retentionBatchId must not be empty");
    }
    if (retentionIds) {
      if (retentionIds.length !== samples.length) {
        throw new Error("PPO batched retention IDs must match sample count");
      }
      if (retentionIds.some((retentionId) => !retentionId)) {
        throw new Error("PPO batched retention IDs must not be empty");
      }
      if (new Set(retentionIds).size !== retentionIds.length) {
        throw new Error("PPO batched retention IDs must be unique");
      }
    }
    const sparseActionTransport =
      this.options.env?.PPO_SPARSE_ACTION_TRANSPORT === "1";
    if (sparseActionTransport && retentionIds) {
      throw new Error(
        "Sparse Action transport does not support per-sample retentionIds",
      );
    }
    const directSparseCount = samples.filter(
      ({ legalActions }) => "sparseActions" in legalActions,
    ).length;
    if (directSparseCount !== 0 && directSparseCount !== samples.length) {
      throw new Error(
        "PPO action batch cannot mix dense and direct sparse legal actions",
      );
    }
    if (directSparseCount && !sparseActionTransport) {
      throw new Error(
        "Direct sparse legal actions require sparse Action transport",
      );
    }

    const packStarted = performance.now();
    const packed = packPpoActBatchInput(
      samples.map(({ observation, legalActions }) =>
        "sparseActions" in legalActions
          ? {
            observation,
            sparseActions: legalActions.sparseActions,
          }
          : {
            observation,
            actions: legalActions.actions,
          }
      ),
      this.featureSpec,
      {
        compactMaskedPrefixes:
          this.options.compactPaddedRows ?? false,
        sparseActions: sparseActionTransport,
      },
    );
    this.recordActPath("node_pack", performance.now() - packStarted);

    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    const writeStarted = performance.now();
    this.sendPreparedPacked(
      {
        type: "packedActBatch",
        requestId,
        ...(retentionIds ? { retentionIds } : {}),
        ...(retentionBatchId ? { retentionBatchId } : {}),
      },
      packed,
    );
    this.recordActPath("node_write_enqueue", performance.now() - writeStarted);
    const waitStarted = performance.now();
    const response = await responsePromise;
    this.recordActPath("node_rpc_wait", performance.now() - waitStarted);
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "actions" || response.requestId !== requestId) throw new Error("Unexpected PPO batch-action response");
    if (
      response.actionIndices.length !== samples.length
      || response.logProbabilities.length !== samples.length
      || response.values.length !== samples.length
    ) throw new Error("PPO returned an action batch with the wrong length");
    return samples.map((sample, index) => {
      const actionIndex = response.actionIndices[index];
      const logProbability = response.logProbabilities[index];
      const value = response.values[index];
      if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= sample.legalActions.actionKeys.length) {
        throw new Error("PPO returned an illegal batched action index");
      }
      if (![logProbability, value].every(Number.isFinite)) throw new Error("PPO returned NaN or Inf in batched action output");
      return {
        actionIndex,
        logProbability,
        value,
        actionKey: sample.legalActions.actionKeys[actionIndex],
      };
    });
  }

  /**
   * Experimental strict-equivalence bulk path.
   * Each sample stays as an independent batch-1 packed payload; Python may
   * overlap only the CUDA forwards, then samples actions in original order.
   */
  async actStreamBatch(samples: Array<{ observation: EncodedObservation; legalActions: EncodedLegalActionsV2 }>) {
    if (!this.featureSpec) throw new Error("Python PPO Feature Spec is not initialized");
    if (!samples.length) throw new Error("PPO stream batch cannot be empty");
    if (!this.process?.stdin.writable) throw new Error("Python PPO process is not running");

    const packed = samples.map(({ observation, legalActions }) =>
      packPpoActInput(observation, legalActions.actions, this.featureSpec!),
    );
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    const totalByteLength = packed.reduce((sum, entry) => sum + entry.payload.byteLength, 0);

    this.process.stdin.write(`${JSON.stringify({
      type: "packedActStreamBatch",
      requestId,
      encoding: "packed-v1-stream-batch",
      byteLength: totalByteLength,
      sampleCount: packed.length,
      samples: packed.map((entry) => ({
        byteLength: entry.payload.byteLength,
        batchSize: entry.batchSize,
        tensors: entry.tensors,
      })),
    })}\n`);
    for (const entry of packed) this.process.stdin.write(entry.payload);

    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "actions" || response.requestId !== requestId) {
      throw new Error("Unexpected PPO stream-batch action response");
    }
    if (
      response.actionIndices.length !== samples.length
      || response.logProbabilities.length !== samples.length
      || response.values.length !== samples.length
    ) throw new Error("PPO returned a stream-action batch with the wrong length");

    return samples.map((sample, index) => {
      const actionIndex = response.actionIndices[index];
      const logProbability = response.logProbabilities[index];
      const value = response.values[index];
      if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= sample.legalActions.actionKeys.length) {
        throw new Error("PPO returned an illegal stream-batched action index");
      }
      if (![logProbability, value].every(Number.isFinite)) {
        throw new Error("PPO returned NaN or Inf in stream-batched action output");
      }
      return {
        actionIndex,
        logProbability,
        value,
        actionKey: sample.legalActions.actionKeys[actionIndex],
      };
    });
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

  async accumulatePrepacked(
    base: PackedBcBatch,
    scalars: PpoUpdateScalarSample[],
  ) {
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.sendPreparedPacked(
      { type: "packedUpdateChunk", requestId },
      appendPpoUpdateScalars(base, scalars),
    );
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (
      response.type !== "updateChunkAccepted"
      || response.requestId !== requestId
      || response.acceptedSamples !== base.batchSize
    ) throw new Error("Unexpected PPO prepacked update-chunk response");
    return response;
  }

  async retainPackedChunk(
    retentionId: string,
    base: PackedBcBatch,
    options: { corruptCompressedByteForTest?: boolean } = {},
  ): Promise<PpoRetentionChunkStats> {
    if (!this.process?.stdin.writable) throw new Error("Python PPO process is not running");
    if (!retentionId) throw new Error("PPO retentionId must not be empty");
    const requestId = this.nextRequestId++;
    const rawSha256 = createHash("sha256").update(base.payload).digest("hex");
    const tensorSignature = createHash("sha256").update(JSON.stringify(base.tensors.map((tensor) => ({
      name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, byteLength: tensor.byteLength,
    })))).digest("hex");
    const compressed = deflateRawSync(base.payload, { level: 1 });
    if (options.corruptCompressedByteForTest && compressed.byteLength) {
      compressed[Math.floor(compressed.byteLength / 2)] ^= 0x01;
    }
    const responsePromise = this.wait();
    this.process.stdin.write(`${JSON.stringify({
      type: "retainPackedChunk",
      requestId,
      retentionId,
      encoding: "packed-deflate-raw-v1",
      codec: "deflate-raw-1",
      byteLength: compressed.byteLength,
      rawByteLength: base.payload.byteLength,
      rawSha256,
      batchSize: base.batchSize,
      tensors: base.tensors,
    })}\n`);
    this.process.stdin.write(compressed);
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (
      response.type !== "retainedChunkStored"
      || response.requestId !== requestId
      || response.retentionId !== retentionId
      || response.batchSize !== base.batchSize
      || response.rawSha256 !== rawSha256
    ) throw new Error("Unexpected PPO retained-chunk response");
    return { ...response, tensorSignature };
  }

  async accumulateRetainedBatches(
    retentionIds: string[],
    scalars: PpoUpdateScalarSample[],
  ) {
    if (!this.process?.stdin.writable) {
      throw new Error("Python PPO process is not running");
    }
    if (!retentionIds.length) {
      throw new Error("PPO retained batch IDs cannot be empty");
    }
    if (retentionIds.some((retentionId) => !retentionId)) {
      throw new Error("PPO retained batch IDs must not be empty");
    }
    if (new Set(retentionIds).size !== retentionIds.length) {
      throw new Error("PPO retained batch IDs must be unique per update chunk");
    }
    if (!scalars.length) {
      throw new Error("PPO retained batch scalar list cannot be empty");
    }
    if (!scalars.every((sample) =>
      [sample.oldLogProbability, sample.advantage, sample.return].every(Number.isFinite)
    )) {
      throw new Error("PPO retained batch scalars contain NaN or Inf");
    }

    const sampleCount = scalars.length;
    const old = Float32Array.from(
      scalars.map((sample) => sample.oldLogProbability),
    );
    const advantages = Float32Array.from(
      scalars.map((sample) => sample.advantage),
    );
    const returns = Float32Array.from(
      scalars.map((sample) => sample.return),
    );
    const payload = Buffer.concat([
      Buffer.from(old.buffer, old.byteOffset, old.byteLength),
      Buffer.from(advantages.buffer, advantages.byteOffset, advantages.byteLength),
      Buffer.from(returns.buffer, returns.byteOffset, returns.byteLength),
    ]);

    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.process.stdin.write(`${JSON.stringify({
      type: "retainedBatchUpdateChunk",
      requestId,
      retentionIds,
      encoding: "ppo-retained-scalars-v1",
      byteLength: payload.byteLength,
      sampleCount,
    })}\n`);
    this.process.stdin.write(payload);

    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (
      response.type !== "updateChunkAccepted"
      || response.requestId !== requestId
      || response.acceptedSamples !== sampleCount
    ) {
      throw new Error("Unexpected PPO retained batch update response");
    }
    return response;
  }

  async accumulateRetained(
    retentionIds: string | string[],
    scalars: PpoUpdateScalarSample[],
  ) {
    if (!this.process?.stdin.writable) throw new Error("Python PPO process is not running");
    if (!scalars.length) throw new Error("PPO retained scalar batch cannot be empty");
    if (!scalars.every((sample) => [sample.oldLogProbability, sample.advantage, sample.return].every(Number.isFinite))) {
      throw new Error("PPO retained scalars contain NaN or Inf");
    }
    const ids = Array.isArray(retentionIds) ? retentionIds : [retentionIds];
    const batchSize = scalars.length;
    if (ids.length !== batchSize) throw new Error("PPO retained IDs must match scalar batch size");
    const old = Float32Array.from(scalars.map((sample) => sample.oldLogProbability));
    const advantages = Float32Array.from(scalars.map((sample) => sample.advantage));
    const returns = Float32Array.from(scalars.map((sample) => sample.return));
    const payload = Buffer.concat([
      Buffer.from(old.buffer, old.byteOffset, old.byteLength),
      Buffer.from(advantages.buffer, advantages.byteOffset, advantages.byteLength),
      Buffer.from(returns.buffer, returns.byteOffset, returns.byteLength),
    ]);
    const requestId = this.nextRequestId++;
    const responsePromise = this.wait();
    this.process.stdin.write(`${JSON.stringify({
      type: "retainedUpdateChunk",
      requestId,
      retentionIds: ids,
      encoding: "ppo-retained-scalars-v1",
      byteLength: payload.byteLength,
      batchSize,
    })}\n`);
    this.process.stdin.write(payload);
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (
      response.type !== "updateChunkAccepted"
      || response.requestId !== requestId
      || response.acceptedSamples !== batchSize
    ) throw new Error("Unexpected PPO retained update-chunk response");
    return response;
  }

  async discardRetained(retentionIds: string[]) {
    if (!retentionIds.length) return { discardedCount: 0 };
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "discardRetained", requestId, retentionIds });
    if (response.type !== "retentionDiscarded" || response.requestId !== requestId) {
      throw new Error("Unexpected PPO retention-discard response");
    }
    return response;
  }

  async retentionStats(): Promise<PpoRetentionStats> {
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "retentionStats", requestId });
    if (response.type !== "retentionStats" || response.requestId !== requestId) {
      throw new Error("Unexpected PPO retention-stats response");
    }
    const { type: _type, requestId: _requestId, ...stats } = response;
    return stats;
  }

  async diagnostics() {
    const requestId = this.nextRequestId++;
    const response = await this.request({ type: "diagnostics", requestId });
    if (response.type !== "diagnostics" || response.requestId !== requestId) {
      throw new Error("Unexpected PPO diagnostics response");
    }
    const { type: _type, requestId: _requestId, ...diagnostics } = response;
    return diagnostics;
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
    this.emitActPathProfile();
  }
}
