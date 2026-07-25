import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { getRlImitationEpisodeFeatureSpec, replayHeuristicImitationEpisode } from "./rlImitationCollector";
import { PythonBcTrainerClient, type BcEncodedSample } from "./pythonBcTrainerClient";
import type { RlFeatureSpec } from "./rlFeatureSpec";
import { readRlImitationEpisodes, type EpisodeRange } from "./rlReplayReader";
import { runParallelBcReplay, type NumberedReplayEpisode } from "./rlBcReplayParallel";

export type BcMetrics = {
  loss: number;
  accuracy: number;
  sampleCount: number;
  elapsedMs: number;
};

export type BcEpochResult = {
  epoch: number;
  train: BcMetrics;
  validation: BcMetrics;
};

export type BehavioralCloningResult = {
  epochs: BcEpochResult[];
  test: BcMetrics;
  bestEpoch: number;
  bestValidationAccuracy: number;
  checkpointPath: string;
  initialParameterHash: string;
  trainedParameterHash: string;
  reloadedParameterHash: string;
  torchThreads: number;
  torchInteropThreads: number;
};

type SplitAccumulator = {
  lossSum: number;
  correct: number;
  count: number;
};

export async function runBehavioralCloning(input: {
  dataPath: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  checkpointPath: string;
  trainRange: EpisodeRange;
  validationRange: EpisodeRange;
  testRange: EpisodeRange;
  seed: number;
  pythonCommand?: string;
  workerCount?: number;
  torchThreads?: number;
  torchInteropThreads?: number;
}): Promise<BehavioralCloningResult> {
  if (!Number.isInteger(input.epochs) || input.epochs <= 0) throw new Error("epochs must be positive");
  if (!Number.isInteger(input.batchSize) || input.batchSize <= 0) throw new Error("batchSize must be positive");
  if (!(input.learningRate > 0) || !Number.isFinite(input.learningRate)) throw new Error("learningRate must be finite and positive");
  const workerCount = input.workerCount ?? 1;
  if (!Number.isInteger(workerCount) || workerCount <= 0) throw new Error("workerCount must be positive");
  for (const [name, value] of [["torchThreads", input.torchThreads], ["torchInteropThreads", input.torchInteropThreads]] as const) {
    if (value !== undefined && (!Number.isInteger(value) || value <= 0)) throw new Error(`${name} must be a positive integer`);
  }
  const checkpointPath = resolve(input.checkpointPath);
  await mkdir(dirname(checkpointPath), { recursive: true });
  const client = new PythonBcTrainerClient();
  let featureSpec: RlFeatureSpec | undefined;
  let initialParameterHash = "";

  const ensureStarted = async (candidate: RlFeatureSpec) => {
    if (!featureSpec) {
      featureSpec = candidate;
      await client.start({
        featureSpec: candidate,
        learningRate: input.learningRate,
        seed: input.seed,
        command: input.pythonCommand,
        torchThreads: input.torchThreads,
        torchInteropThreads: input.torchInteropThreads,
      });
      initialParameterHash = await client.parameterHash();
    } else if (JSON.stringify(featureSpec) !== JSON.stringify(candidate)) {
      throw new Error("Replay episodes use incompatible feature specs");
    }
  };

  const runSplit = async (range: EpisodeRange, train: boolean): Promise<BcMetrics> => {
    const started = performance.now();
    const aggregate: SplitAccumulator = { lossSum: 0, correct: 0, count: 0 };
    let batch: BcEncodedSample[] = [];
    const flush = async () => {
      if (!batch.length) return;
      const result = await client.batch(batch, train);
      aggregate.lossSum += result.lossSum;
      aggregate.correct += result.correct;
      aggregate.count += result.count;
      batch = [];
    };
    let episodeCount = 0;
    if (workerCount === 1) {
      for await (const { episode } of readRlImitationEpisodes(input.dataPath, range)) {
        episodeCount += 1;
        await replayHeuristicImitationEpisode({
          episode,
          onFeatureSpec: ensureStarted,
          onEncodedDecision: async (decision) => {
            if (decision.encodedLegalActions.actionKeys[decision.selectedActionIndex] !== decision.record.selectedActionKey) {
              throw new Error(`Replay target index mismatch for ${decision.record.selectedActionKey}`);
            }
            batch.push({
              observation: decision.encodedObservation,
              actions: decision.encodedLegalActions.actions,
              targetIndex: decision.selectedActionIndex,
            });
            if (batch.length >= input.batchSize) await flush();
          },
        });
      }
      await flush();
    } else {
      const episodes: NumberedReplayEpisode[] = [];
      for await (const item of readRlImitationEpisodes(input.dataPath, range)) episodes.push(item);
      episodeCount = episodes.length;
      if (episodes.length) {
        await ensureStarted(getRlImitationEpisodeFeatureSpec(episodes[0].episode));
        const replay = await runParallelBcReplay({
          episodes,
          workerCount,
          batchSize: input.batchSize,
          onBatch: async (samples) => {
            const result = await client.batch(samples, train);
            aggregate.lossSum += result.lossSum;
            aggregate.correct += result.correct;
            aggregate.count += result.count;
          },
        });
        if (replay.sampleCount !== aggregate.count) {
          throw new Error(`Parallel BC replay sample count mismatch: ${replay.sampleCount}/${aggregate.count}`);
        }
      }
    }
    if (!episodeCount) throw new Error(`No replay episodes found in range ${range.from}-${range.to}`);
    if (!aggregate.count) throw new Error(`No decisions found in range ${range.from}-${range.to}`);
    const metrics = {
      loss: aggregate.lossSum / aggregate.count,
      accuracy: aggregate.correct / aggregate.count,
      sampleCount: aggregate.count,
      elapsedMs: performance.now() - started,
    };
    if (![metrics.loss, metrics.accuracy, metrics.elapsedMs].every(Number.isFinite)) throw new Error("BC metrics contain NaN or Inf");
    return metrics;
  };

  try {
    const epochs: BcEpochResult[] = [];
    let bestEpoch = 0;
    let bestValidationAccuracy = Number.NEGATIVE_INFINITY;
    for (let epoch = 1; epoch <= input.epochs; epoch += 1) {
      const train = await runSplit(input.trainRange, true);
      const validation = await runSplit(input.validationRange, false);
      epochs.push({ epoch, train, validation });
      if (validation.accuracy > bestValidationAccuracy) {
        bestEpoch = epoch;
        bestValidationAccuracy = validation.accuracy;
        await client.save(checkpointPath, { epoch, validationAccuracy: validation.accuracy, seed: input.seed });
      }
    }
    const trainedParameterHash = await client.parameterHash();
    await client.load(checkpointPath);
    const reloadedParameterHash = await client.parameterHash();
    const test = await runSplit(input.testRange, false);
    const appliedThreads = client.getAppliedThreads();
    return {
      epochs,
      test,
      bestEpoch,
      bestValidationAccuracy,
      checkpointPath,
      initialParameterHash,
      trainedParameterHash,
      reloadedParameterHash,
      ...appliedThreads,
    };
  } finally {
    await client.close();
  }
}
