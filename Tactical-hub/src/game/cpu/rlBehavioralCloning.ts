import { access, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { getRlImitationEpisodeFeatureSpec, getRlImitationHeaderFeatureSpec } from "./rlImitationCollector";
import { PythonBcTrainerClient, type BcCheckpointAccumulator, type BcEpisodeCheckpointState } from "./pythonBcTrainerClient";
import type { RlFeatureSpec } from "./rlFeatureSpec";
import { createRlReplayEpisodeIndex, readRlImitationEpisodeAt, type EpisodeRange } from "./rlReplayReader";
import { runDirectBcReplayEpisode } from "./rlBcReplayParallel";
import type { RlSelectedTorchDevice, RlTorchDevice } from "./rlTorchDevice";
import { getDefaultRlReplaySidecarDirectory } from "./rlReplayRngSidecar";

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
export type BehavioralCloningProgress = {
  kind: "episode" | "heartbeat";
  phase: "train" | "validation" | "test";
  epoch: number;
  totalEpochs: number;
  episode: number;
  totalEpisodes: number;
  processedSamples: number;
  processedBatches: number;
  elapsedMs: number;
  recentSamplesPerSecond: number;
};

export type BehavioralCloningResult = {
  epochs: BcEpochResult[];
  test?: BcMetrics;
  bestEpoch: number;
  bestValidationAccuracy: number;
  checkpointPath: string;
  initialParameterHash: string;
  initialOptimizerStateHash: string;
  trainedParameterHash: string;
  optimizerStateHash: string;
  reloadedParameterHash: string;
  torchThreads: number;
  torchInteropThreads: number;
  selectedDevice: RlSelectedTorchDevice;
  completedEpoch: number;
  resumedFrom?: string;
  latestCheckpointPath?: string;
  replayCache: {
    generatedSidecarCount: number;
    reusedSidecarCount: number;
    sidecarPreparationMs: number;
    directReplayMs: number;
  };
};

type SplitAccumulator = {
  lossSum: number;
  correct: number;
  count: number;
};

const emptyAccumulator = (): SplitAccumulator => ({ lossSum: 0, correct: 0, count: 0 });
const metricsFromAccumulator = (aggregate: SplitAccumulator, elapsedMs: number): BcMetrics => {
  if (!aggregate.count) throw new Error("BC split contains no decisions");
  const metrics = { loss: aggregate.lossSum / aggregate.count, accuracy: aggregate.correct / aggregate.count, sampleCount: aggregate.count, elapsedMs };
  if (![metrics.loss, metrics.accuracy, metrics.elapsedMs].every(Number.isFinite)) throw new Error("BC metrics contain NaN or Inf");
  return metrics;
};
const rangeValues = (range: EpisodeRange) => Array.from({ length: range.to - range.from + 1 }, (_, index) => range.from + index);
const assertRanges = (ranges: Array<[string, EpisodeRange]>) => {
  for (const [name, range] of ranges) if (!Number.isInteger(range.from) || !Number.isInteger(range.to) || range.from <= 0 || range.to < range.from) throw new Error(`${name} is invalid`);
  const occupied = new Set<number>();
  for (const [name, range] of ranges) for (const episode of rangeValues(range)) {
    if (occupied.has(episode)) throw new Error(`${name} overlaps another episode range`);
    occupied.add(episode);
  }
};

export async function runBehavioralCloning(input: {
  dataPath: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  checkpointPath: string;
  latestCheckpointPath?: string;
  resumePath?: string;
  trainRange: EpisodeRange;
  validationRange: EpisodeRange;
  testRange: EpisodeRange;
  seed: number;
  pythonCommand?: string;
  workerCount?: number;
  torchThreads?: number;
  torchInteropThreads?: number;
  device?: RlTorchDevice;
  runTest?: boolean;
  onProgress?: (progress: BehavioralCloningProgress) => void;
  onStatus?: (message: string) => void;
  onEpisodeCheckpointSaved?: (
    state: BcEpisodeCheckpointState,
    path: string,
    episodeNumber: number,
    hashes: { model: string; optimizer: string },
  ) => void | Promise<void>;
  onReplayEpisodeStarted?: (episodeNumber: number, execution: "direct" | "worker") => void;
}): Promise<BehavioralCloningResult> {
  if (!Number.isInteger(input.epochs) || input.epochs <= 0) throw new Error("epochs must be positive");
  if (!Number.isInteger(input.batchSize) || input.batchSize <= 0) throw new Error("batchSize must be positive");
  if (!(input.learningRate > 0) || !Number.isFinite(input.learningRate)) throw new Error("learningRate must be finite and positive");
  if (input.device !== undefined && !["auto", "cpu", "cuda"].includes(input.device)) throw new Error("device must be auto, cpu, or cuda");
  const workerCount = input.workerCount ?? 1;
  if (!Number.isInteger(workerCount) || workerCount <= 0) throw new Error("workerCount must be positive");
  if (workerCount !== 1 && (input.latestCheckpointPath || input.resumePath)) {
    throw new Error("Episode-level checkpoint/resume currently requires workers=1");
  }
  assertRanges([["trainRange", input.trainRange], ["validationRange", input.validationRange], ["testRange", input.testRange]]);
  for (const [name, value] of [["torchThreads", input.torchThreads], ["torchInteropThreads", input.torchInteropThreads]] as const) {
    if (value !== undefined && (!Number.isInteger(value) || value <= 0)) throw new Error(`${name} must be a positive integer`);
  }
  const checkpointPath = resolve(input.checkpointPath);
  const latestCheckpointPath = input.latestCheckpointPath ? resolve(input.latestCheckpointPath) : undefined;
  const resumePath = input.resumePath ? resolve(input.resumePath) : undefined;
  const sidecarDirectory = getDefaultRlReplaySidecarDirectory(input.dataPath);
  await mkdir(dirname(checkpointPath), { recursive: true });
  if (latestCheckpointPath) await mkdir(dirname(latestCheckpointPath), { recursive: true });
  if (resumePath) {
    try { await access(resumePath); }
    catch { throw new Error(`Resume checkpoint does not exist: ${resumePath}`); }
  }
  const client = new PythonBcTrainerClient();
  let featureSpec: RlFeatureSpec | undefined;
  let initialParameterHash = "";
  const replayCache = {
    generatedSidecarCount: 0,
    reusedSidecarCount: 0,
    sidecarPreparationMs: 0,
    directReplayMs: 0,
  };
  const trainingMetadata = {
    batchSize: input.batchSize,
    trainRange: input.trainRange,
    validationRange: input.validationRange,
    testRange: input.testRange,
  };

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
        device: input.device ?? "auto",
      });
    } else if (JSON.stringify(featureSpec) !== JSON.stringify(candidate)) {
      throw new Error("Replay episodes use incompatible feature specs");
    }
  };
  const replayIndex = await createRlReplayEpisodeIndex(input.dataPath, Math.max(input.trainRange.to, input.validationRange.to, input.runTest ? input.testRange.to : 0));
  for (const range of [input.trainRange, input.validationRange, ...(input.runTest ? [input.testRange] : [])]) {
    if (!replayIndex.episodes[range.from - 1] || !replayIndex.episodes[range.to - 1]) throw new Error(`Replay episode range ${range.from}-${range.to} is incomplete`);
  }
  await ensureStarted(getRlImitationHeaderFeatureSpec(replayIndex.episodes[input.trainRange.from - 1].header));

  const saveLatest = async (state: BcEpisodeCheckpointState, episodeBoundary = false, episodeNumber?: number) => {
    if (!latestCheckpointPath) return;
    await client.saveTrainingCheckpoint({ path: latestCheckpointPath, state });
    input.onStatus?.(`[BC] latest checkpoint saved epoch=${state.currentEpoch} phase=${state.phase} nextEpisode=${state.nextEpisodeNumber} path=${latestCheckpointPath}`);
    if (episodeBoundary && input.onEpisodeCheckpointSaved) {
      await input.onEpisodeCheckpointSaved(structuredClone(state), latestCheckpointPath, episodeNumber!, {
        model: await client.parameterHash(), optimizer: await client.optimizerHash(),
      });
    }
  };

  const runSplit = async (
    range: EpisodeRange,
    train: boolean,
    phase: BehavioralCloningProgress["phase"],
    epoch: number,
    startEpisode: number,
    aggregate: SplitAccumulator,
    completedEpisodes: number[],
    checkpointState?: BcEpisodeCheckpointState,
    onFinalEpisode?: (metrics: BcMetrics) => void | Promise<void>,
  ): Promise<BcMetrics> => {
    const started = performance.now();
    const totalEpisodes = range.to - range.from + 1;
    const resumedCompletedCount = completedEpisodes.length;
    let completedInInvocation = resumedCompletedCount;
    let processedBatches = 0;
    let previousProgressMs = started;
    let previousProgressSamples = 0;
    const reportProgress = (kind: BehavioralCloningProgress["kind"]) => {
      const now = performance.now();
      const seconds = Math.max((now - previousProgressMs) / 1000, 0.001);
      input.onProgress?.({
        kind,
        phase,
        epoch,
        totalEpochs: input.epochs,
        episode: Math.min(totalEpisodes, kind === "episode" ? completedInInvocation : completedInInvocation + 1),
        totalEpisodes,
        processedSamples: aggregate.count,
        processedBatches,
        elapsedMs: now - started,
        recentSamplesPerSecond: (aggregate.count - previousProgressSamples) / seconds,
      });
      previousProgressMs = now;
      previousProgressSamples = aggregate.count;
    };
    const heartbeat = setInterval(() => reportProgress("heartbeat"), 10_000);
    heartbeat.unref();
    let episodeCount = 0;
    try {
      for (let episodeNumber = startEpisode; episodeNumber <= range.to; episodeNumber += 1) {
        const item = await readRlImitationEpisodeAt(replayIndex, episodeNumber);
        input.onReplayEpisodeStarted?.(episodeNumber, "direct");
        episodeCount += 1;
        await ensureStarted(getRlImitationEpisodeFeatureSpec(item.episode));
        const replay = await runDirectBcReplayEpisode({
          item,
          batchSize: input.batchSize,
          sidecarDirectory,
          onBatch: async (packedBatch) => {
            const result = await client.batchPacked(packedBatch, train);
            aggregate.lossSum += result.lossSum;
            aggregate.correct += result.correct;
            aggregate.count += result.count;
            processedBatches += 1;
          },
        });
        completedEpisodes.push(episodeNumber);
        completedInInvocation += 1;
        replayCache.generatedSidecarCount += Number(replay.sidecarGenerated);
        replayCache.reusedSidecarCount += Number(!replay.sidecarGenerated);
        replayCache.sidecarPreparationMs += replay.sidecarPreparationMs;
        replayCache.directReplayMs += replay.directReplayMs;
        if (checkpointState) {
          checkpointState.nextEpisodeNumber = episodeNumber + 1;
          checkpointState.trainAccumulator = phase === "train" ? { ...aggregate } : checkpointState.trainAccumulator;
          checkpointState.validationAccumulator = phase === "validation" ? { ...aggregate } : checkpointState.validationAccumulator;
          if (episodeNumber === range.to) await onFinalEpisode?.(metricsFromAccumulator(aggregate, performance.now() - started));
        }
        reportProgress("episode");
        if (checkpointState) await saveLatest(checkpointState, true, episodeNumber);
      }
    } finally {
      clearInterval(heartbeat);
    }
    if (startEpisode <= range.to && !episodeCount) throw new Error(`No replay episodes found in range ${startEpisode}-${range.to}`);
    return metricsFromAccumulator(aggregate, performance.now() - started);
  };

  try {
    const epochs: BcEpochResult[] = [];
    let state: BcEpisodeCheckpointState = {
      schemaVersion: 3,
      checkpointKind: "behavioral_cloning_training",
      currentEpoch: 1,
      completedEpoch: 0,
      phase: "train",
      nextEpisodeNumber: input.trainRange.from,
      completedTrainEpisodes: [], completedValidationEpisodes: [],
      trainAccumulator: emptyAccumulator(), validationAccumulator: emptyAccumulator(),
      bestEpoch: null,
      bestValidationAccuracy: null,
      seed: input.seed,
      learningRate: input.learningRate,
      batchSize: input.batchSize,
      trainRange: input.trainRange,
      validationRange: input.validationRange,
      testRange: input.testRange,
      metadata: trainingMetadata,
    };
    if (resumePath) {
      const resumed = await client.resumeTrainingCheckpoint(resumePath, {
        seed: input.seed, learningRate: input.learningRate, batchSize: input.batchSize,
        trainRange: input.trainRange, validationRange: input.validationRange, testRange: input.testRange,
      });
      if (resumed.schemaVersion === 2) {
        state = { ...state, currentEpoch: resumed.completedEpoch + 1, completedEpoch: resumed.completedEpoch, bestEpoch: resumed.bestEpoch, bestValidationAccuracy: resumed.bestValidationAccuracy };
      } else {
        state = resumed;
      }
      const phaseRange = state.phase === "train" ? input.trainRange : input.validationRange;
      const phaseCompleted = state.phase === "train" ? state.completedTrainEpisodes : state.completedValidationEpisodes;
      const expectedCompleted = rangeValues({ from: phaseRange.from, to: Math.max(phaseRange.from - 1, state.nextEpisodeNumber - 1) });
      if (state.nextEpisodeNumber < phaseRange.from || state.nextEpisodeNumber > phaseRange.to + 1
        || JSON.stringify(phaseCompleted) !== JSON.stringify(expectedCompleted)) {
        throw new Error("Resume checkpoint episode position is inconsistent with its phase and completed episodes");
      }
      const allTrain = rangeValues(input.trainRange);
      if (state.phase === "validation" && JSON.stringify(state.completedTrainEpisodes) !== JSON.stringify(allTrain)) {
        throw new Error("Resume checkpoint is missing completed train episodes");
      }
      if (state.currentEpoch > input.epochs && state.completedEpoch < input.epochs) throw new Error("Resume checkpoint currentEpoch exceeds target epochs");
      input.onStatus?.(`[BC] resumed checkpoint=${resumePath}`);
      input.onStatus?.(`[BC] completedEpoch=${state.completedEpoch}`);
      input.onStatus?.(`[BC] nextEpoch=${state.currentEpoch}`);
      input.onStatus?.(`[BC] phase=${state.phase}`);
      input.onStatus?.(`[BC] nextEpisode=${state.nextEpisodeNumber}`);
      input.onStatus?.(`[BC] bestEpoch=${state.bestEpoch}`);
      input.onStatus?.(`[BC] bestValidationAccuracy=${state.bestValidationAccuracy}`);
    }
    initialParameterHash = await client.parameterHash();
    const initialOptimizerStateHash = await client.optimizerHash();
    let pendingTrainMetrics: BcMetrics | undefined;
    if (state.completedEpoch >= input.epochs) {
      input.onStatus?.(`[BC] no training required: completedEpoch=${state.completedEpoch} targetEpochs=${input.epochs}`);
    }
    while (state.currentEpoch <= input.epochs) {
      if (state.phase === "train") {
        pendingTrainMetrics = await runSplit(
          input.trainRange, true, "train", state.currentEpoch, state.nextEpisodeNumber,
          state.trainAccumulator, state.completedTrainEpisodes, state,
          () => { state.phase = "validation"; state.nextEpisodeNumber = input.validationRange.from; },
        );
      }
      if (state.phase === "validation") {
        const epoch = state.currentEpoch;
        const train = pendingTrainMetrics ?? metricsFromAccumulator(state.trainAccumulator, 0);
        await runSplit(
          input.validationRange, false, "validation", epoch, state.nextEpisodeNumber,
          state.validationAccumulator, state.completedValidationEpisodes, state,
          async (validation) => {
            epochs.push({ epoch, train, validation });
            if (state.bestValidationAccuracy === null || validation.accuracy > state.bestValidationAccuracy) {
              state.bestEpoch = epoch;
              state.bestValidationAccuracy = validation.accuracy;
              await client.save(checkpointPath, { epoch, validationAccuracy: validation.accuracy, seed: input.seed });
            }
            state.completedEpoch = epoch;
            state.currentEpoch = epoch + 1;
            state.phase = "train"; state.nextEpisodeNumber = input.trainRange.from;
            state.completedTrainEpisodes = []; state.completedValidationEpisodes = [];
            state.trainAccumulator = emptyAccumulator(); state.validationAccumulator = emptyAccumulator();
            pendingTrainMetrics = undefined;
          },
        );
      }
    }
    if (state.bestEpoch === null || state.bestValidationAccuracy === null) throw new Error("BC training has no completed validation result");
    const trainedParameterHash = await client.parameterHash();
    const optimizerStateHash = await client.optimizerHash();
    let reloadedParameterHash = trainedParameterHash;
    let test: BcMetrics | undefined;
    if (input.runTest) {
      try { await access(checkpointPath); }
      catch { throw new Error(`Best checkpoint does not exist and cannot be evaluated: ${checkpointPath}`); }
      await client.load(checkpointPath);
      reloadedParameterHash = await client.parameterHash();
      test = await runSplit(input.testRange, false, "test", Math.min(input.epochs, state.completedEpoch), input.testRange.from, emptyAccumulator(), []);
    }
    const appliedThreads = client.getAppliedThreads();
    return {
      epochs,
      test,
      bestEpoch: state.bestEpoch,
      bestValidationAccuracy: state.bestValidationAccuracy,
      checkpointPath,
      initialParameterHash,
      initialOptimizerStateHash,
      trainedParameterHash,
      optimizerStateHash,
      reloadedParameterHash,
      ...appliedThreads,
      completedEpoch: state.completedEpoch,
      resumedFrom: resumePath,
      latestCheckpointPath,
      replayCache,
    };
  } finally {
    await client.close();
  }
}
