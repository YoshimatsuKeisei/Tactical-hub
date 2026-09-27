import { createHash } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";
import { packBcEncodedSamples } from "./rlBcPackedBatch";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2, type RlFeatureSpecV2 } from "./rlFeatureSpec";
import { createRlObservationEncoderCache, encodeRlObservationV2 } from "./rlObservationEncoder";
import { PythonPpoClient } from "./pythonPpoClient";
import type { PpoEncodedSample, PpoUpdateScalarSample } from "./rlPpoPackedBatch";
import type { PackedBcBatch } from "./rlBcPackedBatch";
import { DEFAULT_PPO_HYPERPARAMETERS, runPpoSelfPlaySmoke } from "./rlPpoSelfPlay";
import { parseRlTorchDevice } from "./rlTorchDevice";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const positiveInteger = (name: string, fallback: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`);
  return parsed;
};
const required = (name: string) => {
  const result = value(name);
  if (!result) throw new Error(`${name} is required`);
  return result;
};
const seed = positiveInteger("--seed", 7);
const decisions = positiveInteger("--decisions", 512);
const chunkSize = positiveInteger("--replay-chunk-size", 32);
const featureAudit = args.includes("--feature-audit");
const initialCheckpoint = required("--initial-checkpoint");
const resume = required("--resume");
const python = value("--python") ?? "python";
const device = parseRlTorchDevice(value("--device") ?? "cuda");
const workspace = mkdtempSync(join(tmpdir(), "tactical-hub-ppo-retention-"));

process.env.PPO_EQUIVALENCE_DIAGNOSTICS = "1";
process.env.PPO_PHASE_PROFILE = "1";
process.env.PPO_NODE_PROFILE = "1";
delete process.env.PPO_PROFILE;
delete process.env.PPO_ACT_COMPILE_MODE;

const clientOptions = {
  command: python,
  device,
  env: {
    PPO_EQUIVALENCE_DIAGNOSTICS: "1",
    PPO_NODE_PROFILE: "1",
    PPO_PHASE_PROFILE: "1",
    PPO_PROFILE: "0",
  },
};

const sha256 = (value: Uint8Array) => createHash("sha256").update(value).digest("hex");

const auditPackedBase = (base: PackedBcBatch) => ({
  batchSize: base.batchSize,
  rawBytes: base.payload.byteLength,
  rawSha256: sha256(base.payload),
  tensorSignature: sha256(Buffer.from(JSON.stringify(base.tensors.map((descriptor) => ({
    name: descriptor.name,
    dtype: descriptor.dtype,
    shape: descriptor.shape,
    byteLength: descriptor.byteLength,
  }))))),
});

const auditScalars = (samples: PpoUpdateScalarSample[]) => {
  const old = Float32Array.from(samples.map((sample) => sample.oldLogProbability));
  const advantages = Float32Array.from(samples.map((sample) => sample.advantage));
  const returns = Float32Array.from(samples.map((sample) => sample.return));
  const payload = Buffer.concat([
    Buffer.from(old.buffer, old.byteOffset, old.byteLength),
    Buffer.from(advantages.buffer, advantages.byteOffset, advantages.byteLength),
    Buffer.from(returns.buffer, returns.byteOffset, returns.byteLength),
  ]);
  return { batchSize: samples.length, sha256: sha256(payload) };
};

class AuditedPpoClient extends PythonPpoClient {
  private auditedFeatureSpec?: RlFeatureSpecV2;
  readonly featureAudits: ReturnType<typeof auditPackedBase>[] = [];
  readonly scalarAudits: ReturnType<typeof auditScalars>[] = [];

  async start(input: Parameters<PythonPpoClient["start"]>[0]) {
    this.auditedFeatureSpec = input.featureSpec;
    return super.start(input);
  }

  async accumulatePacked(samples: PpoEncodedSample[]) {
    if (!this.auditedFeatureSpec) throw new Error("Audited PPO client has no Feature Spec");
    const base = packBcEncodedSamples(
      samples.map(({ observation, actions, targetIndex }) => ({ observation, actions, targetIndex })),
      this.auditedFeatureSpec,
    );
    this.featureAudits.push(auditPackedBase(base));
    this.scalarAudits.push(auditScalars(samples.map(({ oldLogProbability, advantage, return: expectedReturn }) => ({
      oldLogProbability,
      advantage: advantage!,
      return: expectedReturn!,
    }))));
    return super.accumulatePacked(samples);
  }

  async accumulateRetained(retentionIds: string | string[], scalars: PpoUpdateScalarSample[]) {
    this.scalarAudits.push(auditScalars(scalars));
    const response = await super.accumulateRetained(retentionIds, scalars);
    if (!response.featureAudit) {
      throw new Error("Retention feature audit was not returned");
    }
    this.featureAudits.push(response.featureAudit);
    return response;
  }
}

const makeClient = (audit = false) => {
  const options = audit
    ? {
      ...clientOptions,
      env: {
        ...clientOptions.env,
        PPO_RETENTION_EQUIVALENCE: "1",
      },
    }
    : clientOptions;
  return audit
    ? new AuditedPpoClient(options)
    : new PythonPpoClient(options);
};

const comparableSummary = (result: Awaited<ReturnType<typeof runPpoSelfPlaySmoke>>) => {
  const episode = result.adjudicated[0] ?? result.completed[0];
  if (!episode) throw new Error("Probe produced no completed/adjudicated episode");
  return {
    seed: episode.seed,
    decisionCount: episode.decisionCount,
    finalStateHash: episode.finalStateHash,
    outcomeKind: episode.outcomeKind,
    reason: episode.reason,
    limitReason: episode.limitReason,
    adjudication: episode.adjudication,
    environmentResult: episode.environmentResult,
  };
};
async function runVariant(label: "oracle" | "retention", retainTrajectory: boolean) {
  const outputCheckpoint = join(workspace, `${label}.pt`);
  const client = makeClient(featureAudit);
  const start = performance.now();
  const result = await runPpoSelfPlaySmoke({
    seed,
    episodes: 1,
    initialCheckpoint,
    outputCheckpoint,
    resume,
    safetyMaxTurns: 1_000,
    safetyMaxActions: decisions,
    replayChunkSize: chunkSize,
    memoryLogInterval: Math.max(decisions, 1),
    retainTrajectory,
    fastRlMovement: true,
    client,
  });
  const audited = client instanceof AuditedPpoClient ? client : undefined;
  return {
    result,
    outputCheckpoint,
    wallMs: performance.now() - start,
    featureAudits: audited?.featureAudits,
    scalarAudits: audited?.scalarAudits,
  };
}

async function nextResumeAction(checkpoint: string, episodeCount: number) {
  const nextGameSeed = seed + episodeCount;
  const environment = new RlEnvironmentV2(undefined, true);
  const first = environment.reset(nextGameSeed, 4);
  const featureSpec = createRlFeatureSpecV2(first);
  const actor = environment.getCurrentActorTeamId();
  if (!actor) throw new Error("No actor for resume-next-step probe");
  const observation = environment.getObservationForEncoding(actor);
  const legal = environment.getLegalActionsForEncoding(actor);
  const cache = createRlObservationEncoderCache();
  const encodedObservation = encodeRlObservationV2(observation, cache);
  const encodedActions = encodeRlLegalActionsV2(observation, legal);
  const client = makeClient();
  await client.start({
    seed,
    featureSpec,
    hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
    initialCheckpoint,
    resume: checkpoint,
  });
  try {
    const diagnostics = await client.diagnostics();
    const action = await client.act(encodedObservation, encodedActions);
    return { nextGameSeed, diagnostics, action };
  } finally {
    await client.close();
  }
}

async function expectRejected(operation: () => Promise<unknown>, contains: string) {
  try {
    await operation();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes(contains)) {
      throw new Error(`Expected rejection containing "${contains}", got: ${message}`);
    }
    return message;
  }
  throw new Error(`Expected rejection containing "${contains}"`);
}

async function runNegativeGuards(checkpoint: string, episodeCount: number) {
  const environment = new RlEnvironmentV2(undefined, true);
  const first = environment.reset(seed + episodeCount, 4);
  const featureSpec = createRlFeatureSpecV2(first);
  const actor = environment.getCurrentActorTeamId()!;
  const observation = environment.getObservationForEncoding(actor);
  const legal = environment.getLegalActionsForEncoding(actor);
  const encodedObservation = encodeRlObservationV2(observation, createRlObservationEncoderCache());
  const encodedActions = encodeRlLegalActionsV2(observation, legal);
  const base = packBcEncodedSamples([{
    observation: encodedObservation,
    actions: encodedActions.actions,
    targetIndex: 0,
  }], featureSpec);
  const client = makeClient();
  await client.start({
    seed,
    featureSpec,
    hyperparameters: DEFAULT_PPO_HYPERPARAMETERS,
    initialCheckpoint,
    resume: checkpoint,
  });
  try {
    const before = await client.diagnostics();
    await client.retainPackedChunk("duplicate-id", base);
    const duplicate = await expectRejected(
      () => client.retainPackedChunk("duplicate-id", base),
      "Duplicate PPO retentionId",
    );
    await client.discardRetained(["duplicate-id"]);
    const corrupt = await expectRejected(
      () => client.retainPackedChunk("corrupt-id", base, { corruptCompressedByteForTest: true }),
      "PPO retained",
    );
    await client.beginUpdate(1);
    const missing = await expectRejected(
      () => client.accumulateRetained("missing-id", [{
        oldLogProbability: 0,
        advantage: 1,
        return: 1,
      }]),
      "Unknown PPO retentionId",
    );
    const after = await client.diagnostics();
    if (before.parameterHash !== after.parameterHash || before.optimizerHash !== after.optimizerHash) {
      throw new Error("Negative retention guard changed model/optimizer state");
    }
    return { duplicate, corrupt, missing, noOptimizerStep: true, saveAttempted: false };
  } finally {
    await client.close();
  }
}

try {
  console.log("PPO-RETENTION-01: running deterministic oracle...");
  const oracle = await runVariant("oracle", false);
  console.log("PPO-RETENTION-01: running Python-retention variant...");
  const retention = await runVariant("retention", true);

  const oracleSummary = comparableSummary(oracle.result);
  const retentionSummary = comparableSummary(retention.result);
  if (JSON.stringify(oracleSummary) !== JSON.stringify(retentionSummary)) {
    throw new Error("Oracle and retention episode summaries differ");
  }
  if (oracleSummary.seed !== 9) {
    throw new Error(`Expected update2 resume to produce game seed 9, got ${oracleSummary.seed}`);
  }
  if (featureAudit) {
    if (JSON.stringify(oracle.featureAudits) !== JSON.stringify(retention.featureAudits)) {
      throw new Error("Oracle/retention packed feature bytes, tensor shapes/masks, or action ordering differ");
    }
    if (JSON.stringify(oracle.scalarAudits) !== JSON.stringify(retention.scalarAudits)) {
      throw new Error("Oracle/retention PPO scalar bytes (old log-probability/GAE/return) differ");
    }
  }
  const oracleDiagnostics = oracle.result.update.diagnostics;
  const retentionDiagnostics = retention.result.update.diagnostics;
  if (!oracleDiagnostics || !retentionDiagnostics) {
    throw new Error("Equivalence diagnostics were not returned");
  }
  for (const key of ["gradientHash", "parameterHash", "optimizerHash", "rngHash"] as const) {
    if (oracleDiagnostics[key] !== retentionDiagnostics[key]) {
      throw new Error(`Oracle/retention ${key} mismatch`);
    }
  }

  const oracleNext = await nextResumeAction(
    oracle.outputCheckpoint,
    oracle.result.update.episodeCount,
  );
  const retentionNext = await nextResumeAction(
    retention.outputCheckpoint,
    retention.result.update.episodeCount,
  );
  if (JSON.stringify(oracleNext) !== JSON.stringify(retentionNext)) {
    throw new Error("Resume-next-step result differs");
  }

  const negativeGuards = await runNegativeGuards(
    retention.outputCheckpoint,
    retention.result.update.episodeCount,
  );
  const retentionStats = retention.result.trajectoryRetentionStats;
  if (!retentionStats) throw new Error("Retention stats are missing");
  const oraclePhase = oracle.result.phaseProfile as any;
  const retentionPhase = retention.result.phaseProfile as any;
  const rolloutOverheadRatio = featureAudit
    ? null
    : retentionPhase.stages.rolloutMs / oraclePhase.stages.rolloutMs;
  const averageCompressedKiB = retentionStats.averageCompressedBytesPerSample / 1024;
  const result = {
    probe: "ppo_python_retention_equivalence",
    status: "passed",
    seed,
    gameSeed: oracleSummary.seed,
    decisions,
    chunkSize,
    featureAudit,
    exactEpisodeSummary: true,
    exactFeatureBytesAndDescriptors: featureAudit ? true : "not_audited_in_timing_run",
    exactActionOrderAndMasks: featureAudit ? true : "not_audited_in_timing_run",
    exactPpoScalarBytes: featureAudit ? true : "not_audited_in_timing_run",
    exactGradientHash: true,
    exactParameterHash: true,
    exactOptimizerHash: true,
    exactRngHash: true,
    exactResumeNextStep: true,
    negativeGuards,
    retention: {
      ...retentionStats,
      averageCompressedKiBPerSample: averageCompressedKiB,
      under40KiBPerSample: averageCompressedKiB <= 40,
    },
    timing: {
      comparable: !featureAudit,
      oracleWallMs: oracle.wallMs,
      retentionWallMs: retention.wallMs,
      oraclePhase,
      retentionPhase,
      rolloutOverheadRatio,
      rolloutOverheadWithin10Percent: rolloutOverheadRatio === null
        ? "not_evaluated_with_feature_audit"
        : rolloutOverheadRatio <= 1.10,
      fiveKTotalGate: decisions === 5_000 && !featureAudit
        ? retention.wallMs <= oracle.wallMs * 0.8
        : "not_evaluated_in_this_run",
    },
    finalStateHash: oracleSummary.finalStateHash,
    outcomeKind: oracleSummary.outcomeKind,
    updateCount: retention.result.update.updateCount,
    episodeCount: retention.result.update.episodeCount,
  };
  console.log(JSON.stringify(result, null, 2));
  console.log("PPO-RETENTION-01 SUCCESS");
} finally {
  rmSync(workspace, { recursive: true, force: true });
}
