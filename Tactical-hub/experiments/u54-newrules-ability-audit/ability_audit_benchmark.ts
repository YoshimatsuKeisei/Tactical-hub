import { performance } from "node:perf_hooks";
import { PpoAbilityAuditTracker, aggregatePpoAbilityAudits } from "../../src/game/cpu/rlAbilityAudit";
import { RlEnvironmentV2 } from "../../src/game/cpu/rlEnvironment";

type StepRecord = {
  seed: number;
  actionKey: string;
  legalHash: string;
  stateHashBefore: string;
  stateHashAfter: string;
  rngStateAfter: number;
  terminal: boolean;
};

type EpisodeBoundary = { seed: number; reason: "terminal" | "no_actor"; actionCount: number };

const argValue = (name: string) => {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const decisionTarget = Number(argValue("--decisions") ?? "5000");
const firstSeed = Number(argValue("--seed") ?? "417");
if (!Number.isInteger(decisionTarget) || decisionTarget <= 0) throw new Error("--decisions must be a positive integer");
if (!Number.isInteger(firstSeed)) throw new Error("--seed must be an integer");

function hashText(text: string) {
  let hash = 2166136261;
  for (const character of text) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619) >>> 0;
  return hash.toString(16).padStart(8, "0");
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return { rss: memory.rss, heapUsed: memory.heapUsed };
}

function cpuMilliseconds(start: NodeJS.CpuUsage) {
  const elapsed = process.cpuUsage(start);
  return (elapsed.user + elapsed.system) / 1000;
}

function createEnvironment(ref?: { current?: PpoAbilityAuditTracker }, observerTiming?: { milliseconds: number }) {
  const measured = (operation: () => void) => {
    const started = performance.now();
    operation();
    if (observerTiming) observerTiming.milliseconds += performance.now() - started;
  };
  return new RlEnvironmentV2(undefined, true, {
    cpuStep: {
      rlInPlacePhaseTransitions: true,
      rlPrevalidatedMovement: true,
      rlInPlaceProduction: true,
      ...(ref ? { onBattleResolved: (events: Parameters<PpoAbilityAuditTracker["observeBattleResolution"]>[0]) => measured(() => ref.current?.observeBattleResolution(events)) } : {}),
    },
    ...(ref ? {
      onBeforeApply: (state, decision) => measured(() => ref.current?.beforeApply(state, decision)),
      onAfterApply: (state, decision) => measured(() => ref.current?.afterApply(state, decision)),
    } : {}),
  });
}

function chooseIndex(step: number, seed: number, length: number) {
  const mixed = (Math.imul(step + 1, 1103515245) + Math.imul(seed, 12345)) >>> 0;
  return mixed % length;
}

function runBaseline() {
  const environment = createEnvironment();
  const records: StepRecord[] = [];
  const boundaries: EpisodeBoundary[] = [];
  let seed = firstSeed;
  let matchCount = 0;
  let episodeActionCount = 0;
  let peak = memorySnapshot();
  const startMemory = memorySnapshot();
  const cpuStart = process.cpuUsage();
  const started = performance.now();
  environment.reset(seed, 4);
  matchCount += 1;
  for (let step = 0; step < decisionTarget; step += 1) {
    for (;;) {
      if (environment.isTerminal()) {
        boundaries.push({ seed, reason: "terminal", actionCount: episodeActionCount });
      } else if (environment.getCurrentActorTeamId()) {
        break;
      } else {
        boundaries.push({ seed, reason: "no_actor", actionCount: episodeActionCount });
      }
      seed += 1;
      episodeActionCount = 0;
      environment.reset(seed, 4);
      matchCount += 1;
    }
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`Baseline has no actor at step ${step}`);
    const legal = environment.getLegalActionsForEncoding(actor);
    if (!legal.length) throw new Error(`Baseline has no legal action at step ${step}`);
    const action = legal[chooseIndex(step, seed, legal.length)];
    const record: StepRecord = {
      seed,
      actionKey: action.actionKey,
      legalHash: hashText(legal.map((entry) => entry.actionKey).join("\n")),
      stateHashBefore: environment.getStateHash(),
      stateHashAfter: "",
      rngStateAfter: 0,
      terminal: false,
    };
    environment.stepWithoutObservation(action.actionKey);
    record.stateHashAfter = environment.getStateHash();
    record.rngStateAfter = environment.getRngStateForValidation();
    record.terminal = environment.isTerminal();
    records.push(record);
    episodeActionCount += 1;
    if ((step + 1) % 100 === 0) {
      const current = memorySnapshot();
      peak = { rss: Math.max(peak.rss, current.rss), heapUsed: Math.max(peak.heapUsed, current.heapUsed) };
    }
  }
  return {
    records,
    boundaries,
    matchCount,
    wallMilliseconds: performance.now() - started,
    cpuMilliseconds: cpuMilliseconds(cpuStart),
    memory: { start: startMemory, peak, end: memorySnapshot() },
  };
}

function runAudited(records: readonly StepRecord[], boundaries: readonly EpisodeBoundary[]) {
  const ref: { current?: PpoAbilityAuditTracker } = {};
  const observerTiming = { milliseconds: 0 };
  const environment = createEnvironment(ref, observerTiming);
  const audits: Array<{ environmentIndex: number; seed: number; audit: ReturnType<PpoAbilityAuditTracker["finalize"]> }> = [];
  let seed = records[0]?.seed ?? firstSeed;
  let matchIndex = 0;
  let boundaryIndex = 0;
  let peak = memorySnapshot();
  const startMemory = memorySnapshot();
  const cpuStart = process.cpuUsage();
  const started = performance.now();
  environment.reset(seed, 4);
  ref.current = new PpoAbilityAuditTracker(environment.getStateForValidation());
  for (let step = 0; step < records.length; step += 1) {
    const expected = records[step];
    if (expected.seed !== seed) {
      const boundary = boundaries[boundaryIndex];
      if (!boundary || boundary.seed !== seed) throw new Error(`Missing baseline boundary for seed ${seed}`);
      if (boundary.reason === "terminal" ? !environment.isTerminal() : environment.isTerminal() || environment.getCurrentActorTeamId()) {
        throw new Error(`Episode boundary mismatch for seed ${seed}: ${boundary.reason}`);
      }
      boundaryIndex += 1;
      audits.push({ environmentIndex: matchIndex, seed, audit: ref.current.finalize(environment.getStateForValidation()) });
      matchIndex += 1;
      seed = expected.seed;
      ref.current = undefined;
      environment.reset(seed, 4);
      ref.current = new PpoAbilityAuditTracker(environment.getStateForValidation());
    }
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`Audited run has no actor at step ${step}`);
    const legal = environment.getLegalActionsForEncoding(actor);
    const legalHash = hashText(legal.map((entry) => entry.actionKey).join("\n"));
    if (legalHash !== expected.legalHash) throw new Error(`Legal Action order mismatch at step ${step}`);
    if (environment.getStateHash() !== expected.stateHashBefore) throw new Error(`Pre-action state hash mismatch at step ${step}`);
    if (!legal.some((entry) => entry.actionKey === expected.actionKey)) throw new Error(`Selected Action missing at step ${step}: ${expected.actionKey}`);
    const observeStarted = performance.now();
    ref.current.observeDecisionPoint(
      environment.getStateForValidation(),
      legal,
      environment.getDecisionAuditHintsForValidation(actor),
    );
    observerTiming.milliseconds += performance.now() - observeStarted;
    environment.stepWithoutObservation(expected.actionKey);
    if (environment.getStateHash() !== expected.stateHashAfter) throw new Error(`Post-action state hash mismatch at step ${step}`);
    if (environment.getRngStateForValidation() !== expected.rngStateAfter) throw new Error(`RNG cursor mismatch at step ${step}`);
    if (environment.isTerminal() !== expected.terminal) throw new Error(`Terminal result mismatch at step ${step}`);
    if ((step + 1) % 100 === 0) {
      const current = memorySnapshot();
      peak = { rss: Math.max(peak.rss, current.rss), heapUsed: Math.max(peak.heapUsed, current.heapUsed) };
    }
  }
  audits.push({ environmentIndex: matchIndex, seed, audit: ref.current.finalize(environment.getStateForValidation()) });
  return {
    aggregate: aggregatePpoAbilityAudits(audits),
    verifiedBoundaryCount: boundaryIndex,
    matchCount: audits.length,
    wallMilliseconds: performance.now() - started,
    cpuMilliseconds: cpuMilliseconds(cpuStart),
    observerMilliseconds: observerTiming.milliseconds,
    memory: { start: startMemory, peak, end: memorySnapshot() },
  };
}

function runPlainReplay(records: readonly StepRecord[], boundaries: readonly EpisodeBoundary[]) {
  const environment = createEnvironment();
  let seed = records[0]?.seed ?? firstSeed;
  let boundaryIndex = 0;
  let peak = memorySnapshot();
  const startMemory = memorySnapshot();
  const cpuStart = process.cpuUsage();
  const started = performance.now();
  environment.reset(seed, 4);
  for (let step = 0; step < records.length; step += 1) {
    const expected = records[step];
    if (expected.seed !== seed) {
      const boundary = boundaries[boundaryIndex];
      if (!boundary || boundary.seed !== seed) throw new Error(`Missing plain baseline boundary for seed ${seed}`);
      if (boundary.reason === "terminal" ? !environment.isTerminal() : environment.isTerminal() || environment.getCurrentActorTeamId()) {
        throw new Error(`Plain episode boundary mismatch for seed ${seed}: ${boundary.reason}`);
      }
      boundaryIndex += 1;
      seed = expected.seed;
      environment.reset(seed, 4);
    }
    const actor = environment.getCurrentActorTeamId();
    if (!actor) throw new Error(`Plain replay has no actor at step ${step}`);
    const legal = environment.getLegalActionsForEncoding(actor);
    if (hashText(legal.map((entry) => entry.actionKey).join("\n")) !== expected.legalHash) throw new Error(`Plain legal Action order mismatch at step ${step}`);
    if (environment.getStateHash() !== expected.stateHashBefore) throw new Error(`Plain pre-action state hash mismatch at step ${step}`);
    environment.stepWithoutObservation(expected.actionKey);
    if (environment.getStateHash() !== expected.stateHashAfter) throw new Error(`Plain post-action state hash mismatch at step ${step}`);
    if (environment.getRngStateForValidation() !== expected.rngStateAfter) throw new Error(`Plain RNG cursor mismatch at step ${step}`);
    if (environment.isTerminal() !== expected.terminal) throw new Error(`Plain terminal result mismatch at step ${step}`);
    if ((step + 1) % 100 === 0) {
      const current = memorySnapshot();
      peak = { rss: Math.max(peak.rss, current.rss), heapUsed: Math.max(peak.heapUsed, current.heapUsed) };
    }
  }
  return {
    wallMilliseconds: performance.now() - started,
    cpuMilliseconds: cpuMilliseconds(cpuStart),
    verifiedBoundaryCount: boundaryIndex,
    memory: { start: startMemory, peak, end: memorySnapshot() },
  };
}

const trace = runBaseline();
const baseline = runPlainReplay(trace.records, trace.boundaries);
const audited = runAudited(trace.records, trace.boundaries);
const additionalMilliseconds = audited.wallMilliseconds - baseline.wallMilliseconds;
const output = {
  schemaVersion: 1,
  measuredAt: new Date().toISOString(),
  environment: { node: process.version, platform: process.platform, architecture: process.arch },
  decisionCount: decisionTarget,
  firstSeed,
  lastSeed: trace.records.at(-1)?.seed,
  matchCount: trace.matchCount,
  completedEpisodeBoundaries: trace.boundaries,
  determinism: {
    actionKeysAndOrderMatch: true,
    stateHashesMatch: true,
    rngCursorsMatch: true,
    terminalResultsMatch: true,
    episodeBoundariesMatch: audited.verifiedBoundaryCount === trace.boundaries.length
      && baseline.verifiedBoundaryCount === trace.boundaries.length,
  },
  traceGeneration: {
    wallMilliseconds: trace.wallMilliseconds,
    cpuMilliseconds: trace.cpuMilliseconds,
  },
  baseline: {
    wallMilliseconds: baseline.wallMilliseconds,
    cpuMilliseconds: baseline.cpuMilliseconds,
    memory: baseline.memory,
  },
  audited: {
    wallMilliseconds: audited.wallMilliseconds,
    cpuMilliseconds: audited.cpuMilliseconds,
    observerMilliseconds: audited.observerMilliseconds,
    memory: audited.memory,
  },
  overhead: {
    additionalMilliseconds,
    relativePercent: baseline.wallMilliseconds ? additionalMilliseconds / baseline.wallMilliseconds * 100 : 0,
    observerSharePercent: audited.wallMilliseconds ? audited.observerMilliseconds / audited.wallMilliseconds * 100 : 0,
  },
  abilityAuditInvariant: {
    productionSelections: audited.aggregate.production.selectionCount,
    productionResolved: audited.aggregate.production.successCount + audited.aggregate.production.failureCount,
    noUnresolvedProductionSelections: audited.aggregate.production.selectionCount === audited.aggregate.production.successCount + audited.aggregate.production.failureCount,
  },
};

console.log(JSON.stringify(output, null, 2));
