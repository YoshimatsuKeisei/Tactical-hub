import { createInitialGameState } from "../initialState";
import type { GameState, Unit } from "../types";
import { positionKey } from "../utils/position";
import { advanceCpuOneStep } from "./cpuStep";
import type { HeuristicCpuPolicy } from "./heuristicCpuPolicy";
import { stableDiagnosticHash, summarizeTraceState, type HeadlessTraceEntry } from "./headlessTrace";
import { withLegalProfileSink } from "./legalEnumerationProfile";
import { createProfiledRandomCpuPolicy, getRandomCpuDecision } from "./randomCpuPolicy";
import { getCpuDecisionActionKey } from "./rlEnvironment";
import { createCpuRuntime, type CpuActionLog, type CpuDecision, type CpuPolicy, type CpuRuntime, type CpuTeamSettings } from "./types";

export type HeadlessEndReason = "victory" | "turn_limit" | "exception" | "invariant_violation" | "phase_stall" | "action_limit";
export type HeadlessMode = "debug" | "sweep" | "training";
export type HeadlessProfileCategory = "legalEnumeration" | "policySelection" | "actionApplication" | "invariantChecks" | "stallDetection" | "actionLogging" | "otherRunner" | "total";
export type HeadlessProfileEntry = { calls: number; totalMs: number; averageMs: number; maxMs: number; percentage: number };
export type HeadlessProfile = Record<HeadlessProfileCategory, HeadlessProfileEntry>;
export type LegalEnumerationProfileEntry = Omit<HeadlessProfileEntry, "percentage">;
export type LegalEnumerationBreakdown = { byCategory: Record<string, LegalEnumerationProfileEntry>; byPhase: Record<string, LegalEnumerationProfileEntry> };
export type HeadlessProfilingSection = { name: string; totalMs: number; callCount: number; averageMs: number; percentOfMatchElapsed: number };
export type HeadlessTurnBucket = { fromTurn: number; toTurn: number | null; elapsedMs: number; actionCount: number; msPerAction: number; legalActionGenerationCount: number; pathfindingCount: number; averageLivingUnitCount: number };
export type HeadlessProfiling = { enabled: true; matchElapsedMs: number; overlapWarning: string; sections: HeadlessProfilingSection[]; turnBuckets: HeadlessTurnBucket[]; heuristicDistanceCache?: { searchCount: number; hitCount: number; missCount: number; hitRate: number } };
export type HeadlessMatchOptions = {
  participantCount: 3 | 4;
  seed: number;
  maxTurns: number;
  maxActions?: number;
  maxActionsPerPhase?: number;
  historyLimit?: number;
  mode?: HeadlessMode;
  profile?: boolean;
  trace?: boolean;
  initialState?: GameState;
  policy?: CpuPolicy;
  onProgress?: () => void;
};
export type HeadlessMatchResult = {
  seed: number;
  participantCount: number;
  endReason: HeadlessEndReason;
  winnerTeamId?: string;
  endTurn: number;
  phase: GameState["phase"];
  currentMovementTeamId?: string;
  actionCount: number;
  actionSequenceHash: string;
  violations: string[];
  error?: string;
  recentActions?: CpuActionLog[];
  invariantCheckCount: number;
  profile?: HeadlessProfile;
  legalEnumerationBreakdown?: LegalEnumerationBreakdown;
  profiling?: HeadlessProfiling;
  trace?: HeadlessTraceEntry[];
  finalState: GameState;
};
export type HeadlessBatchResult = {
  matches: HeadlessMatchResult[];
  settledCount: number;
  turnLimitCount: number;
  exceptionCount: number;
  invariantViolationCount: number;
  phaseStallCount: number;
  actionLimitCount: number;
};

function removeParticipant(state: GameState, teamId: string) {
  const removedIds = new Set(state.units.filter((unit) => unit.ownerTeamId === teamId).map((unit) => unit.id));
  state.units = state.units.map((unit) => removedIds.has(unit.id) ? { ...unit, hp: 0, position: { kind: "removed", reason: "team_defeat" }, statuses: [] } as Unit : unit);
  for (const base of state.bases) for (const slot of base.slots) if (slot.unitId && removedIds.has(slot.unitId)) slot.unitId = undefined;
  const team = state.teams.find((entry) => entry.id === teamId);
  for (const baseId of team?.controlledBaseIds ?? []) {
    const base = state.bases.find((entry) => entry.id === baseId);
    if (base) base.ownerTeamId = "neutral";
  }
  if (team) { team.status = "defeated"; team.controlledBaseIds = []; }
}

export function createHeadlessInitialState(participantCount: 3 | 4) {
  const state = createInitialGameState();
  if (participantCount === 3) removeParticipant(state, "team-4");
  const activeIds = state.teams.filter((team) => !team.isNeutral && team.status === "active").map((team) => team.id);
  state.config.playerCount = participantCount;
  state.movementSeatOrderTeamIds = activeIds;
  state.movementOrderTeamIds = activeIds;
  state.currentMovementTeamId = activeIds[0];
  return state;
}

export function checkHeadlessInvariants(state: GameState, runtime?: CpuRuntime) {
  const violations: string[] = [];
  const living = state.units.filter((unit) => unit.hp > 0 && unit.position.kind !== "removed");
  const livingIds = new Set<string>();
  for (const unit of living) {
    if (livingIds.has(unit.id)) violations.push(`duplicate living unit id: ${unit.id}`);
    livingIds.add(unit.id);
    if (unit.position.kind === "water" && unit.type !== "ninja") violations.push(`non-ninja on water: ${unit.id}`);
  }
  for (const unit of state.units.filter((entry) => entry.hp <= 0 || entry.position.kind === "removed")) {
    if (unit.position.kind !== "removed") violations.push(`dead unit retains board position: ${unit.id}/${positionKey(unit.position)}`);
    if (state.bases.some((base) => base.slots.some((slot) => slot.unitId === unit.id))) violations.push(`removed/dead unit occupies BaseSlot: ${unit.id}`);
  }
  const occupiedByTeam = new Map<string, string>();
  for (const unit of living) {
    const key = `${unit.ownerTeamId}:${positionKey(unit.position)}`;
    if (occupiedByTeam.has(key)) violations.push(`same-team position collision: ${occupiedByTeam.get(key)} / ${unit.id} at ${positionKey(unit.position)}`);
    occupiedByTeam.set(key, unit.id);
  }
  for (const base of state.bases) for (const slot of base.slots) {
    if (!slot.unitId) continue;
    const unit = state.units.find((entry) => entry.id === slot.unitId);
    if (!unit) violations.push(`BaseSlot references missing unit: ${base.id}/${slot.id}/${slot.unitId}`);
    else if (unit.hp <= 0 || unit.position.kind !== "base" || unit.position.baseId !== base.id || unit.position.slotId !== slot.id) violations.push(`BaseSlot/unit position mismatch: ${base.id}/${slot.id}/${unit.id}`);
  }
  for (const unit of living.filter((entry) => entry.position.kind === "base")) {
    const position = unit.position as Extract<Unit["position"], { kind: "base" }>;
    if (state.bases.find((base) => base.id === position.baseId)?.slots.find((slot) => slot.id === position.slotId)?.unitId !== unit.id) violations.push(`base unit missing reciprocal slot: ${unit.id}`);
  }
  const rewardInterruptsMovement = state.phase === "reward_placement" && state.phaseAfterRewards === "movement_input";
  if (state.phase === "movement_input" || rewardInterruptsMovement) {
    const team = state.teams.find((entry) => entry.id === state.currentMovementTeamId);
    if (!team || team.status !== "active") violations.push(`movement assigned to inactive/missing team: ${state.currentMovementTeamId ?? "none"}`);
  } else if (state.currentMovementTeamId) violations.push(`current movement team exists outside movement phase: ${state.currentMovementTeamId}`);
  if (state.turnState.phase !== state.phase) violations.push(`phase mismatch: state=${state.phase} turnState=${state.turnState.phase}`);
  if (state.teleportIntents.length && state.phase !== "movement_input" && !rewardInterruptsMovement) violations.push(`teleport intents exist outside movement phase: ${state.phase}`);
  if (state.phase === "strategist_action_resolution" && !state.teams.filter((team) => team.status === "active").every((team) => state.strategistSubmittedTeamIds.includes(team.id))) violations.push("strategist resolution started before all active teams submitted");
  const unitById = new Map(state.units.map((unit) => [unit.id, unit]));
  const constructionById = new Map(state.constructions.map((entry) => [entry.id, entry]));
  for (const group of state.turnState.actionIntents) {
    if (!state.teams.some((team) => team.id === group.teamId)) violations.push(`ActionIntent references missing team: ${group.teamId}`);
    for (const choice of group.productionChoices) if (!state.bases.some((base) => base.id === choice.baseId)) violations.push(`production references missing base: ${choice.baseId}`);
    for (const intent of group.movementIntents) if (!unitById.has(intent.unitId)) violations.push(`movement references missing unit: ${intent.unitId}`);
    for (const intent of group.attackIntents ?? []) {
      if (!unitById.has(intent.attackerUnitId)) violations.push(`attack references missing attacker: ${intent.attackerUnitId}`);
      if (intent.target && !unitById.has(intent.target.unitId)) violations.push(`attack references missing target: ${intent.target.unitId}`);
    }
  }
  for (const intent of state.teleportIntents) if (!unitById.has(intent.strategistUnitId) || !unitById.has(intent.targetUnitId)) violations.push(`teleport references missing unit: ${intent.strategistUnitId}/${intent.targetUnitId}`);
  for (const intent of state.strategistActionIntents) {
    if (!unitById.has(intent.strategistUnitId)) violations.push(`strategist intent references missing unit: ${intent.strategistUnitId}`);
    if (intent.constructionId && !constructionById.has(intent.constructionId)) violations.push(`strategist intent references missing construction: ${intent.constructionId}`);
  }
  for (const construction of state.constructions) {
    if (construction.ownerTeamId && !state.teams.some((team) => team.id === construction.ownerTeamId)) violations.push(`construction owner missing: ${construction.id}/${construction.ownerTeamId}`);
    if (construction.managerUnitId) {
      const manager = unitById.get(construction.managerUnitId);
      if (!manager || manager.hp <= 0 || manager.position.kind === "removed" || manager.type !== "strategist" || manager.role !== "builder") violations.push(`construction manager invalid: ${construction.id}/${construction.managerUnitId}`);
      else if (construction.ownerTeamId !== manager.ownerTeamId) violations.push(`construction owner/manager team mismatch: ${construction.id}`);
    }
  }
  for (const intent of runtime?.hiddenAttackIntents ?? []) if (!unitById.has(intent.attackerUnitId) || (intent.target && !unitById.has(intent.target.unitId))) violations.push(`hidden CPU attack references missing unit: ${intent.attackerUnitId}`);
  if (state.phase === "reward_placement" && !state.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) violations.push("reward phase has no pending request");
  return [...new Set(violations)];
}

type MutableProfileEntry = { calls: number; totalMs: number; maxMs: number };
function profileCollector(enabled: boolean) {
  const categories = ["legalEnumeration", "policySelection", "actionApplication", "invariantChecks", "stallDetection", "actionLogging"] as const;
  const values = Object.fromEntries(categories.map((key) => [key, { calls: 0, totalMs: 0, maxMs: 0 }])) as Record<typeof categories[number], MutableProfileEntry>;
  const add = (key: typeof categories[number], milliseconds: number) => {
    if (!enabled) return;
    const entry = values[key]; entry.calls += 1; entry.totalMs += milliseconds; entry.maxMs = Math.max(entry.maxMs, milliseconds);
  };
  const legalCategoryNames = ["production", "movement", "retreat", "ninjaMovement", "teleport", "attack", "reward", "constructionStrategist", "confirmPass", "movementRangePathSearch", "attackTargetSearch", "boardOccupancyGeneration", "unitBaseEquipmentSearch", "postProcessing", "attackLivingEnemyList", "attackUnitCoordinateBaseSearch", "attackStaticRangePrefilter", "attackRangeDistance", "attackRoadSectionConnection", "attackAcrossBaseBlocking", "attackBaseBlocking", "attackBridgeConnection", "attackLakeNinjaRule", "attackBasicFilter", "attackCandidateIdGeneration", "attackPostProcessing", "attackFinalLegalCheck"];
  const legalCategories = new Map<string, MutableProfileEntry>(legalCategoryNames.map((key) => [key, { calls: 0, totalMs: 0, maxMs: 0 }]));
  const legalPhases = new Map<string, MutableProfileEntry>();
  const sections = new Map<string, MutableProfileEntry>();
  const bucketDefinitions = [[1, 50], [51, 100], [101, 150], [151, 200], [201, null]] as const;
  type MutableBucket = { elapsedMs: number; actionCount: number; legalActionGenerationCount: number; pathfindingCount: number; livingUnitTotal: number; livingUnitSamples: number };
  const buckets = bucketDefinitions.map((): MutableBucket => ({ elapsedMs: 0, actionCount: 0, legalActionGenerationCount: 0, pathfindingCount: 0, livingUnitTotal: 0, livingUnitSamples: 0 }));
  const bucketIndex = (turn: number) => turn <= 50 ? 0 : turn <= 100 ? 1 : turn <= 150 ? 2 : turn <= 200 ? 3 : 4;
  const addDynamic = (target: Map<string, MutableProfileEntry>, key: string, milliseconds: number) => {
    if (!enabled) return;
    const entry = target.get(key) ?? { calls: 0, totalMs: 0, maxMs: 0 };
    entry.calls += 1; entry.totalMs += milliseconds; entry.maxMs = Math.max(entry.maxMs, milliseconds); target.set(key, entry);
  };
  const addSection = (name: string, milliseconds: number) => addDynamic(sections, name, milliseconds);
  const summarize = (target: Map<string, MutableProfileEntry>) => Object.fromEntries([...target].map(([key, entry]) => [key, { ...entry, averageMs: entry.calls ? entry.totalMs / entry.calls : 0 }]));
  const finish = (totalMs: number): HeadlessProfile | undefined => {
    if (!enabled) return undefined;
    const measured = Object.values(values).reduce((sum, entry) => sum + entry.totalMs, 0);
    const all: Record<string, MutableProfileEntry> = { ...values, otherRunner: { calls: 1, totalMs: Math.max(0, totalMs - measured), maxMs: Math.max(0, totalMs - measured) }, total: { calls: 1, totalMs, maxMs: totalMs } };
    return Object.fromEntries(Object.entries(all).map(([key, entry]) => [key, { ...entry, averageMs: entry.calls ? entry.totalMs / entry.calls : 0, percentage: totalMs ? entry.totalMs / totalMs * 100 : 0 }])) as HeadlessProfile;
  };
  const finishLegal = (): LegalEnumerationBreakdown | undefined => enabled ? { byCategory: summarize(legalCategories), byPhase: summarize(legalPhases) } : undefined;
  const finishDetailed = (totalMs: number): HeadlessProfiling | undefined => {
    if (!enabled) return undefined;
    const detailedSections = [...sections.entries()].map(([name, entry]) => ({
      name,
      totalMs: entry.totalMs,
      callCount: entry.calls,
      averageMs: entry.calls ? entry.totalMs / entry.calls : 0,
      percentOfMatchElapsed: totalMs ? entry.totalMs / totalMs * 100 : 0,
    })).sort((left, right) => right.totalMs - left.totalMs || left.name.localeCompare(right.name));
    return {
      enabled: true,
      matchElapsedMs: totalMs,
      overlapWarning: "Sections include nested/inclusive measurements; their totalMs values overlap and must not be summed to match matchElapsedMs.",
      sections: detailedSections,
      turnBuckets: buckets.map((bucket, index) => ({
        fromTurn: bucketDefinitions[index][0],
        toTurn: bucketDefinitions[index][1],
        elapsedMs: bucket.elapsedMs,
        actionCount: bucket.actionCount,
        msPerAction: bucket.actionCount ? bucket.elapsedMs / bucket.actionCount : 0,
        legalActionGenerationCount: bucket.legalActionGenerationCount,
        pathfindingCount: bucket.pathfindingCount,
        averageLivingUnitCount: bucket.livingUnitSamples ? bucket.livingUnitTotal / bucket.livingUnitSamples : 0,
      })),
      heuristicDistanceCache: (() => {
        const hitCount = sections.get("legal.heuristicDistanceCacheHit")?.calls ?? 0;
        const missCount = sections.get("legal.heuristicDistanceCacheMiss")?.calls ?? 0;
        const searchCount = sections.get("legal.heuristicDistanceSearch")?.calls ?? 0;
        return hitCount + missCount ? { searchCount, hitCount, missCount, hitRate: hitCount / (hitCount + missCount) } : undefined;
      })(),
    };
  };
  return {
    add: (key: typeof categories[number], ms: number) => { add(key, ms); addSection(key, ms); },
    addSection,
    addLegalCategory: (key: string, ms: number, turn?: number) => {
      addDynamic(legalCategories, key, ms); addSection(`legal.${key}`, ms);
      if (turn !== undefined && /(path|distance|range)/i.test(key)) buckets[bucketIndex(turn)].pathfindingCount += 1;
    },
    addLegalPhase: (key: string, ms: number) => { addDynamic(legalPhases, key, ms); addSection(`legal.phase.${key}`, ms); },
    recordLegalGeneration: (turn: number, ms: number) => { buckets[bucketIndex(turn)].legalActionGenerationCount += 1; addSection("legalActionGeneration", ms); },
    recordAction: (turn: number, elapsedMs: number, livingUnits: number) => { const bucket = buckets[bucketIndex(turn)]; bucket.elapsedMs += elapsedMs; bucket.actionCount += 1; bucket.livingUnitTotal += livingUnits; bucket.livingUnitSamples += 1; },
    measure<T>(name: string, operation: () => T): T { if (!enabled) return operation(); const started = performance.now(); const value = operation(); addSection(name, performance.now() - started); return value; },
    finish,
    finishLegal,
    finishDetailed,
  };
}

export function getHeadlessProgressSignature(state: GameState, runtime: CpuRuntime) {
  const living = state.units.filter((unit) => unit.hp > 0 && unit.position.kind !== "removed").length;
  const intents = state.turnState.actionIntents.reduce((sum, entry) => sum + entry.productionChoices.length + entry.movementIntents.length + (entry.attackIntents?.length ?? 0), 0);
  const completedProduction = state.productionCompletedTeamIdsThisTurn.slice().sort().join(",");
  return [state.turnNumber, state.phase, state.currentMovementTeamId ?? "-", state.teams.filter((team) => team.status === "active").length, living, intents, completedProduction, state.teleportIntents.length, state.strategistActionIntents.length, state.strategistSubmittedTeamIds.length, state.rewardPlacementRequests.filter((entry) => !entry.completed && !entry.expired).length, state.constructions.filter((entry) => entry.active).length, runtime.rngState, runtime.contextKey, runtime.processedKeys.length, runtime.completedProductionTeamIds.length, runtime.completedAttackTeamIds.length, runtime.hiddenAttackIntents.length].join(":");
}

function importantResolutionSignature(state: GameState) {
  return `${state.turnNumber}:${state.phase}:${state.teams.map((team) => `${team.id}=${team.status}`).join(",")}:${state.bases.map((base) => `${base.id}=${base.ownerTeamId}`).join(",")}:${state.constructions.map((entry) => `${entry.id}=${entry.active}/${entry.ownerTeamId}/${entry.managerUnitId}`).join(",")}`;
}

function decisionSignature(decision: CpuDecision) {
  switch (decision.kind) {
    case "production": return `${decision.kind}:${decision.teamId}:${decision.actorKey}:${decision.choice ? `${decision.choice.baseId}/${decision.choice.unitType}` : "pass"}`;
    case "movement": return `${decision.kind}:${decision.teamId}:${decision.unitId}:${decision.to ? positionKey(decision.to) : "pass"}`;
    case "teleport": return `${decision.kind}:${decision.teamId}:${decision.strategistUnitId}:${decision.intent ? `${decision.intent.targetUnitId}/${positionKey(decision.intent.to)}` : "pass"}`;
    case "attack": return `${decision.kind}:${decision.teamId}:${decision.intent.attackerUnitId}:${decision.intent.target?.unitId ?? "pass"}`;
    case "reward": return `${decision.kind}:${decision.teamId}:${decision.requestId}:${decision.baseId}:${decision.unitType}`;
    case "strategist": return `${decision.kind}:${decision.teamId}:${decision.actorKey}:${decision.intent.action}:${decision.intent.constructionId ?? ""}:${decision.intent.tiles?.map((tile) => `${tile.x},${tile.y}`).join("/") ?? ""}`;
    case "submit_team_production": case "submit_movement": case "complete_attack_team": case "submit_strategist": return `${decision.kind}:${decision.teamId}`;
    default: return decision.kind;
  }
}

function updateActionHash(hash: number, decision: CpuDecision) {
  for (const character of decisionSignature(decision)) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619) >>> 0;
  return hash;
}

function result(options: HeadlessMatchOptions, state: GameState, runtime: CpuRuntime, endReason: HeadlessEndReason, actionHash: number, invariantCheckCount: number, profile: HeadlessProfile | undefined, legalEnumerationBreakdown: LegalEnumerationBreakdown | undefined, profiling: HeadlessProfiling | undefined, trace: HeadlessTraceEntry[] | undefined, violations: string[] = [], error?: string): HeadlessMatchResult {
  const active = state.teams.filter((team) => !team.isNeutral && team.status === "active");
  const mode = options.mode ?? "debug";
  const failed = !["victory", "turn_limit"].includes(endReason);
  const recentActions = mode === "debug" || (mode === "sweep" && failed) ? runtime.logs.slice(-(options.historyLimit ?? 50)) : undefined;
  return { seed: options.seed, participantCount: options.participantCount, endReason, winnerTeamId: active.length === 1 ? active[0].id : undefined, endTurn: state.turnNumber, phase: state.phase, currentMovementTeamId: state.currentMovementTeamId, actionCount: runtime.appliedStepCount, actionSequenceHash: actionHash.toString(16).padStart(8, "0"), violations, error, recentActions, invariantCheckCount, profile, legalEnumerationBreakdown, profiling, trace, finalState: state };
}

export function runHeadlessMatch(options: HeadlessMatchOptions): HeadlessMatchResult {
  const matchStarted = performance.now();
  const timings = profileCollector(Boolean(options.profile));
  let state = timings.measure("initialStateGeneration", () => structuredClone(options.initialState ?? createHeadlessInitialState(options.participantCount)) as GameState);
  let runtime = createCpuRuntime(options.seed, options.maxActions ?? 100_000);
  const mode = options.mode ?? "debug";
  let invariantCheckCount = 0;
  let actionHash = 2166136261;
  const traceEntries: HeadlessTraceEntry[] | undefined = options.trace ? [] : undefined;
  const inspect = () => {
    const start = options.profile ? performance.now() : 0;
    const violations = checkHeadlessInvariants(state, runtime);
    invariantCheckCount += 1;
    timings.add("invariantChecks", options.profile ? performance.now() - start : 0);
    return violations;
  };
  const settings: CpuTeamSettings = Object.fromEntries(state.teams.filter((team) => !team.isNeutral && team.status === "active").map((team) => [team.id, "random_cpu"]));
  const recordPolicyProfile = (enumerationMs: number, policyTotalMs: number, details: { category: string; phase: GameState["phase"]; milliseconds: number }[]) => {
    timings.addSection("policyTotal", policyTotalMs);
    timings.add("legalEnumeration", enumerationMs);
    timings.add("policySelection", Math.max(0, policyTotalMs - enumerationMs));
    timings.addLegalPhase(details[0]?.phase ?? state.phase, enumerationMs);
    timings.recordLegalGeneration(state.turnNumber, enumerationMs);
    for (const detail of details) timings.addLegalCategory(detail.category, detail.milliseconds, state.turnNumber);
  };
  let policy: CpuPolicy;
  if (!options.policy) policy = options.profile ? createProfiledRandomCpuPolicy(recordPolicyProfile) : getRandomCpuDecision;
  else if (!options.profile) policy = options.policy;
  else {
    const sourcePolicy = options.policy;
    policy = (policyState, policyRuntime, policySettings) => {
      const details: { category: string; phase: GameState["phase"]; milliseconds: number }[] = [];
      const started = performance.now();
      const decision = withLegalProfileSink((category, milliseconds) => details.push({ category, phase: policyState.phase, milliseconds }), () => sourcePolicy(policyState, policyRuntime, policySettings));
      const totalMs = performance.now() - started;
      const topLevel = new Set(["production", "movement", "retreat", "ninjaMovement", "teleport", "attack", "reward", "constructionStrategist"]);
      const enumerationMs = details.filter((detail) => topLevel.has(detail.category)).reduce((sum, detail) => sum + detail.milliseconds, 0);
      recordPolicyProfile(enumerationMs, totalMs, details);
      return decision;
    };
  }
  (options.policy as Partial<HeuristicCpuPolicy> | undefined)?.setDecisionDiagnosticsEnabled?.(Boolean(traceEntries));
  const phaseCounts = new Map<string, number>();
  let previous = getHeadlessProgressSignature(state, runtime);
  let previousImportant = importantResolutionSignature(state);
  const finish = (endReason: HeadlessEndReason, violations: string[] = [], error?: string) => {
    const finalReason = violations.length ? "invariant_violation" : endReason;
    const built = timings.measure("resultAggregation", () => result(options, state, runtime, finalReason, actionHash, invariantCheckCount, undefined, undefined, undefined, traceEntries, violations, error));
    const totalMs = performance.now() - matchStarted;
    built.profile = timings.finish(totalMs);
    built.legalEnumerationBreakdown = timings.finishLegal();
    built.profiling = timings.finishDetailed(totalMs);
    return built;
  };
  try {
    if (state.teams.filter((team) => !team.isNeutral && team.status === "active").length <= 1) return finish("victory");
    if (mode !== "training") {
      const initialViolations = inspect();
      if (initialViolations.length) return finish("invariant_violation", initialViolations);
    }
    while (true) {
      const active = state.teams.filter((team) => !team.isNeutral && team.status === "active");
      if (active.length <= 1) return finish("victory");
      if (state.turnNumber > options.maxTurns) return finish("turn_limit");
      if (runtime.appliedStepCount >= (options.maxActions ?? 100_000)) return finish("action_limit", [], `match action limit reached`);
      const phaseKey = `${state.turnNumber}:${state.phase}:${state.currentMovementTeamId ?? "-"}`;
      const phaseCount = (phaseCounts.get(phaseKey) ?? 0) + 1;
      phaseCounts.set(phaseKey, phaseCount);
      if (phaseCount > (options.maxActionsPerPhase ?? 5_000)) return finish("action_limit", [], `phase action limit reached: ${phaseKey}`);
      const actionTurn = state.turnNumber;
      const actionPhase = state.phase;
      const actionIndex = runtime.appliedStepCount;
      const traceStateBefore = traceEntries ? summarizeTraceState(state) : undefined;
      const traceRngBefore = runtime.rngState;
      let selectedDecision: CpuDecision | undefined;
      const livingUnitCount = options.profile ? state.units.filter((unit) => unit.hp > 0 && unit.position.kind !== "removed").length : 0;
      const actionStarted = options.profile ? performance.now() : 0;
      const step = advanceCpuOneStep(state, runtime, settings, policy, {
        logMode: mode === "debug" ? "full" : mode === "sweep" ? "ring" : "none",
        logLimit: options.historyLimit ?? 50,
        onApply: options.profile ? (ms, decision, phaseBefore, phaseAfter, turnBefore, turnAfter) => {
          timings.add("actionApplication", ms);
          if (decision.kind === "resolve_battle") timings.addSection("battleResolution", ms);
          if (phaseBefore !== phaseAfter) timings.addSection("phaseTransition", ms);
          if (turnBefore !== turnAfter) timings.addSection("turnTransition", ms);
        } : undefined,
        onLog: options.profile ? (ms) => timings.add("actionLogging", ms) : undefined,
        onDecision: (decision) => {
          selectedDecision = decision;
          if (options.profile) timings.measure("actionSequenceHash", () => { actionHash = updateActionHash(actionHash, decision); });
          else actionHash = updateActionHash(actionHash, decision);
        },
      });
      state = step.state; runtime = step.runtime;
      if (options.onProgress && runtime.appliedStepCount % 128 === 0) options.onProgress();
      if (!step.applied) return finish(runtime.stoppedReason ? "action_limit" : "phase_stall", [], runtime.stoppedReason ?? "CPU policy returned no action");
      if (traceEntries && traceStateBefore && selectedDecision) {
        const diagnosticSource = options.policy as Partial<HeuristicCpuPolicy> | undefined;
        const diagnostics = diagnosticSource?.getLastDecisionDiagnostics?.();
        traceEntries.push({
          actionIndex,
          seed: options.seed,
          turn: actionTurn,
          phase: actionPhase,
          actorTeamId: selectedDecision.teamId,
          targetBaseId: diagnostics?.targetBaseId ?? (selectedDecision.teamId !== "all" ? diagnosticSource?.getTargetBaseId?.(selectedDecision.teamId, options.seed) : undefined),
          legalActionCount: diagnostics?.legalActionKeys.length,
          legalActionKeys: diagnostics ? [...diagnostics.legalActionKeys] : undefined,
          legalActionKeysHash: diagnostics ? stableDiagnosticHash(diagnostics.legalActionKeys) : undefined,
          selectedActionKey: getCpuDecisionActionKey(selectedDecision),
          rngStateBefore: traceRngBefore,
          rngStateAfter: runtime.rngState,
          stateBefore: traceStateBefore,
          stateAfter: summarizeTraceState(state),
          distanceCache: diagnostics ? { ...diagnostics.distanceCache } : undefined,
        });
      }
      if (options.profile) timings.recordAction(actionTurn, performance.now() - actionStarted, livingUnitCount);
      const important = importantResolutionSignature(state);
      const shouldInspect = mode === "debug" || (mode === "sweep" && important !== previousImportant);
      previousImportant = important;
      if (shouldInspect) {
        const violations = inspect();
        if (violations.length) return finish("invariant_violation", violations);
      }
      const stallStarted = options.profile ? performance.now() : 0;
      const current = getHeadlessProgressSignature(state, runtime);
      const stalled = current === previous;
      timings.add("stallDetection", options.profile ? performance.now() - stallStarted : 0);
      if (stalled) return finish("phase_stall", [], "CPU action produced no state or policy progress");
      previous = current;
    }
  } catch (caught) {
    return finish("exception", [], caught instanceof Error ? `${caught.name}: ${caught.message}` : String(caught));
  }
}

export function runHeadlessBatch(input: Omit<HeadlessMatchOptions, "seed"> & { matchCount: number; seedStart: number }): HeadlessBatchResult {
  const matches = Array.from({ length: input.matchCount }, (_, index) => runHeadlessMatch({ ...input, seed: input.seedStart + index }));
  return summarizeHeadlessMatches(matches);
}

export function summarizeHeadlessMatches(matches: HeadlessMatchResult[]): HeadlessBatchResult {
  return {
    matches,
    settledCount: matches.filter((match) => match.endReason === "victory").length,
    turnLimitCount: matches.filter((match) => match.endReason === "turn_limit").length,
    exceptionCount: matches.filter((match) => match.endReason === "exception").length,
    invariantViolationCount: matches.filter((match) => match.endReason === "invariant_violation").length,
    phaseStallCount: matches.filter((match) => match.endReason === "phase_stall").length,
    actionLimitCount: matches.filter((match) => match.endReason === "action_limit").length,
  };
}
