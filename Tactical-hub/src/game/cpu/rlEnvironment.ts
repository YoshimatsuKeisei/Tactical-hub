import { getAttackCandidates, getTeamAttackerUnitIds } from "../engine/battle";
import { getBuilderUnits, getStrategistActionCandidatesForUnit } from "../engine/construction";
import { getMovementCandidates, getTeamMovementUnitIds, type MovementSemantics } from "../engine/movement";
import { getProductionCandidatesForBase } from "../engine/production";
import { isTeamProductionPending } from "../engine/productionSchedule";
import { getRewardPlacementCandidates } from "../engine/reward";
import { getTeleportDestinationCandidates, getTeleportStrategists, getTeleportTargetCandidates } from "../engine/teleport";
import type { GameState, StrategistActionKind, StrategistRole, UnitPosition } from "../types";
import { positionKey } from "../utils/position";
import { advanceCpuOneStep, syncCpuContext } from "./cpuStep";
import { createHeadlessInitialState } from "./headlessSimulation";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";
import { createTeamVisibleState } from "../visibility";
import { createCpuRuntime } from "./types";
import { getHeavyInfantryMergeCandidates } from "../engine/heavyInfantry";

export type RlActionType = CpuDecision["kind"];
export type RlLegalAction = {
  actionKey: string;
  actionType: RlActionType;
  actorTeamId: string;
  isPass: boolean;
  unitId?: string;
  targetId?: string;
  partnerUnitId?: string;
  tileId?: string;
  tileIds?: string[];
  baseId?: string;
  slotId?: string;
  unitType?: string;
  strategistRole?: StrategistRole;
  strategistActionKind?: StrategistActionKind;
  requestId?: string;
  constructionId?: string;
};

export type RlObservation = {
  config: GameState["config"];
  map: GameState["map"];
  turnNumber: number;
  phase: GameState["phase"];
  actorTeamId?: string;
  observingTeamId: string;
  actionIntents: GameState["turnState"]["actionIntents"];
  currentMovementTeamId?: string;
  movementSeatOrderTeamIds: string[];
  movementOrderStartIndex: number;
  movementOrderTeamIds: string[];
  movementCompletedTeamIds: string[];
  movedUnitIdsThisMovementPhase: string[];
  productionCompletedTeamIdsThisTurn: string[];
  teams: GameState["teams"];
  units: GameState["units"];
  bases: GameState["bases"];
  unitTurnFlags: GameState["unitTurnFlags"];
  siegeStates: GameState["siegeStates"];
  kingCampaignStates: GameState["kingCampaignStates"];
  constructions: GameState["constructions"];
  strategistActionIntents: GameState["strategistActionIntents"];
  strategistSubmittedTeamIds: string[];
  strategistCooldowns: GameState["strategistCooldowns"];
  teleportIntents: GameState["teleportIntents"];
  teleportCooldowns: GameState["teleportCooldowns"];
  rewardPlacementRequests: GameState["rewardPlacementRequests"];
  pendingRewardRequestIds: string[];
  phaseAfterRewards?: GameState["phaseAfterRewards"];
};

export type RlResult = {
  terminal: boolean;
  winnerTeamId?: string;
  loserTeamIds: string[];
  endReason: "ongoing" | "victory" | "stopped";
  actionCount: number;
  rewards: Record<string, number>;
};

export type RlRewardFunction = (state: GameState, result: Omit<RlResult, "rewards">) => Record<string, number>;

export type EnumeratedDecision = { action: RlLegalAction; decision: CpuDecision };

const activeTeamIds = (state: GameState) => state.teams.filter((team) => !team.isNeutral && team.status === "active").map((team) => team.id);
const tileId = (position: UnitPosition) => positionKey(position);
export const getCpuDecisionActionKey = (decision: CpuDecision) => {
  switch (decision.kind) {
    case "production": return `production:${decision.teamId}:${decision.actorKey}:${decision.choice ? `${decision.choice.baseId}:${decision.choice.unitType}:${decision.choice.strategistRole ?? ""}` : "pass"}`;
    case "movement": return `movement:${decision.teamId}:${decision.unitId}:${decision.to ? tileId(decision.to) : "pass"}`;
    case "merge_infantry": return `merge_infantry:${decision.teamId}:${decision.primaryUnitId}:${decision.partnerUnitId}`;
    case "teleport": return `teleport:${decision.teamId}:${decision.strategistUnitId}:${decision.intent ? `${decision.intent.targetUnitId}:${tileId(decision.intent.to)}` : "pass"}`;
    case "attack": return `attack:${decision.teamId}:${decision.intent.attackerUnitId}:${decision.intent.target?.unitId ?? "pass"}`;
    case "reward": return `reward:${decision.teamId}:${decision.requestId}:${decision.baseId}:${decision.unitType}`;
    case "strategist": return `strategist:${decision.teamId}:${decision.actorKey}:${decision.intent.action}:${decision.intent.constructionId ?? ""}:${decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join("/") ?? ""}`;
    default: return `${decision.kind}:${decision.teamId}`;
  }
};

export function describeRlDecision(decision: CpuDecision): RlLegalAction {
  const isPass = (decision.kind === "production" && !decision.choice)
    || (decision.kind === "movement" && !decision.to)
    || (decision.kind === "teleport" && !decision.intent)
    || (decision.kind === "attack" && decision.intent.pass)
    || (decision.kind === "strategist" && decision.intent.action === "pass");
  const base: RlLegalAction = { actionKey: getCpuDecisionActionKey(decision), actionType: decision.kind, actorTeamId: decision.teamId, isPass };
  switch (decision.kind) {
    case "production": return { ...base, baseId: decision.choice?.baseId, unitType: decision.choice?.unitType, strategistRole: decision.choice?.strategistRole };
    case "movement": return { ...base, unitId: decision.unitId, tileId: decision.to ? tileId(decision.to) : undefined };
    case "merge_infantry": return { ...base, unitId: decision.primaryUnitId, partnerUnitId: decision.partnerUnitId };
    case "teleport": return { ...base, unitId: decision.strategistUnitId, targetId: decision.intent?.targetUnitId, tileId: decision.intent ? tileId(decision.intent.to) : undefined };
    case "attack": return { ...base, unitId: decision.intent.attackerUnitId, targetId: decision.intent.target?.unitId, baseId: decision.intent.target?.baseId, slotId: decision.intent.target?.slotId };
    case "reward": return { ...base, requestId: decision.requestId, baseId: decision.baseId, unitType: decision.unitType };
    case "strategist": {
      const tileIds = decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`);
      return {
        ...base,
        unitId: decision.intent.strategistUnitId,
        constructionId: decision.intent.constructionId,
        strategistActionKind: decision.intent.action,
        tileId: tileIds?.join("/"),
        tileIds,
      };
    }
    default: return base;
  }
}

function wrap(decisions: CpuDecision[]): EnumeratedDecision[] {
  return decisions.map((decision) => ({ decision, action: describeRlDecision(decision) }));
}

export function enumerateRlDecisions(state: GameState, runtime: CpuRuntime, teamEligible: (teamId: string) => boolean = () => true, movementSemantics: MovementSemantics = "current"): EnumeratedDecision[] {
  syncCpuContext(runtime, state);
  const active = activeTeamIds(state);
  const eligible = active.filter(teamEligible);
  if (state.phase === "production") {
    const teamId = eligible.find((id) => !runtime.completedProductionTeamIds.includes(id));
    if (teamId) {
      for (const baseId of state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort()) {
        const actorKey = `production:${teamId}:${baseId}`;
        if (runtime.processedKeys.includes(actorKey)) continue;
        const candidates = getProductionCandidatesForBase(state, teamId, baseId);
        if (candidates.length) return wrap(candidates.map((choice) => ({ kind: "production", teamId, actorKey, choice })));
      }
      runtime.completedProductionTeamIds.push(teamId);
      return enumerateRlDecisions(state, runtime, teamEligible, movementSemantics);
    }
    return eligible.length === active.length ? wrap([{ kind: "resolve_production", teamId: "all" }]) : [];
  }
  if (state.phase === "movement_input") {
    const teamId = state.currentMovementTeamId;
    if (!teamId || !eligible.includes(teamId)) return [];
    const visibleState = createTeamVisibleState(state, teamId);
    if (isTeamProductionPending(state, teamId)) {
      for (const baseId of state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort()) {
        const actorKey = `movement-production:${teamId}:${baseId}`;
        if (runtime.processedKeys.includes(actorKey)) continue;
        const candidates = getProductionCandidatesForBase(state, teamId, baseId);
        if (candidates.length) return wrap(candidates.map((choice) => ({ kind: "production", teamId, actorKey, choice })));
      }
      return wrap([{ kind: "submit_team_production", teamId }]);
    }
    const unitId = getTeamMovementUnitIds(visibleState, teamId).find((id) => !runtime.processedKeys.includes(`movement:${teamId}:${id}`));
    if (unitId) {
      const actorKey = `movement:${teamId}:${unitId}`;
      const destinations = getMovementCandidates(visibleState, unitId, movementSemantics).sort((left, right) => tileId(left).localeCompare(tileId(right)));
      return wrap([
        { kind: "movement", teamId, actorKey, unitId },
        ...destinations.map((to): CpuDecision => ({ kind: "movement", teamId, actorKey, unitId, to })),
      ]);
    }
    const strategistUnitId = getTeleportStrategists(state, teamId).map((unit) => unit.id).find((id) => !runtime.processedKeys.includes(`teleport:${teamId}:${id}`));
    if (strategistUnitId) {
      const actorKey = `teleport:${teamId}:${strategistUnitId}`;
      const targets = getTeleportTargetCandidates(state, strategistUnitId);
      const destinations = getTeleportDestinationCandidates(state, strategistUnitId);
      return wrap([
        { kind: "teleport", teamId, actorKey, strategistUnitId },
        ...targets.flatMap((target) => destinations.map((to): CpuDecision => ({ kind: "teleport", teamId, actorKey, strategistUnitId, intent: { teamId, strategistUnitId, targetUnitId: target.id, to } }))),
      ]);
    }
    return wrap([{ kind: "submit_movement", teamId }]);
  }
  if (state.phase === "attack_input") {
    const teamId = eligible.find((id) => !runtime.completedAttackTeamIds.includes(id));
    if (teamId) {
      const attackerUnitId = getTeamAttackerUnitIds(state, teamId).find((id) => !runtime.processedKeys.includes(`attack:${teamId}:${id}`));
      if (!attackerUnitId) return wrap([{ kind: "complete_attack_team", teamId }]);
      const actorKey = `attack:${teamId}:${attackerUnitId}`;
      const targets = getAttackCandidates(state, attackerUnitId);
      return wrap([
        { kind: "attack", teamId, actorKey, intent: { teamId, attackerUnitId, pass: true } },
        ...targets.map((target): CpuDecision => ({ kind: "attack", teamId, actorKey, intent: { teamId, attackerUnitId, target, pass: false } })),
      ]);
    }
    return eligible.length === active.length ? wrap([{ kind: "resolve_battle", teamId: "all" }]) : [];
  }
  if (state.phase === "reward_placement") {
    return wrap(eligible.flatMap((teamId) => getRewardPlacementCandidates(state, teamId).map((candidate): CpuDecision => ({ kind: "reward", teamId, ...candidate }))))
      .sort((left, right) => left.action.actionKey.localeCompare(right.action.actionKey));
  }
  if (state.phase === "strategist_action_input") {
    const teamId = eligible.find((id) => !state.strategistSubmittedTeamIds.includes(id));
    if (!teamId) return [];
    const strategistUnitId = getBuilderUnits(state, teamId).map((unit) => unit.id).sort().find((id) => !runtime.processedKeys.includes(`strategist:${teamId}:${id}`));
    if (!strategistUnitId) return wrap([{ kind: "submit_strategist", teamId }]);
    const actorKey = `strategist:${teamId}:${strategistUnitId}`;
    return wrap(getStrategistActionCandidatesForUnit(state, teamId, strategistUnitId).map((intent) => ({ kind: "strategist", teamId, actorKey, intent })));
  }
  if (state.phase === "strategist_action_resolution") return eligible.length === active.length ? wrap([{ kind: "resolve_strategists", teamId: "all" }]) : [];
  return [];
}

export const enumerateRlDecisionsV1 = enumerateRlDecisions;

export function enumerateRlDecisionsV2(state: GameState, runtime: CpuRuntime, teamEligible: (teamId: string) => boolean = () => true): EnumeratedDecision[] {
  const decisions = enumerateRlDecisions(state, runtime, teamEligible);
  const teamId = state.currentMovementTeamId;
  if (state.phase !== "movement_input" || !teamId || !teamEligible(teamId)) return decisions;
  const mergeDecisions = state.units
    .filter((unit) => unit.ownerTeamId === teamId)
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((unit) => getHeavyInfantryMergeCandidates(state, unit.id)
      .filter((partner) => unit.id.localeCompare(partner.id) < 0)
      .map((partner): CpuDecision => ({
        kind: "merge_infantry",
        teamId,
        actorKey: `movement:${teamId}:${unit.id}`,
        primaryUnitId: unit.id,
        partnerUnitId: partner.id,
      })));
  return [...decisions, ...wrap(mergeDecisions)];
}

const defaultRewards: RlRewardFunction = (state, result) => Object.fromEntries(
  state.teams.filter((team) => !team.isNeutral).map((team) => [team.id, result.terminal ? (team.id === result.winnerTeamId ? 1 : -1) : 0]),
);

export function buildRlObservation(state: GameState, teamId: string, actorTeamId?: string): RlObservation {
  if (!state.teams.some((team) => team.id === teamId && !team.isNeutral)) throw new Error(`Unknown observing team: ${teamId}`);
  const visibleState = createTeamVisibleState(state, teamId);
  return {
    config: visibleState.config,
    map: visibleState.map,
    turnNumber: visibleState.turnNumber,
    phase: visibleState.phase,
    actorTeamId,
    observingTeamId: teamId,
    actionIntents: state.turnState.actionIntents.filter((intent) => intent.teamId === teamId),
    currentMovementTeamId: state.currentMovementTeamId,
    movementSeatOrderTeamIds: state.movementSeatOrderTeamIds,
    movementOrderStartIndex: state.movementOrderStartIndex,
    movementOrderTeamIds: state.movementOrderTeamIds,
    movementCompletedTeamIds: state.movementCompletedTeamIds,
    movedUnitIdsThisMovementPhase: visibleState.movedUnitIdsThisMovementPhase,
    productionCompletedTeamIdsThisTurn: state.productionCompletedTeamIdsThisTurn,
    teams: state.teams,
    units: visibleState.units,
    bases: state.bases,
    unitTurnFlags: visibleState.unitTurnFlags,
    siegeStates: state.siegeStates,
    kingCampaignStates: state.kingCampaignStates,
    constructions: state.constructions,
    strategistActionIntents: state.strategistActionIntents.filter((intent) => intent.teamId === teamId),
    strategistSubmittedTeamIds: state.strategistSubmittedTeamIds,
    strategistCooldowns: state.strategistCooldowns,
    teleportIntents: state.teleportIntents.filter((intent) => intent.teamId === teamId),
    teleportCooldowns: state.teleportCooldowns,
    rewardPlacementRequests: state.rewardPlacementRequests,
    pendingRewardRequestIds: state.rewardPlacementRequests.filter((request) => !request.completed && !request.expired).map((request) => request.id),
    phaseAfterRewards: state.phaseAfterRewards,
  };
}

export class RlEnvironment {
  private state!: GameState;
  private runtime!: CpuRuntime;
  private decisions: EnumeratedDecision[] = [];
  private readonly rewardFunction: RlRewardFunction;

  constructor(
    rewardFunction: RlRewardFunction = defaultRewards,
    private readonly schemaVersion: 1 | 2 = 1,
    private readonly movementSemantics: MovementSemantics = "current",
  ) { this.rewardFunction = rewardFunction; }

  reset(seed: number, participantCount: 3 | 4 = 4, initialState?: GameState) {
    this.state = structuredClone(initialState ?? createHeadlessInitialState(participantCount)) as GameState;
    this.runtime = createCpuRuntime(seed, 100_000);
    this.advanceAutomatic();
    return this.getObservation(this.getCurrentActorTeamId() ?? activeTeamIds(this.state)[0]);
  }

  private apply(decision: CpuDecision) {
    const settings: CpuTeamSettings = Object.fromEntries(activeTeamIds(this.state).map((teamId) => [teamId, "random_cpu"]));
    const result = advanceCpuOneStep(this.state, this.runtime, settings, () => decision, { logMode: "none", movementSemantics: this.movementSemantics });
    if (!result.applied) throw new Error("RL action was not applied");
    this.state = result.state;
    this.runtime = result.runtime;
  }

  private advanceAutomatic() {
    for (let guard = 0; guard < 100; guard += 1) {
      if (this.isTerminal()) { this.decisions = []; return; }
      this.decisions = this.schemaVersion === 1
        ? enumerateRlDecisions(this.state, this.runtime, () => true, this.movementSemantics)
        : enumerateRlDecisionsV2(this.state, this.runtime);
      if (this.decisions.length !== 1 || !["resolve_production", "resolve_battle", "resolve_strategists"].includes(this.decisions[0].decision.kind)) return;
      this.apply(this.decisions[0].decision);
    }
    throw new Error("RL automatic processing safety limit reached");
  }

  getCurrentActorTeamId() { return this.decisions[0]?.action.actorTeamId === "all" ? undefined : this.decisions[0]?.action.actorTeamId; }

  private buildObservation(teamId: string): RlObservation {
    return buildRlObservation(this.state, teamId, this.getCurrentActorTeamId());
  }

  getObservation(teamId: string): RlObservation {
    return structuredClone(this.buildObservation(teamId));
  }

  /**
   * Transient read-only view for the synchronous encoder/replay hot path.
   * Callers must not retain or mutate it; public observations remain cloned.
   */
  getObservationForEncoding(teamId: string): RlObservation {
    return this.buildObservation(teamId);
  }

  getLegalActions(teamId: string): RlLegalAction[] {
    if (teamId !== this.getCurrentActorTeamId()) return [];
    return structuredClone(this.decisions.map((entry) => entry.action));
  }

  step(actionKeyValue: string) {
    const selected = this.decisions.find((entry) => entry.action.actionKey === actionKeyValue);
    if (!selected) throw new Error(`Illegal or stale RL actionKey: ${actionKeyValue}`);
    this.apply(selected.decision);
    this.advanceAutomatic();
    return { observation: this.getCurrentActorTeamId() ? this.getObservation(this.getCurrentActorTeamId()!) : undefined, result: this.getResult() };
  }

  stepReplayAction(actionKeyValue: string, rngStateAfterPolicy: number) {
    if (!Number.isInteger(rngStateAfterPolicy) || rngStateAfterPolicy < 0 || rngStateAfterPolicy > 0xffff_ffff) {
      throw new Error(`Invalid replay rngState: ${rngStateAfterPolicy}`);
    }
    const selected = this.decisions.find((entry) => entry.action.actionKey === actionKeyValue);
    if (!selected) throw new Error(`Illegal or stale RL actionKey: ${actionKeyValue}`);
    this.runtime.rngState = rngStateAfterPolicy;
    this.apply(selected.decision);
    this.advanceAutomatic();
  }

  stepWithPolicy(policy: CpuPolicy = getRandomCpuDecision) {
    return this.stepWithPolicyForReplay(policy).stepResult;
  }

  stepWithPolicyForReplay(policy: CpuPolicy = getRandomCpuDecision) {
    const policyRuntime = structuredClone(this.runtime) as CpuRuntime;
    const controller = policy.controller ?? "random_cpu";
    const settings: CpuTeamSettings = Object.fromEntries(activeTeamIds(this.state).map((teamId) => [teamId, controller]));
    const decision = policy(this.state, policyRuntime, settings);
    if (!decision) throw new Error("RL policy returned no action at a decision point");
    const selected = this.decisions.find((entry) => entry.action.actionKey === getCpuDecisionActionKey(decision));
    if (!selected) throw new Error(`RL policy selected an action outside the legal list: ${getCpuDecisionActionKey(decision)}`);
    this.runtime = policyRuntime;
    const rngStateAfterPolicy = policyRuntime.rngState;
    return {
      actionKey: selected.action.actionKey,
      rngStateAfterPolicy,
      stepResult: this.step(selected.action.actionKey),
    };
  }

  isTerminal() { return activeTeamIds(this.state).length <= 1 || Boolean(this.runtime.stoppedReason); }

  getResult(): RlResult {
    const active = activeTeamIds(this.state);
    const basic = {
      terminal: this.isTerminal(),
      winnerTeamId: active.length === 1 ? active[0] : undefined,
      loserTeamIds: this.state.teams.filter((team) => !team.isNeutral && team.status !== "active").map((team) => team.id),
      endReason: (this.runtime.stoppedReason ? "stopped" : active.length <= 1 ? "victory" : "ongoing") as RlResult["endReason"],
      actionCount: this.runtime.appliedStepCount,
    };
    return { ...basic, rewards: this.rewardFunction(this.state, basic) };
  }

  getStateHash() {
    const text = JSON.stringify({ state: this.state, runtime: { rngState: this.runtime.rngState, contextKey: this.runtime.contextKey, processedKeys: this.runtime.processedKeys, completedProductionTeamIds: this.runtime.completedProductionTeamIds, completedAttackTeamIds: this.runtime.completedAttackTeamIds, hiddenAttackIntents: this.runtime.hiddenAttackIntents, appliedStepCount: this.runtime.appliedStepCount } });
    let hash = 2166136261;
    for (const character of text) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619) >>> 0;
    return hash.toString(16).padStart(8, "0");
  }
}

export class RlEnvironmentV2 extends RlEnvironment {
  constructor(rewardFunction: RlRewardFunction = defaultRewards) { super(rewardFunction, 2); }
}

/** Schema-v1 environment retained only for replaying pre-immediate-movement data. */
export class LegacyReplayRlEnvironment extends RlEnvironment {
  constructor(rewardFunction: RlRewardFunction = defaultRewards) { super(rewardFunction, 1, "legacy_batched"); }
}
