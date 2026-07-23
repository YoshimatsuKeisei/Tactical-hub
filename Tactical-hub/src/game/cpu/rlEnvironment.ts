import { getAttackCandidates, getTeamAttackerUnitIds } from "../engine/battle";
import { getBuilderUnits, getStrategistActionCandidatesForUnit } from "../engine/construction";
import { getMovementCandidates, getTeamMovementUnitIds } from "../engine/movement";
import { getProductionCandidatesForBase } from "../engine/production";
import { isTeamProductionPending } from "../engine/productionSchedule";
import { getRewardPlacementCandidates } from "../engine/reward";
import { getTeleportDestinationCandidates, getTeleportStrategists, getTeleportTargetCandidates } from "../engine/teleport";
import type { GameState, UnitPosition } from "../types";
import { positionKey } from "../utils/position";
import { advanceCpuOneStep, syncCpuContext } from "./cpuStep";
import { createHeadlessInitialState } from "./headlessSimulation";
import { getRandomCpuDecision } from "./randomCpuPolicy";
import type { CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";
import { createCpuRuntime } from "./types";

export type RlActionType = CpuDecision["kind"];
export type RlLegalAction = {
  actionKey: string;
  actionType: RlActionType;
  actorTeamId: string;
  unitId?: string;
  targetId?: string;
  tileId?: string;
  baseId?: string;
  unitType?: string;
  requestId?: string;
  constructionId?: string;
};

export type RlObservation = {
  turnNumber: number;
  phase: GameState["phase"];
  actorTeamId?: string;
  observingTeamId: string;
  currentMovementTeamId?: string;
  movementOrderTeamIds: string[];
  movementCompletedTeamIds: string[];
  teams: GameState["teams"];
  units: GameState["units"];
  bases: GameState["bases"];
  constructions: GameState["constructions"];
  pendingRewardRequestIds: string[];
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
    case "teleport": return `teleport:${decision.teamId}:${decision.strategistUnitId}:${decision.intent ? `${decision.intent.targetUnitId}:${tileId(decision.intent.to)}` : "pass"}`;
    case "attack": return `attack:${decision.teamId}:${decision.intent.attackerUnitId}:${decision.intent.target?.unitId ?? "pass"}`;
    case "reward": return `reward:${decision.teamId}:${decision.requestId}:${decision.baseId}:${decision.unitType}`;
    case "strategist": return `strategist:${decision.teamId}:${decision.actorKey}:${decision.intent.action}:${decision.intent.constructionId ?? ""}:${decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join("/") ?? ""}`;
    default: return `${decision.kind}:${decision.teamId}`;
  }
};

function describe(decision: CpuDecision): RlLegalAction {
  const base: RlLegalAction = { actionKey: getCpuDecisionActionKey(decision), actionType: decision.kind, actorTeamId: decision.teamId };
  switch (decision.kind) {
    case "production": return { ...base, baseId: decision.choice?.baseId, unitType: decision.choice?.unitType };
    case "movement": return { ...base, unitId: decision.unitId, tileId: decision.to ? tileId(decision.to) : undefined };
    case "teleport": return { ...base, unitId: decision.strategistUnitId, targetId: decision.intent?.targetUnitId, tileId: decision.intent ? tileId(decision.intent.to) : undefined };
    case "attack": return { ...base, unitId: decision.intent.attackerUnitId, targetId: decision.intent.target?.unitId };
    case "reward": return { ...base, requestId: decision.requestId, baseId: decision.baseId, unitType: decision.unitType };
    case "strategist": return { ...base, unitId: decision.intent.strategistUnitId, constructionId: decision.intent.constructionId, tileId: decision.intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join("/") };
    default: return base;
  }
}

function wrap(decisions: CpuDecision[]): EnumeratedDecision[] {
  return decisions.map((decision) => ({ decision, action: describe(decision) }));
}

export function enumerateRlDecisions(state: GameState, runtime: CpuRuntime): EnumeratedDecision[] {
  syncCpuContext(runtime, state);
  const active = activeTeamIds(state);
  if (state.phase === "production") {
    const teamId = active.find((id) => !runtime.completedProductionTeamIds.includes(id));
    if (teamId) {
      for (const baseId of state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort()) {
        const actorKey = `production:${teamId}:${baseId}`;
        if (runtime.processedKeys.includes(actorKey)) continue;
        const candidates = getProductionCandidatesForBase(state, teamId, baseId);
        if (candidates.length) return wrap(candidates.map((choice) => ({ kind: "production", teamId, actorKey, choice })));
      }
      runtime.completedProductionTeamIds.push(teamId);
      return enumerateRlDecisions(state, runtime);
    }
    return wrap([{ kind: "resolve_production", teamId: "all" }]);
  }
  if (state.phase === "movement_input") {
    const teamId = state.currentMovementTeamId;
    if (!teamId || !active.includes(teamId)) return [];
    if (isTeamProductionPending(state, teamId)) {
      for (const baseId of state.bases.filter((base) => base.ownerTeamId === teamId).map((base) => base.id).sort()) {
        const actorKey = `movement-production:${teamId}:${baseId}`;
        if (runtime.processedKeys.includes(actorKey)) continue;
        const candidates = getProductionCandidatesForBase(state, teamId, baseId);
        if (candidates.length) return wrap(candidates.map((choice) => ({ kind: "production", teamId, actorKey, choice })));
      }
      return wrap([{ kind: "submit_team_production", teamId }]);
    }
    const unitId = getTeamMovementUnitIds(state, teamId).find((id) => !runtime.processedKeys.includes(`movement:${teamId}:${id}`));
    if (unitId) {
      const actorKey = `movement:${teamId}:${unitId}`;
      const destinations = getMovementCandidates(state, unitId).sort((left, right) => tileId(left).localeCompare(tileId(right)));
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
    const teamId = active.find((id) => !runtime.completedAttackTeamIds.includes(id));
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
    return wrap([{ kind: "resolve_battle", teamId: "all" }]);
  }
  if (state.phase === "reward_placement") {
    return wrap(active.flatMap((teamId) => getRewardPlacementCandidates(state, teamId).map((candidate): CpuDecision => ({ kind: "reward", teamId, ...candidate }))))
      .sort((left, right) => left.action.actionKey.localeCompare(right.action.actionKey));
  }
  if (state.phase === "strategist_action_input") {
    const teamId = active.find((id) => !state.strategistSubmittedTeamIds.includes(id));
    if (!teamId) return [];
    const strategistUnitId = getBuilderUnits(state, teamId).map((unit) => unit.id).sort().find((id) => !runtime.processedKeys.includes(`strategist:${teamId}:${id}`));
    if (!strategistUnitId) return wrap([{ kind: "submit_strategist", teamId }]);
    const actorKey = `strategist:${teamId}:${strategistUnitId}`;
    return wrap(getStrategistActionCandidatesForUnit(state, teamId, strategistUnitId).map((intent) => ({ kind: "strategist", teamId, actorKey, intent })));
  }
  if (state.phase === "strategist_action_resolution") return wrap([{ kind: "resolve_strategists", teamId: "all" }]);
  return [];
}

const defaultRewards: RlRewardFunction = (state, result) => Object.fromEntries(
  state.teams.filter((team) => !team.isNeutral).map((team) => [team.id, result.terminal ? (team.id === result.winnerTeamId ? 1 : -1) : 0]),
);

export class RlEnvironment {
  private state!: GameState;
  private runtime!: CpuRuntime;
  private decisions: EnumeratedDecision[] = [];
  private readonly rewardFunction: RlRewardFunction;

  constructor(rewardFunction: RlRewardFunction = defaultRewards) { this.rewardFunction = rewardFunction; }

  reset(seed: number, participantCount: 3 | 4 = 4, initialState?: GameState) {
    this.state = structuredClone(initialState ?? createHeadlessInitialState(participantCount)) as GameState;
    this.runtime = createCpuRuntime(seed, 100_000);
    this.advanceAutomatic();
    return this.getObservation(this.getCurrentActorTeamId() ?? activeTeamIds(this.state)[0]);
  }

  private apply(decision: CpuDecision) {
    const settings: CpuTeamSettings = Object.fromEntries(activeTeamIds(this.state).map((teamId) => [teamId, "random_cpu"]));
    const result = advanceCpuOneStep(this.state, this.runtime, settings, () => decision, { logMode: "none" });
    if (!result.applied) throw new Error("RL action was not applied");
    this.state = result.state;
    this.runtime = result.runtime;
  }

  private advanceAutomatic() {
    for (let guard = 0; guard < 100; guard += 1) {
      if (this.isTerminal()) { this.decisions = []; return; }
      this.decisions = enumerateRlDecisions(this.state, this.runtime);
      if (this.decisions.length !== 1 || !["resolve_production", "resolve_battle", "resolve_strategists"].includes(this.decisions[0].decision.kind)) return;
      this.apply(this.decisions[0].decision);
    }
    throw new Error("RL automatic processing safety limit reached");
  }

  getCurrentActorTeamId() { return this.decisions[0]?.action.actorTeamId === "all" ? undefined : this.decisions[0]?.action.actorTeamId; }

  getObservation(teamId: string): RlObservation {
    if (!this.state.teams.some((team) => team.id === teamId && !team.isNeutral)) throw new Error(`Unknown observing team: ${teamId}`);
    return structuredClone({
      turnNumber: this.state.turnNumber,
      phase: this.state.phase,
      actorTeamId: this.getCurrentActorTeamId(),
      observingTeamId: teamId,
      currentMovementTeamId: this.state.currentMovementTeamId,
      movementOrderTeamIds: this.state.movementOrderTeamIds,
      movementCompletedTeamIds: this.state.movementCompletedTeamIds,
      teams: this.state.teams,
      units: this.state.units,
      bases: this.state.bases,
      constructions: this.state.constructions,
      pendingRewardRequestIds: this.state.rewardPlacementRequests.filter((request) => !request.completed && !request.expired).map((request) => request.id),
    });
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

  stepWithPolicy(policy: CpuPolicy = getRandomCpuDecision) {
    const policyRuntime = structuredClone(this.runtime) as CpuRuntime;
    const settings: CpuTeamSettings = Object.fromEntries(activeTeamIds(this.state).map((teamId) => [teamId, "random_cpu"]));
    const decision = policy(this.state, policyRuntime, settings);
    if (!decision) throw new Error("RL policy returned no action at a decision point");
    const selected = this.decisions.find((entry) => entry.action.actionKey === getCpuDecisionActionKey(decision));
    if (!selected) throw new Error(`RL policy selected an action outside the legal list: ${getCpuDecisionActionKey(decision)}`);
    this.runtime = policyRuntime;
    return this.step(selected.action.actionKey);
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
