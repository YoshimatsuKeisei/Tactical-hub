import type { GameState, UnitPosition, UnitType } from "../types";
import { positionKey } from "../utils/position";
import type { CpuDecision } from "./types";
import type { RlDecisionAuditHint, RlLegalAction } from "./rlEnvironment";
import type { BattleResolutionAuditEvent } from "../engine/battle";

export const PPO_ABILITY_AUDIT_SCHEMA_VERSION = 1;
export const AUDITED_PRODUCTION_UNIT_TYPES = [
  "infantry",
  "cavalry",
  "archer",
  "engineer",
  "ninja",
  "strategist",
] as const satisfies readonly UnitType[];
const STRATEGIST_ROLES = ["encourage", "builder", "teleporter"] as const;

type ProductionUnitType = (typeof AUDITED_PRODUCTION_UNIT_TYPES)[number];
type StrategistRole = (typeof STRATEGIST_ROLES)[number];
type EncouragementSource = "none" | "strategist" | "king" | "both";

export type PpoAbilityAuditMatch = {
  schemaVersion: typeof PPO_ABILITY_AUDIT_SCHEMA_VERSION;
  production: {
    selectionCount: number;
    passCount: number;
    counts: Record<ProductionUnitType, number>;
    shares: Record<ProductionUnitType, number>;
    successCount: number;
    failureCount: number;
    successCounts: Record<ProductionUnitType, number>;
    failureCounts: Record<ProductionUnitType, number>;
    successRates: Record<ProductionUnitType, number>;
    successfulShares: Record<ProductionUnitType, number>;
    strategistRoleCounts: Record<StrategistRole, number>;
    specialUnitCount: number;
    specialUnitShare: number;
  };
  strategist: {
    bridge: { instructionCount: number; successCount: number; failureCount: number; placedCount: number; placedTileCount: number };
    obstacle: { instructionCount: number; successCount: number; failureCount: number; placedCount: number };
    teleport: { opportunityCount: number; instructionCount: number; successCount: number; failureCount: number; teleportedUnitCount: number };
  };
  encouragement: {
    totalAttackCount: number;
    appliedAttackCount: number;
    hitCount: number;
    missCount: number;
    withoutEncouragementCount: number;
    strategistAppliedAttackCount: number;
    kingAppliedAttackCount: number;
    bothAppliedAttackCount: number;
    applicationRate: number;
  };
  ninja: {
    appearedUnitCount: number;
    appearedUnitIds: string[];
    submersionStartCount: number;
    submersionCompleteCount: number;
    submersionIncompleteCount: number;
    forcedExposureCount: number;
    underwaterMoveCount: number;
    underwaterTurnCount: number;
    teleportOpportunityCount: number;
    teleportSuccessCount: number;
    teleportUserCount: number;
    teleportUserIds: string[];
    obstacleJumpOpportunityCount: number;
    obstacleJumpSuccessCount: number;
    obstacleJumpUserCount: number;
    obstacleJumpUserIds: string[];
    incompleteReasons: Record<string, number>;
  };
};

export type PpoAbilityAuditAggregate = PpoAbilityAuditMatch & {
  matchCount: number;
  production: PpoAbilityAuditMatch["production"] & {
    averagePerMatch: Record<ProductionUnitType, number>;
  };
  perMatch: Array<{ environmentIndex: number; seed: number; audit: PpoAbilityAuditMatch }>;
};

const emptyRecord = <K extends string>(keys: readonly K[]) =>
  Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;

type PendingMovement = {
  unitId: string;
  destinationKey: string;
  ninjaTeleport: boolean;
  obstacleJump: boolean;
};

type PendingProduction = {
  teamId: string;
  baseId: string;
  unitType: ProductionUnitType;
};

const logSignature = (log: GameState["logs"][number]) => JSON.stringify([
  log.id,
  log.turnNumber,
  log.type,
  log.message,
  log.relatedIds ?? [],
]);

export class PpoAbilityAuditTracker {
  private readonly productionCounts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  private readonly productionSuccessCounts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  private readonly productionFailureCounts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  private readonly strategistRoleCounts = emptyRecord(STRATEGIST_ROLES);
  private productionSelectionCount = 0;
  private productionPassCount = 0;
  private bridgeInstructions = 0;
  private bridgeSuccess = 0;
  private bridgeFailures = 0;
  private bridgePlaced = 0;
  private bridgeTiles = 0;
  private obstacleInstructions = 0;
  private obstacleSuccess = 0;
  private obstacleFailures = 0;
  private obstaclePlaced = 0;
  private strategistTeleportOpportunities = 0;
  private strategistTeleportInstructions = 0;
  private strategistTeleportSuccess = 0;
  private strategistTeleportFailures = 0;
  private strategistTeleportedUnits = 0;
  private totalAttacks = 0;
  private encouragedAttacks = 0;
  private encouragedHits = 0;
  private encouragedMisses = 0;
  private unencouragedAttacks = 0;
  private strategistEncouragedAttacks = 0;
  private kingEncouragedAttacks = 0;
  private bothEncouragedAttacks = 0;
  private ninjaSubmersionStarts = 0;
  private ninjaSubmersionCompletes = 0;
  private ninjaSubmersionIncomplete = 0;
  private ninjaForcedExposures = 0;
  private ninjaUnderwaterMoves = 0;
  private ninjaUnderwaterTurns = 0;
  private ninjaTeleportOpportunities = 0;
  private ninjaTeleportSuccess = 0;
  private ninjaObstacleOpportunities = 0;
  private ninjaObstacleSuccess = 0;
  private readonly ninjaIds = new Set<string>();
  private readonly ninjaTeleportUsers = new Set<string>();
  private readonly ninjaObstacleUsers = new Set<string>();
  private readonly activeSubmersions = new Map<string, { enteredTurn: number; lastTurn: number }>();
  private readonly incompleteReasons: Record<string, number> = {};
  private readonly observedBattleEventKeys = new Set<string>();
  private readonly pendingProductionByTeam = new Map<string, PendingProduction>();
  private readonly pendingConstructionKinds = new Map<string, "bridge" | "obstacle">();
  private readonly observedConstructionIds = new Set<string>();
  private readonly observedOpportunityKeys = new Set<string>();
  private readonly movementHints = new Map<string, RlDecisionAuditHint>();
  private logSnapshot: string[];
  private logArrayReference: GameState["logs"];
  private beforePositions = new Map<string, UnitPosition>();
  private beforeConstructionIds = new Set<string>();
  private beforeUnitIds = new Set<string>();
  private resolvingProductionChoices: PendingProduction[] = [];
  private pendingMovement?: PendingMovement;

  constructor(initialState: GameState) {
    this.trackNinjas(initialState);
    for (const ninja of initialState.units.filter((unit) => unit.type === "ninja" && unit.position.kind === "water")) {
      this.activeSubmersions.set(ninja.id, { enteredTurn: initialState.turnNumber, lastTurn: initialState.turnNumber });
    }
    this.logSnapshot = initialState.logs.map(logSignature);
    this.logArrayReference = initialState.logs;
    for (const construction of initialState.constructions) this.observedConstructionIds.add(construction.id);
  }

  private trackNinjas(state: GameState) {
    for (const unit of state.units) if (unit.type === "ninja") this.ninjaIds.add(unit.id);
  }

  observeDecisionPoint(
    state: GameState,
    legal: readonly RlLegalAction[],
    movementAuditHints: readonly RlDecisionAuditHint[] = [],
  ) {
    this.trackNinjas(state);
    this.movementHints.clear();
    for (const hint of movementAuditHints) this.movementHints.set(`${hint.unitId}:${hint.destinationKey}`, hint);
    const movementUnitId = movementAuditHints[0]?.unitId;
    if (movementUnitId) {
      const baseKey = `${state.turnNumber}:${state.phase}:${movementUnitId}`;
      if (movementAuditHints.some((hint) => hint.ninjaTeleport) && !this.observedOpportunityKeys.has(`${baseKey}:ninja-teleport`)) {
        this.observedOpportunityKeys.add(`${baseKey}:ninja-teleport`);
        this.ninjaTeleportOpportunities += 1;
      }
      if (movementAuditHints.some((hint) => hint.obstacleJump) && !this.observedOpportunityKeys.has(`${baseKey}:obstacle-jump`)) {
        this.observedOpportunityKeys.add(`${baseKey}:obstacle-jump`);
        this.ninjaObstacleOpportunities += 1;
      }
    }
    const strategistTeleport = legal.some((action) => action.actionType === "teleport" && !action.isPass);
    const strategist = legal.find((action) => action.actionType === "teleport")?.unitId;
    const key = `${state.turnNumber}:${state.phase}:${strategist ?? "none"}:strategist-teleport`;
    if (strategistTeleport && !this.observedOpportunityKeys.has(key)) {
      this.observedOpportunityKeys.add(key);
      this.strategistTeleportOpportunities += 1;
    }
  }

  observeBattleResolution(events: readonly BattleResolutionAuditEvent[]) {
    for (const event of events) {
      const key = `${event.turnNumber}:${event.id}:${event.attackerUnitId}:${event.targetUnitId}`;
      if (this.observedBattleEventKeys.has(key)) continue;
      this.observedBattleEventKeys.add(key);
      this.totalAttacks += 1;
      const source: EncouragementSource = event.encouraged ? event.encouragementSource : "none";
      if (source === "none") {
        this.unencouragedAttacks += 1;
        continue;
      }
      this.encouragedAttacks += 1;
      if (event.hit) this.encouragedHits += 1;
      else this.encouragedMisses += 1;
      if (source === "strategist" || source === "both") this.strategistEncouragedAttacks += 1;
      if (source === "king" || source === "both") this.kingEncouragedAttacks += 1;
      if (source === "both") this.bothEncouragedAttacks += 1;
    }
  }

  beforeApply(state: GameState, decision: CpuDecision) {
    this.trackNinjas(state);
    this.alignLogSnapshot(state.logs);
    this.beforePositions = new Map(state.units.filter((unit) => unit.type === "ninja").map((unit) => [unit.id, structuredClone(unit.position) as UnitPosition]));
    this.beforeConstructionIds = new Set(state.constructions.map((entry) => entry.id));
    this.beforeUnitIds = new Set(state.units.map((unit) => unit.id));
    this.resolvingProductionChoices = [];
    this.pendingMovement = undefined;

    if (decision.kind === "resolve_production" || decision.kind === "submit_team_production") {
      const teamId = decision.kind === "submit_team_production" ? decision.teamId : undefined;
      const seenTeams = new Set<string>();
      this.resolvingProductionChoices = state.turnState.actionIntents
        .filter((intent) => !teamId || intent.teamId === teamId)
        .flatMap((intent) => intent.productionChoices)
        .filter((choice) => {
          if (seenTeams.has(choice.teamId)) return false;
          seenTeams.add(choice.teamId);
          return AUDITED_PRODUCTION_UNIT_TYPES.includes(choice.unitType as ProductionUnitType);
        })
        .map((choice) => ({ teamId: choice.teamId, baseId: choice.baseId, unitType: choice.unitType as ProductionUnitType }));
    }

    if (decision.kind === "production") {
      if (!decision.choice) this.productionPassCount += 1;
      else if (AUDITED_PRODUCTION_UNIT_TYPES.includes(decision.choice.unitType as ProductionUnitType)) {
        const previous = this.pendingProductionByTeam.get(decision.choice.teamId);
        if (previous) this.failProduction(previous);
        this.productionSelectionCount += 1;
        this.productionCounts[decision.choice.unitType as ProductionUnitType] += 1;
        this.pendingProductionByTeam.set(decision.choice.teamId, {
          teamId: decision.choice.teamId,
          baseId: decision.choice.baseId,
          unitType: decision.choice.unitType as ProductionUnitType,
        });
        if (decision.choice.unitType === "strategist" && decision.choice.strategistRole) {
          this.strategistRoleCounts[decision.choice.strategistRole] += 1;
        }
      }
    } else if (decision.kind === "strategist") {
      if (decision.intent.action === "place_bridge") {
        this.bridgeInstructions += 1;
        this.pendingConstructionKinds.set(decision.intent.strategistUnitId, "bridge");
      }
      if (decision.intent.action === "place_obstacle") {
        this.obstacleInstructions += 1;
        this.pendingConstructionKinds.set(decision.intent.strategistUnitId, "obstacle");
      }
    } else if (decision.kind === "teleport" && decision.intent) {
      this.strategistTeleportInstructions += 1;
    } else if (decision.kind === "movement" && decision.to) {
      const destination = decision.to;
      const unit = state.units.find((entry) => entry.id === decision.unitId);
      if (unit?.type === "ninja") {
        const hint = this.movementHints.get(`${unit.id}:${positionKey(destination)}`);
        this.pendingMovement = {
          unitId: unit.id,
          destinationKey: positionKey(destination),
          ninjaTeleport: hint?.ninjaTeleport ?? false,
          obstacleJump: hint?.obstacleJump ?? false,
        };
      }
    }
  }

  afterApply(state: GameState, decision: CpuDecision) {
    this.trackNinjas(state);
    if (this.resolvingProductionChoices.length) {
      const newUnits = state.units.filter((unit) => !this.beforeUnitIds.has(unit.id));
      for (const choice of this.resolvingProductionChoices) {
        const pending = this.pendingProductionByTeam.get(choice.teamId);
        if (!pending || pending.baseId !== choice.baseId || pending.unitType !== choice.unitType) continue;
        const produced = newUnits.some((unit) => unit.ownerTeamId === choice.teamId
          && unit.type === choice.unitType
          && unit.position.kind === "base"
          && unit.position.baseId === choice.baseId);
        if (produced) this.productionSuccessCounts[choice.unitType] += 1;
        else this.failProduction(choice);
        this.pendingProductionByTeam.delete(choice.teamId);
      }
      this.resolvingProductionChoices = [];
    }
    const newConstructions = state.constructions.filter((entry) =>
      !this.beforeConstructionIds.has(entry.id) && !this.observedConstructionIds.has(entry.id),
    );
    for (const construction of newConstructions) {
      this.observedConstructionIds.add(construction.id);
      if (construction.kind === "bridge") {
        this.bridgeSuccess += 1;
        this.bridgePlaced += 1;
        this.bridgeTiles += construction.tiles.length;
      } else {
        this.obstacleSuccess += 1;
        this.obstaclePlaced += 1;
      }
      if (construction.managerUnitId) this.pendingConstructionKinds.delete(construction.managerUnitId);
    }

    for (const log of this.consumeNewLogs(state.logs)) {
      if (log.type === "construction" && /placement failed|prerequisite disappeared/.test(log.message)) {
        const strategistId = log.relatedIds?.find((id) => this.pendingConstructionKinds.has(id));
        const kind = strategistId ? this.pendingConstructionKinds.get(strategistId) : undefined;
        if (kind === "bridge") this.bridgeFailures += 1;
        if (kind === "obstacle") this.obstacleFailures += 1;
        if (strategistId) this.pendingConstructionKinds.delete(strategistId);
      }
      if (log.type === "movement" && /teleport failed revalidation/.test(log.message)) this.strategistTeleportFailures += 1;
      if (log.type === "movement" && / teleported .* to /.test(log.message)) {
        this.strategistTeleportSuccess += 1;
        this.strategistTeleportedUnits += 1;
      }
    }

    const pending = this.pendingMovement;
    if (pending) {
      const unit = state.units.find((entry) => entry.id === pending.unitId);
      const moved = unit && unit.position.kind !== "removed" && positionKey(unit.position) === pending.destinationKey;
      if (moved && pending.ninjaTeleport && state.ninjaTeleportUsedUnitIds?.includes(pending.unitId)) {
        this.ninjaTeleportSuccess += 1;
        this.ninjaTeleportUsers.add(pending.unitId);
      }
      if (moved && pending.obstacleJump) {
        this.ninjaObstacleSuccess += 1;
        this.ninjaObstacleUsers.add(pending.unitId);
      }
    }

    for (const ninjaId of new Set([...this.beforePositions.keys(), ...state.units.filter((unit) => unit.type === "ninja").map((unit) => unit.id)])) {
      const before = this.beforePositions.get(ninjaId);
      const after = state.units.find((unit) => unit.id === ninjaId)?.position;
      if (!before || !after) continue;
      if (before.kind !== "water" && after.kind === "water") {
        this.ninjaSubmersionStarts += 1;
        this.activeSubmersions.set(ninjaId, { enteredTurn: state.turnNumber, lastTurn: state.turnNumber });
      } else if (before.kind === "water" && after.kind === "water") {
        if (before.x !== after.x || before.y !== after.y) this.ninjaUnderwaterMoves += 1;
      } else if (before.kind === "water" && after.kind !== "water") {
        const cycle = this.activeSubmersions.get(ninjaId);
        if (cycle) this.ninjaUnderwaterTurns += Math.max(0, state.turnNumber - cycle.lastTurn);
        this.activeSubmersions.delete(ninjaId);
        const forcedBridgeExposure = decision.kind === "resolve_strategists" && after.kind === "bridge";
        if (forcedBridgeExposure) {
          this.ninjaForcedExposures += 1;
        } else if (after.kind === "removed") {
          this.recordIncomplete("removed_while_submerged");
        } else {
          this.ninjaSubmersionCompletes += 1;
        }
      }
      const cycle = this.activeSubmersions.get(ninjaId);
      if (cycle && state.turnNumber > cycle.lastTurn) {
        this.ninjaUnderwaterTurns += state.turnNumber - cycle.lastTurn;
        cycle.lastTurn = state.turnNumber;
      }
    }
    this.pendingMovement = undefined;
  }

  private failProduction(choice: PendingProduction) {
    this.productionFailureCounts[choice.unitType] += 1;
  }

  private alignLogSnapshot(logs: GameState["logs"]) {
    if (logs === this.logArrayReference && logs.length === this.logSnapshot.length) return;
    const signatures = logs.map(logSignature);
    const matches = signatures.length === this.logSnapshot.length
      && signatures.every((signature, index) => signature === this.logSnapshot[index]);
    if (!matches) this.logSnapshot = signatures;
    this.logArrayReference = logs;
  }

  private consumeNewLogs(logs: GameState["logs"]) {
    let prefixMatches = logs.length >= this.logSnapshot.length;
    if (prefixMatches && this.logSnapshot.length) {
      const indices = new Set([0, Math.floor((this.logSnapshot.length - 1) / 2), this.logSnapshot.length - 1]);
      prefixMatches = [...indices].every((index) => logSignature(logs[index]) === this.logSnapshot[index]);
    }
    if (prefixMatches) {
      const added = logs.slice(this.logSnapshot.length);
      this.logSnapshot.push(...added.map(logSignature));
      this.logArrayReference = logs;
      return added;
    }

    const signatures = logs.map(logSignature);
    let commonPrefix = 0;
    while (commonPrefix < signatures.length
      && commonPrefix < this.logSnapshot.length
      && signatures[commonPrefix] === this.logSnapshot[commonPrefix]) commonPrefix += 1;
    const added = logs.slice(commonPrefix);
    this.logSnapshot = signatures;
    this.logArrayReference = logs;
    return added;
  }

  private recordIncomplete(reason: string) {
    this.ninjaSubmersionIncomplete += 1;
    this.incompleteReasons[reason] = (this.incompleteReasons[reason] ?? 0) + 1;
  }

  finalize(state: GameState): PpoAbilityAuditMatch {
    this.trackNinjas(state);
    for (const pending of this.pendingProductionByTeam.values()) this.failProduction(pending);
    this.pendingProductionByTeam.clear();
    for (const ninjaId of this.activeSubmersions.keys()) {
      this.recordIncomplete("match_ended_while_submerged");
      this.activeSubmersions.delete(ninjaId);
    }
    for (const kind of this.pendingConstructionKinds.values()) {
      if (kind === "bridge") this.bridgeFailures += 1;
      else this.obstacleFailures += 1;
    }
    this.pendingConstructionKinds.clear();
    this.bridgeFailures += Math.max(0, this.bridgeInstructions - this.bridgeSuccess - this.bridgeFailures);
    this.obstacleFailures += Math.max(0, this.obstacleInstructions - this.obstacleSuccess - this.obstacleFailures);
    this.strategistTeleportFailures += Math.max(0, this.strategistTeleportInstructions - this.strategistTeleportSuccess - this.strategistTeleportFailures);
    const specialUnitCount = this.productionCounts.engineer + this.productionCounts.ninja + this.productionCounts.strategist;
    const productionSuccessCount = Object.values(this.productionSuccessCounts).reduce((total, count) => total + count, 0);
    const productionFailureCount = Object.values(this.productionFailureCounts).reduce((total, count) => total + count, 0);
    return {
      schemaVersion: PPO_ABILITY_AUDIT_SCHEMA_VERSION,
      production: {
        selectionCount: this.productionSelectionCount,
        passCount: this.productionPassCount,
        counts: { ...this.productionCounts },
        shares: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, this.productionSelectionCount ? this.productionCounts[type] / this.productionSelectionCount : 0])) as Record<ProductionUnitType, number>,
        successCount: productionSuccessCount,
        failureCount: productionFailureCount,
        successCounts: { ...this.productionSuccessCounts },
        failureCounts: { ...this.productionFailureCounts },
        successRates: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, this.productionCounts[type] ? this.productionSuccessCounts[type] / this.productionCounts[type] : 0])) as Record<ProductionUnitType, number>,
        successfulShares: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, productionSuccessCount ? this.productionSuccessCounts[type] / productionSuccessCount : 0])) as Record<ProductionUnitType, number>,
        strategistRoleCounts: { ...this.strategistRoleCounts },
        specialUnitCount,
        specialUnitShare: this.productionSelectionCount ? specialUnitCount / this.productionSelectionCount : 0,
      },
      strategist: {
        bridge: { instructionCount: this.bridgeInstructions, successCount: this.bridgeSuccess, failureCount: this.bridgeFailures, placedCount: this.bridgePlaced, placedTileCount: this.bridgeTiles },
        obstacle: { instructionCount: this.obstacleInstructions, successCount: this.obstacleSuccess, failureCount: this.obstacleFailures, placedCount: this.obstaclePlaced },
        teleport: { opportunityCount: this.strategistTeleportOpportunities, instructionCount: this.strategistTeleportInstructions, successCount: this.strategistTeleportSuccess, failureCount: this.strategistTeleportFailures, teleportedUnitCount: this.strategistTeleportedUnits },
      },
      encouragement: {
        totalAttackCount: this.totalAttacks,
        appliedAttackCount: this.encouragedAttacks,
        hitCount: this.encouragedHits,
        missCount: this.encouragedMisses,
        withoutEncouragementCount: this.unencouragedAttacks,
        strategistAppliedAttackCount: this.strategistEncouragedAttacks,
        kingAppliedAttackCount: this.kingEncouragedAttacks,
        bothAppliedAttackCount: this.bothEncouragedAttacks,
        applicationRate: this.totalAttacks ? this.encouragedAttacks / this.totalAttacks : 0,
      },
      ninja: {
        appearedUnitCount: this.ninjaIds.size,
        appearedUnitIds: [...this.ninjaIds].sort(),
        submersionStartCount: this.ninjaSubmersionStarts,
        submersionCompleteCount: this.ninjaSubmersionCompletes,
        submersionIncompleteCount: this.ninjaSubmersionIncomplete,
        forcedExposureCount: this.ninjaForcedExposures,
        underwaterMoveCount: this.ninjaUnderwaterMoves,
        underwaterTurnCount: this.ninjaUnderwaterTurns,
        teleportOpportunityCount: this.ninjaTeleportOpportunities,
        teleportSuccessCount: this.ninjaTeleportSuccess,
        teleportUserCount: this.ninjaTeleportUsers.size,
        teleportUserIds: [...this.ninjaTeleportUsers].sort(),
        obstacleJumpOpportunityCount: this.ninjaObstacleOpportunities,
        obstacleJumpSuccessCount: this.ninjaObstacleSuccess,
        obstacleJumpUserCount: this.ninjaObstacleUsers.size,
        obstacleJumpUserIds: [...this.ninjaObstacleUsers].sort(),
        incompleteReasons: { ...this.incompleteReasons },
      },
    };
  }
}

export function aggregatePpoAbilityAudits(
  matches: Array<{ environmentIndex: number; seed: number; audit: PpoAbilityAuditMatch }>,
): PpoAbilityAuditAggregate {
  const counts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  const successCounts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  const failureCounts = emptyRecord(AUDITED_PRODUCTION_UNIT_TYPES);
  const roles = emptyRecord(STRATEGIST_ROLES);
  const sum = (select: (audit: PpoAbilityAuditMatch) => number) => matches.reduce((total, match) => total + select(match.audit), 0);
  for (const { audit } of matches) {
    for (const type of AUDITED_PRODUCTION_UNIT_TYPES) {
      counts[type] += audit.production.counts[type];
      successCounts[type] += audit.production.successCounts[type];
      failureCounts[type] += audit.production.failureCounts[type];
    }
    for (const role of STRATEGIST_ROLES) roles[role] += audit.production.strategistRoleCounts[role];
  }
  const selectionCount = sum((audit) => audit.production.selectionCount);
  const specialUnitCount = counts.engineer + counts.ninja + counts.strategist;
  const productionSuccessCount = Object.values(successCounts).reduce((total, count) => total + count, 0);
  const productionFailureCount = Object.values(failureCounts).reduce((total, count) => total + count, 0);
  const ninjaIds = new Set(matches.flatMap((match) => match.audit.ninja.appearedUnitIds.map((id) => `${match.environmentIndex}:${id}`)));
  const teleportIds = new Set(matches.flatMap((match) => match.audit.ninja.teleportUserIds.map((id) => `${match.environmentIndex}:${id}`)));
  const obstacleIds = new Set(matches.flatMap((match) => match.audit.ninja.obstacleJumpUserIds.map((id) => `${match.environmentIndex}:${id}`)));
  const incompleteReasons: Record<string, number> = {};
  for (const { audit } of matches) for (const [reason, count] of Object.entries(audit.ninja.incompleteReasons)) incompleteReasons[reason] = (incompleteReasons[reason] ?? 0) + count;
  const totalAttackCount = sum((audit) => audit.encouragement.totalAttackCount);
  const appliedAttackCount = sum((audit) => audit.encouragement.appliedAttackCount);
  const aggregate: PpoAbilityAuditAggregate = {
    schemaVersion: PPO_ABILITY_AUDIT_SCHEMA_VERSION,
    matchCount: matches.length,
    production: {
      selectionCount,
      passCount: sum((audit) => audit.production.passCount),
      counts,
      shares: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, selectionCount ? counts[type] / selectionCount : 0])) as Record<ProductionUnitType, number>,
      successCount: productionSuccessCount,
      failureCount: productionFailureCount,
      successCounts,
      failureCounts,
      successRates: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, counts[type] ? successCounts[type] / counts[type] : 0])) as Record<ProductionUnitType, number>,
      successfulShares: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, productionSuccessCount ? successCounts[type] / productionSuccessCount : 0])) as Record<ProductionUnitType, number>,
      averagePerMatch: Object.fromEntries(AUDITED_PRODUCTION_UNIT_TYPES.map((type) => [type, matches.length ? counts[type] / matches.length : 0])) as Record<ProductionUnitType, number>,
      strategistRoleCounts: roles,
      specialUnitCount,
      specialUnitShare: selectionCount ? specialUnitCount / selectionCount : 0,
    },
    strategist: {
      bridge: { instructionCount: sum((audit) => audit.strategist.bridge.instructionCount), successCount: sum((audit) => audit.strategist.bridge.successCount), failureCount: sum((audit) => audit.strategist.bridge.failureCount), placedCount: sum((audit) => audit.strategist.bridge.placedCount), placedTileCount: sum((audit) => audit.strategist.bridge.placedTileCount) },
      obstacle: { instructionCount: sum((audit) => audit.strategist.obstacle.instructionCount), successCount: sum((audit) => audit.strategist.obstacle.successCount), failureCount: sum((audit) => audit.strategist.obstacle.failureCount), placedCount: sum((audit) => audit.strategist.obstacle.placedCount) },
      teleport: { opportunityCount: sum((audit) => audit.strategist.teleport.opportunityCount), instructionCount: sum((audit) => audit.strategist.teleport.instructionCount), successCount: sum((audit) => audit.strategist.teleport.successCount), failureCount: sum((audit) => audit.strategist.teleport.failureCount), teleportedUnitCount: sum((audit) => audit.strategist.teleport.teleportedUnitCount) },
    },
    encouragement: {
      totalAttackCount,
      appliedAttackCount,
      hitCount: sum((audit) => audit.encouragement.hitCount),
      missCount: sum((audit) => audit.encouragement.missCount),
      withoutEncouragementCount: sum((audit) => audit.encouragement.withoutEncouragementCount),
      strategistAppliedAttackCount: sum((audit) => audit.encouragement.strategistAppliedAttackCount),
      kingAppliedAttackCount: sum((audit) => audit.encouragement.kingAppliedAttackCount),
      bothAppliedAttackCount: sum((audit) => audit.encouragement.bothAppliedAttackCount),
      applicationRate: totalAttackCount ? appliedAttackCount / totalAttackCount : 0,
    },
    ninja: {
      appearedUnitCount: ninjaIds.size,
      appearedUnitIds: [...ninjaIds].sort(),
      submersionStartCount: sum((audit) => audit.ninja.submersionStartCount),
      submersionCompleteCount: sum((audit) => audit.ninja.submersionCompleteCount),
      submersionIncompleteCount: sum((audit) => audit.ninja.submersionIncompleteCount),
      forcedExposureCount: sum((audit) => audit.ninja.forcedExposureCount),
      underwaterMoveCount: sum((audit) => audit.ninja.underwaterMoveCount),
      underwaterTurnCount: sum((audit) => audit.ninja.underwaterTurnCount),
      teleportOpportunityCount: sum((audit) => audit.ninja.teleportOpportunityCount),
      teleportSuccessCount: sum((audit) => audit.ninja.teleportSuccessCount),
      teleportUserCount: teleportIds.size,
      teleportUserIds: [...teleportIds].sort(),
      obstacleJumpOpportunityCount: sum((audit) => audit.ninja.obstacleJumpOpportunityCount),
      obstacleJumpSuccessCount: sum((audit) => audit.ninja.obstacleJumpSuccessCount),
      obstacleJumpUserCount: obstacleIds.size,
      obstacleJumpUserIds: [...obstacleIds].sort(),
      incompleteReasons,
    },
    perMatch: matches,
  };
  return aggregate;
}

export function formatPpoAbilityAuditMarkdown(audit: PpoAbilityAuditAggregate) {
  const lines = [
    `# PPO Ability Audit (schema v${audit.schemaVersion})`,
    "",
    `Matches: ${audit.matchCount}`,
    "",
    "## Production",
    "",
    "| Unit | Selected | Selection share | Successful | Failed | Success rate | Successful share | Per match |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    ...AUDITED_PRODUCTION_UNIT_TYPES.map((type) => `| ${type} | ${audit.production.counts[type]} | ${(audit.production.shares[type] * 100).toFixed(2)}% | ${audit.production.successCounts[type]} | ${audit.production.failureCounts[type]} | ${(audit.production.successRates[type] * 100).toFixed(2)}% | ${(audit.production.successfulShares[type] * 100).toFixed(2)}% | ${audit.production.averagePerMatch[type].toFixed(3)} |`),
    "",
    `Selections / successful / failed: ${audit.production.selectionCount} / ${audit.production.successCount} / ${audit.production.failureCount}`,
    `New three unit share: ${(audit.production.specialUnitShare * 100).toFixed(2)}%`,
    "",
    "## Strategist",
    "",
    `- Bridge: ${audit.strategist.bridge.successCount}/${audit.strategist.bridge.instructionCount} successful (${audit.strategist.bridge.placedCount} bridges, ${audit.strategist.bridge.placedTileCount} tiles)`,
    `- Obstacle: ${audit.strategist.obstacle.successCount}/${audit.strategist.obstacle.instructionCount} successful`,
    `- Teleport: ${audit.strategist.teleport.successCount}/${audit.strategist.teleport.instructionCount} successful; ${audit.strategist.teleport.teleportedUnitCount} units`,
    "",
    "## Encouragement",
    "",
    `- Applied attacks: ${audit.encouragement.appliedAttackCount} (hit ${audit.encouragement.hitCount}, miss ${audit.encouragement.missCount})`,
    `- Strategist / king / overlap: ${audit.encouragement.strategistAppliedAttackCount} / ${audit.encouragement.kingAppliedAttackCount} / ${audit.encouragement.bothAppliedAttackCount}`,
    `- Application rate: ${(audit.encouragement.applicationRate * 100).toFixed(2)}%`,
    "",
    "## Ninja",
    "",
    `- Submersion start / complete / incomplete / forced exposure: ${audit.ninja.submersionStartCount} / ${audit.ninja.submersionCompleteCount} / ${audit.ninja.submersionIncompleteCount} / ${audit.ninja.forcedExposureCount}`,
    `- Ninja teleport: ${audit.ninja.teleportSuccessCount} successes from ${audit.ninja.teleportOpportunityCount} opportunities`,
    `- Obstacle jump: ${audit.ninja.obstacleJumpSuccessCount} successes from ${audit.ninja.obstacleJumpOpportunityCount} opportunities`,
  ];
  return lines.join("\n") + "\n";
}
