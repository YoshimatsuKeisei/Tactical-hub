import { PRODUCIBLE_UNIT_TYPES, UNIT_STATS } from "../constants";
import type { GameState, ProductionChoice, StrategistRole, Unit, UnitType } from "../types";
import { beginMovementPhase } from "./movement";
import { isTeamProductionPending } from "./productionSchedule";

export const STRATEGIST_ROLES: StrategistRole[] = ["builder", "encourage", "teleporter"];

function nextUnitId(state: GameState, baseId: string, unitType: UnitType) {
  const count = state.units.filter((unit) => unit.id.startsWith(`${baseId}-${unitType}`)).length;
  return `${baseId}-${unitType}-${count + 1}`;
}

export function getAvailableProductionTypes(state: GameState, teamId: string, baseId: string): UnitType[] {
  const base = state.bases.find((candidate) => candidate.id === baseId);
  if (!base || base.ownerTeamId !== teamId || !base.slots.some((slot) => !slot.unitId)) return [];

  return PRODUCIBLE_UNIT_TYPES.filter((unitType) => {
    if (unitType === "ninja") {
      return state.units.filter((unit) => unit.ownerTeamId === teamId && unit.type === "ninja" && unit.hp > 0 && unit.position.kind !== "removed").length < 2;
    }
    if (unitType === "archer") {
      const defeatedTeams = state.teams.filter((team) => !team.isNeutral && team.status !== "active").length;
      const limit = 3 + defeatedTeams;
      return state.units.filter((unit) => unit.ownerTeamId === teamId && unit.type === "archer" && unit.hp > 0 && unit.position.kind !== "removed").length < limit;
    }
    if (unitType === "strategist") {
      return state.units.filter(
        (unit) => unit.ownerTeamId === teamId && unit.type === "strategist" && unit.hp > 0 && unit.position.kind !== "removed",
      ).length < 2;
    }
    return true;
  });
}

export function getProductionCandidates(state: GameState, teamId: string): ProductionChoice[] {
  const canProduce = state.phase === "production" || isTeamProductionPending(state, teamId);
  if (!canProduce || state.teams.find((team) => team.id === teamId)?.status !== "active") return [];
  return state.bases
    .filter((base) => base.ownerTeamId === teamId)
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((base) => getProductionCandidatesForBase(state, teamId, base.id));
}

export function getProductionCandidatesForBase(state: GameState, teamId: string, baseId: string): ProductionChoice[] {
  const canProduce = state.phase === "production" || isTeamProductionPending(state, teamId);
  if (!canProduce || state.teams.find((team) => team.id === teamId)?.status !== "active" || state.bases.find((base) => base.id === baseId)?.ownerTeamId !== teamId) return [];
  return getAvailableProductionTypes(state, teamId, baseId).flatMap((unitType): ProductionChoice[] =>
    unitType === "strategist"
      ? STRATEGIST_ROLES.map((strategistRole) => ({ teamId, baseId, unitType, strategistRole }))
      : [{ teamId, baseId, unitType }],
  );
}

export function saveProductionChoice(state: GameState, choice: ProductionChoice): GameState {
  const actionIntents = upsertProductionChoice(state, choice);
  return { ...state, turnState: { ...state.turnState, actionIntents } };
}

function upsertProductionChoice(state: GameState, choice: ProductionChoice) {
  const existing = state.turnState.actionIntents.find((intent) => intent.teamId === choice.teamId);
  if (!existing) {
    return [...state.turnState.actionIntents, { teamId: choice.teamId, productionChoices: [choice], movementIntents: [], attackIntents: [] }];
  }
  return state.turnState.actionIntents.map((intent) =>
    intent.teamId === choice.teamId
      ? {
          ...intent,
          productionChoices: [
            ...intent.productionChoices.filter((candidate) => candidate.baseId !== choice.baseId),
            choice,
          ],
        }
      : intent,
  );
}

function applyProductionChoices(next: GameState, choices: ProductionChoice[]) {
  for (const choice of choices) {
    const base = next.bases.find((candidate) => candidate.id === choice.baseId);
    const legalTypes = getAvailableProductionTypes(next, choice.teamId, choice.baseId);
    const slot = base?.slots.find((candidate) => !candidate.unitId);

    if (!base || !slot || !legalTypes.includes(choice.unitType)) {
      next.logs.push({
        id: `log-production-failed-${next.logs.length}`,
        turnNumber: next.turnNumber,
        type: "production",
        message: `Production failed: ${choice.teamId} ${choice.unitType} at ${choice.baseId}.`,
      });
      continue;
    }

    const unit: Unit = {
      id: nextUnitId(next, base.id, choice.unitType),
      ownerTeamId: choice.teamId,
      type: choice.unitType,
      hp: UNIT_STATS[choice.unitType].hp,
      position: { kind: "base", baseId: base.id, slotId: slot.id },
      statuses: [],
    };
    if (choice.unitType === "strategist") unit.role = choice.strategistRole ?? "encourage";
    slot.unitId = unit.id;
    next.units.push(unit);
    next.logs.push({
      id: `log-production-${unit.id}`,
      turnNumber: next.turnNumber,
      type: "production",
      message: `${choice.teamId} produced ${choice.unitType} in ${base.id}/${slot.id}.`,
      relatedIds: [unit.id, base.id],
    });
  }
}

export function submitTeamProduction(state: GameState, teamId: string): GameState {
  if (!isTeamProductionPending(state, teamId)) return state;
  const next = structuredClone(state) as GameState;
  const choices = next.turnState.actionIntents
    .filter((intent) => intent.teamId === teamId)
    .flatMap((intent) => intent.productionChoices);
  applyProductionChoices(next, choices);
  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) =>
    intent.teamId === teamId ? { ...intent, productionChoices: [] } : intent,
  );
  next.productionCompletedTeamIdsThisTurn = [...new Set([...next.productionCompletedTeamIdsThisTurn, teamId])];
  return next;
}

export function resolveProduction(state: GameState): GameState {
  const next = structuredClone(state) as GameState;
  const choices = next.turnState.actionIntents.flatMap((intent) => intent.productionChoices);
  applyProductionChoices(next, choices);

  next.turnState.actionIntents = next.turnState.actionIntents.map((intent) => ({ ...intent, productionChoices: [] }));
  return state.phase === "production" ? beginMovementPhase(next) : next;
}
