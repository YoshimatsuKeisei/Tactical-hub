import type { GameState, Unit } from "./types";

export function isWaterNinja(unit: Unit | undefined): unit is Unit {
  return Boolean(unit?.type === "ninja" && unit.position.kind === "water" && unit.hp > 0);
}

export function isUnitVisibleToTeam(state: GameState, unit: Unit, viewerTeamId: string) {
  if (!isWaterNinja(unit) || unit.ownerTeamId === viewerTeamId) return true;
  return Boolean(state.ninjaRevealStates?.some(
    (reveal) => reveal.ninjaUnitId === unit.id && reveal.visibleToTeamIds.includes(viewerTeamId),
  ));
}

export function createTeamVisibleState(state: GameState, viewerTeamId: string): GameState {
  const visibleUnits = state.units.filter((unit) => isUnitVisibleToTeam(state, unit, viewerTeamId));
  const visibleIds = new Set(visibleUnits.map((unit) => unit.id));
  return {
    ...state,
    units: visibleUnits,
    unitTurnFlags: state.unitTurnFlags.filter((flag) => visibleIds.has(flag.unitId)),
    movedUnitIdsThisMovementPhase: state.movedUnitIdsThisMovementPhase.filter((unitId) => visibleIds.has(unitId)),
    ninjaRevealStates: state.ninjaRevealStates?.filter((reveal) => visibleIds.has(reveal.ninjaUnitId)),
  };
}

export function revealNinjasToEachOther(state: GameState, first: Unit, second: Unit) {
  const additions = [
    { ninjaUnitId: first.id, teamId: second.ownerTeamId },
    { ninjaUnitId: second.id, teamId: first.ownerTeamId },
  ];
  const reveals = [...(state.ninjaRevealStates ?? [])];
  for (const addition of additions) {
    const existing = reveals.find((reveal) => reveal.ninjaUnitId === addition.ninjaUnitId);
    if (existing) existing.visibleToTeamIds = [...new Set([...existing.visibleToTeamIds, addition.teamId])].sort();
    else reveals.push({ ninjaUnitId: addition.ninjaUnitId, visibleToTeamIds: [addition.teamId] });
  }
  state.ninjaRevealStates = reveals.sort((left, right) => left.ninjaUnitId.localeCompare(right.ninjaUnitId));
}
