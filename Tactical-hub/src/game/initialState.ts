import { UNIT_STATS } from "./constants";
import { testMap4p } from "./maps/testMap4p";
import type { Base, BaseSlot, GameState, Team, Unit, UnitType } from "./types";

const teams: Team[] = [
  { id: "team-1", name: "Team 1", color: "#d94a4a", status: "active", homeBaseId: "home-1", controlledBaseIds: ["home-1"], defeatedUnitCount: 0, conqueredTeamIds: [] },
  { id: "team-2", name: "Team 2", color: "#3e7bd8", status: "active", homeBaseId: "home-2", controlledBaseIds: ["home-2"], defeatedUnitCount: 0, conqueredTeamIds: [] },
  { id: "team-3", name: "Team 3", color: "#36a166", status: "active", homeBaseId: "home-3", controlledBaseIds: ["home-3"], defeatedUnitCount: 0, conqueredTeamIds: [] },
  { id: "team-4", name: "Team 4", color: "#c58a2b", status: "active", homeBaseId: "home-4", controlledBaseIds: ["home-4"], defeatedUnitCount: 0, conqueredTeamIds: [] },
  { id: "neutral", name: "Neutral Guard", color: "#767b86", status: "neutral", controlledBaseIds: [], isNeutral: true },
];

function placeUnit(base: Base, ownerTeamId: string, type: UnitType, slotId: string): Unit {
  const unit: Unit = {
    id: `${base.id}-${type}`,
    ownerTeamId,
    type,
    hp: UNIT_STATS[type].hp,
    position: { kind: "base", baseId: base.id, slotId },
    statuses: [],
  };
  if (type === "strategist") unit.role = "encourage";
  return unit;
}

const homeStrategistSlots: Record<string, { localRow: 0 | 1; localCol: 0 | 1 }> = {
  "home-1": { localRow: 1, localCol: 1 },
  "home-2": { localRow: 1, localCol: 0 },
  "home-3": { localRow: 0, localCol: 0 },
  "home-4": { localRow: 0, localCol: 1 },
};

function findSlot(base: Base, local: { localRow: 0 | 1; localCol: 0 | 1 }): BaseSlot {
  const slot = base.slots.find((candidate) => candidate.localRow === local.localRow && candidate.localCol === local.localCol);
  if (!slot) throw new Error(`Missing slot ${base.id} ${local.localRow}/${local.localCol}`);
  return slot;
}

export function createInitialGameState(): GameState {
  const bases = structuredClone(testMap4p.bases);
  const units: Unit[] = [];

  for (const base of bases) {
    if (base.type === "home") {
      const kingSlot = base.slots.find((candidate) => candidate.id === base.protectedSlotId) ?? base.slots[0];
      const strategistSlot = findSlot(base, homeStrategistSlots[base.id]);
      [
        ["king", kingSlot] as const,
        ["strategist", strategistSlot] as const,
      ].forEach(([unitType, slot]) => {
        const unit = placeUnit(base, base.ownerTeamId, unitType, slot.id);
        slot.unitId = unit.id;
        units.push(unit);
      });
      continue;
    }

    const unitTypes: UnitType[] = ["infantry", "cavalry", "archer"];
    unitTypes.forEach((unitType, index) => {
      const slot = base.slots[index];
      const unit = placeUnit(base, base.ownerTeamId, unitType, slot.id);
      slot.unitId = unit.id;
      units.push(unit);
    });
  }

  return {
    config: { playerCount: 4, productionInterval: 5, mapId: testMap4p.id },
    map: testMap4p,
    turnNumber: 1,
    phase: "movement_input",
    teams: structuredClone(teams),
    units,
    bases,
    unitTurnFlags: [],
    turnState: { turnNumber: 1, phase: "movement_input", actionIntents: [] },
    logs: [{ id: "log-setup", turnNumber: 1, type: "setup", message: "Phase 1 initial state created." }],
    siegeStates: [],
    rewardPlacementRequests: [],
    kingCampaignStates: [],
    constructions: [],
    strategistActionIntents: [],
    strategistSubmittedTeamIds: [],
    strategistCooldowns: [],
    movementSeatOrderTeamIds: ["team-1", "team-2", "team-3", "team-4"],
    movementOrderStartIndex: 0,
    movementOrderTeamIds: ["team-1", "team-2", "team-3", "team-4"],
    currentMovementTeamId: "team-1",
    movementCompletedTeamIds: [],
    productionCompletedTeamIdsThisTurn: [],
    teleportIntents: [],
    teleportCooldowns: [],
    movedUnitIdsThisMovementPhase: [],
  };
}
