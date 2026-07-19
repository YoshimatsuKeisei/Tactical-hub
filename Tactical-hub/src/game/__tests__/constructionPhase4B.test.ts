import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import {
  assignConstructionCapacityBonus,
  assignConstructionManager,
  clearDeadConstructionManagers,
  getConstructionManagementLimit,
  resolveStrategistActions,
  saveStrategistActionIntent,
} from "../engine/construction";
import { resolveKingDefeats } from "../engine/defeat";
import { createInitialGameState } from "../initialState";
import type { Construction, GameState, KingCampaignState, Unit, UnitPosition } from "../types";

function builder(state: GameState, id = "home-1-strategist") {
  const unit = state.units.find((candidate) => candidate.id === id)!;
  unit.role = "builder";
  return unit;
}

function clearSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function move(state: GameState, unit: Unit, position: UnitPosition) {
  clearSlot(state, unit.position);
  unit.position = position;
}

function addUnit(state: GameState, id: string, teamId: string, type: Unit["type"], position: UnitPosition) {
  const unit: Unit = { id, ownerTeamId: teamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit);
  return unit;
}

function bridge(state: GameState, managerUnitId = "home-1-strategist", id = "flood-bridge", tiles = [{ x: 4, y: 2 }, { x: 4, y: 3 }]) {
  const construction: Construction = { id, kind: "bridge", ownerTeamId: "team-1", managerUnitId, tiles, placedTurn: 1, active: true };
  state.constructions.push(construction);
  return construction;
}

function resolveReset(state: GameState, constructionId = "flood-bridge", strategistUnitId = "home-1-strategist") {
  state.phase = state.turnState.phase = "strategist_action_resolution";
  state.strategistActionIntents = [{ teamId: "team-1", strategistUnitId, action: "reset_bridge", constructionId }];
  return resolveStrategistActions(state, () => 0);
}

function blockAllRoads(state: GameState, teamId: string) {
  for (const tile of state.map.tiles.filter((candidate) => candidate.terrain === "road"))
    state.constructions.push({ id: `block-${teamId}-${tile.x}-${tile.y}`, kind: "obstacle", ownerTeamId: "team-1", tiles: [{ x: tile.x, y: tile.y }], placedTurn: 1, active: true });
}

describe("Phase 4-B bridge reset and construction inheritance", () => {
  it("blocks reset when the managing builder or resetting team's king is on the bridge", () => {
    for (const blockedUnitId of ["home-1-strategist", "home-1-king"]) {
      const state = createInitialGameState();
      const manager = builder(state);
      const construction = bridge(state);
      move(state, state.units.find((unit) => unit.id === blockedUnitId)!, { kind: "bridge", bridgeId: construction.id, cellIndex: 0 });
      const saved = saveStrategistActionIntent(state, { teamId: "team-1", strategistUnitId: manager.id, action: "reset_bridge", constructionId: construction.id });
      expect(saved.strategistActionIntents).toEqual([]);
    }
  });

  it("allows other bridge occupants and resolves ninja, enemy, and friendly units correctly", () => {
    const state = createInitialGameState();
    builder(state); bridge(state);
    const ninja = addUnit(state, "enemy-ninja", "team-2", "ninja", { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 0 });
    const enemy = addUnit(state, "enemy-infantry", "team-2", "infantry", { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 1 });
    const friendly = addUnit(state, "friendly-infantry", "team-1", "infantry", { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 1 });
    const resolved = resolveReset(state);
    expect(resolved.units.find((unit) => unit.id === ninja.id)?.position).toEqual({ kind: "water", x: 4, y: 2 });
    expect(resolved.units.find((unit) => unit.id === enemy.id)?.position.kind).toBe("removed");
    expect(resolved.units.find((unit) => unit.id === friendly.id)?.position.kind).toBe("removed");
    expect(resolved.teams.find((team) => team.id === "team-1")?.defeatedUnitCount).toBe(1);
  });

  it("damages a king and retreats it to the nearest open road anywhere on the board", () => {
    const state = createInitialGameState(); builder(state); bridge(state);
    const king = state.units.find((unit) => unit.id === "home-2-king")!;
    move(state, king, { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 0 });
    const resolved = resolveReset(state);
    const nextKing = resolved.units.find((unit) => unit.id === king.id)!;
    expect(nextKing.hp).toBe(2);
    expect(nextKing.position.kind).toBe("tile");
    expect(resolved.kingCampaignStates.find((campaign) => campaign.kingUnitId === king.id)?.contributions).toContainEqual({ teamId: "team-1", cumulativeDamage: 1, effectiveAttackTurns: 0 });
  });

  it("falls back to the home base when tied nearest bases have slots and roads are unavailable", () => {
    const state = createInitialGameState(); builder(state);
    bridge(state, "home-1-strategist", "flood-bridge", [{ x: 15, y: 3 }]);
    const king = state.units.find((unit) => unit.id === "home-2-king")!;
    move(state, king, { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 0 });
    const relay = state.bases.find((base) => base.id === "neutral-north")!;
    relay.ownerTeamId = "team-2";
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push(relay.id);
    blockAllRoads(state, "team-2");
    const resolved = resolveReset(state);
    expect(resolved.units.find((unit) => unit.id === king.id)?.position).toMatchObject({ kind: "base", baseId: "home-2" });
  });

  it("kills a surviving king when neither road nor owned base slot is available", () => {
    const state = createInitialGameState(); builder(state); bridge(state);
    const king = state.units.find((unit) => unit.id === "home-2-king")!;
    move(state, king, { kind: "bridge", bridgeId: "flood-bridge", cellIndex: 0 });
    blockAllRoads(state, "team-2");
    const home = state.bases.find((base) => base.id === "home-2")!;
    for (const slot of home.slots.filter((entry) => !entry.unitId)) {
      const occupant = addUnit(state, `fill-${slot.id}`, "team-2", "infantry", { kind: "base", baseId: home.id, slotId: slot.id });
      slot.unitId = occupant.id;
    }
    const resolved = resolveReset(state);
    expect(resolved.units.find((unit) => unit.id === king.id)?.position.kind).toBe("removed");
    expect(resolved.teams.find((team) => team.id === "team-2")?.status).toBe("defeated");
  });

  it("force-resets bridge obstacles and applies their manager cooldown", () => {
    const state = createInitialGameState(); builder(state); bridge(state); state.turnNumber = 8;
    state.constructions.push({ id: "bridge-obstacle", kind: "obstacle", ownerTeamId: "team-2", managerUnitId: "home-2-strategist", tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    const resolved = resolveReset(state);
    expect(resolved.constructions.find((entry) => entry.id === "bridge-obstacle")?.active).toBe(false);
    expect(resolved.strategistCooldowns).toContainEqual({ strategistUnitId: "home-2-strategist", kind: "obstacle", availableFromTurn: 13 });
  });

  it("orphans a dead builder's equipment and allows a successor to inherit within capacity", () => {
    const state = createInitialGameState(); const dead = builder(state); const construction = bridge(state);
    dead.hp = 0; dead.position = { kind: "removed", reason: "defeated" };
    clearDeadConstructionManagers(state);
    expect(construction.active).toBe(true);
    expect(state.constructions[0].managerUnitId).toBeUndefined();
    const successor = addUnit(state, "successor", "team-1", "strategist", { kind: "tile", x: 5, y: 1 }); successor.role = "builder";
    const inherited = assignConstructionManager(state, construction.id, successor.id);
    expect(inherited.constructions[0].managerUnitId).toBe(successor.id);
  });

  it("transfers equipment on conquest and leaves it ownerless on simultaneous neutralization", () => {
    const conquered = createInitialGameState(); bridge(conquered, "home-2-strategist"); conquered.constructions[0].ownerTeamId = "team-2";
    const campaign: KingCampaignState = { kingUnitId: "home-2-king", kingTeamId: "team-2", contributions: [{ teamId: "team-1", cumulativeDamage: 1, effectiveAttackTurns: 1 }] };
    resolveKingDefeats(conquered, [{ kingUnitId: "home-2-king", kingTeamId: "team-2", candidateTeamIds: ["team-1"], campaign }], [], () => 0);
    expect(conquered.constructions[0]).toMatchObject({ ownerTeamId: "team-1", managerUnitId: undefined, active: true });
    expect(conquered.teams.find((team) => team.id === "team-1")?.conqueredTeamIds).toContain("team-2");

    const neutralized = createInitialGameState(); bridge(neutralized, "home-2-strategist"); neutralized.constructions[0].ownerTeamId = "team-2";
    const second: KingCampaignState = { kingUnitId: "home-3-king", kingTeamId: "team-3", contributions: [] };
    resolveKingDefeats(neutralized, [
      { kingUnitId: "home-2-king", kingTeamId: "team-2", candidateTeamIds: ["team-1"], campaign },
      { kingUnitId: "home-3-king", kingTeamId: "team-3", candidateTeamIds: ["team-1"], campaign: second },
    ], [], () => 0);
    expect(neutralized.constructions[0]).toMatchObject({ managerUnitId: undefined, active: true });
    expect(neutralized.constructions[0].ownerTeamId).toBeUndefined();
    expect(neutralized.teams.find((team) => team.id === "team-1")?.conqueredTeamIds ?? []).toHaveLength(0);
  });

  it("uses conquest counts 0, 1, and 2 for selectable per-builder limits", () => {
    const state = createInitialGameState(); const first = builder(state); const second = addUnit(state, "second-builder", "team-1", "strategist", { kind: "tile", x: 5, y: 1 }); second.role = "builder";
    expect(getConstructionManagementLimit(state, first.id, "bridge")).toBe(1);
    state.teams.find((team) => team.id === "team-1")!.conqueredTeamIds = ["team-2"];
    const assigned = assignConstructionCapacityBonus(state, "team-1", second.id);
    expect(getConstructionManagementLimit(assigned, first.id, "bridge")).toBe(1);
    expect(getConstructionManagementLimit(assigned, second.id, "bridge")).toBe(2);
    assigned.teams.find((team) => team.id === "team-1")!.conqueredTeamIds = ["team-2", "team-3"];
    expect(getConstructionManagementLimit(assigned, first.id, "obstacle")).toBe(2);
    expect(getConstructionManagementLimit(assigned, second.id, "obstacle")).toBe(2);
  });

  it("processes a unit only once when multiple overlapping bridges reset simultaneously", () => {
    const state = createInitialGameState(); builder(state); const second = addUnit(state, "second-builder", "team-1", "strategist", { kind: "tile", x: 5, y: 1 }); second.role = "builder";
    bridge(state, "home-1-strategist", "bridge-a", [{ x: 4, y: 2 }]);
    bridge(state, second.id, "bridge-b", [{ x: 4, y: 2 }]);
    addUnit(state, "single-victim", "team-2", "infantry", { kind: "bridge", bridgeId: "bridge-a", cellIndex: 0 });
    state.phase = state.turnState.phase = "strategist_action_resolution";
    state.strategistActionIntents = [
      { teamId: "team-1", strategistUnitId: "home-1-strategist", action: "reset_bridge", constructionId: "bridge-a" },
      { teamId: "team-1", strategistUnitId: second.id, action: "reset_bridge", constructionId: "bridge-b" },
    ];
    const resolved = resolveStrategistActions(state, () => 0);
    expect(resolved.teams.find((team) => team.id === "team-1")?.defeatedUnitCount).toBe(1);
  });
});
