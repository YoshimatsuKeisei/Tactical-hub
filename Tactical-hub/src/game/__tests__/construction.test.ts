import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getAttackCandidates } from "../engine/battle";
import {
  getBridgeCandidates,
  getObstacleCandidates,
  getOperationalAreaTiles,
  getOperationalRoadTiles,
  getOperationalRoadSectionIds,
  getOwnStrategistPreview,
  resolveStrategistActions,
  saveStrategistActionIntent,
  submitStrategistActions,
} from "../engine/construction";
import { getMovementCandidates } from "../engine/movement";
import { getAvailableProductionTypes } from "../engine/production";
import { saveProductionChoice } from "../engine/production";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";

function addBuilder(state: GameState, id: string, teamId = "team-1") {
  const unit: Unit = { id, ownerTeamId: teamId, type: "strategist", role: "builder", hp: UNIT_STATS.strategist.hp, position: { kind: "tile", x: teamId === "team-1" ? 4 : 17, y: 1 }, statuses: [] };
  state.units.push(unit); return unit;
}
function addUnit(state: GameState, id: string, teamId: string, position: UnitPosition, type: Unit["type"] = "cavalry") {
  const unit: Unit = { id, ownerTeamId: teamId, type, hp: UNIT_STATS[type].hp, position, statuses: [] }; state.units.push(unit); return unit;
}

describe("Phase 4-A construction", () => {
  function candidateCountState() {
    const state = createInitialGameState();
    const builder = state.units.find((unit) => unit.id === "home-1-strategist")!;
    builder.role = "builder";
    state.units = [builder];
    return { state, builder };
  }

  it("collects all three Player 1 home road sections and the exact 27 obstacle / 28 bridge candidates", () => {
    const { state, builder } = candidateCountState();
    expect([...getOperationalRoadSectionIds(state, "team-1")].sort()).toEqual([
      "road-home-1-neutral-center",
      "road-home-1-neutral-north",
      "road-home-1-neutral-west",
    ]);
    const obstacles = getObstacleCandidates(state, builder.id);
    const bridges = getBridgeCandidates(state, builder.id);
    expect(obstacles).toHaveLength(27);
    expect(bridges).toHaveLength(28);
    expect(new Set(obstacles.map((cell) => `${cell.x},${cell.y}`)).size).toBe(27);
    expect(new Set(bridges.map((cells) => cells.map((cell) => `${cell.x},${cell.y}`).join("|"))).size).toBe(28);
  });

  it("accepts production and strategist intents for every active team", () => {
    let state = createInitialGameState();
    for (const team of state.teams.filter((entry) => entry.status === "active")) {
      state = saveProductionChoice(state, { teamId: team.id, baseId: team.homeBaseId!, unitType: "infantry" });
      state.units.find((unit) => unit.id === `${team.homeBaseId}-strategist`)!.role = "builder";
      state = saveStrategistActionIntent(state, { teamId: team.id, strategistUnitId: `${team.homeBaseId}-strategist`, action: "pass" });
    }
    expect(new Set(state.turnState.actionIntents.flatMap((intent) => intent.productionChoices).map((choice) => choice.teamId))).toEqual(new Set(["team-1", "team-2", "team-3", "team-4"]));
    expect(new Set(state.strategistActionIntents.map((intent) => intent.teamId))).toEqual(new Set(["team-1", "team-2", "team-3", "team-4"]));
  });

  it("keeps two builders' intents separate and scopes preview data to the selected team", () => {
    let state = createInitialGameState();
    const first = addBuilder(state, "team-1-builder-a");
    const second = addBuilder(state, "team-1-builder-b");
    const enemy = addBuilder(state, "team-2-builder", "team-2");
    state = saveStrategistActionIntent(state, { teamId: "team-1", strategistUnitId: first.id, action: "pass" });
    const obstacle = getObstacleCandidates(state, second.id)[0];
    state = saveStrategistActionIntent(state, { teamId: "team-1", strategistUnitId: second.id, action: "place_obstacle", tiles: [obstacle] });
    const enemyObstacle = getObstacleCandidates(state, enemy.id)[0];
    state = saveStrategistActionIntent(state, { teamId: "team-2", strategistUnitId: enemy.id, action: "place_obstacle", tiles: [enemyObstacle] });
    expect(state.strategistActionIntents.filter((intent) => intent.teamId === "team-1")).toHaveLength(2);
    expect(getOwnStrategistPreview(state, "team-1").map((cell) => `${cell.x},${cell.y}`)).toEqual([`${obstacle.x},${obstacle.y}`]);
    expect(getOwnStrategistPreview(state, "team-2").map((cell) => `${cell.x},${cell.y}`)).toEqual([`${enemyObstacle.x},${enemyObstacle.y}`]);
  });

  it("limits living strategists to two for production while ignoring removed strategists", () => {
    const state = createInitialGameState();
    addBuilder(state, "builder-2");
    expect(getAvailableProductionTypes(state, "team-1", "home-1")).not.toContain("strategist");
    state.units.find((unit) => unit.id === "builder-2")!.position = { kind: "removed", reason: "defeated" };
    expect(getAvailableProductionTypes(state, "team-1", "home-1")).toContain("strategist");
  });

  it("generates operational-road obstacles and straight lake bridges without mutating state", () => {
    const state = createInitialGameState(); const builder = addBuilder(state, "builder");
    const before = structuredClone(state);
    expect(getOperationalRoadTiles(state, "team-1").length).toBeGreaterThan(0);
    expect(getObstacleCandidates(state, builder.id)).toContainEqual({ x: 5, y: 1 });
    const bridges = getBridgeCandidates(state, builder.id);
    expect(bridges.length).toBeGreaterThan(0);
    expect(bridges.every((cells) => cells.every((cell, index) => !index || cell.x === cells[0].x || cell.y === cells[0].y))).toBe(true);
    expect(state).toEqual(before);
  });

  it("offers every unoccupied cell of a connected bridge as an obstacle candidate", () => {
    const state = createInitialGameState();
    const builder = addBuilder(state, "bridge-obstacle-builder");
    state.constructions.push({
      id: "candidate-bridge",
      kind: "bridge",
      ownerTeamId: "team-2",
      managerUnitId: "enemy-builder",
      tiles: [
        { x: 4, y: 2 },
        { x: 4, y: 3 },
        { x: 4, y: 4 },
      ],
      placedTurn: 1,
      active: true,
    });
    addUnit(state, "bridge-occupant", "team-2", {
      kind: "bridge",
      bridgeId: "candidate-bridge",
      cellIndex: 1,
    });
    state.constructions.push({
      id: "existing-bridge-obstacle",
      kind: "obstacle",
      ownerTeamId: "team-2",
      managerUnitId: "enemy-builder",
      tiles: [{ x: 4, y: 4 }],
      placedTurn: 1,
      active: true,
    });

    const candidates = getObstacleCandidates(state, builder.id);
    expect(candidates).toContainEqual({ x: 4, y: 2 });
    expect(candidates).not.toContainEqual({ x: 4, y: 3 });
    expect(candidates).not.toContainEqual({ x: 4, y: 4 });
  });

  it("includes a connected active bridge, but not the road section across it, in the operational area", () => {
    const state = createInitialGameState();
    const builder = addBuilder(state, "operational-area-builder");
    state.constructions.push({
      id: "operational-area-bridge",
      kind: "bridge",
      ownerTeamId: "team-2",
      tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }],
      placedTurn: 1,
      active: true,
    });

    const operationalRoadKeys = new Set(
      getOperationalRoadTiles(state, builder.ownerTeamId).map((tile) => `${tile.x},${tile.y}`),
    );
    const area = getOperationalAreaTiles(state, builder.ownerTeamId);
    expect(area).toEqual(expect.arrayContaining([{ x: 4, y: 2 }, { x: 4, y: 3 }]));
    expect(
      area
        .filter((cell) => state.map.tiles.find((tile) => tile.x === cell.x && tile.y === cell.y)?.terrain === "road")
        .every((cell) => operationalRoadKeys.has(`${cell.x},${cell.y}`)),
    ).toBe(true);
  });

  it("prevents one builder from maintaining two bridges and blocks same-team overlap at input", () => {
    const state = createInitialGameState(); const first = addBuilder(state, "builder-a"); const second = addBuilder(state, "builder-b");
    const bridge = getBridgeCandidates(state, first.id)[0];
    const saved = saveStrategistActionIntent(state, { teamId: "team-1", strategistUnitId: first.id, action: "place_bridge", tiles: bridge });
    expect(getBridgeCandidates(saved, second.id).some((candidate) => candidate.some((cell) => bridge.some((other) => cell.x === other.x && cell.y === other.y)))).toBe(false);
    saved.constructions.push({ id: "owned", kind: "bridge", ownerTeamId: "team-1", managerUnitId: first.id, tiles: bridge, placedTurn: 1, active: true });
    expect(getBridgeCandidates(saved, first.id)).toEqual([]);
  });

  it("makes all cross-team overlapping placements fail without cooldown and independent of order", () => {
    const make = (reverse: boolean) => { const state = createInitialGameState(); addBuilder(state, "a", "team-1"); addBuilder(state, "b", "team-2"); state.phase = state.turnState.phase = "strategist_action_resolution"; const intents = [
      { teamId: "team-1", strategistUnitId: "a", action: "place_obstacle" as const, tiles: [{ x: 4, y: 1 }] },
      { teamId: "team-2", strategistUnitId: "b", action: "place_obstacle" as const, tiles: [{ x: 4, y: 1 }] },
    ]; state.strategistActionIntents = reverse ? intents.reverse() : intents; return resolveStrategistActions(state); };
    const a = make(false), b = make(true);
    expect(a.constructions).toEqual([]); expect(a.strategistCooldowns).toEqual([]);
    expect(b.constructions).toEqual(a.constructions);
  });

  it("uses independent reset cooldowns with the exact T+5 boundary", () => {
    const state = createInitialGameState(); const builder = addBuilder(state, "builder"); state.turnNumber = 10; state.phase = state.turnState.phase = "strategist_action_resolution";
    state.constructions.push({ id: "bridge", kind: "bridge", ownerTeamId: "team-1", managerUnitId: builder.id, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    state.strategistActionIntents = [{ teamId: "team-1", strategistUnitId: builder.id, action: "reset_bridge", constructionId: "bridge" }];
    const reset = resolveStrategistActions(state);
    expect(reset.strategistCooldowns).toContainEqual({ strategistUnitId: builder.id, kind: "bridge", availableFromTurn: 15 });
    expect(getObstacleCandidates(reset, builder.id).length).toBeGreaterThan(0);
    reset.turnNumber = 14; expect(getBridgeCandidates(reset, builder.id)).toEqual([]);
    reset.turnNumber = 15; expect(getBridgeCandidates(reset, builder.id).length).toBeGreaterThan(0);
  });

  it("waits for every active team submission before strategist resolution", () => {
    let state = createInitialGameState(); state.phase = state.turnState.phase = "strategist_action_input";
    for (const team of state.teams.filter((entry) => entry.status === "active").slice(0, -1)) {
      state = submitStrategistActions(state, team.id);
      expect(state.phase).toBe("strategist_action_input");
    }
    state = submitStrategistActions(state, "team-4");
    expect(state.phase).toBe("strategist_action_resolution");
    expect(resolveStrategistActions(state).phase).toBe("movement_input");
  });

  it("blocks movement but not attacks through an obstacle", () => {
    const state = createInitialGameState(); addUnit(state, "mover", "team-1", { kind: "tile", x: 3, y: 1 }, "archer"); addUnit(state, "enemy", "team-2", { kind: "tile", x: 5, y: 1 });
    state.constructions.push({ id: "obs", kind: "obstacle", ownerTeamId: "team-1", managerUnitId: "builder", tiles: [{ x: 4, y: 1 }], placedTurn: 1, active: true });
    expect(getMovementCandidates(state, "mover")).not.toContainEqual({ kind: "tile", x: 4, y: 1 });
    expect(getMovementCandidates(state, "mover")).not.toContainEqual({ kind: "tile", x: 5, y: 1 });
    expect(getAttackCandidates(state, "mover").map((target) => target.unitId)).toContain("enemy");
  });

  it("treats active bridges as neutral dynamic road connections for movement and attacks", () => {
    const state = createInitialGameState();
    state.constructions.push({ id: "bridge", kind: "bridge", ownerTeamId: "team-2", managerUnitId: "enemy-builder", tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }], placedTurn: 1, active: true });
    addUnit(state, "rider", "team-1", { kind: "tile", x: 4, y: 1 });
    addUnit(state, "archer", "team-1", { kind: "tile", x: 4, y: 1 }, "archer");
    addUnit(state, "enemy-across", "team-2", { kind: "tile", x: 4, y: 4 });
    expect(getMovementCandidates(state, "rider")).toContainEqual({ kind: "bridge", bridgeId: "bridge", cellIndex: 0 });
    expect(getMovementCandidates(state, "rider")).toContainEqual({ kind: "bridge", bridgeId: "bridge", cellIndex: 1 });
    expect(getAttackCandidates(state, "archer").map((target) => target.unitId)).toContain("enemy-across");

    state.constructions.push({ id: "bridge-obstacle", kind: "obstacle", ownerTeamId: "team-1", managerUnitId: "builder", tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true });
    expect(getMovementCandidates(state, "rider")).not.toContainEqual({ kind: "bridge", bridgeId: "bridge", cellIndex: 0 });
    expect(getAttackCandidates(state, "archer").map((target) => target.unitId)).toContain("enemy-across");
  });
});
