import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { getAttackCandidates } from "../engine/battle";
import { getMovementCandidates, saveMovementIntent, submitMovement } from "../engine/movement";
import { getOperationalRoadTiles } from "../engine/construction";
import { getTeleportDestinationCandidates, getTeleportTargetCandidates, isTeleportAvailable, saveTeleportIntent } from "../engine/teleport";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function add(state: GameState, id: string, teamId: string, type: Unit["type"], position: UnitPosition, role?: Unit["role"]) {
  const unit: Unit = { id, ownerTeamId: teamId, type, role, hp: UNIT_STATS[type].hp, position, statuses: [] };
  state.units.push(unit); return unit;
}
function fixture() {
  const state = createInitialGameState();
  const teleporter = add(state, "teleporter-a", "team-1", "strategist", { kind: "tile", x: 4, y: 1 }, "teleporter");
  const target = add(state, "teleport-target", "team-1", "infantry", { kind: "tile", x: 5, y: 1 });
  return { state, teleporter, target };
}

describe("Phase 4-C teleporter strategist", () => {
  it.each([
    ["friendly regular", "infantry", undefined, "friendly", true],
    ["another teleporter", "strategist", "teleporter", "friendly", false],
    ["encouragement strategist", "strategist", "encourage", "friendly", false],
    ["builder strategist", "strategist", "builder", "friendly", false],
    ["king", "king", undefined, "friendly", false],
    ["engineer", "engineer", undefined, "friendly", false],
    ["enemy", "infantry", undefined, "enemy", false],
  ] as const)("target decision table: %s => %s", (_, type, role, allegiance, expected) => {
    const { state, teleporter } = fixture();
    const candidate = add(state, `decision-${type}-${role ?? allegiance}`, allegiance === "enemy" ? "team-2" : "team-1", type, { kind: "tile", x: 3, y: 1 }, role);
    expect(getTeleportTargetCandidates(state, teleporter.id).some((unit) => unit.id === candidate.id)).toBe(expected);
  });

  it.each([
    ["dead", "dead"],
    ["removed", "removed"],
    ["normally moved", "moved"],
    ["normal-move reserved", "move-reserved"],
    ["other teleport reserved", "teleport-reserved"],
  ] as const)("target state decision table: %s is unavailable", (_, stateKind) => {
    let { state, teleporter, target } = fixture();
    if (stateKind === "dead") target.hp = 0;
    if (stateKind === "removed") target.position = { kind: "removed", reason: "defeated" };
    if (stateKind === "moved") state.movedUnitIdsThisMovementPhase.push(target.id);
    if (stateKind === "move-reserved") state = saveMovementIntent(state, { teamId: "team-1", unitId: target.id, from: target.position, to: target.position, stay: true });
    if (stateKind === "teleport-reserved") {
      const other = add(state, "decision-other-teleporter", "team-1", "strategist", { kind: "tile", x: 4, y: 1 }, "teleporter");
      const destination = getTeleportDestinationCandidates(state, other.id)[0];
      state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: other.id, targetUnitId: target.id, to: destination });
    }
    expect(getTeleportTargetCandidates(state, teleporter.id).some((unit) => unit.id === target.id)).toBe(false);
  });

  it("target decision table excludes the acting teleporter itself", () => {
    const { state, teleporter } = fixture();
    expect(getTeleportTargetCandidates(state, teleporter.id).some((unit) => unit.id === teleporter.id)).toBe(false);
  });

  it("reuses encouragement radius 1 near an owned base and radius 2 elsewhere", () => {
    const near = fixture();
    add(near.state, "near-edge", "team-1", "infantry", { kind: "tile", x: 3, y: 1 });
    add(near.state, "near-outside", "team-1", "infantry", { kind: "tile", x: 2, y: 1 });
    expect(getTeleportTargetCandidates(near.state, near.teleporter.id).map((unit) => unit.id)).toContain("near-edge");
    expect(getTeleportTargetCandidates(near.state, near.teleporter.id).map((unit) => unit.id)).not.toContain("near-outside");

    const far = fixture(); far.teleporter.position = { kind: "tile", x: 11, y: 6 }; far.target.position = { kind: "tile", x: 13, y: 6 };
    const outside = add(far.state, "far-outside", "team-1", "infantry", { kind: "tile", x: 14, y: 6 });
    expect(getTeleportTargetCandidates(far.state, far.teleporter.id).map((unit) => unit.id)).toContain(far.target.id);
    expect(getTeleportTargetCandidates(far.state, far.teleporter.id).map((unit) => unit.id)).not.toContain(outside.id);
  });

  it("excludes prohibited unit categories and reserved or moved units", () => {
    const { state, teleporter } = fixture();
    add(state, "other-teleporter", "team-1", "strategist", { kind: "tile", x: 5, y: 1 }, "teleporter");
    add(state, "encourager", "team-1", "strategist", { kind: "tile", x: 3, y: 2 }, "encourage");
    add(state, "builder", "team-1", "strategist", { kind: "tile", x: 4, y: 2 }, "builder");
    add(state, "engineer", "team-1", "engineer", { kind: "tile", x: 5, y: 2 });
    add(state, "enemy", "team-2", "infantry", { kind: "tile", x: 3, y: 1 });
    const ids = getTeleportTargetCandidates(state, teleporter.id).map((unit) => unit.id);
    expect(ids).not.toEqual(expect.arrayContaining([teleporter.id, "other-teleporter", "encourager", "builder", "engineer", "enemy", "home-1-king"]));
    const target = state.units.find((unit) => unit.id === "teleport-target")!;
    const planned = saveMovementIntent(state, { teamId: "team-1", unitId: target.id, from: target.position, to: { kind: "tile", x: 6, y: 1 }, stay: false });
    expect(getTeleportTargetCandidates(planned, teleporter.id).map((unit) => unit.id)).not.toContain(target.id);
  });

  it("offers operational roads, active bridges and owned base slots but not the opposite road section", () => {
    const { state, teleporter } = fixture();
    state.constructions.push({ id: "teleport-bridge", kind: "bridge", ownerTeamId: "team-2", tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }], placedTurn: 1, active: true });
    const destinations = getTeleportDestinationCandidates(state, teleporter.id);
    expect(destinations.some((position) => position.kind === "tile")).toBe(true);
    expect(destinations).toContainEqual({ kind: "bridge", bridgeId: "teleport-bridge", cellIndex: 0 });
    expect(destinations.some((position) => position.kind === "base" && position.baseId === "home-1")).toBe(true);
    expect(destinations.some((position) => position.kind === "base" && position.baseId !== "home-1")).toBe(false);
    const operationalRoads = new Set(getOperationalRoadTiles(state, "team-1").map((tile) => `${tile.x},${tile.y}`));
    expect(destinations.filter((position) => position.kind === "tile").every((position) => operationalRoads.has(`${position.x},${position.y}`))).toBe(true);
  });

  it("prevents target and destination conflicts with normal moves and another teleporter", () => {
    let { state, teleporter, target } = fixture();
    const second = add(state, "teleporter-b", "team-1", "strategist", { kind: "tile", x: 4, y: 1 }, "teleporter");
    const destination = getTeleportDestinationCandidates(state, teleporter.id).find((position) => positionKey(position) !== positionKey(target.position))!;
    state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: destination });
    expect(getTeleportTargetCandidates(state, second.id).map((unit) => unit.id)).not.toContain(target.id);
    expect(getTeleportDestinationCandidates(state, second.id).map(positionKey)).not.toContain(positionKey(destination));
    expect(saveMovementIntent(state, { teamId: "team-1", unitId: target.id, from: target.position, to: destination, stay: false })).toBe(state);
    expect(getMovementCandidates(state, target.id)).toEqual([]);
  });

  it("moves only on successful revalidation, marks moved, and applies T+5 cooldown", () => {
    let { state, teleporter, target } = fixture(); state.turnNumber = 10;
    target.statuses = [{ kind: "retreating", retreatTargetBaseId: "home-1" }];
    const destination = getTeleportDestinationCandidates(state, teleporter.id).find((position) => position.kind === "base")!;
    expect(isTeleportAvailable(state, teleporter.id)).toBe(true);
    state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: destination });
    expect(state.teleportIntents).toContainEqual({ teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: destination });
    const resolved = submitMovement(state, "team-1");
    expect(resolved.units.find((unit) => unit.id === target.id)?.position).toEqual(destination);
    expect(resolved.movedUnitIdsThisMovementPhase).toContain(target.id);
    expect(resolved.units.find((unit) => unit.id === target.id)?.statuses.some((status) => status.kind === "retreating")).toBe(false);
    expect(resolved.teleportCooldowns).toContainEqual({ strategistUnitId: teleporter.id, availableFromTurn: 15 });
    resolved.turnNumber = 14; expect(isTeleportAvailable(resolved, teleporter.id)).toBe(false);
    resolved.turnNumber = 15; expect(isTeleportAvailable(resolved, teleporter.id)).toBe(true);
  });

  it("fails an invalidated destination without moving or starting cooldown", () => {
    let { state, teleporter, target } = fixture();
    const destination = getTeleportDestinationCandidates(state, teleporter.id).find((position) => position.kind === "tile")!;
    state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: destination });
    expect(state.teleportIntents).toHaveLength(1);
    if (destination.kind === "tile") add(state, "late-occupant", "team-1", "infantry", destination);
    const resolved = submitMovement(state, "team-1");
    expect(resolved.units.find((unit) => unit.id === target.id)?.position).toEqual(target.position);
    expect(resolved.teleportCooldowns).toEqual([]);
  });

  it("lets two teleporters move different targets, then continues sequential movement into attack input", () => {
    let { state, teleporter, target } = fixture();
    const second = add(state, "teleporter-b", "team-1", "strategist", { kind: "tile", x: 4, y: 1 }, "teleporter");
    const secondTarget = add(state, "teleport-target-b", "team-1", "cavalry", { kind: "tile", x: 3, y: 1 });
    const destinations = getTeleportDestinationCandidates(state, teleporter.id).filter((position) => position.kind === "base");
    state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: teleporter.id, targetUnitId: target.id, to: destinations[0] });
    state = saveTeleportIntent(state, { teamId: "team-1", strategistUnitId: second.id, targetUnitId: secondTarget.id, to: destinations[1] });
    state = submitMovement(state, "team-1");
    expect(state.units.find((unit) => unit.id === target.id)?.position).toEqual(destinations[0]);
    expect(state.units.find((unit) => unit.id === secondTarget.id)?.position).toEqual(destinations[1]);
    expect(state.currentMovementTeamId).toBe("team-2");
    while (state.phase === "movement_input" && state.currentMovementTeamId) state = submitMovement(state, state.currentMovementTeamId);
    expect(state.phase).toBe("attack_input");
  });

  it("teleports a regular unit to an operational road and leaves it able to attack after movement", () => {
    let { state, teleporter, target } = fixture();
    const destination = { kind: "tile", x: 4, y: 4 } as const;
    const enemy = add(state, "road-teleport-enemy", "team-2", "infantry", { kind: "tile", x: 5, y: 5 });
    expect(getTeleportDestinationCandidates(state, teleporter.id)).toContainEqual(destination);

    state = saveTeleportIntent(state, {
      teamId: "team-1",
      strategistUnitId: teleporter.id,
      targetUnitId: target.id,
      to: destination,
    });
    state = submitMovement(state, "team-1");
    expect(state.units.find((unit) => unit.id === target.id)?.position).toEqual(destination);
    expect(state.movedUnitIdsThisMovementPhase).toContain(target.id);

    while (state.phase === "movement_input" && state.currentMovementTeamId) state = submitMovement(state, state.currentMovementTeamId);
    expect(state.phase).toBe("attack_input");
    expect(getAttackCandidates(state, target.id).map((candidate) => candidate.unitId)).toContain(enemy.id);
  });

  it("recalculates later-team move and teleport destinations after an earlier teleport occupies a road", () => {
    let { state, teleporter, target } = fixture();
    const occupiedDestination = { kind: "tile", x: 4, y: 4 } as const;
    const laterMover = add(state, "later-mover", "team-2", "infantry", { kind: "tile", x: 5, y: 5 });
    const laterTeleporter = add(state, "later-teleporter", "team-2", "strategist", { kind: "tile", x: 6, y: 6 }, "teleporter");
    state.teams.find((team) => team.id === "team-2")!.controlledBaseIds.push("home-1");

    state = saveTeleportIntent(state, {
      teamId: "team-1",
      strategistUnitId: teleporter.id,
      targetUnitId: target.id,
      to: occupiedDestination,
    });
    state = submitMovement(state, "team-1");

    expect(state.currentMovementTeamId).toBe("team-2");
    expect(state.units.find((unit) => unit.id === target.id)?.position).toEqual(occupiedDestination);
    expect(getOperationalRoadTiles(state, "team-2").map(({ x, y }) => `${x},${y}`)).toContain("4,4");
    expect(getMovementCandidates(state, laterMover.id).map(positionKey)).not.toContain(positionKey(occupiedDestination));
    expect(getTeleportDestinationCandidates(state, laterTeleporter.id).map(positionKey)).not.toContain(positionKey(occupiedDestination));

    while (state.phase === "movement_input" && state.currentMovementTeamId) state = submitMovement(state, state.currentMovementTeamId);
    expect(state.phase).toBe("attack_input");
  });
});
