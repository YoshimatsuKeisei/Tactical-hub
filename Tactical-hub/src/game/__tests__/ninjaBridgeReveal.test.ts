import { describe, expect, it } from "vitest";
import { getBridgeCandidates, resolveStrategistActions, saveStrategistActionIntent } from "../engine/construction";
import { getMovementCandidates } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { Unit } from "../types";

describe("bridge construction over water ninjas", () => {
  it("rejects the whole bridge candidate and intent when an own water ninja occupies any planned tile", () => {
    const state = createInitialGameState();
    const builder = state.units.find((unit) => unit.id === "home-1-strategist")!;
    builder.role = "builder";
    const tiles = getBridgeCandidates(state, builder.id).find((candidate) => candidate.length >= 2)!;
    expect(tiles).toBeDefined();
    const occupiedCell = tiles.at(-1)!;
    const ownNinja: Unit = {
      id: "own-water-ninja",
      ownerTeamId: builder.ownerTeamId,
      type: "ninja",
      hp: 1,
      position: { kind: "water", ...occupiedCell },
      statuses: [],
    };
    state.units.push(ownNinja);
    expect(getBridgeCandidates(state, builder.id)).not.toContainEqual(tiles);
    const saved = saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "place_bridge", tiles });
    expect(saved).toBe(state);
    expect(saved.strategistActionIntents).toHaveLength(0);
  });

  it("allows an enemy water ninja, saves the intent, and places that ninja on the correct new bridge cell", () => {
    const state = createInitialGameState();
    const builder = state.units.find((unit) => unit.id === "home-1-strategist")!;
    builder.role = "builder";
    const tiles = getBridgeCandidates(state, builder.id).find((candidate) => candidate.length >= 2)!;
    const occupiedCellIndex = 1;
    const enemyNinja: Unit = {
      id: "enemy-water-ninja",
      ownerTeamId: "team-2",
      type: "ninja",
      hp: 1,
      position: { kind: "water", ...tiles[occupiedCellIndex] },
      statuses: [],
    };
    state.units.push(enemyNinja);
    expect(getBridgeCandidates(state, builder.id)).toContainEqual(tiles);
    const saved = saveStrategistActionIntent(state, { teamId: "team-1", strategistUnitId: builder.id, action: "place_bridge", tiles });
    expect(saved.strategistActionIntents).toHaveLength(1);
    saved.phase = saved.turnState.phase = "strategist_action_resolution";
    const resolved = resolveStrategistActions(saved, () => 0);
    const bridge = resolved.constructions.find((construction) => construction.kind === "bridge" && construction.active)!;
    expect(resolved.units.find((unit) => unit.id === enemyNinja.id)).toMatchObject({
      hp: enemyNinja.hp,
      position: { kind: "bridge", bridgeId: bridge.id, cellIndex: occupiedCellIndex },
    });
    expect(resolved.phase).toBe("movement_input");
    expect(resolved.movedUnitIdsThisMovementPhase).not.toContain(enemyNinja.id);
    resolved.currentMovementTeamId = enemyNinja.ownerTeamId;
    expect(getMovementCandidates(resolved, enemyNinja.id).length).toBeGreaterThan(0);
  });
});
