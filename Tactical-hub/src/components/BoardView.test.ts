import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../game/constants";
import { getMovementCandidates } from "../game/engine/movement";
import { createInitialGameState } from "../game/initialState";
import type { Unit } from "../game/types";
import { getMovementCandidateByBoardCell } from "./BoardView";

describe("BoardView movement candidate mapping", () => {
  it("maps a horizontally adjacent bridge candidate to its rendered board cell", () => {
    const state = createInitialGameState();
    state.constructions.push({
      id: "horizontal-bridge",
      kind: "bridge",
      ownerTeamId: "team-2",
      managerUnitId: "team-2-builder",
      tiles: [
        { x: 4, y: 3 },
        { x: 5, y: 3 },
        { x: 6, y: 3 },
        { x: 7, y: 3 },
        { x: 8, y: 3 },
      ],
      placedTurn: 1,
      active: true,
    });
    const cavalry: Unit = {
      id: "bridge-left-cavalry",
      ownerTeamId: "team-1",
      type: "cavalry",
      hp: UNIT_STATS.cavalry.hp,
      position: { kind: "tile", x: 3, y: 3 },
      statuses: [],
    };
    state.units.push(cavalry);

    const candidates = getMovementCandidates(state, cavalry.id);
    const byCell = getMovementCandidateByBoardCell(state, candidates);

    expect(candidates).toContainEqual({
      kind: "bridge",
      bridgeId: "horizontal-bridge",
      cellIndex: 0,
    });
    expect(byCell.get("4,3")).toEqual({
      kind: "bridge",
      bridgeId: "horizontal-bridge",
      cellIndex: 0,
    });
  });
});
