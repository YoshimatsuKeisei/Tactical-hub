import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { UNIT_STATS } from "../game/constants";
import { createInitialGameState } from "../game/initialState";
import type { Unit } from "../game/types";
import { BoardView } from "./BoardView";
import { UnitToken } from "./UnitToken";

function movementState() {
  const state = createInitialGameState();
  state.phase = state.turnState.phase = "movement_input";
  state.currentMovementTeamId = "team-1";
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  const ninja: Unit = {
    id: "ui-ninja", ownerTeamId: "team-1", type: "ninja", hp: UNIT_STATS.ninja.hp,
    position: { kind: "tile", x: 3, y: 1 }, statuses: [],
  };
  const enemy: Unit = {
    id: "ui-enemy", ownerTeamId: "team-2", type: "infantry", hp: UNIT_STATS.infantry.hp,
    position: { kind: "tile", x: 6, y: 1 }, statuses: [],
  };
  state.units.push(ninja, enemy);
  return { state, ninja };
}

describe("ninja teleport UI", () => {
  it("highlights a teleport-only destination with ordinary movement cells", () => {
    const { state, ninja } = movementState();
    const html = renderToStaticMarkup(<BoardView
      state={state}
      selectedUnitId={ninja.id}
      onSelectUnit={vi.fn()}
      onChooseDestination={vi.fn()}
      onChooseAttackTarget={vi.fn()}
      manualTeamId="team-1"
      onChooseConstruction={vi.fn()}
    />);
    const tileButtons = [...html.matchAll(/<button class="tile [^"]*"/g)].map((match) => match[0]);
    const ordinaryIndex = state.map.tiles.findIndex((tile) => tile.x === 4 && tile.y === 1);
    const teleportIndex = state.map.tiles.findIndex((tile) => tile.x === 7 && tile.y === 1);

    expect(tileButtons[ordinaryIndex]).toContain("highlighted");
    expect(tileButtons[teleportIndex]).toContain("highlighted");
  });

  it("shows whether the one-use ability is available or used", () => {
    const { ninja } = movementState();
    const available = renderToStaticMarkup(<UnitToken unit={ninja} />);
    const used = renderToStaticMarkup(<UnitToken unit={ninja} ninjaTeleportUsed />);

    expect(available).toContain("Ninja teleport available");
    expect(available).toContain("瞬");
    expect(used).toContain("Ninja teleport used");
    expect(used).toContain("済");
  });
});
