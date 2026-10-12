import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { createInitialGameState } from "../game/initialState";
import { BoardView } from "./BoardView";
import { GameDebugPanel } from "./GameDebugPanel";

describe("unified strategist UI", () => {
  it("shows encourage and construction controls for a legacy teleporter role", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    strategist.role = "teleporter";
    state.phase = state.turnState.phase = "strategist_action_input";

    const html = renderToStaticMarkup(<BoardView
      state={state}
      selectedUnitId={strategist.id}
      onSelectUnit={vi.fn()}
      onChooseDestination={vi.fn()}
      onChooseAttackTarget={vi.fn()}
      manualTeamId="team-1"
      constructionMode="obstacle"
      onChooseConstruction={vi.fn()}
    />);

    expect(html).toContain("encourage-highlighted");
    expect(html).toContain("construction-preview");
  });

  it("labels a legacy builder role with all three abilities", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!;
    strategist.role = "builder";
    state.phase = state.turnState.phase = "movement_input";
    state.currentMovementTeamId = "team-1";
    state.productionCompletedTeamIdsThisTurn = ["team-1"];

    const html = renderToStaticMarkup(<GameDebugPanel
      state={state}
      manualTeamId="team-1"
      onManualTeamChange={vi.fn()}
      onConstructionModeChange={vi.fn()}
      onResolveMovement={vi.fn()}
      onResolveBattle={vi.fn()}
      onResolveProduction={vi.fn()}
      onStateChange={vi.fn()}
    />);

    expect(html).toContain("home-1-strategist");
    expect(html).toContain("abilities: encourage / build / teleport");
  });
});
