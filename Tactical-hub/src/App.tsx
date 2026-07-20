import { useMemo, useState } from "react";
import { BoardView } from "./components/BoardView";
import { GameDebugPanel } from "./components/GameDebugPanel";
import { resolveBattle, saveAttackIntent } from "./game/engine/battle";
import { saveMovementIntent, resolveMovement } from "./game/engine/movement";
import { resolveProduction } from "./game/engine/production";
import { isRetreating } from "./game/engine/retreat";
import { createInitialGameState } from "./game/initialState";
import type { AttackTarget, UnitPosition } from "./game/types";
import { saveStrategistActionIntent } from "./game/engine/construction";

export default function App() {
  const [state, setState] = useState(createInitialGameState);
  const [selectedUnitId, setSelectedUnitId] = useState<string>();
  const [manualTeamId, setManualTeamId] = useState("team-1");
  const [constructionMode, setConstructionMode] = useState<"bridge" | "obstacle">();
  const selectedUnit = useMemo(
    () => state.units.find((unit) => unit.id === selectedUnitId),
    [selectedUnitId, state.units],
  );
  const effectiveManualTeamId = state.phase === "movement_input" && state.currentMovementTeamId
    ? state.currentMovementTeamId
    : manualTeamId;

  function chooseDestination(position: UnitPosition) {
    if (!selectedUnit) return;
    setState(
      saveMovementIntent(state, {
        teamId: selectedUnit.ownerTeamId,
        unitId: selectedUnit.id,
        from: selectedUnit.position,
        to: position,
        stay: false,
      }),
    );
  }

  function chooseAttackTarget(target: AttackTarget) {
    if (!selectedUnit) return;
    if (isRetreating(selectedUnit)) return;
    setState(
      saveAttackIntent(state, {
        teamId: selectedUnit.ownerTeamId,
        attackerUnitId: selectedUnit.id,
        target,
        pass: false,
      }),
    );
  }

  return (
    <main className="app-shell">
      <div className="play-area">
        <header>
          <h1>Tactical Hub Phase 1</h1>
          <p>Local logic sandbox for map, base slots, production intents, and simultaneous movement resolution.</p>
        </header>
        <div className="board-scroll">
          <BoardView
            state={state}
            selectedUnitId={selectedUnitId}
            onSelectUnit={setSelectedUnitId}
            onChooseDestination={chooseDestination}
            onChooseAttackTarget={chooseAttackTarget}
            manualTeamId={effectiveManualTeamId}
            constructionMode={constructionMode}
            onChooseConstruction={(unitId, kind, tiles) => {
              const unit = state.units.find((candidate) => candidate.id === unitId);
              if (!unit) return;
              setState(saveStrategistActionIntent(state, { teamId: unit.ownerTeamId, strategistUnitId: unit.id, action: kind === "bridge" ? "place_bridge" : "place_obstacle", tiles }));
            }}
          />
        </div>
      </div>
      <GameDebugPanel
        state={state}
        selectedUnitId={selectedUnitId}
        manualTeamId={effectiveManualTeamId}
        onManualTeamChange={setManualTeamId}
        constructionMode={constructionMode}
        onConstructionModeChange={setConstructionMode}
        onResolveProduction={() => setState(resolveProduction(state))}
        onResolveMovement={() => {
          const next = resolveMovement(state);
          setState(next);
          if (next.currentMovementTeamId) setManualTeamId(next.currentMovementTeamId);
          setSelectedUnitId(undefined);
        }}
        onResolveBattle={() => {
          setState(resolveBattle(state));
          setSelectedUnitId(undefined);
        }}
        onStateChange={setState}
      />
    </main>
  );
}
