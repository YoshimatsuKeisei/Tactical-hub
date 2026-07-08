import { useMemo, useState } from "react";
import { BoardView } from "./components/BoardView";
import { GameDebugPanel } from "./components/GameDebugPanel";
import { resolveBattle, saveAttackIntent } from "./game/engine/battle";
import { saveMovementIntent, resolveMovement } from "./game/engine/movement";
import { resolveProduction } from "./game/engine/production";
import { createInitialGameState } from "./game/initialState";
import type { AttackTarget, UnitPosition } from "./game/types";

export default function App() {
  const [state, setState] = useState(createInitialGameState);
  const [selectedUnitId, setSelectedUnitId] = useState<string>();
  const selectedUnit = useMemo(
    () => state.units.find((unit) => unit.id === selectedUnitId),
    [selectedUnitId, state.units],
  );

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
          />
        </div>
      </div>
      <GameDebugPanel
        state={state}
        selectedUnitId={selectedUnitId}
        onResolveProduction={() => setState(resolveProduction(state))}
        onResolveMovement={() => {
          setState(resolveMovement(state));
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
