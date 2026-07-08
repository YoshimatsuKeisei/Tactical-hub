import { UNIT_STATS } from "../game/constants";
import { getAttackCandidates, saveAttackIntent } from "../game/engine/battle";
import { getAvailableProductionTypes, saveProductionChoice } from "../game/engine/production";
import type { GameState } from "../game/types";

type Props = {
  state: GameState;
  selectedUnitId?: string;
  onResolveMovement: () => void;
  onResolveBattle: () => void;
  onResolveProduction: () => void;
  onStateChange: (state: GameState) => void;
};

export function GameDebugPanel({ state, selectedUnitId, onResolveMovement, onResolveBattle, onResolveProduction, onStateChange }: Props) {
  const selectedUnit = state.units.find((unit) => unit.id === selectedUnitId);
  const movementIntents = state.turnState.actionIntents.flatMap((intent) => intent.movementIntents);
  const attackIntents = state.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []);
  const productionIntents = state.turnState.actionIntents.flatMap((intent) => intent.productionChoices);
  const attackCandidates = selectedUnitId ? getAttackCandidates(state, selectedUnitId) : [];
  const activeTeam = state.teams.find((team) => team.id === "team-1")!;
  const controlledBases = state.bases.filter((base) => activeTeam.controlledBaseIds.includes(base.id));

  return (
    <aside className="debug-panel">
      <section className="status-card">
        <h2>Turn</h2>
        <div className="status-grid">
          <span>Turn</span>
          <strong>{state.turnNumber}</strong>
          <span>Phase</span>
          <strong>{state.phase}</strong>
          <span>Selected</span>
          <strong>{selectedUnit ? `${selectedUnit.id} (${selectedUnit.type})` : "none"}</strong>
          <span>Moves</span>
          <strong>{movementIntents.length}</strong>
          <span>Attacks</span>
          <strong>{attackIntents.length}</strong>
        </div>
        <button className="primary sticky-action" onClick={onResolveMovement}>
          Resolve Movement
        </button>
      </section>

      <section>
        <h2>Legend</h2>
        <div className="legend-grid">
          <span className="legend-swatch base" /> Home / Relay Base
          <span className="legend-swatch baseGate" /> Base Gate
          <span className="legend-swatch road" /> Road / Diagonal
          <span className="legend-swatch lake" /> Lake
          <span className="legend-swatch reorganize" /> Reorganize Area
          <span className="legend-swatch outside" /> Outside
        </div>
      </section>

      <section>
        <h2>Production</h2>
        {controlledBases.map((base) => {
          const types = getAvailableProductionTypes(state, activeTeam.id, base.id);
          return (
            <div key={base.id} className="production-row">
              <strong>{base.id}</strong>
              <div className="button-row">
                {types.length ? (
                  types.map((unitType) => (
                    <button
                      key={unitType}
                      onClick={() => onStateChange(saveProductionChoice(state, { teamId: activeTeam.id, baseId: base.id, unitType }))}
                    >
                      {UNIT_STATS[unitType].label}
                    </button>
                  ))
                ) : (
                  <span>no slot</span>
                )}
              </div>
            </div>
          );
        })}
        <button className="primary" onClick={onResolveProduction}>
          Resolve Production
        </button>
      </section>

      <section>
        <h2>Saved Intents</h2>
        <div className="intent-list">
          {movementIntents.length ? (
            movementIntents.map((intent) => (
              <div key={intent.unitId} className="intent-item">
                <strong>{intent.unitId}</strong>
                <span>{intent.stay ? "stay" : JSON.stringify(intent.to)}</span>
              </div>
            ))
          ) : (
            <p>No movement intents saved.</p>
          )}
        </div>
        <details>
          <summary>Raw intents</summary>
          <pre>{JSON.stringify({ productionIntents, movementIntents, attackIntents }, null, 2)}</pre>
        </details>
      </section>

      <section>
        <h2>Attack</h2>
        <div className="intent-list">
          {attackCandidates.length ? (
            attackCandidates.map((target) => (
              <button
                key={target.unitId}
                className="intent-item command-item"
                onClick={() => {
                  if (!selectedUnit) return;
                  onStateChange(
                    saveAttackIntent(state, {
                      teamId: selectedUnit.ownerTeamId,
                      attackerUnitId: selectedUnit.id,
                      target,
                      pass: false,
                    }),
                  );
                }}
              >
                <strong>{target.unitId}</strong>
                <span>{target.baseId ? `${target.baseId}/${target.slotId}` : "board"}</span>
              </button>
            ))
          ) : (
            <p>No attack candidates.</p>
          )}
          {selectedUnit ? (
            <button
              className="secondary"
              onClick={() =>
                onStateChange(
                  saveAttackIntent(state, {
                    teamId: selectedUnit.ownerTeamId,
                    attackerUnitId: selectedUnit.id,
                    pass: true,
                  }),
                )
              }
            >
              Save No Attack
            </button>
          ) : null}
        </div>
        <div className="intent-list">
          {attackIntents.length ? (
            attackIntents.map((intent) => (
              <div key={intent.attackerUnitId} className="intent-item">
                <strong>{intent.attackerUnitId}</strong>
                <span>{intent.pass ? "no attack" : intent.target?.unitId}</span>
              </div>
            ))
          ) : (
            <p>No attack intents saved.</p>
          )}
        </div>
        <button className="primary battle-action" onClick={onResolveBattle}>
          Resolve Battle
        </button>
      </section>

      <section>
        <h2>State Summary</h2>
        <p>Units: {state.units.filter((unit) => unit.position.kind !== "removed").length}</p>
        <p>Bases: {state.bases.length}</p>
      </section>

      <section>
        <h2>Home Slots</h2>
        <div className="slot-debug">
          {state.bases
            .filter((base) => base.type === "home")
            .map((base) => (
              <div key={base.id} className="slot-debug-base">
                <strong>{base.id}</strong>
                {base.slots.map((slot) => (
                  <span key={slot.id}>
                    {slot.localRow},{slot.localCol}: {slot.unitId ?? "empty"}
                  </span>
                ))}
              </div>
            ))}
        </div>
      </section>

      <section>
        <h2>Game Log</h2>
        <ol className="log-list" reversed>
          {state.logs.slice(-12).map((log) => (
            <li key={log.id}>{log.message}</li>
          ))}
        </ol>
      </section>
    </aside>
  );
}
