import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BoardView } from "./components/BoardView";
import { CpuControlPanel, type CpuRunnerSpeed } from "./components/CpuControlPanel";
import { GameDebugPanel } from "./components/GameDebugPanel";
import { resolveBattle, saveAttackIntent } from "./game/engine/battle";
import { saveMovementIntent, resolveMovement } from "./game/engine/movement";
import { resolveProduction, submitTeamProduction } from "./game/engine/production";
import { isRetreating } from "./game/engine/retreat";
import { createInitialGameState } from "./game/initialState";
import type { AttackTarget, StrategistRole, UnitPosition } from "./game/types";
import { saveStrategistActionIntent } from "./game/engine/construction";
import { advanceVisualCpuOneStep, resolveBattleWithHiddenCpuIntents } from "./game/cpu/visualCpuRunner";
import { createCpuRuntime, type CpuRuntime, type CpuTeamSettings, type TeamController } from "./game/cpu/types";

export default function App() {
  const [state, setState] = useState(createInitialGameState);
  const [selectedUnitId, setSelectedUnitId] = useState<string>();
  const [manualTeamId, setManualTeamId] = useState("team-1");
  const [constructionMode, setConstructionMode] = useState<"bridge" | "obstacle">();
  const [cpuSettings, setCpuSettings] = useState<CpuTeamSettings>(() => Object.fromEntries(createInitialGameState().teams.filter((team) => !team.isNeutral).map((team) => [team.id, "human"])));
  const [cpuRuntime, setCpuRuntime] = useState<CpuRuntime>(() => createCpuRuntime(1));
  const [cpuRunning, setCpuRunning] = useState(false);
  const [cpuPaused, setCpuPaused] = useState(false);
  const [cpuSpeed, setCpuSpeed] = useState<CpuRunnerSpeed>("normal");
  const stateRef = useRef(state);
  const runtimeRef = useRef(cpuRuntime);
  const settingsRef = useRef(cpuSettings);
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { runtimeRef.current = cpuRuntime; }, [cpuRuntime]);
  useEffect(() => { settingsRef.current = cpuSettings; }, [cpuSettings]);
  const selectedUnit = useMemo(
    () => state.units.find((unit) => unit.id === selectedUnitId),
    [selectedUnitId, state.units],
  );
  const effectiveManualTeamId = state.phase === "movement_input" && state.currentMovementTeamId
    ? state.currentMovementTeamId
    : manualTeamId;
  const initialStrategistRolesLocked = state.turnNumber !== 1 || state.movementCompletedTeamIds.length > 0 || state.productionCompletedTeamIdsThisTurn.length > 0 || state.movedUnitIdsThisMovementPhase.length > 0;
  const initialStrategistRoles = Object.fromEntries(state.units.filter((unit) => unit.type === "strategist" && unit.id.startsWith("home-")).map((unit) => [unit.id, unit.role ?? "encourage"])) as Record<string, StrategistRole>;

  function seededInitialRole(teamId: string) {
    const roles: StrategistRole[] = ["builder", "encourage", "teleporter"];
    const hash = [...teamId].reduce((value, character) => Math.imul(value ^ character.charCodeAt(0), 16777619) >>> 0, cpuRuntime.seed >>> 0);
    return roles[hash % roles.length];
  }

  const advanceCpu = useCallback(() => {
    const result = advanceVisualCpuOneStep(stateRef.current, runtimeRef.current, settingsRef.current);
    runtimeRef.current = result.runtime;
    setCpuRuntime(result.runtime);
    if (result.state !== stateRef.current) {
      stateRef.current = result.state;
      setState(result.state);
    }
    if (result.runtime.stoppedReason) setCpuRunning(false);
    return result.applied;
  }, []);

  useEffect(() => {
    if (!cpuRunning || cpuPaused) return;
    const delay = cpuSpeed === "normal" ? 700 : cpuSpeed === "fast" ? 150 : 10;
    const timer = window.setInterval(advanceCpu, delay);
    return () => window.clearInterval(timer);
  }, [advanceCpu, cpuPaused, cpuRunning, cpuSpeed]);

  function chooseDestination(position: UnitPosition) {
    if (!selectedUnit) return;
    if ((cpuSettings[selectedUnit.ownerTeamId] ?? "human") !== "human") return;
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
    if ((cpuSettings[selectedUnit.ownerTeamId] ?? "human") !== "human") return;
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
          <div className="turn-phase-banner">
            <span>ターン <strong>{state.turnNumber}</strong></span>
            <span>フェーズ <strong>{state.phase}</strong></span>
            {state.currentMovementTeamId ? <span>移動担当 <strong>{state.currentMovementTeamId}</strong></span> : null}
          </div>
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
        onResolveProduction={() => setState(
          state.phase === "movement_input" && state.currentMovementTeamId
            ? submitTeamProduction(state, state.currentMovementTeamId)
            : resolveProduction(state),
        )}
        onResolveMovement={() => {
          const next = resolveMovement(state);
          setState(next);
          if (next.currentMovementTeamId) setManualTeamId(next.currentMovementTeamId);
          setSelectedUnitId(undefined);
        }}
        onResolveBattle={() => {
          const resolved = resolveBattleWithHiddenCpuIntents(state, cpuRuntime);
          runtimeRef.current = resolved.runtime;
          setCpuRuntime(resolved.runtime);
          setState(resolved.state);
          setSelectedUnitId(undefined);
        }}
        battleResolveDisabled={state.phase === "attack_input" && state.teams.some((team) => team.status === "active" && cpuSettings[team.id] === "random_cpu" && !cpuRuntime.completedAttackTeamIds.includes(team.id))}
        onStateChange={setState}
        cpuSettingsControls={<CpuControlPanel
          view="settings"
          teams={state.teams}
          settings={cpuSettings}
          onControllerChange={(teamId: string, controller: TeamController) => {
            setCpuSettings((current) => ({ ...current, [teamId]: controller }));
            if (controller === "random_cpu" && !initialStrategistRolesLocked) {
              const role = seededInitialRole(teamId);
              setState((current) => ({ ...current, units: current.units.map((unit) => unit.ownerTeamId === teamId && unit.type === "strategist" && unit.id.startsWith("home-") ? { ...unit, role } : unit) }));
            }
            const reset = createCpuRuntime(cpuRuntime.seed);
            runtimeRef.current = reset;
            setCpuRuntime(reset);
          }}
          initialStrategistRoles={initialStrategistRoles}
          initialStrategistRolesLocked={initialStrategistRolesLocked}
          onInitialStrategistRoleChange={(unitId, role) => {
            if (initialStrategistRolesLocked) return;
            setState((current) => ({ ...current, units: current.units.map((unit) => unit.id === unitId ? { ...unit, role } : unit) }));
          }}
          running={cpuRunning}
          paused={cpuPaused}
          onStart={() => { setCpuPaused(false); setCpuRunning(true); }}
          onPause={() => setCpuPaused(true)}
          onResume={() => setCpuPaused(false)}
          onStep={advanceCpu}
          speed={cpuSpeed}
          onSpeedChange={setCpuSpeed}
          currentCpuTeamId={state.phase === "movement_input" && state.currentMovementTeamId && cpuSettings[state.currentMovementTeamId] === "random_cpu" ? state.currentMovementTeamId : cpuRuntime.logs.at(-1)?.teamId}
          seed={cpuRuntime.seed}
          onSeedChange={(seed) => {
            const reset = createCpuRuntime(seed);
            runtimeRef.current = reset;
            setCpuRuntime(reset);
          }}
          logs={cpuRuntime.logs}
          stoppedReason={cpuRuntime.stoppedReason}
        />}
        cpuLogControls={<CpuControlPanel
          view="logs"
          teams={state.teams}
          settings={cpuSettings}
          onControllerChange={() => undefined}
          running={cpuRunning}
          paused={cpuPaused}
          onStart={() => undefined}
          onPause={() => undefined}
          onResume={() => undefined}
          onStep={() => undefined}
          speed={cpuSpeed}
          onSpeedChange={() => undefined}
          currentCpuTeamId={cpuRuntime.logs.at(-1)?.teamId}
          seed={cpuRuntime.seed}
          onSeedChange={() => undefined}
          logs={cpuRuntime.logs}
          stoppedReason={cpuRuntime.stoppedReason}
        />}
      />
    </main>
  );
}
