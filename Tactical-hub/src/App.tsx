import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BoardView } from "./components/BoardView";
import { CpuControlPanel, type CpuRunnerSpeed } from "./components/CpuControlPanel";
import { GameDebugPanel } from "./components/GameDebugPanel";
import { getAttackCandidates, getTeamAttackCandidates, saveAttackIntent } from "./game/engine/battle";
import { commitUnitMovement } from "./game/engine/movement";
import { resolveProduction, submitTeamProduction } from "./game/engine/production";
import { isRetreating } from "./game/engine/retreat";
import { createInitialGameState } from "./game/initialState";
import type { AttackTarget, StrategistRole, UnitPosition } from "./game/types";
import { saveStrategistActionIntent } from "./game/engine/construction";
import { resolveBattleWithHiddenCpuIntents, resolveLocalMovement, resolveLocalStrategistActions } from "./game/cpu/visualCpuRunner";
import { createCpuRuntime, type CpuRuntime, type CpuTeamSettings, type TeamController } from "./game/cpu/types";
import { createVisualCpuPolicyRouter, isCpuController } from "./game/cpu/cpuPolicyRouter";
import { createTeamVisibleState, isUnitVisibleToTeam } from "./game/visibility";
import { HttpBrowserBcInferenceClient } from "./game/cpu/browserBcClient";
import { advanceVisualCpuOneStepWithBc } from "./game/cpu/browserBcPolicy";
import { AppNavigation, type AppScreen } from "./components/AppNavigation";
import { IndexedDbLocalGameSaveRepository } from "./game/save/indexedDbLocalGameSaveRepository";
import {
  installLocalGameSessionLifecycle,
  LocalGameSession,
  type ResumedLocalGameSession,
} from "./game/save/localGameSession";
import type { LocalGameSaveRepository } from "./game/save/localGameSaveStorageTypes";

const initialTeams = createInitialGameState().teams;

function createDefaultLocalCpuSettings(): CpuTeamSettings {
  return Object.fromEntries(initialTeams.filter((team) => !team.isNeutral).map((team, index) => [team.id, index === 0 ? "human" : "random_cpu"]));
}

export default function App() {
  const [screen, setScreen] = useState<AppScreen>("home");
  const [localCpuSettings, setLocalCpuSettings] = useState<CpuTeamSettings>(createDefaultLocalCpuSettings);
  const [localSaveRepository] = useState<LocalGameSaveRepository>(() => new IndexedDbLocalGameSaveRepository());
  const [resumedSession, setResumedSession] = useState<ResumedLocalGameSession>();

  if (screen === "play") return <PlayScreen initialCpuSettings={localCpuSettings} repository={localSaveRepository} resumedSession={resumedSession} />;

  return <AppNavigation
    screen={screen}
    teams={initialTeams}
    localCpuSettings={localCpuSettings}
    localSaveRepository={localSaveRepository}
    onNavigate={setScreen}
    onLocalCpuChange={(teamId, controller) => setLocalCpuSettings((current) => ({ ...current, [teamId]: controller }))}
    onResumeLocal={(session) => {
      setResumedSession(session);
      setLocalCpuSettings(session.cpuSettings);
      setScreen("play");
    }}
    onStartLocal={() => {
      setResumedSession(undefined);
      setScreen("play");
    }}
  />;
}

type PlayScreenProps = {
  initialCpuSettings: CpuTeamSettings;
  repository: LocalGameSaveRepository;
  resumedSession?: ResumedLocalGameSession;
};

export function PlayScreen({ initialCpuSettings, repository, resumedSession }: PlayScreenProps) {
  const [initial] = useState(() => {
    if (resumedSession) return {
      state: resumedSession.state,
      runtime: resumedSession.cpuRuntime,
      settings: resumedSession.cpuSettings,
      policy: resumedSession.visualCpuPolicy,
      session: resumedSession.session,
      paused: resumedSession.cpuPaused,
      resumed: true,
    };
    const state = createInitialGameState();
    const runtime = createCpuRuntime(1);
    const settings = { ...initialCpuSettings };
    return {
      state,
      runtime,
      settings,
      policy: createVisualCpuPolicyRouter(),
      session: new LocalGameSession({ repository, cpuSettings: settings }),
      paused: false,
      resumed: false,
    };
  });
  const [state, setState] = useState(initial.state);
  const [selectedUnitId, setSelectedUnitId] = useState<string>();
  const [manualTeamId, setManualTeamId] = useState("team-1");
  const [constructionMode, setConstructionMode] = useState<"bridge" | "obstacle">();
  const [cpuSettings, setCpuSettings] = useState<CpuTeamSettings>(initial.settings);
  const [cpuRuntime, setCpuRuntime] = useState<CpuRuntime>(initial.runtime);
  const [cpuRunning, setCpuRunning] = useState(false);
  const [cpuPaused, setCpuPaused] = useState(initial.paused);
  const [cpuSpeed, setCpuSpeed] = useState<CpuRunnerSpeed>("normal");
  const [visualCpuPolicy] = useState(() => initial.policy);
  const [localGameSession] = useState(() => initial.session);
  const [autosaveState, setAutosaveState] = useState(() => localGameSession.getStatus());
  const [bcInferenceClient] = useState(() => new HttpBrowserBcInferenceClient());
  const cpuAdvancePendingRef = useRef(false);
  const stateRef = useRef(state);
  const runtimeRef = useRef(cpuRuntime);
  const settingsRef = useRef(cpuSettings);
  const sessionRevisionRef = useRef(0);
  const initialAutosaveStartedRef = useRef(false);
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { runtimeRef.current = cpuRuntime; }, [cpuRuntime]);
  useEffect(() => { settingsRef.current = cpuSettings; }, [cpuSettings]);

  const commitStableLocalState = useCallback((nextState: typeof state, nextRuntime: CpuRuntime, nextSettings: CpuTeamSettings) => {
    const revision = ++sessionRevisionRef.current;
    stateRef.current = nextState;
    runtimeRef.current = nextRuntime;
    settingsRef.current = nextSettings;
    setState(nextState);
    setCpuRuntime(nextRuntime);
    setCpuSettings(nextSettings);
    localGameSession.commit({
      gameState: { revision, value: nextState },
      cpuRuntime: { revision, value: nextRuntime },
      cpuSettings: { revision, value: nextSettings },
      heuristicPolicyState: { revision, value: visualCpuPolicy.snapshotHeuristicState() },
      resumeUi: { viewerTeamId: "team-1" },
    });
  }, [localGameSession, visualCpuPolicy]);

  useEffect(() => {
    if (!initial.resumed && !initialAutosaveStartedRef.current) {
      initialAutosaveStartedRef.current = true;
      commitStableLocalState(stateRef.current, runtimeRef.current, settingsRef.current);
    }
    const detachLifecycle = installLocalGameSessionLifecycle(localGameSession);
    const unsubscribeStatus = localGameSession.subscribeStatus(setAutosaveState);
    return () => {
      unsubscribeStatus();
      detachLifecycle();
      localGameSession.dispose();
    };
  }, [commitStableLocalState, initial.resumed, localGameSession]);
  useEffect(() => {
    if (!initial.resumed) return;
    setCpuPaused(false);
    setCpuRunning(true);
  }, [initial.resumed]);
  const selectedUnit = useMemo(
    () => state.units.find((unit) => unit.id === selectedUnitId),
    [selectedUnitId, state.units],
  );
  const effectiveManualTeamId = state.phase === "movement_input" && state.currentMovementTeamId
    ? state.currentMovementTeamId
    : manualTeamId;
  const visibleState = useMemo(() => {
    const visible = createTeamVisibleState(state, effectiveManualTeamId);
    if (state.phase !== "attack_input" || selectedUnit?.ownerTeamId !== effectiveManualTeamId
      || selectedUnit.type !== "ninja" || selectedUnit.position.kind !== "water") return visible;
    const visibleIds = new Set(visible.units.map((unit) => unit.id));
    const targetIds = new Set(getAttackCandidates(state, selectedUnit.id).map((target) => target.unitId));
    const revealedTargets = state.units.filter((unit) => targetIds.has(unit.id) && !visibleIds.has(unit.id));
    return revealedTargets.length ? { ...visible, units: [...visible.units, ...revealedTargets] } : visible;
  }, [effectiveManualTeamId, selectedUnit, state]);
  useEffect(() => {
    if (selectedUnit && !isUnitVisibleToTeam(state, selectedUnit, effectiveManualTeamId)) setSelectedUnitId(undefined);
  }, [effectiveManualTeamId, selectedUnit, state]);
  const initialStrategistRolesLocked = state.turnNumber !== 1 || state.movementCompletedTeamIds.length > 0 || state.productionCompletedTeamIdsThisTurn.length > 0 || state.movedUnitIdsThisMovementPhase.length > 0;
  const initialStrategistRoles = Object.fromEntries(state.units.filter((unit) => unit.type === "strategist" && unit.id.startsWith("home-")).map((unit) => [unit.id, unit.role ?? "encourage"])) as Record<string, StrategistRole>;

  function seededInitialRole(teamId: string) {
    const roles: StrategistRole[] = ["builder", "encourage", "teleporter"];
    const hash = [...teamId].reduce((value, character) => Math.imul(value ^ character.charCodeAt(0), 16777619) >>> 0, cpuRuntime.seed >>> 0);
    return roles[hash % roles.length];
  }

  const advanceCpu = useCallback(async () => {
    if (cpuAdvancePendingRef.current) return false;
    cpuAdvancePendingRef.current = true;
    const sourceState = stateRef.current;
    const sourceRuntime = runtimeRef.current;
    const sourceSettings = settingsRef.current;
    try {
      const result = await advanceVisualCpuOneStepWithBc(
        sourceState,
        sourceRuntime,
        sourceSettings,
        visualCpuPolicy,
        bcInferenceClient,
        () => stateRef.current === sourceState && runtimeRef.current === sourceRuntime && settingsRef.current === sourceSettings,
      );
      if (stateRef.current !== sourceState || runtimeRef.current !== sourceRuntime || settingsRef.current !== sourceSettings) return false;
      if (result.state !== sourceState || result.runtime !== sourceRuntime) commitStableLocalState(result.state, result.runtime, sourceSettings);
      if (result.runtime.stoppedReason) setCpuRunning(false);
      return result.applied;
    } finally {
      cpuAdvancePendingRef.current = false;
    }
  }, [bcInferenceClient, commitStableLocalState, visualCpuPolicy]);

  useEffect(() => {
    if (!cpuRunning || cpuPaused) return;
    const delay = state.phase === "attack_input" ? 10 : cpuSpeed === "normal" ? 700 : cpuSpeed === "fast" ? 150 : 10;
    const timer = window.setInterval(() => { void advanceCpu(); }, delay);
    return () => window.clearInterval(timer);
  }, [advanceCpu, cpuPaused, cpuRunning, cpuSpeed, state.phase]);

  function chooseDestination(position: UnitPosition) {
    if (!selectedUnit) return;
    if ((cpuSettings[selectedUnit.ownerTeamId] ?? "human") !== "human") return;
    commitStableLocalState(
      commitUnitMovement(state, {
        teamId: selectedUnit.ownerTeamId,
        unitId: selectedUnit.id,
        from: selectedUnit.position,
        to: position,
        stay: false,
      }), runtimeRef.current, settingsRef.current,
    );
  }

  function chooseAttackTarget(target: AttackTarget) {
    if (!selectedUnit) return;
    if ((cpuSettings[selectedUnit.ownerTeamId] ?? "human") !== "human") return;
    if (isRetreating(selectedUnit)) return;
    if (state.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []).some((intent) => intent.attackerUnitId === selectedUnit.id)) return;
    commitStableLocalState(
      saveAttackIntent(state, {
        teamId: selectedUnit.ownerTeamId,
        attackerUnitId: selectedUnit.id,
        target,
        pass: false,
      }), runtimeRef.current, settingsRef.current,
    );
    setSelectedUnitId(undefined);
  }

  useEffect(() => {
    if (state.phase !== "attack_input") return;
    let next = state;
    for (const team of state.teams.filter((entry) => entry.status === "active" && (cpuSettings[entry.id] ?? "human") === "human")) {
      for (const attacker of getTeamAttackCandidates(next, team.id).filter((entry) => entry.targets.length === 1)) {
        const alreadySaved = next.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []).some((intent) => intent.attackerUnitId === attacker.attackerUnitId);
        if (!alreadySaved) next = saveAttackIntent(next, { teamId: team.id, attackerUnitId: attacker.attackerUnitId, target: attacker.targets[0], pass: false });
      }
    }
    if (next !== state) commitStableLocalState(next, runtimeRef.current, settingsRef.current);
  }, [commitStableLocalState, cpuSettings, state]);

  function chooseDeterministicAttackTarget(unitId: string, targets: AttackTarget[]) {
    const hash = [...unitId].reduce((value, character) => Math.imul(value ^ character.charCodeAt(0), 16777619) >>> 0, cpuRuntime.rngState >>> 0);
    return targets[hash % targets.length];
  }

  function resolveBattleAfterCompletingHumanChoices() {
    let completed = state;
    for (const team of state.teams.filter((entry) => entry.status === "active" && (cpuSettings[entry.id] ?? "human") === "human")) {
      for (const attacker of getTeamAttackCandidates(completed, team.id).filter((entry) => entry.targets.length > 0)) {
        const alreadySaved = completed.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []).some((intent) => intent.attackerUnitId === attacker.attackerUnitId);
        if (!alreadySaved) completed = saveAttackIntent(completed, { teamId: team.id, attackerUnitId: attacker.attackerUnitId, target: chooseDeterministicAttackTarget(attacker.attackerUnitId, attacker.targets), pass: false });
      }
    }
    const resolved = resolveBattleWithHiddenCpuIntents(completed, cpuRuntime);
    commitStableLocalState(resolved.state, resolved.runtime, settingsRef.current);
    setSelectedUnitId(undefined);
  }

  useEffect(() => {
    if (state.phase !== "attack_input") return;
    const humanTeams = state.teams.filter((team) => team.status === "active" && (cpuSettings[team.id] ?? "human") === "human");
    const hasHumanChoice = humanTeams.some((team) => getTeamAttackCandidates(state, team.id).some((entry) => entry.targets.length > 1));
    const cpuTeams = state.teams.filter((team) => team.status === "active" && isCpuController(cpuSettings[team.id]));
    const cpuReady = cpuTeams.every((team) => cpuRuntime.completedAttackTeamIds.includes(team.id));
    if (!hasHumanChoice && cpuReady) resolveBattleAfterCompletingHumanChoices();
  }, [cpuRuntime.completedAttackTeamIds, cpuSettings, state]);

  return (
    <main className="app-shell">
      <div className="play-area">
        <header>
          <h1>Tactical Hub Phase 1</h1>
          <p>Local logic sandbox for map, base slots, production intents, and simultaneous movement resolution.</p>
          <div className={`autosave-status autosave-status-${autosaveState.status}`} role="status">
            {autosaveState.status === "saved" ? "保存済み" : null}
            {autosaveState.status === "pending" || autosaveState.status === "saving" ? "保存中…" : null}
            {autosaveState.status === "error" ? "自動保存に失敗しました。ゲームは続けられますが、現在の進行が保存されていない可能性があります。" : null}
            {autosaveState.status === "disabled" ? "このCPU構成では自動保存・再開は利用できません。" : null}
          </div>
          {initial.resumed && resumedSession?.recoverySource === "previous" ? <div className="resume-recovery-notice" role="status">
            最新の自動保存を読み込めなかったため、直前の正常な保存から復帰しました。
          </div> : null}
          <div className="turn-phase-banner">
            <span>ターン <strong>{state.turnNumber}</strong></span>
            <span>フェーズ <strong>{state.phase}</strong></span>
            {state.currentMovementTeamId ? <span>移動担当 <strong>{state.currentMovementTeamId}</strong></span> : null}
          </div>
        </header>
        <div className="board-scroll">
          <BoardView
            state={visibleState}
            selectedUnitId={selectedUnitId}
            onSelectUnit={setSelectedUnitId}
            onChooseDestination={chooseDestination}
            onChooseAttackTarget={chooseAttackTarget}
            manualTeamId={effectiveManualTeamId}
            constructionMode={constructionMode}
            onChooseConstruction={(unitId, kind, tiles) => {
              const unit = state.units.find((candidate) => candidate.id === unitId);
              if (!unit) return;
              if ((cpuSettings[unit.ownerTeamId] ?? "human") !== "human") return;
              commitStableLocalState(saveStrategistActionIntent(state, { teamId: unit.ownerTeamId, strategistUnitId: unit.id, action: kind === "bridge" ? "place_bridge" : "place_obstacle", tiles }), runtimeRef.current, settingsRef.current);
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
        onResolveProduction={() => commitStableLocalState(
          state.phase === "movement_input" && state.currentMovementTeamId
            ? submitTeamProduction(state, state.currentMovementTeamId)
            : resolveProduction(state),
          runtimeRef.current,
          settingsRef.current,
        )}
        onResolveMovement={() => {
          const resolved = resolveLocalMovement(state, cpuRuntime);
          commitStableLocalState(resolved.state, resolved.runtime, settingsRef.current);
          if (resolved.state.currentMovementTeamId) setManualTeamId(resolved.state.currentMovementTeamId);
          setSelectedUnitId(undefined);
        }}
        onResolveBattle={() => {
          resolveBattleAfterCompletingHumanChoices();
        }}
        onResolveStrategistActions={() => {
          const resolved = resolveLocalStrategistActions(state, cpuRuntime);
          commitStableLocalState(resolved.state, resolved.runtime, settingsRef.current);
        }}
        battleResolveDisabled={state.phase === "attack_input" && state.teams.some((team) => team.status === "active" && isCpuController(cpuSettings[team.id]) && !cpuRuntime.completedAttackTeamIds.includes(team.id))}
        manualUnitInteractionEnabled={Boolean(selectedUnit && (cpuSettings[selectedUnit.ownerTeamId] ?? "human") === "human")}
        onStateChange={(nextState) => commitStableLocalState(nextState, runtimeRef.current, settingsRef.current)}
        cpuSettingsControls={<CpuControlPanel
          view="settings"
          teams={state.teams}
          settings={cpuSettings}
          onControllerChange={(teamId: string, controller: TeamController) => {
            const nextSettings = { ...settingsRef.current, [teamId]: controller };
            let nextState = stateRef.current;
            if (isCpuController(controller) && !initialStrategistRolesLocked) {
              const role = seededInitialRole(teamId);
              nextState = { ...nextState, units: nextState.units.map((unit) => unit.ownerTeamId === teamId && unit.type === "strategist" && unit.id.startsWith("home-") ? { ...unit, role } : unit) };
            }
            const reset = createCpuRuntime(cpuRuntime.seed);
            commitStableLocalState(nextState, reset, nextSettings);
          }}
          initialStrategistRoles={initialStrategistRoles}
          initialStrategistRolesLocked={initialStrategistRolesLocked}
          onInitialStrategistRoleChange={(unitId, role) => {
            if (initialStrategistRolesLocked) return;
            const nextState = { ...stateRef.current, units: stateRef.current.units.map((unit) => unit.id === unitId ? { ...unit, role } : unit) };
            commitStableLocalState(nextState, runtimeRef.current, settingsRef.current);
          }}
          running={cpuRunning}
          paused={cpuPaused}
          onStart={() => { setCpuPaused(false); setCpuRunning(true); }}
          onPause={() => setCpuPaused(true)}
          onResume={() => setCpuPaused(false)}
          onStep={advanceCpu}
          speed={cpuSpeed}
          onSpeedChange={setCpuSpeed}
          currentCpuTeamId={state.phase === "movement_input" && state.currentMovementTeamId && isCpuController(cpuSettings[state.currentMovementTeamId]) ? state.currentMovementTeamId : cpuRuntime.logs.at(-1)?.teamId}
          seed={cpuRuntime.seed}
          onSeedChange={(seed) => {
            const reset = createCpuRuntime(seed);
            commitStableLocalState(stateRef.current, reset, settingsRef.current);
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
