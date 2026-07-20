import { UNIT_STATS } from "../game/constants";
import { getAttackCandidates, saveAttackIntent } from "../game/engine/battle";
import { getEncourageRadius, getEncouragedUnitIds, getEncouragedUnitIdsByStrategist, isUnitEncouraged } from "../game/engine/encouragement";
import { getMovementCandidates, saveMovementIntent } from "../game/engine/movement";
import { getAvailableProductionTypes, saveProductionChoice, STRATEGIST_ROLES } from "../game/engine/production";
import { isTeamProductionPending } from "../game/engine/productionSchedule";
import { getPendingRewardRequests, placeRewardUnit } from "../game/engine/reward";
import {
  getNearestEnemyBaseDistance,
  getNearestFriendlyBaseDistance,
  getRetreatDebugInfo,
  getRetreatMoveEffect,
  getUnitTurnFlags,
  isRetreating,
  isUnitRetreatEligible,
} from "../game/engine/retreat";
import type { GameState } from "../game/types";
import { positionKey } from "../game/utils/position";
import { assignConstructionCapacityBonus, assignConstructionManager, getBridgeCandidates, getBuilderUnits, getManagedConstructions, getObstacleCandidates, resolveStrategistActions, saveStrategistActionIntent, submitStrategistActions } from "../game/engine/construction";
import { cancelTeleportIntent, getTeleportDestinationCandidates, getTeleportStrategists, getTeleportTargetCandidates, isTeleportAvailable, saveTeleportIntent } from "../game/engine/teleport";
import { useState, type ReactNode } from "react";

type Props = {
  state: GameState;
  selectedUnitId?: string;
  manualTeamId: string;
  onManualTeamChange: (teamId: string) => void;
  constructionMode?: "bridge" | "obstacle";
  onConstructionModeChange: (mode: "bridge" | "obstacle" | undefined) => void;
  onResolveMovement: () => void;
  onResolveBattle: () => void;
  onResolveProduction: () => void;
  onStateChange: (state: GameState) => void;
  battleResolveDisabled?: boolean;
  cpuSettingsControls?: ReactNode;
  cpuLogControls?: ReactNode;
};

export function GameDebugPanel({ state, selectedUnitId, manualTeamId, onManualTeamChange, constructionMode, onConstructionModeChange, onResolveMovement, onResolveBattle, onResolveProduction, onStateChange, battleResolveDisabled, cpuSettingsControls, cpuLogControls }: Props) {
  const [panelTab, setPanelTab] = useState<"settings" | "phase" | "logs">("settings");
  const [teleportTargets, setTeleportTargets] = useState<Record<string, string>>({});
  const selectedUnit = state.units.find((unit) => unit.id === selectedUnitId);
  const movementIntents = state.turnState.actionIntents.flatMap((intent) => intent.movementIntents);
  const attackIntents = state.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []);
  const productionIntents = state.turnState.actionIntents.flatMap((intent) => intent.productionChoices);
  const attackCandidates = selectedUnitId ? getAttackCandidates(state, selectedUnitId) : [];
  const movementCandidates = selectedUnitId ? getMovementCandidates(state, selectedUnitId) : [];
  const movementTeamId = state.phase === "movement_input" ? state.currentMovementTeamId : undefined;
  const activeTeam = state.teams.find((team) => team.id === (movementTeamId ?? manualTeamId) && team.status === "active") ?? state.teams.find((team) => team.status === "active")!;
  const currentMovementIndex = state.currentMovementTeamId ? state.movementOrderTeamIds.indexOf(state.currentMovementTeamId) : -1;
  const nextMovementTeamId = state.movementOrderTeamIds.slice(currentMovementIndex + 1).find((teamId) => !state.movementCompletedTeamIds.includes(teamId));
  const controlledBases = state.bases.filter((base) => activeTeam.controlledBaseIds.includes(base.id) || base.ownerTeamId === activeTeam.id);
  const encouragedUnitIds = getEncouragedUnitIds(state);
  const encourageStrategists = state.units.filter(
    (unit) => unit.type === "strategist" && unit.role === "encourage" && unit.position.kind !== "removed",
  );
  const selectedRetreatFlags = selectedUnit ? getUnitTurnFlags(state, selectedUnit.id) : undefined;
  const selectedRetreatDebug = selectedUnitId ? getRetreatDebugInfo(state, selectedUnitId) : undefined;
  const selectedFriendlyBaseDistance = selectedUnit
    ? getNearestFriendlyBaseDistance(state, selectedUnit.ownerTeamId, selectedUnit.position)
    : undefined;
  const selectedEnemyBaseDistance = selectedUnit ? getNearestEnemyBaseDistance(state, selectedUnit.ownerTeamId, selectedUnit.position) : undefined;
  const pendingRewards = getPendingRewardRequests(state);
  const productionPending = state.currentMovementTeamId
    ? isTeamProductionPending(state, state.currentMovementTeamId)
    : false;

  function attackIntentSummary(attackerUnitId: string, targetUnitId?: string) {
    if (!targetUnitId) return undefined;
    return getAttackCandidates(state, attackerUnitId).find((target) => target.unitId === targetUnitId);
  }

  function rewardLabel(rewardType: (typeof state.rewardPlacementRequests)[number]["rewardType"]) {
    if (rewardType === "capture_reward") return "占領褒賞";
    if (rewardType === "contribution_compensation") return "攻略功労補償";
    if (rewardType === "king_conquest_reward") return "王撃破褒賞";
    if (rewardType === "king_contribution_compensation") return "王攻略補償";
    return "占領褒賞変換補償";
  }

  return (
    <aside className="debug-panel">
      <nav className="debug-tabs" aria-label="Right panel tabs">
        <button className={panelTab === "settings" ? "primary" : "secondary"} onClick={() => setPanelTab("settings")}>プレイヤー構成・自動進行</button>
        <button className={panelTab === "phase" ? "primary" : "secondary"} onClick={() => setPanelTab("phase")}>フェーズ操作</button>
        <button className={panelTab === "logs" ? "primary" : "secondary"} onClick={() => setPanelTab("logs")}>ログ</button>
      </nav>
      {panelTab === "settings" ? cpuSettingsControls : null}
      {panelTab === "logs" ? cpuLogControls : null}
      <section className="status-card" hidden={panelTab !== "phase"}>
        <h2>Current Phase Actions</h2>
        <div className="status-grid">
          <span>Selected</span>
          <strong>{selectedUnit ? `${selectedUnit.id} (${selectedUnit.type})` : "none"}</strong>
          <span>Encouraged</span>
          <strong>{selectedUnit ? (isUnitEncouraged(state, selectedUnit) ? "yes" : "no") : "-"}</strong>
          <span>Retreat</span>
          <strong>
            {selectedUnit
              ? isRetreating(selectedUnit)
                ? "retreating"
                : isUnitRetreatEligible(state, selectedUnit)
                  ? "eligible"
                  : "no"
              : "-"}
          </strong>
          {state.phase === "movement_input" ? <>
            <span>Moves</span><strong>{movementIntents.length}</strong>
            <span>Movement order</span><strong>{state.movementOrderTeamIds.join(" → ") || "-"}</strong>
            <span>Current mover</span><strong>{state.currentMovementTeamId ?? "-"}</strong>
            <span>Completed</span><strong>{state.movementCompletedTeamIds.join(", ") || "none"}</strong>
            <span>Next mover</span><strong>{nextMovementTeamId ?? "-"}</strong>
          </> : null}
          {state.phase === "attack_input" ? <><span>Attacks</span><strong>{attackIntents.length}</strong></> : null}
        </div>
        {state.phase === "movement_input" && !productionPending ? <button className="primary sticky-action" onClick={onResolveMovement} disabled={!state.currentMovementTeamId}>
          Confirm Movement / Pass ({state.currentMovementTeamId ?? "-"})
        </button> : null}
        {productionPending ? <p>生産を確定またはスキップしてから移動してください。</p> : null}
      </section>

      <section hidden={panelTab !== "phase"}>
        <h2>操作チーム</h2>
        <div className="button-row">
          {state.teams.filter((team) => team.status === "active").map((team) => (
            <button
              key={team.id}
              className={team.id === activeTeam.id ? "primary" : "secondary"}
              disabled={state.phase === "movement_input" && team.id !== state.currentMovementTeamId}
              onClick={() => onManualTeamChange(team.id)}
            >{team.name}</button>
          ))}
        </div>
      </section>

      {panelTab === "phase" && state.phase === "reward_placement" ? (
        <section>
          <h2>褒賞配置</h2>
          {pendingRewards.map((request) => (
            <div className="intent-item" key={request.id}>
              <strong>{rewardLabel(request.rewardType)}</strong>
              <span>対象: {request.teamId} / 発生元: {request.sourceKingUnitId ?? request.sourceBaseId}</span>
              {request.eligibleBaseIds.map((baseId) => (
                <div className="button-row" key={baseId}>
                  <span>{baseId}</span>
                  {getAvailableProductionTypes(state, request.teamId, baseId).map((unitType) => (
                    <button key={unitType} onClick={() => onStateChange(placeRewardUnit(state, request.id, baseId, unitType))}>{UNIT_STATS[unitType].label}</button>
                  ))}
                </div>
              ))}
            </div>
          ))}
        </section>
      ) : null}

      {panelTab === "phase" && state.phase === "movement_input" && state.currentMovementTeamId && !productionPending ? (
        <section>
          <h2>Teleport</h2>
          {getTeleportStrategists(state, state.currentMovementTeamId).map((strategist) => {
            const targets = getTeleportTargetCandidates(state, strategist.id);
            const destinations = getTeleportDestinationCandidates(state, strategist.id);
            const selectedTargetId = teleportTargets[strategist.id];
            const saved = state.teleportIntents.find((intent) => intent.strategistUnitId === strategist.id);
            return <div className="intent-item" key={strategist.id}>
              <strong>{strategist.id}: {isTeleportAvailable(state, strategist.id) ? "available" : "cooldown"}</strong>
              <span>targets: {targets.map((unit) => unit.id).join(", ") || "none"}</span>
              <div className="button-row">{targets.map((unit) => <button key={unit.id} onClick={() => setTeleportTargets((current) => ({ ...current, [strategist.id]: unit.id }))}>{unit.id}</button>)}</div>
              <span>destinations: {destinations.length}</span>
              {selectedTargetId ? <div className="button-row">{destinations.map((to) => <button key={positionKey(to)} onClick={() => onStateChange(saveTeleportIntent(state, { teamId: state.currentMovementTeamId!, strategistUnitId: strategist.id, targetUnitId: selectedTargetId, to }))}>{positionKey(to)}</button>)}</div> : null}
              <span>saved: {saved ? `${saved.targetUnitId} → ${positionKey(saved.to)}` : "none"}</span>
              {saved ? <button onClick={() => onStateChange(cancelTeleportIntent(state, strategist.id))}>Cancel Teleport</button> : null}
            </div>;
          })}
        </section>
      ) : null}

      <section hidden={panelTab !== "phase" || state.phase !== "movement_input" || productionPending}>
        <h2>Retreat</h2>
        {selectedUnit ? (
          <div className="intent-list">
            <div className="intent-item">
              <strong>{selectedUnit.id}</strong>
              <span>status: {isRetreating(selectedUnit) ? "retreating" : "normal"}</span>
              <span>eligible: {selectedRetreatDebug?.eligible ? "yes" : "no"}</span>
              <span>
                friendly base:{" "}
                {selectedFriendlyBaseDistance
                  ? `${selectedFriendlyBaseDistance.baseIds.join(", ")} / distance ${selectedFriendlyBaseDistance.distance}`
                  : "none"}
              </span>
              <span>
                enemy base:{" "}
                {selectedEnemyBaseDistance
                  ? `${selectedEnemyBaseDistance.baseIds.join(", ")} / distance ${selectedEnemyBaseDistance.distance}`
                  : "none"}
              </span>
              {selectedRetreatFlags ? (
                <>
                  <span>
                    last battle: attacked {selectedRetreatFlags.attackedInPreviousBattle ? "yes" : "no"} / targeted{" "}
                    {selectedRetreatFlags.wasTargetedInPreviousBattle ? "yes" : "no"} / survived{" "}
                    {selectedRetreatFlags.survivedPreviousBattle ? "yes" : "no"}
                  </span>
                  {selectedRetreatDebug ? (
                    <>
                      <span>
                        debug: attacker {selectedRetreatDebug.wasAttacker ? "yes" : "no"} / targeted{" "}
                        {selectedRetreatDebug.wasTargeted ? "yes" : "no"} / participated{" "}
                        {selectedRetreatDebug.participatedInBattle ? "yes" : "no"}
                      </span>
                      <span>battle position kind: {selectedRetreatDebug.battlePositionKind ?? "-"}</span>
                      <span>
                        nearest hostile base: {selectedRetreatDebug.nearestHostileBaseId ?? "-"} / controller:{" "}
                        {selectedRetreatDebug.nearestHostileBaseController ?? "-"}
                      </span>
                      <span>
                        hostile base distance at battle: {selectedRetreatDebug.nearestHostileBaseDistanceAtBattle ?? "-"} / within 3:{" "}
                        {selectedRetreatDebug.withinHostileBaseRangeAtBattle ? "yes" : "no"}
                      </span>
                      <span>failure reasons: {selectedRetreatDebug.failureReasons.join(", ") || "-"}</span>
                    </>
                  ) : null}
                  <span>
                    eligibility reason: {selectedRetreatFlags.retreatEligibilityReason ?? (selectedRetreatFlags.retreatEligible ? "eligible" : "unknown")}
                  </span>
                  <span>
                    enemy base within 3 at battle: {selectedRetreatFlags.enemyBaseWithin3AtBattleStart ? "yes" : "no"} / battle distance:{" "}
                    {selectedRetreatFlags.enemyBaseDistanceAtBattleStart ?? "-"}
                  </span>
                  <span>position at battle: {selectedRetreatFlags.positionAtBattleStart ? positionKey(selectedRetreatFlags.positionAtBattleStart) : "-"}</span>
                </>
              ) : (
                <span>last battle: none</span>
              )}
            </div>
            {isRetreating(selectedUnit) ? (
              <button
                className="secondary"
                onClick={() =>
                  onStateChange(
                    saveMovementIntent(state, {
                      teamId: selectedUnit.ownerTeamId,
                      unitId: selectedUnit.id,
                      from: selectedUnit.position,
                      to: selectedUnit.position,
                      stay: true,
                    }),
                  )
                }
              >
                Save Stay / End Retreat
              </button>
            ) : null}
            {movementCandidates.length ? (
              movementCandidates.map((position) => (
                <div key={positionKey(position)} className="intent-item">
                  <strong>{positionKey(position)}</strong>
                  <span>retreat: {getRetreatMoveEffect(state, selectedUnit, selectedUnit.position, position)}</span>
                </div>
              ))
            ) : (
              <p>No movement candidates.</p>
            )}
          </div>
        ) : (
          <p>No unit selected.</p>
        )}
      </section>

      <section hidden={panelTab !== "settings"}>
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

      <section hidden={panelTab !== "phase" || (state.phase !== "movement_input" && state.phase !== "attack_input")}>
        <h2>Encourage</h2>
        <div className="intent-list">
          {encourageStrategists.length ? (
            encourageStrategists.map((strategist) => (
              <div key={strategist.id} className="intent-item">
                <strong>{strategist.id}</strong>
                <span>role: encourage / radius: {getEncourageRadius(state, strategist)}</span>
                <span>targets: {getEncouragedUnitIdsByStrategist(state, strategist).join(", ") || "none"}</span>
              </div>
            ))
          ) : (
            <p>No encourage strategists.</p>
          )}
          <p>Encouraged units: {[...encouragedUnitIds].sort((a, b) => a.localeCompare(b)).join(", ") || "none"}</p>
        </div>
      </section>

      <section hidden={panelTab !== "phase" || (state.phase !== "production" && !productionPending)}>
        <h2>Production</h2>
        {controlledBases.map((base) => {
          const types = getAvailableProductionTypes(state, activeTeam.id, base.id);
          return (
            <div key={base.id} className="production-row">
              <strong>{base.id}</strong>
              <div className="button-row">
                {types.length ? (
                  types.flatMap((unitType) => unitType === "strategist"
                    ? STRATEGIST_ROLES.map((strategistRole) => (
                      <button key={`${unitType}-${strategistRole}`} onClick={() => onStateChange(saveProductionChoice(state, { teamId: activeTeam.id, baseId: base.id, unitType, strategistRole }))}>
                        {UNIT_STATS[unitType].label} ({strategistRole})
                      </button>
                    ))
                    : [<button key={unitType} onClick={() => onStateChange(saveProductionChoice(state, { teamId: activeTeam.id, baseId: base.id, unitType }))}>{UNIT_STATS[unitType].label}</button>])
                ) : (
                  <span>no slot</span>
                )}
              </div>
            </div>
          );
        })}
        <button className="primary" onClick={onResolveProduction} disabled={state.phase === "reward_placement"}>
          {productionPending ? `生産を確定／スキップ (${state.currentMovementTeamId})` : "Resolve Production"}
        </button>
      </section>

      <section hidden={panelTab !== "phase" || !["production", "movement_input", "attack_input"].includes(state.phase)}>
        <h2>Saved Intents</h2>
        <div className="intent-list">
          {state.phase === "movement_input" && movementIntents.length ? (
            movementIntents.map((intent) => (
              <div key={intent.unitId} className="intent-item">
                <strong>{intent.unitId}</strong>
                <span>{intent.stay ? "stay" : JSON.stringify(intent.to)}</span>
              </div>
            ))
          ) : state.phase === "movement_input" ? (
            <p>No movement intents saved.</p>
          ) : null}
          {state.phase === "production" || productionPending ? <p>Production intents: {productionIntents.length}</p> : null}
          {state.phase === "attack_input" ? <p>Attack intents: {attackIntents.length}</p> : null}
        </div>
        <details>
          <summary>Raw intents</summary>
          <pre>{JSON.stringify(state.phase === "production" ? { productionIntents } : state.phase === "movement_input" ? { movementIntents } : { attackIntents }, null, 2)}</pre>
        </details>
      </section>

      <section hidden={panelTab !== "phase" || state.phase !== "attack_input"}>
        <h2>Attack</h2>
        <div className="intent-list">
          {selectedUnit && isRetreating(selectedUnit) ? (
            <p>Retreating units cannot attack.</p>
          ) : attackCandidates.length ? (
            attackCandidates.map((target) => {
              const targetUnit = state.units.find((unit) => unit.id === target.unitId);
              return (
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
                  <strong>
                    1/{target.finalSuccessDenominator ?? "-"} {target.unitId}
                  </strong>
                  <span>
                    {targetUnit ? `${targetUnit.type} HP:${targetUnit.hp}` : "unknown"} /{" "}
                    {target.baseId ? `${target.baseId}/${target.slotId}` : targetUnit?.position.kind ?? "board"}
                  </span>
                  <span>
                    base: 1/{target.baseSuccessDenominator ?? "-"} / encouraged: {target.encouraged ? "yes" : "no"}
                  </span>
                </button>
              );
            })
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
            attackIntents.map((intent) => {
              const summary = attackIntentSummary(intent.attackerUnitId, intent.target?.unitId);
              return (
                <div key={intent.attackerUnitId} className="intent-item">
                  <strong>{intent.attackerUnitId}</strong>
                  <span>{intent.pass ? "no attack" : intent.target?.unitId}</span>
                  {summary ? (
                    <span>
                      base: 1/{summary.baseSuccessDenominator} / encouraged: {summary.encouraged ? "yes" : "no"} / final: 1/
                      {summary.finalSuccessDenominator}
                    </span>
                  ) : null}
                </div>
              );
            })
          ) : (
            <p>No attack intents saved.</p>
          )}
        </div>
        <button className="primary battle-action" onClick={onResolveBattle} disabled={state.phase === "reward_placement" || battleResolveDisabled}>
          Resolve Battle
        </button>
      </section>

      <section hidden={panelTab !== "phase" || (state.phase !== "strategist_action_input" && state.phase !== "strategist_action_resolution")}>
        <h2>Strategist Actions</h2>
        <p>Operating: {activeTeam.name}</p>
        {getBuilderUnits(state, activeTeam.id).length ? <>
          <p>盤面で建設型軍師を選択してから、設置する設備を選んでください。</p>
          <div className="button-row"><button className={constructionMode === "bridge" ? "primary" : "secondary"} onClick={() => onConstructionModeChange("bridge")}>Choose bridge on board</button><button className={constructionMode === "obstacle" ? "primary" : "secondary"} onClick={() => onConstructionModeChange("obstacle")}>Choose obstacle on board</button><button onClick={() => onConstructionModeChange(undefined)}>Clear board mode</button></div>
        </> : <p>This team has no builder strategist.</p>}
        {getBuilderUnits(state, activeTeam.id).map((builder) => {
          const bridges = getManagedConstructions(state, builder.id, "bridge");
          const obstacles = getManagedConstructions(state, builder.id, "obstacle");
          return <div className="intent-item" key={builder.id}>
            <strong>{builder.id}</strong>
            {getBridgeCandidates(state, builder.id).map((tiles) => <button key={JSON.stringify(tiles)} onClick={() => onStateChange(saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "place_bridge", tiles }))}>Bridge {tiles.map((cell) => `${cell.x},${cell.y}`).join("-")}</button>)}
            {bridges.map((bridge) => <button key={bridge.id} onClick={() => onStateChange(saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "reset_bridge", constructionId: bridge.id }))}>Reset bridge {bridge.id}</button>)}
            {getObstacleCandidates(state, builder.id).map((cell) => <button key={`${cell.x},${cell.y}`} onClick={() => onStateChange(saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "place_obstacle", tiles: [cell] }))}>Obstacle {cell.x},{cell.y}</button>)}
            {obstacles.map((obstacle) => <button key={obstacle.id} onClick={() => onStateChange(saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "reset_obstacle", constructionId: obstacle.id }))}>Reset obstacle {obstacle.id}</button>)}
            {(activeTeam.conqueredTeamIds?.length ?? 0) === 1 ? <button onClick={() => onStateChange(assignConstructionCapacityBonus(state, activeTeam.id, builder.id))}>Assign extra construction slots</button> : null}
            <button onClick={() => onStateChange(saveStrategistActionIntent(state, { teamId: builder.ownerTeamId, strategistUnitId: builder.id, action: "pass" }))}>Pass</button>
          </div>;
        })}
        {state.constructions.filter((construction) => construction.active && construction.ownerTeamId === activeTeam.id && !construction.managerUnitId).map((construction) => <div className="intent-item" key={`unmanaged-${construction.id}`}>
          <strong>Unmanaged {construction.kind}: {construction.id}</strong>
          {getBuilderUnits(state, activeTeam.id).map((builder) => <button key={builder.id} onClick={() => onStateChange(assignConstructionManager(state, construction.id, builder.id))}>Assign to {builder.id}</button>)}
        </div>)}
        <h3>Saved intents: {activeTeam.name}</h3>
        {state.strategistActionIntents.filter((intent) => intent.teamId === activeTeam.id).map((intent) => <div className="intent-item" key={intent.strategistUnitId}><strong>{intent.strategistUnitId}</strong><span>{intent.action}</span><span>{intent.tiles?.map((cell) => `${cell.x},${cell.y}`).join(" / ") ?? intent.constructionId ?? "-"}</span></div>)}
        {state.teams.filter((team) => team.status === "active").map((team) => { const submitted = state.strategistSubmittedTeamIds.includes(team.id); return <div className="intent-item" key={team.id}><strong>{team.name}</strong><span>{submitted ? "Submitted" : "Not submitted"}</span><button onClick={() => onStateChange(submitStrategistActions(state, team.id))} disabled={submitted || state.phase !== "strategist_action_input"}>{submitted ? "Submitted" : `Submit ${team.name}`}</button></div>; })}
        <button onClick={() => onStateChange(resolveStrategistActions(state))} disabled={!state.teams.filter((team) => team.status === "active").every((team) => state.strategistSubmittedTeamIds.includes(team.id)) || state.phase !== "strategist_action_resolution"}>Resolve Strategists</button>
      </section>

      <section hidden={panelTab !== "logs"}>
        <h2>State Summary</h2>
        <p>Units: {state.units.filter((unit) => unit.position.kind !== "removed").length}</p>
        <p>Bases: {state.bases.length}</p>
      </section>

      <section hidden={panelTab !== "logs"}>
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

      <section hidden={panelTab !== "logs"}>
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
