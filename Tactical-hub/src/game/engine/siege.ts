import { SIEGE_INACTIVITY_RESET_TURNS } from "../constants";
import type { GameState, SiegeState } from "../types";

export function getSiegeState(state: GameState, baseId: string) {
  return state.siegeStates.find((siege) => siege.baseId === baseId);
}

function ensureSiege(state: GameState, baseId: string, defendingTeamId: string): SiegeState {
  let siege = getSiegeState(state, baseId);
  if (!siege || siege.defendingTeamId !== defendingTeamId) {
    siege = { baseId, defendingTeamId, teamRecords: [], active: true, defenderLossOccurred: false, fallCandidateTeamIds: [] };
    state.siegeStates = [...state.siegeStates.filter((entry) => entry.baseId !== baseId), siege];
    state.logs.push({ id: `log-siege-start-${state.logs.length}`, turnNumber: state.turnNumber, type: "siege", message: `攻略状態開始: ${baseId}（守備 ${defendingTeamId}）`, relatedIds: [baseId, defendingTeamId] });
  }
  return siege;
}

function teamRecord(siege: SiegeState, teamId: string) {
  let record = siege.teamRecords.find((entry) => entry.teamId === teamId);
  if (!record) {
    record = { teamId, defenderKills: 0, effectiveAttackTurns: 0 };
    siege.teamRecords.push(record);
  }
  return record;
}

export function recordEffectiveBaseAttacks(state: GameState, attacks: { baseId: string; defendingTeamId: string; attackingTeamId: string }[]) {
  const unique = new Map(attacks.map((attack) => [`${attack.baseId}:${attack.attackingTeamId}`, attack]));
  for (const attack of unique.values()) {
    const siege = ensureSiege(state, attack.baseId, attack.defendingTeamId);
    teamRecord(siege, attack.attackingTeamId).effectiveAttackTurns += 1;
    siege.lastEffectiveAttackTurn = state.turnNumber;
    siege.active = true;
    state.logs.push({ id: `log-siege-attack-${state.logs.length}`, turnNumber: state.turnNumber, type: "siege", message: `有効攻撃ターン記録: ${attack.attackingTeamId} → ${attack.baseId}`, relatedIds: [attack.baseId, attack.attackingTeamId] });
  }
}

export function recordDefenderKill(state: GameState, baseId: string, defendingTeamId: string, attackingTeamIds: string[]) {
  const siege = ensureSiege(state, baseId, defendingTeamId);
  for (const teamId of new Set(attackingTeamIds)) teamRecord(siege, teamId).defenderKills += 1;
  siege.defenderLossOccurred = true;
  state.logs.push({ id: `log-siege-kill-${state.logs.length}`, turnNumber: state.turnNumber, type: "siege", message: `拠点内守備駒撃破数更新: ${baseId} / ${[...new Set(attackingTeamIds)].join(", ")}`, relatedIds: [baseId, ...attackingTeamIds] });
}

export function resetSiegeState(state: GameState, baseId: string, reason: "inactivity" | "owner_changed") {
  if (!getSiegeState(state, baseId)) return;
  state.siegeStates = state.siegeStates.filter((entry) => entry.baseId !== baseId);
  state.logs.push({ id: `log-siege-reset-${state.logs.length}`, turnNumber: state.turnNumber, type: "siege", message: `${reason === "inactivity" ? "10ターン無攻撃" : "所有者変更"}による攻略状態リセット: ${baseId}`, relatedIds: [baseId] });
}

export function resetInactiveSieges(state: GameState) {
  for (const siege of [...state.siegeStates]) {
    if (siege.lastEffectiveAttackTurn !== undefined && state.turnNumber - siege.lastEffectiveAttackTurn >= SIEGE_INACTIVITY_RESET_TURNS) resetSiegeState(state, siege.baseId, "inactivity");
  }
}
