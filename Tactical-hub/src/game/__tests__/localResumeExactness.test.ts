import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { saveAttackIntent } from "../engine/battle";
import { completeSiegeCapture } from "../engine/capture";
import { createInitialGameState } from "../initialState";
import type { GameState, SiegeState, UnitPosition } from "../types";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { createVisualCpuPolicyRouter } from "../cpu/cpuPolicyRouter";
import { createLocalGameRng } from "../cpu/localGameRng";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { createCpuRuntime, type CpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { resolveBattleWithHiddenCpuIntents, resolveLocalStrategistActions } from "../cpu/visualCpuRunner";

const allHeuristic: CpuTeamSettings = {
  "team-1": "heuristic_cpu",
  "team-2": "heuristic_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "heuristic_cpu",
};
const allRandom: CpuTeamSettings = {
  "team-1": "random_cpu",
  "team-2": "random_cpu",
  "team-3": "random_cpu",
  "team-4": "random_cpu",
};

function persisted<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function battleSnapshot() {
  let state = createInitialGameState();
  state.phase = state.turnState.phase = "attack_input";
  const attacker = state.units.find((unit) => unit.id === "home-1-strategist")!;
  clearBaseSlot(state, attacker.position);
  attacker.type = "infantry";
  attacker.role = undefined;
  attacker.position = { kind: "tile", x: 4, y: 1 };
  state.units.push({
    id: "resume-battle-target",
    ownerTeamId: "team-2",
    type: "infantry",
    hp: UNIT_STATS.infantry.hp,
    position: { kind: "tile", x: 5, y: 1 },
    statuses: [],
  });
  state = saveAttackIntent(state, {
    teamId: "team-1",
    attackerUnitId: attacker.id,
    target: { kind: "unit", unitId: "resume-battle-target" },
    pass: false,
  });
  return { state, runtime: createCpuRuntime(0x1234abcd) };
}

function captureSnapshot() {
  const state = createInitialGameState();
  for (const unit of state.units.filter((entry) => entry.ownerTeamId === "team-1" || entry.ownerTeamId === "team-2")) {
    clearBaseSlot(state, unit.position);
    unit.hp = 0;
    unit.position = { kind: "removed", reason: "defeated" };
  }
  const siege: SiegeState = {
    baseId: "neutral-north",
    defendingTeamId: "neutral",
    active: true,
    defenderLossOccurred: true,
    fallCandidateTeamIds: ["team-1", "team-2"],
    teamRecords: [
      { teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 },
      { teamId: "team-2", defenderKills: 1, effectiveAttackTurns: 1 },
    ],
  };
  state.siegeStates.push(siege);
  return { state, runtime: createCpuRuntime(0x0badcafe) };
}

function resolveCapture(stateSnapshot: GameState, runtimeSnapshot: CpuRuntime) {
  const state = structuredClone(stateSnapshot) as GameState;
  const runtime = structuredClone(runtimeSnapshot) as CpuRuntime;
  const siege = state.siegeStates.find((entry) => entry.baseId === "neutral-north")!;
  completeSiegeCapture(state, siege, ["team-1", "team-2"], "annihilation", createLocalGameRng(runtime));
  return { state, runtime };
}

function strategistSnapshot() {
  const state = createInitialGameState();
  const builder = state.units.find((unit) => unit.id === "home-1-strategist")!;
  builder.role = "builder";
  state.constructions.push({
    id: "resume-bridge",
    kind: "bridge",
    ownerTeamId: "team-1",
    managerUnitId: builder.id,
    tiles: [{ x: 4, y: 2 }, { x: 4, y: 3 }],
    placedTurn: 1,
    active: true,
  });
  const king = state.units.find((unit) => unit.id === "home-2-king")!;
  clearBaseSlot(state, king.position);
  king.position = { kind: "bridge", bridgeId: "resume-bridge", cellIndex: 0 };
  state.phase = state.turnState.phase = "strategist_action_resolution";
  state.strategistActionIntents = [{
    teamId: "team-1",
    strategistUnitId: builder.id,
    action: "reset_bridge",
    constructionId: "resume-bridge",
  }];
  return { state, runtime: createCpuRuntime(0x2468ace0) };
}

describe("LOCAL exact resume foundation", () => {
  it("produces the same RNG state sequence for the same seed and action sequence", () => {
    const first = createCpuRuntime(1);
    const restored = persisted(first);
    const firstRng = createLocalGameRng(first);
    const restoredRng = createLocalGameRng(restored);
    const firstStates = Array.from({ length: 12 }, () => ({ value: firstRng(), rngState: first.rngState }));
    const restoredStates = Array.from({ length: 12 }, () => ({ value: restoredRng(), rngState: restored.rngState }));
    expect(restoredStates).toEqual(firstStates);
    expect(firstStates.slice(0, 3).map((entry) => entry.rngState)).toEqual([1015568748, 1586005467, 2165703038]);
  });

  it("restores immediately before battle with the same result and final RNG state", () => {
    const snapshot = battleSnapshot();
    const uninterrupted = resolveBattleWithHiddenCpuIntents(structuredClone(snapshot.state), structuredClone(snapshot.runtime));
    const restored = resolveBattleWithHiddenCpuIntents(persisted(snapshot.state), persisted(snapshot.runtime));
    expect(restored.state).toEqual(uninterrupted.state);
    expect(restored.runtime.rngState).toBe(uninterrupted.runtime.rngState);
    expect(restored.runtime.rngState).not.toBe(snapshot.runtime.rngState);
    expect(snapshot.runtime.rngState).toBe(0x1234abcd);
  });

  it("restores immediately before a capture tie-break with the same owner and RNG state", () => {
    const snapshot = captureSnapshot();
    const uninterrupted = resolveCapture(snapshot.state, snapshot.runtime);
    const restored = resolveCapture(persisted(snapshot.state), persisted(snapshot.runtime));
    expect(restored.state.bases.find((base) => base.id === "neutral-north")?.ownerTeamId)
      .toBe(uninterrupted.state.bases.find((base) => base.id === "neutral-north")?.ownerTeamId);
    expect(restored.runtime.rngState).toBe(uninterrupted.runtime.rngState);
    expect(restored.runtime.rngState).not.toBe(snapshot.runtime.rngState);
    expect(snapshot.runtime.rngState).toBe(0x0badcafe);
  });

  it("restores random CPU before a decision with the same next action and RNG state", () => {
    const state = createInitialGameState();
    const snapshot = createCpuRuntime(271828);
    const uninterruptedRuntime = structuredClone(snapshot) as CpuRuntime;
    const restoredRuntime = persisted(snapshot);
    const uninterrupted = getRandomCpuDecision(state, uninterruptedRuntime, allRandom);
    const restored = getRandomCpuDecision(structuredClone(state), restoredRuntime, allRandom);
    expect(restored).toEqual(uninterrupted);
    expect(restoredRuntime.rngState).toBe(uninterruptedRuntime.rngState);
  });

  it("snapshots and restores a selected heuristic target without changing the next action or RNG", () => {
    const state = createInitialGameState();
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    state.units.push({
      id: "a-resume-infantry",
      ownerTeamId: "team-1",
      type: "infantry",
      hp: UNIT_STATS.infantry.hp,
      position: { kind: "tile", x: 4, y: 1 },
      statuses: [],
    });
    const original = createHeuristicCpuPolicy();
    const originalRuntime = createCpuRuntime(424242);
    expect(original(state, originalRuntime, allHeuristic)?.kind).toBe("movement");
    const selectedTarget = original.getTargetBaseId("team-1", originalRuntime.seed);
    expect(selectedTarget).toBeDefined();

    const serializedPolicyState = persisted(original.snapshotState());
    const restored = createHeuristicCpuPolicy();
    restored.restoreState(serializedPolicyState);
    expect(restored.snapshotState()).toEqual(original.snapshotState());
    expect(restored.getTargetBaseId("team-1", originalRuntime.seed)).toBe(selectedTarget);

    const restoredRuntime = structuredClone(originalRuntime) as CpuRuntime;
    const nextOriginal = original(state, originalRuntime, allHeuristic);
    const nextRestored = restored(structuredClone(state), restoredRuntime, allHeuristic);
    expect(nextRestored).toEqual(nextOriginal);
    expect(restored.snapshotState()).toEqual(original.snapshotState());
    expect(restoredRuntime.rngState).toBe(originalRuntime.rngState);

    const router = createVisualCpuPolicyRouter({ heuristicPolicy: original });
    const restoredRouter = createVisualCpuPolicyRouter();
    restoredRouter.restoreHeuristicState(persisted(router.snapshotHeuristicState()));
    expect(restoredRouter.snapshotHeuristicState()).toEqual(router.snapshotHeuristicState());
  });

  it("restores strategist resolution with the same result and final RNG state", () => {
    const snapshot = strategistSnapshot();
    const uninterrupted = resolveLocalStrategistActions(structuredClone(snapshot.state), structuredClone(snapshot.runtime));
    const restored = resolveLocalStrategistActions(persisted(snapshot.state), persisted(snapshot.runtime));
    expect(restored.state).toEqual(uninterrupted.state);
    expect(restored.runtime.rngState).toBe(uninterrupted.runtime.rngState);
    expect(restored.runtime.rngState).not.toBe(snapshot.runtime.rngState);
    expect(snapshot.runtime.rngState).toBe(0x2468ace0);
  });

  it("keeps LOCAL human resolution entry points off unseeded Math.random defaults", () => {
    const app = readFileSync(new URL("../../App.tsx", import.meta.url), "utf8");
    const debugPanel = readFileSync(new URL("../../components/GameDebugPanel.tsx", import.meta.url), "utf8");
    expect(app).toContain("resolveLocalMovement(state, cpuRuntime)");
    expect(app).toContain("resolveLocalStrategistActions(state, cpuRuntime)");
    expect(app).not.toMatch(/resolveMovement\(state\)/);
    expect(debugPanel).not.toMatch(/resolveStrategistActions\(state\)/);
    expect(`${app}\n${debugPanel}`).not.toContain("Math.random");
  });
});
