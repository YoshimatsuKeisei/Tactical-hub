import { describe, expect, it } from "vitest";
import { UNIT_STATS } from "../constants";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { createCpuRuntime, type CpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { advanceVisualCpuOneStep } from "../cpu/visualCpuRunner";
import { commitUnitMovement, getMovementCandidates } from "../engine/movement";
import { createInitialGameState } from "../initialState";
import {
  createLocalMatchRestoreBundle,
  createLocalMatchSaveSnapshot,
  parseLocalMatchSave,
  serializeLocalMatchSave,
} from "../save/localGameSaveSerializer";
import { isLocalMatchContinuable, validateLocalMatchSave } from "../save/localGameSaveValidator";
import type { LocalMatchSaveResult, LocalMatchSaveV1 } from "../save/localGameSaveTypes";
import type { GameState, Unit, UnitPosition } from "../types";

const settings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "random_cpu",
};

function persisted<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function unwrap<T>(result: LocalMatchSaveResult<T>): T {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.path ?? "-"}: ${result.error.message}`);
  return result.value;
}

function expectError<T>(result: LocalMatchSaveResult<T>, code: string) {
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.error.code).toBe(code);
}

function createSave(
  gameState = createInitialGameState(),
  cpuRuntime = createCpuRuntime(12345),
  cpuSettings = settings,
  heuristicPolicyState = createHeuristicCpuPolicy().snapshotState(),
) {
  return unwrap(createLocalMatchSaveSnapshot({
    saveId: "local-save-test-1",
    createdAt: "2026-10-04T00:00:00.000Z",
    updatedAt: "2026-10-04T01:00:00.000Z",
    displayName: "Round-trip fixture",
    gameState,
    cpuSettings,
    cpuRuntime,
    heuristicPolicyState,
    resumeUi: { viewerTeamId: "team-1" },
  }));
}

function roundTrip(
  gameState: GameState,
  cpuRuntime = createCpuRuntime(12345),
  cpuSettings = settings,
  heuristicPolicyState = createHeuristicCpuPolicy().snapshotState(),
) {
  const save = createSave(gameState, cpuRuntime, cpuSettings, heuristicPolicyState);
  const json = unwrap(serializeLocalMatchSave(save));
  const parsed = unwrap(parseLocalMatchSave(json));
  const restored = unwrap(createLocalMatchRestoreBundle(parsed));
  return { save, json, parsed, restored };
}

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId);
  if (slot) delete slot.unitId;
}

function setAttackPhase(state: GameState) {
  state.phase = state.turnState.phase = "attack_input";
  state.currentMovementTeamId = undefined;
  state.movementDefendedBaseIdsAtTeamStart = undefined;
  state.teleportIntents = [];
  state.movedUnitIdsThisMovementPhase = [];
}

function cloneSave(save = createSave()) {
  return persisted(save) as LocalMatchSaveV1;
}

describe("LOCAL match save schema v1", () => {
  it("round-trips the complete initial GameState", () => {
    const state = createInitialGameState();
    const { restored } = roundTrip(state);
    expect(restored.gameState).toEqual(state);
    expect(restored.cpuSettings).toEqual(settings);
    expect(restored.resumeUi).toEqual({ viewerTeamId: "team-1" });
  });

  it("round-trips immediate movement progress and defended-base snapshot", () => {
    let state = createInitialGameState();
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    const mover = state.units.find((unit) => unit.id === "home-1-strategist")!;
    const destination = getMovementCandidates(state, mover.id)[0];
    expect(destination).toBeDefined();
    state = commitUnitMovement(state, { teamId: "team-1", unitId: mover.id, from: mover.position, to: destination, stay: false });
    const restored = roundTrip(state).restored.gameState;
    expect(restored.movedUnitIdsThisMovementPhase).toEqual(state.movedUnitIdsThisMovementPhase);
    expect(restored.movementSeatOrderTeamIds).toEqual(state.movementSeatOrderTeamIds);
    expect(restored.movementOrderTeamIds).toEqual(state.movementOrderTeamIds);
    expect(restored.currentMovementTeamId).toBe(state.currentMovementTeamId);
    expect(restored.movementDefendedBaseIdsAtTeamStart).toEqual(state.movementDefendedBaseIdsAtTeamStart);
  });

  it("round-trips a Human attack intent and hidden CPU attack intents", () => {
    const state = createInitialGameState(); setAttackPhase(state);
    state.turnState.actionIntents = [{
      teamId: "team-1",
      productionChoices: [], movementIntents: [],
      attackIntents: [{ teamId: "team-1", attackerUnitId: "home-1-king", pass: true }],
    }];
    const runtime = createCpuRuntime(88);
    runtime.hiddenAttackIntents = [{ teamId: "team-2", attackerUnitId: "home-2-king", pass: true }];
    runtime.completedAttackTeamIds = ["team-2"];
    const restored = roundTrip(state, runtime).restored;
    expect(restored.gameState.turnState.actionIntents).toEqual(state.turnState.actionIntents);
    expect(restored.cpuRuntime.hiddenAttackIntents).toEqual(runtime.hiddenAttackIntents);
  });

  it("round-trips pending reward placement and its resume phase", () => {
    const state = createInitialGameState();
    state.phase = state.turnState.phase = "reward_placement";
    state.currentMovementTeamId = undefined;
    state.phaseAfterRewards = "strategist_action_input";
    state.rewardPlacementRequests.push({
      id: "save-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1",
      destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false,
    });
    const restored = roundTrip(state).restored.gameState;
    expect(restored.rewardPlacementRequests).toEqual(state.rewardPlacementRequests);
    expect(restored.phaseAfterRewards).toBe("strategist_action_input");
  });

  it("round-trips bridge, obstacle, manager, and cooldown state", () => {
    const state = createInitialGameState();
    const manager = state.units.find((unit) => unit.id === "home-1-strategist")!; manager.role = "builder";
    state.constructions = [
      { id: "save-bridge", kind: "bridge", ownerTeamId: "team-1", managerUnitId: manager.id, tiles: [{ x: 4, y: 2 }], placedTurn: 1, active: true },
      { id: "save-obstacle", kind: "obstacle", ownerTeamId: "team-1", tiles: [{ x: 4, y: 1 }], placedTurn: 1, active: true },
    ];
    state.strategistCooldowns = [{ strategistUnitId: manager.id, kind: "bridge", availableFromTurn: 6 }];
    const restored = roundTrip(state).restored.gameState;
    expect(restored.constructions).toEqual(state.constructions);
    expect(restored.strategistCooldowns).toEqual(state.strategistCooldowns);
  });

  it("round-trips teleport intent and destination reservation", () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!; strategist.role = "teleporter";
    const home = state.bases.find((base) => base.id === "home-1")!;
    const slot = home.slots.find((entry) => !entry.unitId)!;
    const target: Unit = { id: "save-teleport-target", ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "base", baseId: home.id, slotId: slot.id }, statuses: [] };
    slot.unitId = target.id; state.units.push(target);
    state.teleportIntents = [{ teamId: "team-1", strategistUnitId: strategist.id, targetUnitId: target.id, to: { kind: "tile", x: 4, y: 1 } }];
    const restored = roundTrip(state).restored.gameState;
    expect(restored.teleportIntents).toEqual(state.teleportIntents);
  });

  it("round-trips retreat status and turn flags without changing retreat rules", () => {
    const state = createInitialGameState();
    const unit = state.units.find((entry) => entry.id === "home-1-strategist")!;
    clearBaseSlot(state, unit.position); unit.position = { kind: "tile", x: 4, y: 1 };
    unit.statuses = [{ kind: "retreating", retreatTargetBaseId: "home-1", sourceId: "save-test" }];
    state.unitTurnFlags = [{
      unitId: unit.id, battleTurnNumber: 1, positionAtBattleStart: { kind: "tile", x: 4, y: 1 },
      enemyBaseDistanceAtBattleStart: 2, enemyBaseWithin3AtBattleStart: true,
      retreatFriendlyBaseIdsAtEligibility: ["home-1"], retreatHostileBaseIdsAtEligibility: ["neutral-north"],
      wasAliveAtBattleStart: true, survivedPreviousBattle: true, attackedInPreviousBattle: false,
      wasTargetedInPreviousBattle: true, retreatEligible: true, retreatEligibilityReason: "save fixture",
    }];
    const restored = roundTrip(state).restored.gameState;
    expect(restored.units.find((entry) => entry.id === unit.id)?.statuses).toEqual(unit.statuses);
    expect(restored.unitTurnFlags).toEqual(state.unitTurnFlags);
  });

  it("round-trips siege and king-campaign progress", () => {
    const state = createInitialGameState();
    state.siegeStates = [{
      baseId: "neutral-north", defendingTeamId: "neutral", active: true, defenderLossOccurred: true,
      fallCandidateTeamIds: ["team-1"], lastEffectiveAttackTurn: 1,
      teamRecords: [{ teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 }],
    }];
    state.kingCampaignStates = [{
      kingUnitId: "home-2-king", kingTeamId: "team-2",
      contributions: [{ teamId: "team-1", cumulativeDamage: 1, effectiveAttackTurns: 1 }],
    }];
    const restored = roundTrip(state).restored.gameState;
    expect(restored.siegeStates).toEqual(state.siegeStates);
    expect(restored.kingCampaignStates).toEqual(state.kingCampaignStates);
  });

  it("round-trips every CpuRuntime field including rngState", () => {
    const state = createInitialGameState(); setAttackPhase(state);
    const runtime = createCpuRuntime(99, 777);
    runtime.rngState = 0xdeadbeef;
    runtime.contextKey = "1:attack_input";
    runtime.processedKeys = ["attack:team-2:home-2-king"];
    runtime.completedProductionTeamIds = ["team-1"];
    runtime.completedAttackTeamIds = ["team-2"];
    runtime.hiddenAttackIntents = [{ teamId: "team-2", attackerUnitId: "home-2-king", pass: true }];
    runtime.logs = [{ id: "cpu-0", turnNumber: 1, phase: "attack_input", teamId: "team-2", action: "attack planned" }];
    runtime.appliedStepCount = 12;
    runtime.stoppedReason = "paused fixture";
    const restored = roundTrip(state, runtime).restored.cpuRuntime;
    expect(restored).toEqual(runtime);
  });

  it("round-trips heuristic target memory and preserves the next action and RNG", () => {
    const state = createInitialGameState();
    state.productionCompletedTeamIdsThisTurn = ["team-1"];
    state.units.push({ id: "a-save-infantry", ownerTeamId: "team-1", type: "infantry", hp: UNIT_STATS.infantry.hp, position: { kind: "tile", x: 4, y: 1 }, statuses: [] });
    const heuristicSettings: CpuTeamSettings = { "team-1": "heuristic_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    const policy = createHeuristicCpuPolicy();
    const runtime = createCpuRuntime(404);
    expect(policy(state, runtime, heuristicSettings)?.kind).toBe("movement");
    const target = policy.getTargetBaseId("team-1", runtime.seed);
    const restoredBundle = roundTrip(state, runtime, heuristicSettings, policy.snapshotState()).restored;
    const restoredPolicy = createHeuristicCpuPolicy(); restoredPolicy.restoreState(restoredBundle.heuristicPolicyState);
    const originalRuntime = structuredClone(runtime) as CpuRuntime;
    const restoredRuntime = structuredClone(restoredBundle.cpuRuntime) as CpuRuntime;
    expect(restoredPolicy.getTargetBaseId("team-1", runtime.seed)).toBe(target);
    expect(restoredPolicy(restoredBundle.gameState, restoredRuntime, heuristicSettings)).toEqual(policy(state, originalRuntime, heuristicSettings));
    expect(restoredRuntime.rngState).toBe(originalRuntime.rngState);
  });

  it("rejects bc_cpu settings as unsupported", () => {
    expectError(createLocalMatchSaveSnapshot({
      saveId: "bc", createdAt: "2026-10-04T00:00:00Z", updatedAt: "2026-10-04T00:00:00Z", displayName: "BC",
      gameState: createInitialGameState(), cpuSettings: { ...settings, "team-4": "bc_cpu" }, cpuRuntime: createCpuRuntime(1), heuristicPolicyState: { matches: [] },
    }), "UNSUPPORTED_CPU_CONTROLLER");
  });

  it("rejects an unknown save schema version", () => {
    const save = cloneSave() as unknown as { saveSchemaVersion: number };
    save.saveSchemaVersion = 999;
    expectError(validateLocalMatchSave(save), "UNSUPPORTED_SCHEMA_VERSION");
  });

  it("rejects an unknown game rules version", () => {
    const save = cloneSave() as unknown as { compatibility: { gameRulesVersion: number } };
    save.compatibility.gameRulesVersion = 999;
    expectError(validateLocalMatchSave(save), "UNSUPPORTED_RULES_VERSION");
  });

  it("rejects a map mismatch", () => {
    const save = cloneSave(); save.compatibility.mapId = "another-map";
    expectError(validateLocalMatchSave(save), "MAP_MISMATCH");
  });

  it("rejects an unknown map version", () => {
    const save = cloneSave() as unknown as { compatibility: { mapVersion: number } };
    save.compatibility.mapVersion = 999;
    expectError(validateLocalMatchSave(save), "MAP_MISMATCH");
  });

  it("rejects duplicate unit IDs", () => {
    const save = cloneSave(); save.payload.gameState.units.push(persisted(save.payload.gameState.units[0]));
    expectError(validateLocalMatchSave(save), "INVALID_GAME_STATE");
  });

  it("rejects a base owned by a missing team", () => {
    const save = cloneSave(); save.payload.gameState.bases[0].ownerTeamId = "missing-team";
    expectError(validateLocalMatchSave(save), "INVALID_GAME_STATE");
  });

  it("rejects an out-of-bounds unit position", () => {
    const save = cloneSave();
    const unit = save.payload.gameState.units.find((entry) => entry.id === "home-1-strategist")!;
    clearBaseSlot(save.payload.gameState, unit.position);
    unit.position = { kind: "tile", x: save.payload.gameState.map.width, y: 0 };
    expectError(validateLocalMatchSave(save), "INVALID_GAME_STATE");
  });

  it("rejects a broken state/turnState phase relation", () => {
    const save = cloneSave(); save.payload.gameState.turnState.phase = "attack_input";
    expectError(validateLocalMatchSave(save), "INVALID_GAME_STATE");
  });

  it("rejects a CpuRuntime hidden intent with broken references", () => {
    const save = cloneSave();
    save.payload.cpuRuntime.hiddenAttackIntents = [{ teamId: "team-2", attackerUnitId: "missing-unit", pass: true }];
    expectError(validateLocalMatchSave(save), "INVALID_CPU_RUNTIME");
  });

  it("rejects heuristic memory that references a missing team or base", () => {
    const save = cloneSave();
    save.payload.heuristicPolicyState.matches = [{ seed: 1, lastTurn: 1, targetBaseIdByTeamId: { "missing-team": "home-1" } }];
    expectError(validateLocalMatchSave(save), "INVALID_HEURISTIC_STATE");
  });

  it("rejects a missing required gameplay field without applying a default", () => {
    const save = cloneSave();
    delete (save.payload.gameState as unknown as Record<string, unknown>).turnState;
    expectError(validateLocalMatchSave(save), "STRUCTURAL_ERROR");
  });

  it("safely rejects truncated or malformed JSON", () => {
    expectError(parseLocalMatchSave('{"saveSchemaVersion":1,"payload":'), "MALFORMED_JSON");
  });

  it("does not consume rngState while snapshotting, serializing, parsing, validating, or restoring", () => {
    const runtime = createCpuRuntime(123456); const before = runtime.rngState;
    const save = createSave(createInitialGameState(), runtime);
    expect(runtime.rngState).toBe(before);
    const json = unwrap(serializeLocalMatchSave(save)); expect(runtime.rngState).toBe(before);
    const parsed = unwrap(parseLocalMatchSave(json)); expect(runtime.rngState).toBe(before);
    unwrap(createLocalMatchRestoreBundle(parsed)); expect(runtime.rngState).toBe(before);
  });

  it("rejects Map, Set, function, or other non-JSON payload values instead of silently dropping them", () => {
    const save = cloneSave() as unknown as { payload: { cpuRuntime: { processedKeys: unknown } } };
    save.payload.cpuRuntime.processedKeys = new Set(["hidden"]);
    expectError(validateLocalMatchSave(save), "STRUCTURAL_ERROR");
  });

  it("derives metadata from payload and rejects contradictory metadata", () => {
    const save = cloneSave(); save.metadata.turnNumber += 1;
    expectError(validateLocalMatchSave(save), "INVALID_GAME_STATE");
  });

  it("identifies finished matches as non-continuable without duplicating storage behavior", () => {
    const state = createInitialGameState();
    for (const team of state.teams.filter((entry) => !entry.isNeutral && entry.id !== "team-1")) team.status = "defeated";
    expect(isLocalMatchContinuable(state)).toBe(false);
    expectError(createLocalMatchSaveSnapshot({
      saveId: "finished", createdAt: "2026-10-04T00:00:00Z", updatedAt: "2026-10-04T00:00:00Z", displayName: "Finished",
      gameState: state, cpuSettings: settings, cpuRuntime: createCpuRuntime(1), heuristicPolicyState: { matches: [] },
    }), "MATCH_ALREADY_FINISHED");
  });

  it("keeps a restored random CPU decision identical", () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(654321);
    const restored = roundTrip(state, runtime).restored;
    const firstRuntime = structuredClone(runtime) as CpuRuntime;
    expect(getRandomCpuDecision(restored.gameState, restored.cpuRuntime, settings)).toEqual(getRandomCpuDecision(state, firstRuntime, settings));
    expect(restored.cpuRuntime.rngState).toBe(firstRuntime.rngState);
  });

  it("accepts real supported-CPU snapshots across normal phase progression", () => {
    const allRandom: CpuTeamSettings = { "team-1": "random_cpu", "team-2": "random_cpu", "team-3": "random_cpu", "team-4": "random_cpu" };
    let state = createInitialGameState();
    let runtime = createCpuRuntime(20261004);
    let validated = 0;
    for (let step = 0; step < 180 && isLocalMatchContinuable(state); step += 1) {
      const result = advanceVisualCpuOneStep(state, runtime, allRandom);
      state = result.state; runtime = result.runtime;
      const snapshot = createLocalMatchSaveSnapshot({
        saveId: "phase-progression", createdAt: "2026-10-04T00:00:00Z", updatedAt: "2026-10-04T00:00:00Z", displayName: "Phase progression",
        gameState: state, cpuSettings: allRandom, cpuRuntime: runtime, heuristicPolicyState: { matches: [] },
      });
      expect(snapshot.ok, snapshot.ok ? undefined : `${snapshot.error.code}: ${snapshot.error.path}: ${snapshot.error.message}`).toBe(true);
      validated += 1;
      if (!result.applied) break;
    }
    expect(validated).toBeGreaterThan(20);
  });
});
