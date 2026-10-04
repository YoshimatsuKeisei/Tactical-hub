import { webcrypto } from "node:crypto";
import { IDBFactory } from "fake-indexeddb";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getRandomCpuDecision } from "../cpu/randomCpuPolicy";
import { createVisualCpuPolicyRouter } from "../cpu/cpuPolicyRouter";
import { createCpuRuntime, type CpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { advanceVisualCpuOneStep } from "../cpu/visualCpuRunner";
import { commitUnitMovement, getMovementCandidates } from "../engine/movement";
import { saveAttackIntent } from "../engine/battle";
import { createInitialGameState } from "../initialState";
import { IndexedDbLocalGameSaveRepository } from "../save/indexedDbLocalGameSaveRepository";
import {
  assembleLocalGameStableSnapshot,
  installLocalGameSessionLifecycle,
  LocalGameSession,
  resumeLocalGameSession,
  shouldFlushLocalGameSnapshot,
  startLocalGameSession,
  type LocalGameStableSnapshotParts,
} from "../save/localGameSession";
import type {
  DeleteLocalGameSaveResult,
  GetLocalGameSaveResult,
  LocalGameSaveMetadataRecord,
  LocalGameSaveRepository,
  LocalGameSaveRepositoryError,
  LocalGameSaveRepositoryResult,
  PutLocalGameSaveResult,
} from "../save/localGameSaveStorageTypes";
import { LOCAL_GAME_SAVE_STORE_NAME, type LocalGameSaveStorageRecord } from "../save/localGameSaveStorageTypes";
import type { LocalMatchSaveV1 } from "../save/localGameSaveTypes";
import type { GameState, Unit, UnitPosition } from "../types";

const supportedSettings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "random_cpu",
};

function ok<T>(value: T): LocalGameSaveRepositoryResult<T> { return { ok: true, value }; }
function fail<T>(error: LocalGameSaveRepositoryError): LocalGameSaveRepositoryResult<T> { return { ok: false, error }; }

class MemoryRepository implements LocalGameSaveRepository {
  readonly saves = new Map<string, LocalMatchSaveV1>();
  readonly putCalls: LocalMatchSaveV1[] = [];
  readonly deleteCalls: string[] = [];
  putErrors: LocalGameSaveRepositoryError[] = [];
  deleteErrors: LocalGameSaveRepositoryError[] = [];
  putGate?: Promise<void>;
  getSource: "current" | "previous" = "current";

  async put(save: LocalMatchSaveV1): Promise<LocalGameSaveRepositoryResult<PutLocalGameSaveResult>> {
    this.putCalls.push(structuredClone(save));
    if (this.putGate) await this.putGate;
    const error = this.putErrors.shift();
    if (error) return fail(error);
    this.saves.set(save.saveId, structuredClone(save));
    return ok({ saveId: save.saveId, revision: this.putCalls.length });
  }

  async get(saveId: string): Promise<LocalGameSaveRepositoryResult<GetLocalGameSaveResult>> {
    const save = this.saves.get(saveId);
    return save ? ok({ save: structuredClone(save), source: this.getSource, revision: 1 }) : fail({ code: "NOT_FOUND", message: "missing" });
  }

  async list(): Promise<LocalGameSaveRepositoryResult<LocalGameSaveMetadataRecord[]>> {
    return ok([...this.saves.values()].map((save) => ({
      storageRecordVersion: 1,
      saveId: save.saveId,
      createdAt: save.createdAt,
      updatedAt: save.updatedAt,
      metadata: structuredClone(save.metadata),
    })));
  }

  async delete(saveId: string): Promise<LocalGameSaveRepositoryResult<DeleteLocalGameSaveResult>> {
    this.deleteCalls.push(saveId);
    const error = this.deleteErrors.shift();
    if (error) return fail(error);
    const deleted = this.saves.delete(saveId);
    return ok({ saveId, deleted });
  }
}

function fixedOptions(repository: LocalGameSaveRepository, cpuSettings = supportedSettings) {
  return {
    repository,
    cpuSettings,
    uuid: () => "session-save-1",
    clock: () => "2026-10-04T10:00:00.000Z",
    displayName: "Session fixture",
  };
}

function parts(
  revision: number,
  state = createInitialGameState(),
  runtime = createCpuRuntime(123),
  settings = supportedSettings,
  policy = createVisualCpuPolicyRouter(),
): LocalGameStableSnapshotParts {
  return {
    gameState: { revision, value: state },
    cpuRuntime: { revision, value: runtime },
    cpuSettings: { revision, value: settings },
    heuristicPolicyState: { revision, value: policy.snapshotHeuristicState() },
    resumeUi: { viewerTeamId: "team-1" },
  };
}

async function persistAndResume(state: GameState, runtime = createCpuRuntime(123), settings = supportedSettings, policy = createVisualCpuPolicyRouter()) {
  const repository = new MemoryRepository();
  const session = startLocalGameSession(fixedOptions(repository, settings), parts(1, state, runtime, settings, policy));
  await session.flush();
  const resumed = await resumeLocalGameSession({ repository, saveId: session.saveId!, clock: () => "2026-10-04T10:00:00.000Z" });
  if (!resumed.ok) throw new Error(`${resumed.error.code}: ${resumed.error.message}`);
  return { repository, session, resumed: resumed.value };
}

function setAttackPhase(state: GameState) {
  state.phase = state.turnState.phase = "attack_input";
  state.currentMovementTeamId = undefined;
  state.movementDefendedBaseIdsAtTeamStart = undefined;
  state.teleportIntents = [];
  state.movedUnitIdsThisMovementPhase = [];
}

function clearBaseSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId);
  if (slot) delete slot.unitId;
}

afterEach(() => {
  vi.useRealTimers();
});

describe("LocalGameSession autosave and resume orchestration", () => {
  it("creates a saveId and persists an initial supported LOCAL game", async () => {
    const repository = new MemoryRepository();
    const session = startLocalGameSession(fixedOptions(repository), parts(1));
    await session.flush();
    expect(session.saveId).toBe("session-save-1");
    expect(repository.saves.get(session.saveId!)).toBeDefined();
    expect(session.getStatus()).toMatchObject({ status: "saved", lastSavedRevision: 1 });
  });

  it("does not consume game RNG while generating the saveId", async () => {
    const repository = new MemoryRepository();
    const runtime = createCpuRuntime(7); runtime.rngState = 0xdeadbeef;
    const session = startLocalGameSession(fixedOptions(repository), parts(1, createInitialGameState(), runtime));
    await session.flush();
    expect(runtime.rngState).toBe(0xdeadbeef);
  });

  it("autosaves a committed Human movement after the debounce", async () => {
    vi.useFakeTimers();
    const repository = new MemoryRepository();
    const state = createInitialGameState();
    const session = startLocalGameSession(fixedOptions(repository), parts(1, state));
    await session.flush(); repository.putCalls.length = 0;
    const unit = state.units.find((entry) => entry.id === "home-1-strategist")!;
    const destination = getMovementCandidates(state, unit.id)[0];
    const moved = commitUnitMovement(state, { teamId: "team-1", unitId: unit.id, from: unit.position, to: destination, stay: false });
    expect(session.commit(parts(2, moved))).toBe(true);
    expect(repository.putCalls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(repository.putCalls.at(-1)?.payload.gameState).toEqual(moved);
  });

  it("coalesces multiple commits inside one debounce window to the latest snapshot", async () => {
    vi.useFakeTimers();
    const repository = new MemoryRepository();
    const runtime = createCpuRuntime(1);
    const session = startLocalGameSession(fixedOptions(repository), parts(1, createInitialGameState(), runtime));
    await session.flush(); repository.putCalls.length = 0;
    for (let revision = 2; revision <= 4; revision += 1) {
      const nextRuntime = { ...runtime, appliedStepCount: revision };
      session.commit(parts(revision, createInitialGameState(), nextRuntime));
    }
    await vi.advanceTimersByTimeAsync(1_000);
    expect(repository.putCalls).toHaveLength(1);
    expect(repository.putCalls[0].payload.cpuRuntime.appliedStepCount).toBe(4);
  });

  it("serializes writes and saves a newer revision after an in-flight write", async () => {
    const repository = new MemoryRepository();
    let release!: () => void;
    repository.putGate = new Promise<void>((resolve) => { release = resolve; });
    const session = new LocalGameSession(fixedOptions(repository));
    session.commit(parts(1), { immediate: true });
    const flushing = session.flush();
    await Promise.resolve();
    const newerRuntime = createCpuRuntime(1); newerRuntime.appliedStepCount = 2;
    session.commit(parts(2, createInitialGameState(), newerRuntime), { immediate: true });
    release(); repository.putGate = undefined;
    await flushing;
    expect(repository.putCalls.map((save) => save.payload.cpuRuntime.appliedStepCount)).toEqual([0, 2]);
    expect(repository.saves.get(session.saveId!)?.payload.cpuRuntime.appliedStepCount).toBe(2);
  });

  it("flushes immediately after a stable phase transition", async () => {
    const repository = new MemoryRepository();
    const first = createInitialGameState();
    const session = startLocalGameSession(fixedOptions(repository), parts(1, first));
    await session.flush(); repository.putCalls.length = 0;
    const attack = structuredClone(first); setAttackPhase(attack);
    expect(shouldFlushLocalGameSnapshot(first, attack)).toBe(true);
    session.commit(parts(2, attack));
    await session.flush();
    expect(repository.putCalls).toHaveLength(1);
  });

  it("rejects a snapshot whose state/runtime/policy revisions do not match", async () => {
    const repository = new MemoryRepository();
    const session = new LocalGameSession(fixedOptions(repository));
    const mismatched = parts(1); mismatched.cpuRuntime.revision = 2;
    expect(assembleLocalGameStableSnapshot(mismatched)).toBeUndefined();
    expect(session.commit(mismatched, { immediate: true })).toBe(false);
    await session.flush();
    expect(repository.putCalls).toHaveLength(0);
  });

  it("restores random CPU runtime and preserves its next decision and RNG", async () => {
    const state = createInitialGameState();
    const runtime = createCpuRuntime(77);
    const randomSettings = Object.fromEntries(state.teams.filter((team) => !team.isNeutral).map((team) => [team.id, "random_cpu"])) as CpuTeamSettings;
    const { resumed } = await persistAndResume(state, runtime, randomSettings);
    const uninterruptedRuntime = structuredClone(runtime) as CpuRuntime;
    const resumedRuntime = structuredClone(resumed.cpuRuntime) as CpuRuntime;
    expect(getRandomCpuDecision(state, uninterruptedRuntime, randomSettings)).toEqual(getRandomCpuDecision(resumed.state, resumedRuntime, resumed.cpuSettings));
    expect(resumedRuntime.rngState).toBe(uninterruptedRuntime.rngState);
  });

  it("restores heuristic target memory, next action, and RNG", async () => {
    const state = createInitialGameState(); state.productionCompletedTeamIdsThisTurn = ["team-1"];
    state.units.push({ id: "session-infantry", ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "tile", x: 4, y: 1 }, statuses: [] });
    const heuristicSettings: CpuTeamSettings = { "team-1": "heuristic_cpu", "team-2": "human", "team-3": "human", "team-4": "human" };
    const policy = createVisualCpuPolicyRouter(); const runtime = createCpuRuntime(404);
    policy(state, runtime, heuristicSettings);
    const savedPolicyState = policy.snapshotHeuristicState();
    const { resumed } = await persistAndResume(state, runtime, heuristicSettings, policy);
    expect(resumed.visualCpuPolicy.snapshotHeuristicState()).toEqual(savedPolicyState);
    const originalRuntime = structuredClone(runtime) as CpuRuntime;
    const resumedRuntime = structuredClone(resumed.cpuRuntime) as CpuRuntime;
    expect(resumed.visualCpuPolicy(resumed.state, resumedRuntime, resumed.cpuSettings)).toEqual(policy(state, originalRuntime, heuristicSettings));
    expect(resumedRuntime.rngState).toBe(originalRuntime.rngState);
  });

  it("restores a state after multiple immediate unit movements", async () => {
    let state = createInitialGameState();
    for (const unitId of ["home-1-strategist", "home-1-king"]) {
      const unit = state.units.find((entry) => entry.id === unitId)!;
      const destination = getMovementCandidates(state, unit.id)[0];
      state = commitUnitMovement(state, { teamId: "team-1", unitId, from: unit.position, to: destination, stay: false });
    }
    const { resumed } = await persistAndResume(state);
    expect(resumed.state).toEqual(state);
  });

  it("restores a partially-entered Human attack intent", async () => {
    let state = createInitialGameState(); setAttackPhase(state);
    state = saveAttackIntent(state, { teamId: "team-1", attackerUnitId: "home-1-king", pass: true });
    expect((await persistAndResume(state)).resumed.state.turnState.actionIntents).toEqual(state.turnState.actionIntents);
  });

  it("restores pending reward placement", async () => {
    const state = createInitialGameState(); state.phase = state.turnState.phase = "reward_placement"; state.currentMovementTeamId = undefined;
    state.phaseAfterRewards = "strategist_action_input";
    state.rewardPlacementRequests.push({ id: "session-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    const { resumed } = await persistAndResume(state);
    expect(resumed.state.rewardPlacementRequests).toEqual(state.rewardPlacementRequests);
    expect(resumed.state.phaseAfterRewards).toBe("strategist_action_input");
  });

  it("restores strategist action state", async () => {
    const state = createInitialGameState(); state.phase = state.turnState.phase = "strategist_action_input"; state.currentMovementTeamId = undefined; state.movementDefendedBaseIdsAtTeamStart = undefined;
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!; strategist.role = "builder";
    state.strategistActionIntents = [{ teamId: "team-1", strategistUnitId: strategist.id, action: "pass" }];
    state.strategistSubmittedTeamIds = ["team-1"];
    expect((await persistAndResume(state)).resumed.state.strategistActionIntents).toEqual(state.strategistActionIntents);
  });

  it("restores teleport intent and destination reservation", async () => {
    const state = createInitialGameState();
    const strategist = state.units.find((unit) => unit.id === "home-1-strategist")!; strategist.role = "teleporter";
    const home = state.bases.find((base) => base.id === "home-1")!; const slot = home.slots.find((entry) => !entry.unitId)!;
    const target: Unit = { id: "session-target", ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "base", baseId: home.id, slotId: slot.id }, statuses: [] };
    slot.unitId = target.id; state.units.push(target);
    state.teleportIntents = [{ teamId: "team-1", strategistUnitId: strategist.id, targetUnitId: target.id, to: { kind: "tile", x: 4, y: 1 } }];
    expect((await persistAndResume(state)).resumed.state.teleportIntents).toEqual(state.teleportIntents);
  });

  it("restores retreat state", async () => {
    const state = createInitialGameState(); const unit = state.units.find((entry) => entry.id === "home-1-strategist")!;
    clearBaseSlot(state, unit.position); unit.position = { kind: "tile", x: 4, y: 1 }; unit.statuses = [{ kind: "retreating", retreatTargetBaseId: "home-1", sourceId: "session" }];
    state.unitTurnFlags = [{ unitId: unit.id, battleTurnNumber: 1, positionAtBattleStart: unit.position, enemyBaseDistanceAtBattleStart: 2, enemyBaseWithin3AtBattleStart: true, retreatFriendlyBaseIdsAtEligibility: ["home-1"], retreatHostileBaseIdsAtEligibility: ["neutral-north"], wasAliveAtBattleStart: true, survivedPreviousBattle: true, attackedInPreviousBattle: false, wasTargetedInPreviousBattle: true, retreatEligible: true, retreatEligibilityReason: "fixture" }];
    const restored = (await persistAndResume(state)).resumed.state;
    expect(restored.unitTurnFlags).toEqual(state.unitTurnFlags);
    expect(restored.units.find((entry) => entry.id === unit.id)?.statuses).toEqual(unit.statuses);
  });

  it("restores siege and king campaign progress", async () => {
    const state = createInitialGameState();
    state.siegeStates = [{ baseId: "neutral-north", defendingTeamId: "neutral", active: true, defenderLossOccurred: true, fallCandidateTeamIds: ["team-1"], lastEffectiveAttackTurn: 1, teamRecords: [{ teamId: "team-1", defenderKills: 1, effectiveAttackTurns: 1 }] }];
    state.kingCampaignStates = [{ kingUnitId: "home-2-king", kingTeamId: "team-2", contributions: [{ teamId: "team-1", cumulativeDamage: 1, effectiveAttackTurns: 1 }] }];
    const restored = (await persistAndResume(state)).resumed.state;
    expect(restored.siegeStates).toEqual(state.siegeStates); expect(restored.kingCampaignStates).toEqual(state.kingCampaignStates);
  });

  it("returns a resumed session paused and does not advance it implicitly", async () => {
    vi.useFakeTimers();
    const state = createInitialGameState(); const runtime = createCpuRuntime(1);
    const { resumed } = await persistAndResume(state, runtime);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(resumed.cpuPaused).toBe(true);
    expect(resumed.state).toEqual(state); expect(resumed.cpuRuntime).toEqual(runtime);
  });

  it("exposes a previous-revision recovery source from the repository", async () => {
    const repository = new MemoryRepository();
    const session = startLocalGameSession(fixedOptions(repository), parts(1)); await session.flush();
    repository.getSource = "previous";
    const resumed = await resumeLocalGameSession({ repository, saveId: session.saveId! });
    expect(resumed.ok && resumed.value.recoverySource).toBe("previous");
  });

  it("resumes through the real Stage 2 previous fallback when current is corrupted", async () => {
    const factory = new IDBFactory();
    const databaseName = "stage-3-recovery";
    const repository = new IndexedDbLocalGameSaveRepository({ indexedDB: factory, crypto: webcrypto as unknown as Crypto, databaseName });
    const session = startLocalGameSession(fixedOptions(repository), parts(1)); await session.flush();
    const newerRuntime = createCpuRuntime(123); newerRuntime.appliedStepCount = 2;
    session.commit(parts(2, createInitialGameState(), newerRuntime), { immediate: true }); await session.flush();
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(databaseName); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    const transaction = database.transaction(LOCAL_GAME_SAVE_STORE_NAME, "readwrite");
    const store = transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME);
    const record = await new Promise<LocalGameSaveStorageRecord>((resolve, reject) => {
      const request = store.get(session.saveId!); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    record.current.sha256 = "0".repeat(64); store.put(record);
    await new Promise<void>((resolve, reject) => { transaction.oncomplete = () => resolve(); transaction.onabort = () => reject(transaction.error); });
    database.close();
    const resumed = await resumeLocalGameSession({ repository, saveId: session.saveId! });
    expect(resumed.ok && resumed.value.recoverySource).toBe("previous");
    expect(resumed.ok && resumed.value.cpuRuntime.appliedStepCount).toBe(0);
    await new Promise<void>((resolve) => { const request = factory.deleteDatabase(databaseName); request.onsuccess = () => resolve(); request.onerror = () => resolve(); });
  });

  it("keeps gameplay alive and reports quota errors", async () => {
    const repository = new MemoryRepository(); repository.putErrors.push({ code: "QUOTA_EXCEEDED", message: "full" });
    const state = createInitialGameState(); const session = startLocalGameSession(fixedOptions(repository), parts(1, state));
    await session.flush();
    expect(session.getStatus()).toMatchObject({ status: "error", lastError: { code: "QUOTA_EXCEEDED" } });
    expect(state).toEqual(createInitialGameState());
  });

  it("retains a failed transaction snapshot and retries on the next autosave request", async () => {
    vi.useFakeTimers();
    const repository = new MemoryRepository(); repository.putErrors.push({ code: "TRANSACTION_FAILED", message: "fixture" });
    const session = startLocalGameSession(fixedOptions(repository), parts(1)); await session.flush();
    expect(session.getStatus().status).toBe("error");
    const runtime = createCpuRuntime(1); runtime.appliedStepCount = 2;
    session.commit(parts(2, createInitialGameState(), runtime));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(session.getStatus()).toMatchObject({ status: "saved", lastSavedRevision: 2 });
    expect(repository.saves.get(session.saveId!)?.payload.cpuRuntime.appliedStepCount).toBe(2);
  });

  it("disables the session and performs no writes when any team uses bc_cpu", async () => {
    const repository = new MemoryRepository(); const bcSettings = { ...supportedSettings, "team-4": "bc_cpu" } as CpuTeamSettings;
    const session = startLocalGameSession(fixedOptions(repository, bcSettings), parts(1, createInitialGameState(), createCpuRuntime(1), bcSettings));
    await session.flush();
    expect(session.enabled).toBe(false); expect(session.saveId).toBeUndefined(); expect(session.getStatus().status).toBe("disabled"); expect(repository.putCalls).toHaveLength(0);
  });

  it("deletes the continuation save after a terminal state is committed", async () => {
    const repository = new MemoryRepository(); const session = startLocalGameSession(fixedOptions(repository), parts(1)); await session.flush();
    const finished = createInitialGameState();
    for (const team of finished.teams) if (!team.isNeutral && team.id !== "team-1") team.status = "defeated";
    session.commit(parts(2, finished)); await session.flush();
    expect(repository.deleteCalls).toEqual([session.saveId]); expect(repository.saves.has(session.saveId!)).toBe(false);
  });

  it("keeps the terminal game result and reports a delete failure", async () => {
    const repository = new MemoryRepository(); const session = startLocalGameSession(fixedOptions(repository), parts(1)); await session.flush();
    repository.deleteErrors.push({ code: "TRANSACTION_FAILED", message: "delete fixture" });
    const finished = createInitialGameState(); for (const team of finished.teams) if (!team.isNeutral && team.id !== "team-1") team.status = "defeated";
    session.commit(parts(2, finished)); await session.flush();
    expect(finished.teams.filter((team) => !team.isNeutral && team.status === "active").map((team) => team.id)).toEqual(["team-1"]);
    expect(session.getStatus()).toMatchObject({ status: "error", lastError: { code: "TRANSACTION_FAILED" } });
  });

  it("does not consume RNG during repeated session autosaves", async () => {
    const repository = new MemoryRepository(); const runtime = createCpuRuntime(9); runtime.rngState = 0x12345678;
    const session = startLocalGameSession(fixedOptions(repository), parts(1, createInitialGameState(), runtime)); await session.flush();
    session.commit(parts(2, createInitialGameState(), runtime), { immediate: true }); await session.flush();
    expect(runtime.rngState).toBe(0x12345678);
  });

  it("keeps uninterrupted and save/resume CPU futures identical", async () => {
    const settings: CpuTeamSettings = { "team-1": "heuristic_cpu", "team-2": "random_cpu", "team-3": "random_cpu", "team-4": "random_cpu" };
    let uninterruptedState = createInitialGameState(); let uninterruptedRuntime = createCpuRuntime(2026); const uninterruptedPolicy = createVisualCpuPolicyRouter();
    for (let index = 0; index < 8; index += 1) {
      const result = advanceVisualCpuOneStep(uninterruptedState, uninterruptedRuntime, settings, uninterruptedPolicy);
      uninterruptedState = result.state; uninterruptedRuntime = result.runtime;
    }
    const { resumed } = await persistAndResume(uninterruptedState, uninterruptedRuntime, settings, uninterruptedPolicy);
    let resumedState = resumed.state; let resumedRuntime = resumed.cpuRuntime;
    for (let index = 0; index < 12; index += 1) {
      const left = advanceVisualCpuOneStep(uninterruptedState, uninterruptedRuntime, settings, uninterruptedPolicy);
      uninterruptedState = left.state; uninterruptedRuntime = left.runtime;
      const right = advanceVisualCpuOneStep(resumedState, resumedRuntime, resumed.cpuSettings, resumed.visualCpuPolicy);
      resumedState = right.state; resumedRuntime = right.runtime;
    }
    expect(resumedState).toEqual(uninterruptedState);
    expect(resumedRuntime).toEqual(uninterruptedRuntime);
    expect(resumed.visualCpuPolicy.snapshotHeuristicState()).toEqual(uninterruptedPolicy.snapshotHeuristicState());
  });

  it("best-effort flushes pending work on pagehide and hidden visibility", async () => {
    const repository = new MemoryRepository(); const session = new LocalGameSession(fixedOptions(repository));
    class Target extends EventTarget { visibilityState = "visible"; }
    const page = new Target(); const visibility = new Target();
    const detach = installLocalGameSessionLifecycle(session, page, visibility);
    session.commit(parts(1), { immediate: false }); page.dispatchEvent(new Event("pagehide")); await session.flush();
    expect(repository.putCalls).toHaveLength(1);
    const runtime = createCpuRuntime(1); runtime.appliedStepCount = 2; session.commit(parts(2, createInitialGameState(), runtime), { immediate: false });
    visibility.visibilityState = "hidden"; visibility.dispatchEvent(new Event("visibilitychange")); await session.flush();
    expect(repository.putCalls).toHaveLength(2); detach();
  });
});
