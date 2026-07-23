import { describe, expect, it } from "vitest";
import { getAttackCandidates } from "../engine/battle";
import { getEnemyControlledBases } from "../engine/retreat";
import { createHeuristicCpuPolicy, createHeuristicDistanceEvaluator } from "../cpu/heuristicCpuPolicy";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";
import { UNIT_STATS } from "../constants";
import type { GameState, Unit, UnitPosition } from "../types";
import { getRoadAttackDistance } from "../utils/roadTopology";

const allCpu: CpuTeamSettings = { "team-1": "random_cpu", "team-2": "random_cpu", "team-3": "random_cpu", "team-4": "random_cpu" };

function movementState() {
  const state = createHeadlessInitialState(4);
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  return state;
}

function removeTeamUnitsExcept(state: GameState, teamId: string, keepId: string) {
  for (const unit of state.units.filter((entry) => entry.ownerTeamId === teamId && entry.id !== keepId)) {
    const currentPosition = unit.position;
    if (currentPosition.kind === "base") {
      const slot = state.bases.find((base) => base.id === currentPosition.baseId)?.slots.find((entry) => entry.id === currentPosition.slotId);
      if (slot?.unitId === unit.id) slot.unitId = undefined;
    }
    unit.hp = 0; unit.position = { kind: "removed", reason: "defeated" };
  }
}

function setPosition(state: GameState, unit: Unit, position: UnitPosition) {
  const currentPosition = unit.position;
  if (currentPosition.kind === "base") {
    const old = state.bases.find((base) => base.id === currentPosition.baseId)?.slots.find((slot) => slot.id === currentPosition.slotId);
    if (old?.unitId === unit.id) old.unitId = undefined;
  }
  unit.position = position;
}

describe("HeuristicCpuPolicy", () => {
  it("returns the exact existing road distance while reusing one reverse search per target", () => {
    const state = createHeadlessInitialState(4);
    state.constructions.push({ id: "distance-test-bridge", kind: "bridge", ownerTeamId: "team-1", tiles: [2, 3, 4, 5, 6].map((y) => ({ x: 7, y })), placedTurn: 1, active: true });
    const target = state.bases.find((base) => base.id === "neutral-north")!;
    const destination = { kind: "base", baseId: target.id, slotId: target.slots[0].id } as const;
    const positions: UnitPosition[] = [...state.map.tiles
      .filter((tile) => ["road", "baseGate", "reorganize"].includes(tile.terrain))
      .slice(0, 20)
      .map((tile): UnitPosition => ({ kind: "tile", x: tile.x, y: tile.y })),
      ...[0, 1, 2, 3, 4].map((cellIndex): UnitPosition => ({ kind: "bridge", bridgeId: "distance-test-bridge", cellIndex })),
      { kind: "base", baseId: "home-1", slotId: "distance-test" },
      { kind: "water", x: 8, y: 4 }];
    const evaluator = createHeuristicDistanceEvaluator(state);
    const untouchedRuntime = createCpuRuntime(1234);
    const rngBefore = untouchedRuntime.rngState;
    expect(positions.map((position) => evaluator.distance(position, target.id))).toEqual(positions.map((position) => getRoadAttackDistance(state, position, destination)));
    expect(evaluator.stats).toEqual({ requests: positions.length, searches: 1, hits: positions.length - 1, misses: 1 });
    expect(untouchedRuntime.rngState).toBe(rngBefore);
  });

  it("does not reuse a lookup across targets, decisions, state changes, or policy instances", () => {
    const state = createHeadlessInitialState(4);
    const position: UnitPosition = { kind: "tile", x: 4, y: 1 };
    const targets = state.bases.slice(0, 2);
    const firstDecision = createHeuristicDistanceEvaluator(state);
    for (const target of targets) {
      const expected = getRoadAttackDistance(state, position, { kind: "base", baseId: target.id, slotId: target.slots[0].id });
      expect(firstDecision.distance(position, target.id)).toBe(expected);
    }
    expect(firstDecision.stats).toEqual({ requests: 2, searches: 2, hits: 0, misses: 2 });

    state.constructions.push({ id: "new-decision-bridge", kind: "bridge", ownerTeamId: "team-1", tiles: [2, 3, 4, 5, 6].map((y) => ({ x: 7, y })), placedTurn: 1, active: true });
    const nextDecision = createHeuristicDistanceEvaluator(state);
    expect(nextDecision.distance(position, targets[0].id)).toBe(getRoadAttackDistance(state, position, { kind: "base", baseId: targets[0].id, slotId: targets[0].slots[0].id }));
    expect(nextDecision.stats).toEqual({ requests: 1, searches: 1, hits: 0, misses: 1 });
    expect(createHeuristicDistanceEvaluator(state).stats).toEqual({ requests: 0, searches: 0, hits: 0, misses: 0 });
  });

  it("selects a nearest neutral/enemy base, keeps it while valid, and reselects after capture", () => {
    const state = movementState();
    const unit = state.units.find((entry) => entry.id === "home-1-strategist")!;
    unit.role = "encourage";
    removeTeamUnitsExcept(state, "team-1", unit.id);
    const policy = createHeuristicCpuPolicy();
    const runtime = createCpuRuntime(91);
    policy(state, runtime, allCpu);
    const selected = policy.getTargetBaseId("team-1", 91)!;
    const candidates = getEnemyControlledBases(state, "team-1");
    const selectedBase = state.bases.find((base) => base.id === selected)!;
    const selectedDistance = getRoadAttackDistance(state, unit.position, { kind: "base", baseId: selected, slotId: selectedBase.slots[0].id });
    expect(selectedDistance).toBe(Math.min(...candidates.map((base) => getRoadAttackDistance(state, unit.position, { kind: "base", baseId: base.id, slotId: base.slots[0].id }))));

    setPosition(state, unit, { kind: "tile", x: 18, y: 1 });
    policy(state, runtime, allCpu);
    expect(policy.getTargetBaseId("team-1", 91)).toBe(selected);

    selectedBase.ownerTeamId = "team-1";
    state.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push(selected);
    policy(state, runtime, allCpu);
    expect(policy.getTargetBaseId("team-1", 91)).not.toBe(selected);
  });

  it("passes movement for the king and builder strategist", () => {
    const state = movementState();
    const policy = createHeuristicCpuPolicy();
    const runtime = createCpuRuntime(2);
    const king = policy(state, runtime, allCpu);
    expect(king).toMatchObject({ kind: "movement", unitId: "home-1-king" });
    if (king?.kind === "movement") expect(king.to).toBeUndefined();
    runtime.processedKeys.push("movement:team-1:home-1-king");
    state.units.find((unit) => unit.id === "home-1-strategist")!.role = "builder";
    const builder = policy(state, runtime, allCpu);
    expect(builder).toMatchObject({ kind: "movement", unitId: "home-1-strategist" });
    if (builder?.kind === "movement") expect(builder.to).toBeUndefined();
  });

  it("chooses a legal move that strictly reduces distance to the retained target", () => {
    const state = movementState();
    const unit: Unit = { id: "a-team-1-infantry", ownerTeamId: "team-1", type: "infantry", hp: UNIT_STATS.infantry.hp, position: { kind: "tile", x: 4, y: 1 }, statuses: [] };
    state.units.push(unit);
    const policy = createHeuristicCpuPolicy();
    const runtime = createCpuRuntime(13);
    const decision = policy(state, runtime, allCpu);
    expect(decision?.kind).toBe("movement");
    if (decision?.kind !== "movement" || !decision.to) throw new Error("Expected an improving movement");
    const target = policy.getTargetBaseId("team-1", 13)!;
    const base = state.bases.find((entry) => entry.id === target)!;
    const destination = { kind: "base", baseId: target, slotId: base.slots[0].id } as const;
    expect(getRoadAttackDistance(state, decision.to, destination)).toBeLessThan(getRoadAttackDistance(state, unit.position, destination));
  });

  it("prioritizes an attack against an enemy inside the target base", () => {
    const state = createHeadlessInitialState(4);
    const targetBase = state.bases.find((base) => base.id === "neutral-north")!;
    for (const base of state.bases.filter((base) => base.id !== targetBase.id)) {
      base.ownerTeamId = "team-1";
      state.teams.find((team) => team.id === "team-1")!.controlledBaseIds.push(base.id);
    }
    targetBase.ownerTeamId = "team-2";
    for (const slot of targetBase.slots) {
      if (slot.unitId) {
        const unit = state.units.find((entry) => entry.id === slot.unitId);
        if (unit) { unit.hp = 0; unit.position = { kind: "removed", reason: "defeated" }; }
      }
      slot.unitId = undefined;
    }
    const defender: Unit = { id: "target-base-defender", ownerTeamId: "team-2", type: "infantry", hp: 1, position: { kind: "base", baseId: targetBase.id, slotId: targetBase.slots[0].id }, statuses: [] };
    targetBase.slots[0].unitId = defender.id;
    const attacker: Unit = { id: "a-target-attacker", ownerTeamId: "team-1", type: "archer", hp: UNIT_STATS.archer.hp, position: { kind: "tile", x: 1, y: 1 }, statuses: [] };
    state.units.push(attacker, defender);
    removeTeamUnitsExcept(state, "team-1", attacker.id);
    const attackPosition = state.map.tiles.find((tile) => ["road", "baseGate", "reorganize"].includes(tile.terrain) && targetBase.coords.some((cell) => Math.max(Math.abs(cell.x - tile.x), Math.abs(cell.y - tile.y)) <= 3));
    expect(attackPosition).toBeDefined();
    attacker.position = { kind: "tile", x: attackPosition!.x, y: attackPosition!.y };
    state.phase = state.turnState.phase = "attack_input";
    expect(getAttackCandidates(state, attacker.id).map((target) => target.unitId)).toContain(defender.id);
    const decision = createHeuristicCpuPolicy()(state, createCpuRuntime(4), allCpu);
    expect(decision).toMatchObject({ kind: "attack", intent: { attackerUnitId: attacker.id, target: { unitId: defender.id }, pass: false } });
  });

  it("chooses only archer, cavalry or infantry for production and rewards", () => {
    const policy = createHeuristicCpuPolicy();
    const production = movementState();
    production.productionCompletedTeamIdsThisTurn = [];
    const produced = policy(production, createCpuRuntime(5), allCpu);
    expect(produced?.kind).toBe("production");
    if (produced?.kind === "production") expect(["archer", "cavalry", "infantry"]).toContain(produced.choice?.unitType);

    const reward = createHeadlessInitialState(4);
    reward.phase = reward.turnState.phase = "reward_placement";
    reward.phaseAfterRewards = "strategist_action_input";
    reward.rewardPlacementRequests.push({ id: "heuristic-reward", teamId: "team-1", rewardType: "capture_reward", sourceBaseId: "home-1", destinationKind: "fixed", fixedBaseId: "home-1", eligibleBaseIds: ["home-1"], completed: false, expired: false });
    const placed = policy(reward, createCpuRuntime(6), allCpu);
    expect(placed?.kind).toBe("reward");
    if (placed?.kind === "reward") expect(["archer", "cavalry", "infantry"]).toContain(placed.unitType);
  });

  it("is deterministic and runs through RL Environment.stepWithPolicy", () => {
    const first = new RlEnvironment(); first.reset(77, 4);
    const second = new RlEnvironment(); second.reset(77, 4);
    const firstPolicy = createHeuristicCpuPolicy(), secondPolicy = createHeuristicCpuPolicy();
    for (let index = 0; index < 250 && !first.isTerminal(); index += 1) {
      first.stepWithPolicy(firstPolicy); second.stepWithPolicy(secondPolicy);
      expect(second.getStateHash()).toBe(first.getStateHash());
    }
    expect(first.getResult().actionCount).toBeGreaterThan(0);
  });
});
