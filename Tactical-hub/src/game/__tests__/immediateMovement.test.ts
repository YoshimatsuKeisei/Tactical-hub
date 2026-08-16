import { describe, expect, it } from "vitest";
import { advanceCpuOneStep } from "../cpu/cpuStep";
import { enumerateRlDecisions, RlEnvironment, RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createCpuRuntime, type CpuDecision } from "../cpu/types";
import {
  commitUnitMovement,
  getMovementCandidates,
  saveMovementIntent,
  submitLegacyMovement,
  submitMovement,
} from "../engine/movement";
import { createInitialGameState } from "../initialState";
import type { GameState, Unit, UnitPosition } from "../types";
import { positionKey } from "../utils/position";

function clearSlot(state: GameState, position: UnitPosition) {
  if (position.kind !== "base") return;
  const slot = state.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId);
  if (slot) slot.unitId = undefined;
}

function movementFixture() {
  const state = createInitialGameState();
  for (const unit of state.units.filter((entry) => entry.ownerTeamId === "team-1")) {
    clearSlot(state, unit.position);
    unit.position = { kind: "removed", reason: "defeated" };
  }
  const units: Unit[] = [
    { id: "a-front", ownerTeamId: "team-1", type: "strategist", hp: 1, position: { kind: "tile", x: 4, y: 1 }, statuses: [] },
    { id: "b-back", ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "tile", x: 3, y: 1 }, statuses: [] },
  ];
  state.units.push(...units);
  state.productionCompletedTeamIdsThisTurn = ["team-1"];
  return state;
}

function move(state: GameState, unitId: string, to: UnitPosition) {
  const unit = state.units.find((entry) => entry.id === unitId)!;
  return commitUnitMovement(state, { teamId: "team-1", unitId, from: unit.position, to, stay: false });
}

describe("immediate movement semantics", () => {
  it("commits each unit immediately and generates the next candidates from the updated board", () => {
    const initial = movementFixture();
    const afterFront = move(initial, "a-front", { kind: "tile", x: 5, y: 1 });
    expect(afterFront.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
    expect(afterFront.movedUnitIdsThisMovementPhase).toContain("a-front");
    expect(getMovementCandidates(afterFront, "b-back").map(positionKey)).toContain("4,1");
    expect(getMovementCandidates(afterFront, "b-back").map(positionKey)).not.toContain("5,1");

    const afterBack = move(afterFront, "b-back", { kind: "tile", x: 4, y: 1 });
    expect(afterBack.units.find((unit) => unit.id === "b-back")?.position).toEqual({ kind: "tile", x: 4, y: 1 });
    expect(getMovementCandidates(afterBack, "a-front")).toEqual([]);
  });

  it("uses submit_movement only to finish the current team", () => {
    const initial = movementFixture();
    const savedOnly = saveMovementIntent(initial, {
      teamId: "team-1", unitId: "a-front", from: { kind: "tile", x: 4, y: 1 }, to: { kind: "tile", x: 5, y: 1 }, stay: false,
    });
    const submitted = submitMovement(savedOnly, "team-1");
    expect(submitted.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 4, y: 1 });
    expect(submitted.currentMovementTeamId).toBe("team-2");
  });

  it("retains the former intent ordering and submit resolution only in legacy mode", () => {
    let state = movementFixture();
    expect(enumerateRlDecisions(state, createCpuRuntime(2), () => true, "legacy_batched")
      .map((entry) => entry.action.actionKey)).toContain("movement:team-1:a-front:5,1");
    state = saveMovementIntent(state, {
      teamId: "team-1", unitId: "a-front", from: { kind: "tile", x: 4, y: 1 }, to: { kind: "tile", x: 5, y: 1 }, stay: false,
    });
    expect(getMovementCandidates(state, "b-back", "legacy_batched").map(positionKey)).toContain("4,1");
    state = saveMovementIntent(state, {
      teamId: "team-1", unitId: "b-back", from: { kind: "tile", x: 3, y: 1 }, to: { kind: "tile", x: 4, y: 1 }, stay: false,
    });
    expect(state.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 4, y: 1 });
    const resolved = submitLegacyMovement(state, "team-1", () => 0.5);
    expect(resolved.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
    expect(resolved.units.find((unit) => unit.id === "b-back")?.position).toEqual({ kind: "tile", x: 3, y: 1 });
  });

  it("releases and claims the exact BaseSlot immediately", () => {
    const state = movementFixture();
    const base = state.bases.find((entry) => entry.id === "home-1")!;
    const slot = base.slots[0];
    const front = state.units.find((unit) => unit.id === "a-front")!;
    front.position = { kind: "base", baseId: base.id, slotId: slot.id };
    slot.unitId = front.id;
    const exit = getMovementCandidates(state, front.id).find((position) => position.kind !== "base");
    expect(exit).toBeDefined();
    const afterExit = move(state, front.id, exit!);
    expect(afterExit.bases.find((entry) => entry.id === base.id)?.slots.find((entry) => entry.id === slot.id)?.unitId).toBeUndefined();

    const entrant: Unit = { id: "c-entrant", ownerTeamId: "team-1", type: "infantry", hp: 1, position: { kind: "tile", x: 0, y: 0 }, statuses: [] };
    afterExit.units.push(entrant);
    let source: UnitPosition | undefined;
    const destination = { kind: "base", baseId: base.id, slotId: slot.id } as const;
    for (const tile of afterExit.map.tiles) {
      entrant.position = { kind: "tile", x: tile.x, y: tile.y };
      if (getMovementCandidates(afterExit, entrant.id).some((candidate) => positionKey(candidate) === positionKey(destination))) {
        source = entrant.position;
        break;
      }
    }
    expect(source).toBeDefined();
    const afterEntry = commitUnitMovement(afterExit, { teamId: "team-1", unitId: entrant.id, from: source!, to: destination, stay: false });
    expect(afterEntry.units.find((unit) => unit.id === entrant.id)?.position).toEqual(destination);
    expect(afterEntry.bases.find((entry) => entry.id === base.id)?.slots.find((entry) => entry.id === slot.id)?.unitId).toBe(entrant.id);
  });

  it("applies the shared CPU movement decision immediately", () => {
    const state = movementFixture();
    const decision: CpuDecision = { kind: "movement", teamId: "team-1", actorKey: "movement:a-front", unitId: "a-front", to: { kind: "tile", x: 5, y: 1 } };
    const result = advanceCpuOneStep(state, createCpuRuntime(1), { "team-1": "random_cpu" }, () => decision);
    expect(result.state.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
    expect(result.state.movedUnitIdsThisMovementPhase).toContain("a-front");
  });

  it.each([
    ["v1", () => new RlEnvironment()],
    ["v2", () => new RlEnvironmentV2()],
  ])("applies an RL %s movement action before returning the next decision", (_label, createEnvironment) => {
    const environment = createEnvironment();
    environment.reset(7, 4, movementFixture());
    const actor = environment.getCurrentActorTeamId();
    expect(actor).toBe("team-1");
    const action = environment.getLegalActions(actor!).find((entry) => entry.actionType === "movement" && entry.unitId === "a-front" && entry.tileId === "5,1");
    expect(action).toBeDefined();
    environment.step(action!.actionKey);
    const observation = environment.getObservation(environment.getCurrentActorTeamId() ?? "team-1");
    expect(observation.units.find((unit) => unit.id === "a-front")?.position).toEqual({ kind: "tile", x: 5, y: 1 });
  });
});
