import { describe, expect, it } from "vitest";
import { encodeRlLegalActions, getRlActionFeatureWidth, RL_ACTION_ENCODER_VERSION } from "../cpu/rlActionEncoder";
import { describeRlDecision, RlEnvironment, type RlLegalAction } from "../cpu/rlEnvironment";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";

const finiteNumbers = (value: unknown): number[] => typeof value === "number"
  ? [value]
  : Array.isArray(value)
    ? value.flatMap(finiteNumbers)
    : value && typeof value === "object"
      ? Object.values(value).flatMap(finiteNumbers)
      : [];

function initial() {
  const environment = new RlEnvironment();
  const observation = environment.reset(301, 4);
  return { environment, observation };
}

function expectDifferent(observation: ReturnType<typeof initial>["observation"], left: RlLegalAction, right: RlLegalAction) {
  const encoded = encodeRlLegalActions(observation, [left, right]);
  expect(encoded.actions[0]).not.toEqual(encoded.actions[1]);
}

describe("RL-2B Action Encoder", () => {
  it("is deterministic, finite, fixed-width, aligned with actionKeys, and pure", () => {
    const { environment, observation } = initial();
    const actions = environment.getLegalActions(environment.getCurrentActorTeamId()!);
    const observationBefore = structuredClone(observation);
    const actionsBefore = structuredClone(actions);
    const first = encodeRlLegalActions(observation, actions);
    const second = encodeRlLegalActions(observation, actions);

    expect(first).toEqual(second);
    expect(first.schemaVersion).toBe(RL_ACTION_ENCODER_VERSION);
    expect(first.actionKeys).toEqual(actions.map((action) => action.actionKey));
    expect(first.actions).toHaveLength(actions.length);
    expect(new Set(first.actions.map((row) => row.length))).toEqual(new Set([getRlActionFeatureWidth(observation)]));
    expect(finiteNumbers(first).every(Number.isFinite)).toBe(true);
    expect(observation).toEqual(observationBefore);
    expect(actions).toEqual(actionsBefore);
  });

  it("does not use actionKey text as a feature", () => {
    const { observation } = initial();
    const action = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "movement:team-1:u", unitId: "home-1-king" });
    const renamed = { ...action, actionKey: "completely-different-external-key" };
    expect(encodeRlLegalActions(observation, [action]).actions).toEqual(encodeRlLegalActions(observation, [renamed]).actions);
  });

  it("uses entity IDs only to resolve Observation references, not as numeric values", () => {
    const { observation } = initial();
    const unit = observation.units.find((entry) => entry.ownerTeamId === observation.observingTeamId && entry.position.kind !== "removed")!;
    const action = describeRlDecision({ kind: "movement", teamId: observation.observingTeamId, actorKey: "m", unitId: unit.id });
    const renamedObservation = structuredClone(observation);
    const renamedId = "opaque-unit-reference";
    renamedObservation.units.find((entry) => entry.id === unit.id)!.id = renamedId;
    for (const base of renamedObservation.bases) for (const slot of base.slots) if (slot.unitId === unit.id) slot.unitId = renamedId;
    for (const base of renamedObservation.map.bases) for (const slot of base.slots) if (slot.unitId === unit.id) slot.unitId = renamedId;
    for (const flag of renamedObservation.unitTurnFlags) if (flag.unitId === unit.id) flag.unitId = renamedId;
    renamedObservation.movedUnitIdsThisMovementPhase = renamedObservation.movedUnitIdsThisMovementPhase.map((id) => id === unit.id ? renamedId : id);
    const renamedAction = { ...action, unitId: renamedId, actionKey: "renamed-action-key" };

    expect(encodeRlLegalActions(observation, [action]).actions).toEqual(encodeRlLegalActions(renamedObservation, [renamedAction]).actions);
  });

  it("distinguishes movement destinations, attack targets, and pass from normal actions", () => {
    const { observation } = initial();
    const mover = observation.units.find((unit) => unit.ownerTeamId === "team-1")!;
    const targets = observation.units.filter((unit) => unit.ownerTeamId !== "team-1" && unit.position.kind !== "removed").slice(0, 2);
    const movementA = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "m", unitId: mover.id, to: { kind: "tile", x: 3, y: 3 } });
    const movementB = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "m", unitId: mover.id, to: { kind: "tile", x: 4, y: 3 } });
    const movementPass = describeRlDecision({ kind: "movement", teamId: "team-1", actorKey: "m", unitId: mover.id });
    const attackA = describeRlDecision({ kind: "attack", teamId: "team-1", actorKey: "a", intent: { teamId: "team-1", attackerUnitId: mover.id, target: { kind: "unit", unitId: targets[0].id }, pass: false } });
    const attackB = describeRlDecision({ kind: "attack", teamId: "team-1", actorKey: "a", intent: { teamId: "team-1", attackerUnitId: mover.id, target: { kind: "unit", unitId: targets[1].id }, pass: false } });

    expectDifferent(observation, movementA, movementB);
    expectDifferent(observation, movementA, movementPass);
    expectDifferent(observation, attackA, attackB);
  });

  it("distinguishes strategist production roles and strategist action kinds", () => {
    const { observation } = initial();
    const production = (role: "builder" | "teleporter") => describeRlDecision({
      kind: "production", teamId: "team-1", actorKey: "p",
      choice: { teamId: "team-1", baseId: "home-1", unitType: "strategist", strategistRole: role },
    });
    const strategist = (action: "place_bridge" | "place_obstacle") => describeRlDecision({
      kind: "strategist", teamId: "team-1", actorKey: "s",
      intent: { teamId: "team-1", strategistUnitId: "home-1-strategist", action, tiles: [{ x: 4, y: 2 }] },
    });
    expectDifferent(observation, production("builder"), production("teleporter"));
    expectDifferent(observation, strategist("place_bridge"), strategist("place_obstacle"));
  });

  it("encodes a multi-tile strategist target as a lossless fixed board mask", () => {
    const { observation } = initial();
    const action = (tiles: { x: number; y: number }[]) => describeRlDecision({
      kind: "strategist", teamId: "team-1", actorKey: "s",
      intent: { teamId: "team-1", strategistUnitId: "home-1-strategist", action: "place_bridge", tiles },
    });
    const first = action([{ x: 4, y: 2 }, { x: 4, y: 3 }]);
    const reordered = action([{ x: 4, y: 3 }, { x: 4, y: 2 }]);
    const different = action([{ x: 4, y: 2 }, { x: 5, y: 2 }]);

    expect(encodeRlLegalActions(observation, [first]).actions).toEqual(encodeRlLegalActions(observation, [reordered]).actions);
    expectDifferent(observation, first, different);
  });

  it("resolves reward meaning from Observation without encoding requestId", () => {
    const state = createHeadlessInitialState(4);
    state.rewardPlacementRequests.push({
      id: "opaque-request-a",
      teamId: "team-1",
      rewardType: "capture_reward",
      sourceBaseId: "home-1",
      destinationKind: "fixed",
      fixedBaseId: "home-1",
      eligibleBaseIds: ["home-1"],
      completed: false,
      expired: false,
    });
    state.phase = state.turnState.phase = "reward_placement";
    state.phaseAfterRewards = "strategist_action_input";
    const environment = new RlEnvironment();
    const observation = environment.reset(302, 4, state);
    const reward = environment.getLegalActions("team-1")[0];
    const renamedObservation = structuredClone(observation);
    renamedObservation.rewardPlacementRequests[0].id = "renamed-request";
    const renamedAction = { ...reward, requestId: "renamed-request", actionKey: "renamed-key" };

    expect(encodeRlLegalActions(observation, [reward]).actions).toEqual(encodeRlLegalActions(renamedObservation, [renamedAction]).actions);
  });

  it("cannot restore another team's private unresolved intents", () => {
    const state = createHeadlessInitialState(4);
    const own = state.units.find((unit) => unit.ownerTeamId === "team-1")!;
    const enemy = state.units.find((unit) => unit.ownerTeamId === "team-2")!;
    state.phase = state.turnState.phase = "attack_input";
    state.turnState.actionIntents = [
      { teamId: "team-1", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-1", attackerUnitId: own.id, pass: true }] },
      { teamId: "team-2", productionChoices: [], movementIntents: [], attackIntents: [{ teamId: "team-2", attackerUnitId: enemy.id, target: { kind: "unit", unitId: own.id }, pass: false }] },
    ];
    const environment = new RlEnvironment();
    environment.reset(303, 4, state);
    const observation = environment.getObservation("team-1");
    const actions = environment.getLegalActions("team-1");
    const encoded = encodeRlLegalActions(observation, actions);

    expect(observation.actionIntents.map((intent) => intent.teamId)).toEqual(["team-1"]);
    expect(encoded.actions).toHaveLength(actions.length);
    expect(finiteNumbers(encoded).every(Number.isFinite)).toBe(true);
  });
});
