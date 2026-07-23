import { describe, expect, it } from "vitest";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { runHeadlessMatch } from "../cpu/headlessSimulation";
import { compareHeadlessTraces, stableDiagnosticHash } from "../cpu/headlessTrace";

function traced(seed = 2010, profile = false) {
  return runHeadlessMatch({ participantCount: 4, seed, maxTurns: 1, maxActions: 1_000, mode: "training", policy: createHeuristicCpuPolicy(), profile, trace: true });
}

describe("headless deterministic diagnostic trace", () => {
  it("does not change the action sequence, outcome, or RNG series", () => {
    const plain = runHeadlessMatch({ participantCount: 4, seed: 2010, maxTurns: 1, maxActions: 1_000, mode: "training", policy: createHeuristicCpuPolicy() });
    const withTrace = traced();
    const replay = traced();
    expect({ hash: withTrace.actionSequenceHash, count: withTrace.actionCount, reason: withTrace.endReason }).toEqual({ hash: plain.actionSequenceHash, count: plain.actionCount, reason: plain.endReason });
    expect(withTrace.trace?.map((entry) => [entry.selectedActionKey, entry.rngStateBefore, entry.rngStateAfter])).toEqual(replay.trace?.map((entry) => [entry.selectedActionKey, entry.rngStateBefore, entry.rngStateAfter]));
    expect(compareHeadlessTraces(withTrace.trace!, replay.trace!)).toEqual({ equal: true });
  });

  it("keeps profile ON/OFF traces and hashes identical", () => {
    const plain = traced(2020, false), profiled = traced(2020, true);
    expect(profiled.actionSequenceHash).toBe(plain.actionSequenceHash);
    expect(profiled.trace).toEqual(plain.trace);
  });

  it("reports the first selected Action, Legal Action, and State differences", () => {
    const original = traced(2030).trace!;
    const selected = structuredClone(original);
    selected[2].selectedActionKey += ":changed";
    expect(compareHeadlessTraces(original, selected).firstDifference).toMatchObject({ kind: "selected_action", actionIndex: original[2].actionIndex });

    const legal = structuredClone(original);
    legal[1].legalActionKeys = [...(legal[1].legalActionKeys ?? []), "diagnostic-extra"];
    legal[1].legalActionCount = legal[1].legalActionKeys.length;
    legal[1].legalActionKeysHash = stableDiagnosticHash(legal[1].legalActionKeys);
    expect(compareHeadlessTraces(original, legal).firstDifference).toMatchObject({ kind: "legal_actions", actionIndex: original[1].actionIndex, addedLegalActions: ["diagnostic-extra"] });

    const state = structuredClone(original);
    state[0].stateBefore.unitsHash = "ffffffff";
    state[0].stateBefore.stateHash = "eeeeeeee";
    expect(compareHeadlessTraces(original, state).firstDifference).toMatchObject({ kind: "state_before", actionIndex: original[0].actionIndex, differingStateFields: expect.arrayContaining(["unitsHash"]) });
  });
});
