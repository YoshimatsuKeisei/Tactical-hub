import { describe, expect, it } from "vitest";
import type { EncodedLegalActionsV2 } from "../cpu/rlActionEncoder";
import type { EncodedObservation } from "../cpu/rlObservationEncoder";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { finalizeVictoryTrajectory, replayPpoTrajectory, runPpoSelfPlaySmoke, type PpoClientLike, type PpoTrajectoryStep } from "../cpu/rlPpoSelfPlay";

const step = (teamId: string, value = 0): PpoTrajectoryStep => ({
  decisionIndex: 0, turnNumber: 1, phase: "movement_input", selectedActionIndex: 0, selectedActionKey: "action",
  oldLogProbability: 0, value, reward: 99, done: false, teamId,
});

describe("Phase 12B PPO self-play", () => {
  it("computes GAE independently per team and assigns only terminal victory rewards", () => {
    const steps = [step("team-1"), step("team-2"), step("team-1")];
    finalizeVictoryTrajectory(steps, { "team-1": 1, "team-2": -1 }, { gamma: 1, gaeLambda: 1 });
    expect(steps.map(({ reward, done }) => ({ reward, done }))).toEqual([
      { reward: 0, done: false }, { reward: -1, done: true }, { reward: 1, done: true },
    ]);
    expect(steps.map((candidate) => candidate.advantage)).toEqual([1, -1, 1]);
    expect(steps.map((candidate) => candidate.return)).toEqual([1, -1, 1]);
  });

  it("excludes a safety-truncated episode from PPO updates", async () => {
    let actCount = 0;
    let updateCount = 0;
    let saveCount = 0;
    let closeCount = 0;
    const client = {
      async start() { return { type: "ready", selectedDevice: "cpu", updateCount: 0, episodeCount: 0 }; },
      async act(_observation: EncodedObservation, actions: EncodedLegalActionsV2) {
        actCount += 1;
        return { type: "action", requestId: 1, actionIndex: 0, actionKey: actions.actionKeys[0], logProbability: 0, value: 0 };
      },
      async beginUpdate() { updateCount += 1; throw new Error("must not update"); },
      async accumulatePacked() { updateCount += 1; throw new Error("must not update"); },
      async finishUpdate() { updateCount += 1; throw new Error("must not update"); },
      async save() { saveCount += 1; throw new Error("must not save"); },
      async close() { closeCount += 1; },
    } as unknown as PpoClientLike;

    await expect(runPpoSelfPlaySmoke({
      seed: 7, episodes: 1, initialCheckpoint: "unused", outputCheckpoint: "unused",
      safetyMaxActions: 1, client,
    })).rejects.toThrow("no normal victory trajectory");
    expect({ actCount, updateCount, saveCount, closeCount }).toEqual({ actCount: 1, updateCount: 0, saveCount: 0, closeCount: 1 });
  });

  it("replays the same action index/key and final state without retaining encoded rollout data", async () => {
    const seed = 23;
    const original = new RlEnvironmentV2();
    original.reset(seed, 4);
    const teamId = original.getCurrentActorTeamId()!;
    const observation = original.getObservationForEncoding(teamId);
    const legal = original.getLegalActionsForEncoding(teamId);
    const selectedActionIndex = 0;
    const selectedActionKey = legal[selectedActionIndex].actionKey;
    original.stepWithoutObservation(selectedActionKey);
    const result = original.getResult();
    const trajectory: PpoTrajectoryStep[] = [{
      decisionIndex: 0, turnNumber: observation.turnNumber, phase: observation.phase, teamId,
      selectedActionIndex, selectedActionKey, oldLogProbability: -0.5, value: 0.25,
      reward: 0, done: false, advantage: 0.75, return: 1,
    }];
    expect(trajectory[0]).not.toHaveProperty("observation");
    expect(trajectory[0]).not.toHaveProperty("legalActions");
    let received = 0;
    const client = {
      async accumulatePacked(samples: Array<{ targetIndex: number; actions: number[][] }>) {
        received += samples.length;
        expect(samples[0].targetIndex).toBe(selectedActionIndex);
        expect(samples[0].actions.length).toBe(legal.length);
        return { acceptedSamples: samples.length, accumulatedSamples: received };
      },
    } as unknown as PpoClientLike;
    await expect(replayPpoTrajectory({
      rollout: {
        seed, terminal: result.terminal, endReason: result.endReason, winnerTeamId: result.winnerTeamId,
        loserTeamIds: result.loserTeamIds, finalStateHash: original.getStateHash(), trajectory,
      },
      client, chunkSize: 1, memoryLogInterval: 100,
    })).resolves.toBe(1);
    expect(received).toBe(1);
  });
});
