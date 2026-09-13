import { afterEach, describe, expect, it, vi } from "vitest";
import type { EncodedLegalActionsV2 } from "../cpu/rlActionEncoder";
import type { EncodedObservation } from "../cpu/rlObservationEncoder";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { finalizeTerminalTrajectory, finalizeVictoryTrajectory, replayPpoTrajectory, runPpoSelfPlaySmoke, type PpoClientLike, type PpoTrajectoryStep } from "../cpu/rlPpoSelfPlay";
import { adjudicatePpoTimeLimit } from "../cpu/rlPpoAdjudication";
import * as headless from "../cpu/headlessSimulation";
import type { PpoEncodedSample } from "../cpu/rlPpoPackedBatch";

const step = (teamId: string, value = 0): PpoTrajectoryStep => ({
  decisionIndex: 0, turnNumber: 1, phase: "movement_input", selectedActionIndex: 0, selectedActionKey: "action",
  oldLogProbability: 0, value, reward: 99, done: false, teamId,
});

function fakeClient() {
  return {
    start: vi.fn(async () => ({ type: "ready", selectedDevice: "cpu", updateCount: 0, episodeCount: 0 })),
    act: vi.fn(async (_observation: EncodedObservation, actions: EncodedLegalActionsV2) => ({
      type: "action", requestId: 1, actionIndex: 0, actionKey: actions.actionKeys[0], logProbability: -0.5, value: 0.1,
    })),
    beginUpdate: vi.fn(async (totalSamples: number) => ({ totalSamples })),
    accumulatePacked: vi.fn(async (samples: PpoEncodedSample[]) => {
      expect(samples.every((sample) => Number.isFinite(sample.advantage) && Number.isFinite(sample.return))).toBe(true);
      return { acceptedSamples: samples.length };
    }),
    finishUpdate: vi.fn(async (episodeCount: number) => ({ episodeCount })),
    save: vi.fn(async (path: string, _metadata: Record<string, unknown>) => ({ path })),
    close: vi.fn(async () => {}),
  };
}

afterEach(() => vi.restoreAllMocks());

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

  it.each([
    ["safety_action_limit", { safetyMaxActions: 8 }],
    ["safety_turn_limit", { safetyMaxTurns: 1, safetyMaxActions: 1000 }],
  ] as const)("learns and replays a real %s cutoff and saves both checkpoint metadata", async (reason, limits) => {
    const client = fakeClient();
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const result = await runPpoSelfPlaySmoke({
      seed: 7, initialCheckpoint: "unused", outputCheckpoint: "latest", bestCheckpoint: "best",
      ...limits, replayChunkSize: 3, client: client as unknown as PpoClientLike,
    });
    expect(result.completed).toEqual([]);
    expect(result.truncated).toEqual([]);
    expect(result.adjudicated).toHaveLength(1);
    const episode = result.adjudicated[0];
    expect(episode).toMatchObject({ outcomeKind: "time_limit_adjudicated", reason, limitReason: reason });
    expect(episode.environmentResult).toMatchObject({ terminal: false, endReason: "ongoing", winnerTeamId: undefined });
    expect(Object.values(episode.environmentResult.rewards).every((reward) => reward === 0)).toBe(true);
    expect(episode.adjudication).toHaveLength(4);
    expect(result.replayedSamples).toBe(client.act.mock.calls.length);
    expect(client.beginUpdate).toHaveBeenCalledWith(result.replayedSamples);
    expect(client.finishUpdate).toHaveBeenCalledWith(1);
    expect(client.accumulatePacked.mock.calls.every(([samples]) => samples.length <= 3)).toBe(true);
    for (const path of ["latest", "best"]) expect(client.save).toHaveBeenCalledWith(path, expect.objectContaining({
      victoryEpisodeCount: 0, adjudicatedEpisodeCount: 1, truncatedEpisodeCount: 0, replayedSamples: result.replayedSamples,
    }));
    expect(stderr.mock.calls.some(([line]) => String(line).includes(`"limitReason":"${reason}"`))).toBe(true);
    expect(client.close).toHaveBeenCalledOnce();
  });

  it.each(["phase_stall", "no_actor", "no_legal_actions", "exception", "invariant_violation", "stopped"])("excludes %s even at the action limit and logs before throwing", async (reason) => {
    const client = fakeClient();
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    if (reason === "phase_stall") vi.spyOn(RlEnvironmentV2.prototype, "getProgressHash").mockReturnValue("same");
    if (reason === "no_actor") vi.spyOn(RlEnvironmentV2.prototype, "getCurrentActorTeamId").mockReturnValue(undefined);
    if (reason === "no_legal_actions") vi.spyOn(RlEnvironmentV2.prototype, "getLegalActionsForEncoding").mockReturnValue([]);
    if (reason === "exception") client.act.mockRejectedValue(new Error("policy failed"));
    if (reason === "invariant_violation") vi.spyOn(headless, "checkHeadlessInvariants").mockReturnValue(["invalid cutoff"]);
    if (reason === "stopped") {
      const getResult = RlEnvironmentV2.prototype.getResult;
      vi.spyOn(RlEnvironmentV2.prototype, "getResult").mockImplementation(function (this: RlEnvironmentV2) {
        return { ...getResult.call(this), terminal: true, endReason: "stopped" };
      });
    }
    await expect(runPpoSelfPlaySmoke({
      seed: 7, episodes: 1, initialCheckpoint: "unused", outputCheckpoint: "unused",
      safetyMaxActions: 1, client: client as unknown as PpoClientLike,
    })).rejects.toThrow("no learnable trajectory");
    expect(client.beginUpdate).not.toHaveBeenCalled();
    expect(client.accumulatePacked).not.toHaveBeenCalled();
    expect(client.finishUpdate).not.toHaveBeenCalled();
    expect(client.save).not.toHaveBeenCalled();
    expect(client.close).toHaveBeenCalledOnce();
    const line = stderr.mock.calls.map(([line]) => String(line)).find((line) => line.startsWith("[PPO episode]"))!;
    expect(line).toContain('"outcomeKind":"abnormal_truncated"');
    expect(line).toContain(reason === "stopped" ? '"endReason":"stopped"' : `"reason":"${reason}`);
    expect(line).not.toContain('"adjudication":');
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
    const rollout = {
        outcomeKind: "time_limit_adjudicated" as const, limitReason: "safety_action_limit" as const,
        seed, terminal: result.terminal, endReason: result.endReason, winnerTeamId: result.winnerTeamId,
        loserTeamIds: result.loserTeamIds, finalStateHash: original.getStateHash(), trajectory,
    };
    const hash = original.getStateHash();
    const adjudication = adjudicatePpoTimeLimit(original.getStateForValidation());
    finalizeTerminalTrajectory(trajectory, Object.fromEntries(adjudication.map((team) => [team.teamId, team.reward])), { gamma: 0.99, gaeLambda: 0.95 });
    expect(original.getStateHash()).toBe(hash);
    expect(original.getResult()).toEqual(result);
    await expect(replayPpoTrajectory({
      rollout,
      client, chunkSize: 1, memoryLogInterval: 100,
    })).resolves.toBe(1);
    expect(received).toBe(1);
    for (const mismatch of [
      { finalStateHash: "wrong" }, { terminal: true }, { endReason: "victory" as const },
      { winnerTeamId: "team-1" }, { loserTeamIds: ["team-2"] },
    ]) await expect(replayPpoTrajectory({
      rollout: { ...rollout, ...mismatch }, client, chunkSize: 1, memoryLogInterval: 100,
    })).rejects.toThrow("PPO replay final mismatch");
  });

  it("assigns adjudicated reward and done only to each team's last step with finite GAE", () => {
    const state = headless.createHeadlessInitialState(4);
    for (const [index, team] of state.teams.filter((team) => !team.isNeutral).entries()) {
      state.units.find((unit) => unit.ownerTeamId === team.id && unit.type === "king")!.hp = index + 1;
    }
    const adjudication = adjudicatePpoTimeLimit(state);
    const rewards = Object.fromEntries(adjudication.map((team) => [team.teamId, team.reward]));
    const teams = adjudication.map((team) => team.teamId);
    const steps = [...teams, ...teams].map((teamId) => step(teamId, 0.4));
    finalizeTerminalTrajectory(steps, rewards, { gamma: 0.99, gaeLambda: 0.95 });
    for (const [index, entry] of steps.entries()) {
      expect(entry.done).toBe(index >= teams.length);
      expect(entry.reward).toBe(index >= teams.length ? rewards[entry.teamId] : 0);
      expect(Number.isFinite(entry.advantage) && Number.isFinite(entry.return)).toBe(true);
      if (entry.done) expect(entry.return).toBeCloseTo(rewards[entry.teamId]);
    }
  });

  it("keeps abnormal episodes out of a mixed update and includes their count in metadata", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const client = fakeClient();
    client.act.mockRejectedValueOnce(new Error("first episode failed"));
    const result = await runPpoSelfPlaySmoke({
      seed: 7, episodes: 2, initialCheckpoint: "unused", outputCheckpoint: "latest",
      safetyMaxActions: 1, client: client as unknown as PpoClientLike,
    });
    expect(result.truncated[0]).toMatchObject({ seed: 7, outcomeKind: "abnormal_truncated", reason: "exception:first episode failed" });
    expect(result.adjudicated[0]).toMatchObject({ seed: 8, outcomeKind: "time_limit_adjudicated" });
    expect(client.beginUpdate).toHaveBeenCalledWith(1);
    expect(client.finishUpdate).toHaveBeenCalledWith(1);
    expect(client.save).toHaveBeenCalledWith("latest", expect.objectContaining({
      victoryEpisodeCount: 0, adjudicatedEpisodeCount: 1, truncatedEpisodeCount: 1, replayedSamples: 1,
    }));
  });
});
