import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { encodeRlObservation } from "../cpu/rlObservationEncoder";
import { encodeRlLegalActions } from "../cpu/rlActionEncoder";
import { createRlFeatureSpec } from "../cpu/rlFeatureSpec";
import { PythonPolicyClient } from "../cpu/pythonPolicyClient";
import { runRlSmokeMatch, summarizeRlSmokeProgress } from "../cpu/rlSmokeRunner";

const fixture = resolve(
  process.cwd(),
  "src/game/__tests__/fixtures/fakePolicyServer.cjs",
);

function setup() {
  const env = new RlEnvironment();
  const observation = env.reset(202, 4);
  const legalActions = env.getLegalActions(env.getCurrentActorTeamId()!);
  return {
    observation,
    encodedObservation: encodeRlObservation(observation),
    encodedActions: encodeRlLegalActions(observation, legalActions),
    spec: createRlFeatureSpec(observation),
  };
}

describe("PythonPolicyClient JSONL bridge", () => {
  it("summarizes turn and non-neutral team ownership without removed units", () => {
    const { observation } = setup();
    const summary = summarizeRlSmokeProgress(observation);

    expect(summary.turnNumber).toBe(observation.turnNumber);
    expect(summary.teams).toEqual(
      observation.teams
        .filter((team) => !team.isNeutral)
        .map((team) => ({
          teamId: team.id,
          status: team.status,
          unitCount: observation.units.filter(
            (unit) => unit.ownerTeamId === team.id && unit.position.kind !== "removed",
          ).length,
          baseCount: observation.bases.filter((base) => base.ownerTeamId === team.id).length,
        })),
    );
  });

  it("keeps actionKeys[index] correspondence through a persistent JSONL process", async () => {
    const input = setup();
    const client = new PythonPolicyClient({
      command: process.execPath,
      args: [fixture, "valid"],
    });

    await client.start(202, input.spec);
    expect(client.getSelectedDevice()).toBe("cpu");
    const result = await client.act(input.encodedObservation, input.encodedActions);

    expect(result.actionIndex).toBe(input.encodedActions.actionKeys.length - 1);
    expect(result.actionKey).toBe(input.encodedActions.actionKeys[result.actionIndex]);
    expect(result.value).toBe(0.25);
    await client.close();
  });

  it("rejects an actionIndex outside the current legal range", async () => {
    const input = setup();
    const client = new PythonPolicyClient({
      command: process.execPath,
      args: [fixture, "invalid"],
    });

    await client.start(202, input.spec);
    await expect(
      client.act(input.encodedObservation, input.encodedActions),
    ).rejects.toThrow(/actionIndex/);
    await client.close();
  });

  it("uses one persistent process while advancing decisions", async () => {
    const result = await runRlSmokeMatch({
      seed: 303,
      participantCount: 4,
      maxDecisions: 3,
      python: {
        command: process.execPath,
        args: [fixture, "valid"],
      },
    });

    expect(result.decisionCount).toBe(3);
    expect(result.terminal).toBe(false);
    expect(result.endReason).toBe("decision_limit");
    expect(result.pythonAbnormalExit).toBe(false);
    expect(result.selectedDevice).toBe("cpu");
    expect(result.turnNumber).toBeGreaterThanOrEqual(1);
    expect(result.teams).toHaveLength(4);
  });
});
