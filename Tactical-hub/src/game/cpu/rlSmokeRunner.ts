import { encodeRlLegalActions } from "./rlActionEncoder";
import { RlEnvironment, type RlObservation } from "./rlEnvironment";
import { createRlFeatureSpec } from "./rlFeatureSpec";
import { encodeRlObservation } from "./rlObservationEncoder";
import { PythonPolicyClient, type PythonPolicyClientOptions } from "./pythonPolicyClient";

export type RlSmokeResult = {
  seed: number;
  decisionCount: number;
  terminal: boolean;
  turnNumber: number;
  teams: Array<{
    teamId: string;
    status: RlObservation["teams"][number]["status"];
    unitCount: number;
    baseCount: number;
  }>;
  winnerTeamId?: string;
  loserTeamIds: string[];
  endReason: "ongoing" | "victory" | "stopped" | "decision_limit" | "python_error";
  pythonAbnormalExit: boolean;
  error?: string;
};

export function summarizeRlSmokeProgress(observation: RlObservation) {
  return {
    turnNumber: observation.turnNumber,
    teams: observation.teams
      .filter((team) => !team.isNeutral)
      .map((team) => ({
        teamId: team.id,
        status: team.status,
        unitCount: observation.units.filter(
          (unit) => unit.ownerTeamId === team.id && unit.position.kind !== "removed",
        ).length,
        baseCount: observation.bases.filter((base) => base.ownerTeamId === team.id).length,
      })),
  };
}

export async function runRlSmokeMatch(input: {
  seed: number;
  participantCount?: 4;
  maxDecisions?: number;
  python?: PythonPolicyClientOptions;
}): Promise<RlSmokeResult> {
  const environment = new RlEnvironment();
  const firstObservation = environment.reset(input.seed, input.participantCount ?? 4);
  const client = new PythonPolicyClient(input.python);
  let decisionCount = 0;
  const progress = () => summarizeRlSmokeProgress(
    environment.getObservation(firstObservation.observingTeamId),
  );
  try {
    await client.start(input.seed, createRlFeatureSpec(firstObservation));
    while (!environment.isTerminal()) {
      if (decisionCount >= (input.maxDecisions ?? 100_000)) {
        const result = environment.getResult();
        return {
          seed: input.seed,
          decisionCount,
          terminal: false,
          ...progress(),
          loserTeamIds: result.loserTeamIds,
          endReason: "decision_limit",
          pythonAbnormalExit: false,
        };
      }
      const actor = environment.getCurrentActorTeamId();
      if (!actor) throw new Error("RL Environment has no actor at a non-terminal decision point");
      const observation = environment.getObservation(actor);
      const legalActions = environment.getLegalActions(actor);
      if (!legalActions.length) throw new Error(`No legal actions for current actor ${actor}`);
      const encodedObservation = encodeRlObservation(observation);
      const encodedActions = encodeRlLegalActions(observation, legalActions);
      const selected = await client.act(encodedObservation, encodedActions);
      environment.step(selected.actionKey);
      decisionCount += 1;
    }
    const result = environment.getResult();
    return {
      seed: input.seed,
      decisionCount,
      terminal: result.terminal,
      ...progress(),
      winnerTeamId: result.winnerTeamId,
      loserTeamIds: result.loserTeamIds,
      endReason: result.endReason,
      pythonAbnormalExit: false,
    };
  } catch (error) {
    const result = environment.getResult();
    return {
      seed: input.seed,
      decisionCount,
      terminal: result.terminal,
      ...progress(),
      winnerTeamId: result.winnerTeamId,
      loserTeamIds: result.loserTeamIds,
      endReason: "python_error",
      pythonAbnormalExit: true,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await client.close();
  }
}
