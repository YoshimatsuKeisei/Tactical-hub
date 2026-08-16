import type { GameState } from "../types";
import { advanceCpuOneStep, syncCpuContext, type CpuStepResult } from "./cpuStep";
import { getVisualCpuActorTeamId } from "./cpuPolicyRouter";
import { encodeRlLegalActionsV1 } from "./rlActionEncoder";
import { buildRlObservation, enumerateRlDecisionsV1 } from "./rlEnvironment";
import { createRlFeatureSpecV1 } from "./rlFeatureSpec";
import { encodeRlObservationV1 } from "./rlObservationEncoder";
import type { BrowserBcInferenceClient } from "./browserBcClient";
import type { CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";

export function createBcDecisionKey(state: GameState, runtime: CpuRuntime, teamId: string, actionKeys: readonly string[]) {
  return [state.turnNumber, state.phase, teamId, runtime.appliedStepCount, runtime.contextKey, runtime.processedKeys.join(","), actionKeys.join("|")].join(":");
}

export function createBcInferenceRequest(state: GameState, sourceRuntime: CpuRuntime, settings: CpuTeamSettings) {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  syncCpuContext(runtime, state);
  const teamId = getVisualCpuActorTeamId(state, runtime, settings);
  if (!teamId || settings[teamId] !== "bc_cpu") return undefined;
  const decisions = enumerateRlDecisionsV1(state, runtime, (candidateTeamId) => candidateTeamId === teamId);
  if (!decisions.length || decisions.some((entry) => entry.action.actorTeamId !== teamId)) {
    return { teamId, runtime, decisions: [], request: undefined };
  }
  const observation = buildRlObservation(state, teamId, teamId);
  const legalActions = decisions.map((entry) => entry.action);
  const encodedLegalActions = encodeRlLegalActionsV1(observation, legalActions);
  return {
    teamId,
    runtime,
    decisions,
    request: {
      decisionKey: createBcDecisionKey(state, runtime, teamId, encodedLegalActions.actionKeys),
      featureSpec: createRlFeatureSpecV1(observation),
      observation: encodeRlObservationV1(observation),
      legalActions: encodedLegalActions,
    },
  };
}

function failedResult(state: GameState, sourceRuntime: CpuRuntime, teamId: string | undefined, error: unknown): CpuStepResult {
  const runtime = structuredClone(sourceRuntime) as CpuRuntime;
  const message = error instanceof Error ? error.message : String(error);
  runtime.stoppedReason = message.includes("unavailable") ? message : `BC inference error: ${message}`;
  runtime.logs.push({
    id: `cpu-${runtime.logs.length}`,
    turnNumber: state.turnNumber,
    phase: state.phase,
    teamId,
    action: "BC inference",
    error: runtime.stoppedReason,
  });
  return { state, runtime, applied: false };
}

export async function advanceVisualCpuOneStepWithBc(
  state: GameState,
  runtime: CpuRuntime,
  settings: CpuTeamSettings,
  synchronousPolicy: CpuPolicy,
  client: BrowserBcInferenceClient,
  isStillCurrent: () => boolean = () => true,
): Promise<CpuStepResult> {
  const prepared = createBcInferenceRequest(state, runtime, settings);
  if (!prepared) return advanceCpuOneStep(state, runtime, settings, synchronousPolicy);
  if (!prepared.request) return { state, runtime: prepared.runtime, applied: false };
  try {
    const response = await client.infer(prepared.request);
    if (!isStillCurrent()) return { state, runtime, applied: false };
    const selected = prepared.decisions.find((entry) => entry.action.actionKey === response.actionKey);
    if (!selected) return failedResult(state, runtime, prepared.teamId, `BC inference returned illegal or stale actionKey: ${response.actionKey}`);
    const result = advanceCpuOneStep(state, prepared.runtime, settings, () => selected.decision);
    result.runtime.logs.push({
      id: `cpu-${result.runtime.logs.length}`,
      turnNumber: state.turnNumber,
      phase: state.phase,
      teamId: prepared.teamId,
      action: "BC selected",
      detail: response.actionKey,
    });
    return result;
  } catch (error) {
    if (!isStillCurrent()) return { state, runtime, applied: false };
    const message = error instanceof Error ? error.message : String(error);
    return failedResult(state, runtime, prepared.teamId, message.includes("fetch") ? "BC inference server unavailable" : error);
  }
}
