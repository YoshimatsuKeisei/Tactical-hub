import type { GameState, TeamStatus } from "../types";
import type { RlResult } from "./rlEnvironment";

export const PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL = 25;
export const PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS = 120;

export type PpoTeamDefeatDiagnosticV8 = {
  teamId: string;
  status: TeamStatus;
  predicateName: "active_non_neutral_team_without_owned_base";
  defeatPredicateSatisfied: boolean;
  kingDefeatPredicateSatisfied: null;
  predicateInputs: {
    teamIsNeutral: boolean;
    teamStatus: TeamStatus;
    ownedBaseIds: string[];
    kingUnits: Array<{
      unitId: string;
      positionKind: GameState["units"][number]["position"]["kind"];
      kingCampaignPresent: boolean;
    }>;
  };
  blockers: {
    withoutOwnedBasePath: Array<{
      condition: string;
      currentValue: boolean | TeamStatus | string[];
    }>;
    kingDefeatPath: Array<{
      condition: string;
      unitId?: string;
      currentValue: string | number | boolean;
    }>;
  };
};

export type PpoDefeatEnvironmentSnapshotV8 = {
  environmentIndex: number;
  currentEpisodeSeed: number;
  generation: number;
  episodeDecisionCount: number;
  currentTurn: number;
  currentPhase: GameState["phase"];
  terminal: boolean;
  endReason: RlResult["endReason"];
  activeNonNeutralTeamIds: string[];
  nonNeutralTeamStatuses: Array<{
    teamId: string;
    status: TeamStatus;
  }>;
  defeatDiagnostics: PpoTeamDefeatDiagnosticV8[];
};

export type PpoDefeatDiagnosticEventKindV8 =
  | "active_team_count_changed"
  | "active_team_status_changed"
  | "defeat_predicate_inputs_changed"
  | "two_team_entered"
  | "two_team_periodic"
  | "two_team_observation_window_completed";

export type PpoDefeatDiagnosticEventV8 = {
  eventKinds: PpoDefeatDiagnosticEventKindV8[];
  previousActiveTeamCount: number;
  currentActiveTeamCount: number;
  teamStatusChanges: Array<{
    teamId: string;
    previousStatus: TeamStatus;
    currentStatus: TeamStatus;
  }>;
  predicateInputChangedTeamIds: string[];
  twoTeamEntered: boolean;
  observationWindowCompleted: boolean;
  snapshot: PpoDefeatEnvironmentSnapshotV8;
};

export type PpoDefeatDiagnosticTrackerV8 = {
  previousSnapshot: PpoDefeatEnvironmentSnapshotV8;
  twoTeamEntryTurn?: number;
  nextPeriodicTurn?: number;
  observationWindowCompleted: boolean;
};

export type PpoTwoTeamObservationV8 = {
  selectedEnvironmentIndex: number;
  selectedEpisodeSeed: number;
  selectedGeneration: number;
  twoTeamEntryTurn: number;
  entrySnapshot: PpoDefeatEnvironmentSnapshotV8;
  latestSnapshot: PpoDefeatEnvironmentSnapshotV8;
  observationWindowCompleted: boolean;
};

function buildTeamDefeatDiagnostic(
  state: GameState,
  team: GameState["teams"][number],
): PpoTeamDefeatDiagnosticV8 {
  const ownedBaseIds = state.bases
    .filter((base) => base.ownerTeamId === team.id)
    .map((base) => base.id)
    .sort();
  const kingUnits = state.units
    .filter((unit) =>
      unit.ownerTeamId === team.id && unit.type === "king")
    .map((unit) => ({
      unitId: unit.id,
      positionKind: unit.position.kind,
      kingCampaignPresent: state.kingCampaignStates.some(
        (campaign) => campaign.kingUnitId === unit.id,
      ),
    }))
    .sort((left, right) => left.unitId.localeCompare(right.unitId));
  const defeatPredicateSatisfied =
    !team.isNeutral
    && team.status === "active"
    && ownedBaseIds.length === 0;

  const withoutOwnedBasePath:
    PpoTeamDefeatDiagnosticV8["blockers"]["withoutOwnedBasePath"] = [];
  if (team.isNeutral) {
    withoutOwnedBasePath.push({
      condition: "team.isNeutral === false",
      currentValue: true,
    });
  }
  if (team.status !== "active") {
    withoutOwnedBasePath.push({
      condition: 'team.status === "active"',
      currentValue: team.status,
    });
  }
  if (ownedBaseIds.length) {
    withoutOwnedBasePath.push({
      condition: "ownedBaseIds.length === 0",
      currentValue: ownedBaseIds,
    });
  }

  const kingDefeatPath:
    PpoTeamDefeatDiagnosticV8["blockers"]["kingDefeatPath"] = [];
  if (!kingUnits.length) {
    kingDefeatPath.push({
      condition: "king unit must exist in the resolution state",
      currentValue: false,
    });
  }
  for (const king of kingUnits) {
    if (king.positionKind !== "removed") {
      kingDefeatPath.push({
        condition: 'king.position.kind === "removed" during resolution',
        unitId: king.unitId,
        currentValue: king.positionKind,
      });
    } else if (!king.kingCampaignPresent) {
      kingDefeatPath.push({
        condition: "getKingCampaign(state, king.id) is present",
        unitId: king.unitId,
        currentValue: false,
      });
    } else {
      kingDefeatPath.push({
        condition: "resolution-local aliveAtBattleStart and candidateTeamIds are required",
        unitId: king.unitId,
        currentValue: "not_available_in_snapshot",
      });
    }
  }

  return {
    teamId: team.id,
    status: team.status,
    predicateName: "active_non_neutral_team_without_owned_base",
    defeatPredicateSatisfied,
    // The king path also depends on battle/flood resolution-local inputs
    // (aliveAtBattleStart and candidateTeamIds), so a state snapshot alone
    // cannot truthfully evaluate it.
    kingDefeatPredicateSatisfied: null,
    predicateInputs: {
      teamIsNeutral: Boolean(team.isNeutral),
      teamStatus: team.status,
      ownedBaseIds,
      kingUnits,
    },
    blockers: {
      withoutOwnedBasePath,
      kingDefeatPath,
    },
  };
}

export function createPpoDefeatEnvironmentSnapshotV8(input: {
  environmentIndex: number;
  currentEpisodeSeed: number;
  generation: number;
  episodeDecisionCount: number;
  state: GameState;
  result: RlResult;
}): PpoDefeatEnvironmentSnapshotV8 {
  const activeTeams = input.state.teams.filter(
    (team) => !team.isNeutral && team.status === "active",
  );
  return {
    environmentIndex: input.environmentIndex,
    currentEpisodeSeed: input.currentEpisodeSeed,
    generation: input.generation,
    episodeDecisionCount: input.episodeDecisionCount,
    currentTurn: input.state.turnNumber,
    currentPhase: input.state.phase,
    terminal: input.result.terminal,
    endReason: input.result.endReason,
    activeNonNeutralTeamIds: activeTeams.map((team) => team.id),
    nonNeutralTeamStatuses: input.state.teams
      .filter((team) => !team.isNeutral)
      .map((team) => ({ teamId: team.id, status: team.status })),
    defeatDiagnostics: activeTeams.map((team) =>
      buildTeamDefeatDiagnostic(input.state, team)),
  };
}

function diagnosticByTeam(
  snapshot: PpoDefeatEnvironmentSnapshotV8,
) {
  return new Map(
    snapshot.defeatDiagnostics.map((diagnostic) => [
      diagnostic.teamId,
      diagnostic,
    ]),
  );
}

export function createPpoDefeatDiagnosticTrackerV8(
  snapshot: PpoDefeatEnvironmentSnapshotV8,
): PpoDefeatDiagnosticTrackerV8 {
  return {
    previousSnapshot: snapshot,
    observationWindowCompleted: false,
  };
}

export function observePpoDefeatDiagnosticV8(
  tracker: PpoDefeatDiagnosticTrackerV8,
  snapshot: PpoDefeatEnvironmentSnapshotV8,
): PpoDefeatDiagnosticEventV8 | undefined {
  const previous = tracker.previousSnapshot;
  const previousStatus = new Map(
    previous.nonNeutralTeamStatuses.map((team) => [team.teamId, team.status]),
  );
  const teamStatusChanges = snapshot.nonNeutralTeamStatuses.flatMap((team) => {
    const status = previousStatus.get(team.teamId);
    return status !== undefined && status !== team.status
      ? [{
          teamId: team.teamId,
          previousStatus: status,
          currentStatus: team.status,
        }]
      : [];
  });
  const previousDiagnostics = diagnosticByTeam(previous);
  const predicateInputChangedTeamIds = snapshot.defeatDiagnostics
    .filter((diagnostic) =>
      JSON.stringify(previousDiagnostics.get(diagnostic.teamId)?.predicateInputs)
      !== JSON.stringify(diagnostic.predicateInputs))
    .map((diagnostic) => diagnostic.teamId);
  const previousActiveTeamCount =
    previous.activeNonNeutralTeamIds.length;
  const currentActiveTeamCount =
    snapshot.activeNonNeutralTeamIds.length;
  const eventKinds: PpoDefeatDiagnosticEventKindV8[] = [];
  if (
    (previousActiveTeamCount === 4 && currentActiveTeamCount === 3)
    || (previousActiveTeamCount === 3 && currentActiveTeamCount === 2)
    || (previousActiveTeamCount === 2 && currentActiveTeamCount === 1)
  ) {
    eventKinds.push("active_team_count_changed");
  }
  if (teamStatusChanges.length) {
    eventKinds.push("active_team_status_changed");
  }
  if (predicateInputChangedTeamIds.length) {
    eventKinds.push("defeat_predicate_inputs_changed");
  }

  let twoTeamEntered = false;
  if (
    currentActiveTeamCount === 2
    && previousActiveTeamCount !== 2
  ) {
    twoTeamEntered = true;
    tracker.twoTeamEntryTurn = snapshot.currentTurn;
    tracker.nextPeriodicTurn =
      snapshot.currentTurn + PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL;
    tracker.observationWindowCompleted = false;
    eventKinds.push("two_team_entered");
  }

  if (
    currentActiveTeamCount === 2
    && tracker.twoTeamEntryTurn !== undefined
    && tracker.nextPeriodicTurn !== undefined
    && snapshot.currentTurn >= tracker.nextPeriodicTurn
  ) {
    eventKinds.push("two_team_periodic");
    while (tracker.nextPeriodicTurn <= snapshot.currentTurn) {
      tracker.nextPeriodicTurn += PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL;
    }
  }

  let observationWindowCompleted = false;
  if (
    currentActiveTeamCount === 2
    && tracker.twoTeamEntryTurn !== undefined
    && !tracker.observationWindowCompleted
    && snapshot.currentTurn - tracker.twoTeamEntryTurn
      >= PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS
  ) {
    tracker.observationWindowCompleted = true;
    observationWindowCompleted = true;
    eventKinds.push("two_team_observation_window_completed");
  }

  tracker.previousSnapshot = snapshot;
  if (!eventKinds.length) return undefined;
  return {
    eventKinds: [...new Set(eventKinds)],
    previousActiveTeamCount,
    currentActiveTeamCount,
    teamStatusChanges,
    predicateInputChangedTeamIds,
    twoTeamEntered,
    observationWindowCompleted,
    snapshot,
  };
}

export function updatePpoTwoTeamObservationV8(
  current: PpoTwoTeamObservationV8 | undefined,
  event: PpoDefeatDiagnosticEventV8,
): PpoTwoTeamObservationV8 | undefined {
  if (!current) {
    if (!event.twoTeamEntered) return undefined;
    return {
      selectedEnvironmentIndex: event.snapshot.environmentIndex,
      selectedEpisodeSeed: event.snapshot.currentEpisodeSeed,
      selectedGeneration: event.snapshot.generation,
      twoTeamEntryTurn: event.snapshot.currentTurn,
      entrySnapshot: event.snapshot,
      latestSnapshot: event.snapshot,
      observationWindowCompleted: event.observationWindowCompleted,
    };
  }
  if (
    event.snapshot.environmentIndex !== current.selectedEnvironmentIndex
    || event.snapshot.currentEpisodeSeed !== current.selectedEpisodeSeed
    || event.snapshot.generation !== current.selectedGeneration
  ) {
    return current;
  }
  return {
    ...current,
    latestSnapshot: event.snapshot,
    observationWindowCompleted:
      current.observationWindowCompleted
      || event.observationWindowCompleted,
  };
}
