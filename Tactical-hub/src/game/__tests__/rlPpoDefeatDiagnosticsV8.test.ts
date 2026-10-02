import { describe, expect, it } from "vitest";
import { createHeadlessInitialState } from "../cpu/headlessSimulation";
import {
  createPpoDefeatDiagnosticTrackerV8,
  createPpoDefeatEnvironmentSnapshotV8,
  observePpoDefeatDiagnosticV8,
  PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS,
  PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL,
  updatePpoTwoTeamObservationV8,
  type PpoDefeatEnvironmentSnapshotV8,
} from "../cpu/rlPpoDefeatDiagnosticsV8";

function snapshot(input: {
  turn: number;
  activeTeamIds: string[];
  environmentIndex?: number;
}) {
  const state = createHeadlessInitialState(4);
  state.turnNumber = input.turn;
  state.turnState.turnNumber = input.turn;
  for (const team of state.teams) {
    if (
      !team.isNeutral
      && !input.activeTeamIds.includes(team.id)
    ) {
      team.status = "defeated";
    }
  }
  return createPpoDefeatEnvironmentSnapshotV8({
    environmentIndex: input.environmentIndex ?? 0,
    currentEpisodeSeed: 7,
    generation: 0,
    episodeDecisionCount: input.turn * 10,
    state,
    result: {
      terminal: input.activeTeamIds.length <= 1,
      winnerTeamId: input.activeTeamIds.length === 1
        ? input.activeTeamIds[0]
        : undefined,
      loserTeamIds: state.teams
        .filter((team) =>
          !team.isNeutral && team.status === "defeated")
        .map((team) => team.id),
      endReason: input.activeTeamIds.length <= 1
        ? "victory"
        : "ongoing",
      actionCount: input.turn * 10,
      rewards: {},
    },
  });
}

describe("PPO V8 defeat diagnostics", () => {
  it("summarizes the exact no-owned-base predicate inputs", () => {
    const state = createHeadlessInitialState(4);
    state.kingCampaignStates.push({
      kingUnitId: "home-1-king",
      kingTeamId: "team-1",
      contributions: [],
    });
    const before = createPpoDefeatEnvironmentSnapshotV8({
      environmentIndex: 0,
      currentEpisodeSeed: 7,
      generation: 0,
      episodeDecisionCount: 12,
      state,
      result: {
        terminal: false,
        loserTeamIds: [],
        endReason: "ongoing",
        actionCount: 12,
        rewards: {},
      },
    });
    const teamBefore = before.defeatDiagnostics.find(
      (entry) => entry.teamId === "team-1",
    )!;

    expect(teamBefore).toMatchObject({
      teamId: "team-1",
      status: "active",
      predicateName:
        "active_non_neutral_team_without_owned_base",
      defeatPredicateSatisfied: false,
      kingDefeatPredicateSatisfied: null,
      predicateInputs: {
        teamIsNeutral: false,
        teamStatus: "active",
        ownedBaseIds: ["home-1"],
        kingUnits: [{
          unitId: "home-1-king",
          positionKind: "base",
          kingCampaignPresent: true,
        }],
      },
      blockers: {
        withoutOwnedBasePath: [{
          condition: "ownedBaseIds.length === 0",
          currentValue: ["home-1"],
        }],
      },
    });
    expect(teamBefore.predicateInputs.kingUnits[0]).not.toHaveProperty(
      "hp",
    );

    for (const base of state.bases) {
      if (base.ownerTeamId === "team-1") {
        base.ownerTeamId = "neutral";
      }
    }
    const after = createPpoDefeatEnvironmentSnapshotV8({
      environmentIndex: 0,
      currentEpisodeSeed: 7,
      generation: 0,
      episodeDecisionCount: 13,
      state,
      result: {
        terminal: false,
        loserTeamIds: [],
        endReason: "ongoing",
        actionCount: 13,
        rewards: {},
      },
    });
    const teamAfter = after.defeatDiagnostics.find(
      (entry) => entry.teamId === "team-1",
    )!;
    expect(teamAfter.defeatPredicateSatisfied).toBe(true);
    expect(teamAfter.predicateInputs.ownedBaseIds).toEqual([]);
    expect(teamAfter.blockers.withoutOwnedBasePath).toEqual([]);
  });

  it("selects the first two-team episode and completes its bounded turn window", () => {
    expect(PPO_DEFEAT_DIAGNOSTIC_TURN_INTERVAL).toBe(25);
    expect(PPO_DEFEAT_DIAGNOSTIC_OBSERVATION_TURNS).toBe(120);

    const initial = snapshot({
      turn: 90,
      activeTeamIds: ["team-1", "team-2", "team-3", "team-4"],
    });
    const tracker = createPpoDefeatDiagnosticTrackerV8(initial);
    const threeTeams = observePpoDefeatDiagnosticV8(
      tracker,
      snapshot({
        turn: 95,
        activeTeamIds: ["team-1", "team-2", "team-3"],
      }),
    )!;
    expect(threeTeams.eventKinds).toContain(
      "active_team_count_changed",
    );

    const entry = observePpoDefeatDiagnosticV8(
      tracker,
      snapshot({
        turn: 100,
        activeTeamIds: ["team-1", "team-2"],
      }),
    )!;
    expect(entry.eventKinds).toContain("two_team_entered");
    let observation = updatePpoTwoTeamObservationV8(
      undefined,
      entry,
    )!;

    const beforePeriodic = observePpoDefeatDiagnosticV8(
      tracker,
      snapshot({
        turn: 124,
        activeTeamIds: ["team-1", "team-2"],
      }),
    );
    expect(beforePeriodic).toBeUndefined();
    const periodic = observePpoDefeatDiagnosticV8(
      tracker,
      snapshot({
        turn: 125,
        activeTeamIds: ["team-1", "team-2"],
      }),
    )!;
    expect(periodic.eventKinds).toContain("two_team_periodic");
    observation = updatePpoTwoTeamObservationV8(
      observation,
      periodic,
    )!;

    const completion = observePpoDefeatDiagnosticV8(
      tracker,
      snapshot({
        turn: 220,
        activeTeamIds: ["team-1", "team-2"],
      }),
    )!;
    expect(completion.eventKinds).toContain(
      "two_team_observation_window_completed",
    );
    observation = updatePpoTwoTeamObservationV8(
      observation,
      completion,
    )!;
    expect(observation).toMatchObject({
      selectedEnvironmentIndex: 0,
      selectedEpisodeSeed: 7,
      selectedGeneration: 0,
      twoTeamEntryTurn: 100,
      observationWindowCompleted: true,
      latestSnapshot: { currentTurn: 220 },
    });
  });

  it("does not replace the selected episode with another environment", () => {
    const entry = {
      eventKinds: ["two_team_entered" as const],
      previousActiveTeamCount: 3,
      currentActiveTeamCount: 2,
      teamStatusChanges: [],
      predicateInputChangedTeamIds: [],
      twoTeamEntered: true,
      observationWindowCompleted: false,
      snapshot: snapshot({
        turn: 100,
        activeTeamIds: ["team-1", "team-2"],
      }),
    };
    const selected = updatePpoTwoTeamObservationV8(undefined, entry)!;
    const otherSnapshot: PpoDefeatEnvironmentSnapshotV8 = snapshot({
      turn: 150,
      activeTeamIds: ["team-2", "team-3"],
      environmentIndex: 1,
    });
    const unchanged = updatePpoTwoTeamObservationV8(selected, {
      ...entry,
      snapshot: otherSnapshot,
    });
    expect(unchanged).toBe(selected);
  });
});
