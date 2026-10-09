import { describe, expect, it } from "vitest";
import type { GameState } from "../types";
import {
  capturePpoCombatSnapshot,
  createPpoCombatDiagnostics,
  observePpoBattleDefeats,
  shouldCapturePpoCombatDiagnostics,
} from "../cpu/rlPpoCombatDiagnostics";

function state(input: {
  units: Array<{
    id: string;
    ownerTeamId: string;
    type: "infantry" | "cavalry";
    hp: number;
    formation?: "heavy";
    removed?: boolean;
  }>;
  logs?: Array<{
    id: string;
    turnNumber: number;
    type: "battle";
    message: string;
    relatedIds?: string[];
  }>;
}) {
  return {
    teams: [
      { id: "team-1", status: "active" },
      { id: "team-2", status: "active" },
      { id: "neutral", status: "neutral" },
    ],
    units: input.units.map((unit) => ({
      id: unit.id,
      ownerTeamId: unit.ownerTeamId,
      type: unit.type,
      hp: unit.hp,
      formation: unit.formation,
      statuses: [],
      position: unit.removed
        ? { kind: "removed", reason: "defeated" }
        : { kind: "tile", x: 0, y: 0 },
    })),
    logs: input.logs ?? [],
  } as unknown as GameState;
}

describe("PPO combat defeat diagnostics", () => {
  it("captures around attack_input because battle resolution is automatic", () => {
    expect(shouldCapturePpoCombatDiagnostics("attack_input")).toBe(true);
    expect(shouldCapturePpoCombatDiagnostics("production")).toBe(false);
    expect(shouldCapturePpoCombatDiagnostics("movement_input")).toBe(false);
  });

  it("separates heavy infantry and splits shared defeat credit", () => {
    const before = state({
      units: [
        { id: "i", ownerTeamId: "team-1", type: "infantry", hp: 1 },
        { id: "h", ownerTeamId: "team-1", type: "infantry", formation: "heavy", hp: 2 },
        { id: "t1", ownerTeamId: "team-2", type: "cavalry", hp: 2 },
        { id: "t2", ownerTeamId: "team-2", type: "cavalry", hp: 1 },
        { id: "n", ownerTeamId: "neutral", type: "cavalry", hp: 1 },
      ],
      logs: [{
        id: "old",
        turnNumber: 1,
        type: "battle",
        message: "previous",
      }],
    });
    const snapshot = capturePpoCombatSnapshot(before);
    const after = state({
      units: [
        { id: "i", ownerTeamId: "team-1", type: "infantry", hp: 1 },
        { id: "h", ownerTeamId: "team-1", type: "infantry", formation: "heavy", hp: 2 },
        { id: "t1", ownerTeamId: "team-2", type: "cavalry", hp: 0, removed: true },
        { id: "t2", ownerTeamId: "team-2", type: "cavalry", hp: 0, removed: true },
        { id: "n", ownerTeamId: "neutral", type: "cavalry", hp: 0, removed: true },
      ],
      logs: [
        {
          id: "old",
          turnNumber: 1,
          type: "battle",
          message: "previous",
        },
        {
          id: "hit-i",
          turnNumber: 2,
          type: "battle",
          message: "i -> t1 result: success, damage: 1.",
          relatedIds: ["i", "t1"],
        },
        {
          id: "hit-h1",
          turnNumber: 2,
          type: "battle",
          message: "h -> t1 result: success, damage: 1.",
          relatedIds: ["h", "t1"],
        },
        {
          id: "hit-h2",
          turnNumber: 2,
          type: "battle",
          message: "h -> t2 result: success, damage: 1.",
          relatedIds: ["h", "t2"],
        },
        {
          id: "hit-neutral-target",
          turnNumber: 2,
          type: "battle",
          message: "i -> n result: success, damage: 1.",
          relatedIds: ["i", "n"],
        },
      ],
    });

    const diagnostics = createPpoCombatDiagnostics();
    observePpoBattleDefeats(diagnostics, snapshot, after);

    expect(diagnostics.enemyDefeatedUnitCount).toBe(2);
    expect(diagnostics.unattributedEnemyDefeatedUnitCount).toBe(0);
    expect(
      diagnostics.enemyDefeatParticipationCountByAttackerType.infantry,
    ).toBe(1);
    expect(
      diagnostics.enemyDefeatParticipationCountByAttackerType.heavy_infantry,
    ).toBe(2);
    expect(diagnostics.enemyDefeatCreditByAttackerType.infantry).toBeCloseTo(0.5);
    expect(
      diagnostics.enemyDefeatCreditByAttackerType.heavy_infantry,
    ).toBeCloseTo(1.5);
  });

  it("records an unattributed direct defeat when no successful hit log is available", () => {
    const before = state({
      units: [
        { id: "t", ownerTeamId: "team-2", type: "cavalry", hp: 1 },
      ],
    });
    const after = state({
      units: [
        { id: "t", ownerTeamId: "team-2", type: "cavalry", hp: 0, removed: true },
      ],
    });
    const diagnostics = createPpoCombatDiagnostics();
    observePpoBattleDefeats(
      diagnostics,
      capturePpoCombatSnapshot(before),
      after,
    );
    expect(diagnostics.enemyDefeatedUnitCount).toBe(1);
    expect(diagnostics.unattributedEnemyDefeatedUnitCount).toBe(1);
  });
});
