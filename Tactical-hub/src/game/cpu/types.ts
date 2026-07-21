import type { AttackIntent, GameState, ProductionChoice, StrategistActionIntent, TeleportIntent, UnitPosition } from "../types";

export type TeamController = "human" | "random_cpu";
export type CpuTeamSettings = Record<string, TeamController>;

export type CpuActionLog = {
  id: string;
  turnNumber: number;
  phase: GameState["phase"];
  teamId?: string;
  action: string;
  detail?: string;
  error?: string;
};

export type CpuRuntime = {
  seed: number;
  rngState: number;
  contextKey: string;
  processedKeys: string[];
  completedProductionTeamIds: string[];
  completedAttackTeamIds: string[];
  hiddenAttackIntents: AttackIntent[];
  logs: CpuActionLog[];
  appliedStepCount: number;
  maxAppliedSteps: number;
  stoppedReason?: string;
};

export type CpuDecision =
  | { kind: "production"; teamId: string; actorKey: string; choice?: ProductionChoice }
  | { kind: "movement"; teamId: string; actorKey: string; unitId: string; to?: UnitPosition }
  | { kind: "teleport"; teamId: string; actorKey: string; strategistUnitId: string; intent?: TeleportIntent }
  | { kind: "submit_movement"; teamId: string }
  | { kind: "submit_team_production"; teamId: string }
  | { kind: "attack"; teamId: string; actorKey: string; intent: AttackIntent }
  | { kind: "complete_attack_team"; teamId: string }
  | { kind: "reward"; teamId: string; requestId: string; baseId: string; unitType: ProductionChoice["unitType"] }
  | { kind: "strategist"; teamId: string; actorKey: string; intent: StrategistActionIntent }
  | { kind: "submit_strategist"; teamId: string }
  | { kind: "resolve_production"; teamId: string }
  | { kind: "resolve_battle"; teamId: string }
  | { kind: "resolve_strategists"; teamId: string };

export type CpuPolicy = (state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings) => CpuDecision | undefined;

export function createCpuRuntime(seed: number, maxAppliedSteps = 10_000): CpuRuntime {
  const normalized = seed >>> 0;
  return {
    seed: normalized,
    rngState: normalized || 1,
    contextKey: "",
    processedKeys: [],
    completedProductionTeamIds: [],
    completedAttackTeamIds: [],
    hiddenAttackIntents: [],
    logs: [],
    appliedStepCount: 0,
    maxAppliedSteps,
  };
}
