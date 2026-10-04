import type { HeuristicCpuPolicyState } from "../cpu/heuristicCpuPolicy";
import type { CpuRuntime, CpuTeamSettings } from "../cpu/types";
import type { GameState, TurnState } from "../types";
import { testMap4p } from "../maps/testMap4p";

export const LOCAL_MATCH_SAVE_SCHEMA_VERSION = 1 as const;
export const LOCAL_GAME_RULES_VERSION = 1 as const;
export const LOCAL_MAP_ID = testMap4p.id;
export const LOCAL_MAP_VERSION = 1 as const;

/** These DTOs are plain JSON data. The serializer rejects non-JSON values before cloning. */
export type LocalGameStateDtoV1 = GameState;
export type LocalCpuRuntimeDtoV1 = CpuRuntime;
export type LocalHeuristicPolicyStateDtoV1 = HeuristicCpuPolicyState;

export type LocalMatchCompatibilityV1 = {
  gameRulesVersion: typeof LOCAL_GAME_RULES_VERSION;
  mapId: string;
  mapVersion: typeof LOCAL_MAP_VERSION;
};

export type LocalMatchPreviewV1 = {
  mapName: string;
  livingUnitCount: number;
  ownedBaseCountByTeamId: Record<string, number>;
};

export type LocalMatchMetadataV1 = {
  displayName: string;
  turnNumber: number;
  phase: TurnState["phase"];
  humanTeamIds: string[];
  controllersByTeamId: CpuTeamSettings;
  activeTeamIds: string[];
  defeatedTeamIds: string[];
  preview: LocalMatchPreviewV1;
};

export type LocalMatchResumeUiV1 = {
  viewerTeamId?: string;
};

export type LocalMatchSavePayloadV1 = {
  gameState: LocalGameStateDtoV1;
  cpuSettings: CpuTeamSettings;
  cpuRuntime: LocalCpuRuntimeDtoV1;
  heuristicPolicyState: LocalHeuristicPolicyStateDtoV1;
  resumeUi?: LocalMatchResumeUiV1;
};

export type LocalMatchSaveV1 = {
  saveSchemaVersion: typeof LOCAL_MATCH_SAVE_SCHEMA_VERSION;
  saveId: string;
  createdAt: string;
  updatedAt: string;
  compatibility: LocalMatchCompatibilityV1;
  metadata: LocalMatchMetadataV1;
  payload: LocalMatchSavePayloadV1;
};

export type CreateLocalMatchSaveInput = {
  saveId: string;
  createdAt: string;
  updatedAt: string;
  displayName: string;
  gameState: GameState;
  cpuSettings: CpuTeamSettings;
  cpuRuntime: CpuRuntime;
  heuristicPolicyState: HeuristicCpuPolicyState;
  resumeUi?: LocalMatchResumeUiV1;
};

export type LocalMatchRestoreBundle = {
  gameState: GameState;
  cpuSettings: CpuTeamSettings;
  cpuRuntime: CpuRuntime;
  heuristicPolicyState: HeuristicCpuPolicyState;
  resumeUi?: LocalMatchResumeUiV1;
};

export type LocalMatchSaveErrorCode =
  | "MALFORMED_JSON"
  | "STRUCTURAL_ERROR"
  | "UNSUPPORTED_SCHEMA_VERSION"
  | "UNSUPPORTED_RULES_VERSION"
  | "MAP_MISMATCH"
  | "UNSUPPORTED_CPU_CONTROLLER"
  | "MATCH_ALREADY_FINISHED"
  | "INVALID_GAME_STATE"
  | "INVALID_CPU_RUNTIME"
  | "INVALID_HEURISTIC_STATE";

export type LocalMatchSaveError = {
  code: LocalMatchSaveErrorCode;
  message: string;
  path?: string;
};

export type LocalMatchSaveResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: LocalMatchSaveError };
