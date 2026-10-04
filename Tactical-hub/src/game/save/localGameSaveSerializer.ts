import type { GameState } from "../types";
import {
  LOCAL_GAME_RULES_VERSION,
  LOCAL_MAP_VERSION,
  LOCAL_MATCH_SAVE_SCHEMA_VERSION,
  type CreateLocalMatchSaveInput,
  type LocalMatchMetadataV1,
  type LocalMatchRestoreBundle,
  type LocalMatchSaveResult,
  type LocalMatchSaveV1,
} from "./localGameSaveTypes";
import { isLocalMatchContinuable, validateLocalMatchSave } from "./localGameSaveValidator";

function jsonClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

const omitted = Symbol("omitted optional JSON field");

function toSerializableDto(value: unknown, path = "$", ancestors = new Set<object>()): unknown | typeof omitted {
  if (value === undefined) return omitted;
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} must be finite`);
    return value;
  }
  if (typeof value !== "object") throw new Error(`${path} contains unsupported ${typeof value}`);
  if (ancestors.has(value)) throw new Error(`${path} contains a cycle`);
  const prototype = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) throw new Error(`${path} contains a class instance, Map, or Set`);
  ancestors.add(value);
  if (Array.isArray(value)) {
    const converted = value.map((entry, index) => {
      const result = toSerializableDto(entry, `${path}[${index}]`, ancestors);
      if (result === omitted) throw new Error(`${path}[${index}] is undefined`);
      return result;
    });
    ancestors.delete(value);
    return converted;
  }
  const converted: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    const result = toSerializableDto(entry, `${path}.${key}`, ancestors);
    if (result !== omitted) converted[key] = result;
  }
  ancestors.delete(value);
  return converted;
}

export function deriveLocalMatchMetadata(
  displayName: string,
  gameState: GameState,
  cpuSettings: CreateLocalMatchSaveInput["cpuSettings"],
): LocalMatchMetadataV1 {
  const nonNeutralTeams = gameState.teams.filter((team) => !team.isNeutral);
  return {
    displayName,
    turnNumber: gameState.turnNumber,
    phase: gameState.phase,
    humanTeamIds: nonNeutralTeams.filter((team) => cpuSettings[team.id] === "human").map((team) => team.id),
    controllersByTeamId: jsonClone(cpuSettings),
    activeTeamIds: nonNeutralTeams.filter((team) => team.status === "active").map((team) => team.id),
    defeatedTeamIds: nonNeutralTeams.filter((team) => team.status !== "active").map((team) => team.id),
    preview: {
      mapName: gameState.map.name,
      livingUnitCount: gameState.units.filter((unit) => unit.hp > 0 && unit.position.kind !== "removed").length,
      ownedBaseCountByTeamId: Object.fromEntries(nonNeutralTeams.map((team) => [team.id, gameState.bases.filter((base) => base.ownerTeamId === team.id).length])),
    },
  };
}

export function createLocalMatchSaveSnapshot(input: CreateLocalMatchSaveInput): LocalMatchSaveResult<LocalMatchSaveV1> {
  if (!isLocalMatchContinuable(input.gameState)) return { ok: false, error: { code: "MATCH_ALREADY_FINISHED", message: "Finished LOCAL matches are not resumable", path: "$.payload.gameState.teams" } };
  const runtimeCandidate: LocalMatchSaveV1 = {
    saveSchemaVersion: LOCAL_MATCH_SAVE_SCHEMA_VERSION,
    saveId: input.saveId,
    createdAt: input.createdAt,
    updatedAt: input.updatedAt,
    compatibility: {
      gameRulesVersion: LOCAL_GAME_RULES_VERSION,
      mapId: input.gameState.config.mapId,
      mapVersion: LOCAL_MAP_VERSION,
    },
    metadata: deriveLocalMatchMetadata(input.displayName, input.gameState, input.cpuSettings),
    payload: {
      gameState: input.gameState,
      cpuSettings: input.cpuSettings,
      cpuRuntime: input.cpuRuntime,
      heuristicPolicyState: input.heuristicPolicyState,
      ...(input.resumeUi ? { resumeUi: input.resumeUi } : {}),
    },
  };
  let candidate: LocalMatchSaveV1;
  try {
    candidate = toSerializableDto(runtimeCandidate) as LocalMatchSaveV1;
  } catch (error) {
    return { ok: false, error: { code: "STRUCTURAL_ERROR", message: error instanceof Error ? error.message : "Runtime snapshot is not JSON serializable" } };
  }
  const validated = validateLocalMatchSave(candidate);
  if (!validated.ok) return validated;
  return { ok: true, value: jsonClone(validated.value) };
}

export function serializeLocalMatchSave(save: unknown): LocalMatchSaveResult<string> {
  const validated = validateLocalMatchSave(save);
  if (!validated.ok) return validated;
  try {
    return { ok: true, value: JSON.stringify(validated.value) };
  } catch (error) {
    return { ok: false, error: { code: "STRUCTURAL_ERROR", message: error instanceof Error ? error.message : "Save serialization failed" } };
  }
}

export function parseLocalMatchSave(json: string): LocalMatchSaveResult<LocalMatchSaveV1> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    return { ok: false, error: { code: "MALFORMED_JSON", message: error instanceof Error ? error.message : "Save JSON parsing failed" } };
  }
  return validateLocalMatchSave(parsed);
}

export function createLocalMatchRestoreBundle(save: unknown): LocalMatchSaveResult<LocalMatchRestoreBundle> {
  const validated = validateLocalMatchSave(save);
  if (!validated.ok) return validated;
  const payload = validated.value.payload;
  return {
    ok: true,
    value: jsonClone({
      gameState: payload.gameState,
      cpuSettings: payload.cpuSettings,
      cpuRuntime: payload.cpuRuntime,
      heuristicPolicyState: payload.heuristicPolicyState,
      ...(payload.resumeUi ? { resumeUi: payload.resumeUi } : {}),
    }),
  };
}
