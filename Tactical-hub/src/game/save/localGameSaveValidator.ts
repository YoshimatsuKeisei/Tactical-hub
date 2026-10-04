import { testMap4p } from "../maps/testMap4p";
import type { GameState, Unit, UnitPosition } from "../types";
import type { CpuRuntime, CpuTeamSettings, TeamController } from "../cpu/types";
import type { HeuristicCpuPolicyState } from "../cpu/heuristicCpuPolicy";
import {
  LOCAL_GAME_RULES_VERSION,
  LOCAL_MAP_ID,
  LOCAL_MAP_VERSION,
  LOCAL_MATCH_SAVE_SCHEMA_VERSION,
  type LocalMatchSaveError,
  type LocalMatchSaveResult,
  type LocalMatchSaveV1,
} from "./localGameSaveTypes";

const MAX_ARRAY_LENGTH = 100_000;
const MAX_OBJECT_KEYS = 100_000;
const MAX_STRING_LENGTH = 100_000;
const MAX_ID_LENGTH = 256;
const MAX_DEPTH = 40;
const MAX_JSON_NODES = 1_000_000;
const UINT32_MAX = 0xffff_ffff;

const PHASES = ["production", "movement_input", "movement_resolution", "attack_input", "battle_resolution", "capture_resolution", "reward_placement", "strategist_action_input", "strategist_action_resolution"] as const;
const TEAM_STATUSES = ["active", "defeated", "eliminated", "neutral"] as const;
const UNIT_TYPES = ["king", "infantry", "cavalry", "archer", "engineer", "ninja", "apprentice_ninja", "strategist"] as const;
const STRATEGIST_ROLES = ["encourage", "builder", "teleporter"] as const;
const CONTROLLERS: TeamController[] = ["human", "random_cpu", "heuristic_cpu", "bc_cpu"];
const TERRAIN_TYPES = ["outside", "road", "lake", "base", "baseGate", "reorganize"] as const;
const LOG_TYPES = ["setup", "production", "movement", "battle", "siege", "capture", "reward", "construction"] as const;
const REWARD_TYPES = ["capture_reward", "contribution_compensation", "king_conquest_reward", "king_contribution_compensation", "overridden_capture_compensation"] as const;
const STRATEGIST_ACTIONS = ["place_bridge", "reset_bridge", "place_obstacle", "reset_obstacle", "pass"] as const;
const CONSTRUCTION_KINDS = ["bridge", "obstacle"] as const;

type UnknownRecord = Record<string, unknown>;
type ShapeIssue = { path: string; message: string };

function isRecord(value: unknown): value is UnknownRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function inspectJsonValue(value: unknown): ShapeIssue | undefined {
  let nodes = 0;
  const ancestors = new Set<object>();
  const visit = (current: unknown, path: string, depth: number): ShapeIssue | undefined => {
    nodes += 1;
    if (nodes > MAX_JSON_NODES) return { path, message: "JSON value is too large" };
    if (depth > MAX_DEPTH) return { path, message: "JSON nesting is too deep" };
    if (current === null || typeof current === "boolean") return undefined;
    if (typeof current === "string") return current.length <= MAX_STRING_LENGTH ? undefined : { path, message: "string is too long" };
    if (typeof current === "number") return Number.isFinite(current) ? undefined : { path, message: "number must be finite" };
    if (typeof current !== "object") return { path, message: `unsupported JSON value: ${typeof current}` };
    if (ancestors.has(current)) return { path, message: "cyclic value is not serializable" };
    ancestors.add(current);
    if (Array.isArray(current)) {
      if (current.length > MAX_ARRAY_LENGTH) return { path, message: "array is too large" };
      for (let index = 0; index < current.length; index += 1) {
        const issue = visit(current[index], `${path}[${index}]`, depth + 1);
        if (issue) return issue;
      }
    } else {
      if (!isRecord(current)) return { path, message: "class instances, Map, and Set are not supported" };
      const entries = Object.entries(current);
      if (entries.length > MAX_OBJECT_KEYS) return { path, message: "object has too many fields" };
      for (const [key, entry] of entries) {
        const issue = visit(entry, `${path}.${key}`, depth + 1);
        if (issue) return issue;
      }
    }
    ancestors.delete(current);
    return undefined;
  };
  return visit(value, "$", 0);
}

class ShapeChecker {
  issue?: ShapeIssue;

  fail(path: string, message: string): false {
    this.issue ??= { path, message };
    return false;
  }

  record(value: unknown, path: string): value is UnknownRecord {
    return isRecord(value) || this.fail(path, "must be an object");
  }

  array(value: unknown, path: string): value is unknown[] {
    return Array.isArray(value) || this.fail(path, "must be an array");
  }

  required(record: UnknownRecord, keys: readonly string[], path: string) {
    for (const key of keys) if (!Object.prototype.hasOwnProperty.call(record, key)) return this.fail(`${path}.${key}`, "required field is missing");
    return true;
  }

  string(value: unknown, path: string, allowEmpty = false) {
    return (typeof value === "string" && value.length <= MAX_STRING_LENGTH && (allowEmpty || value.length > 0))
      || this.fail(path, allowEmpty ? "must be a string" : "must be a non-empty string");
  }

  id(value: unknown, path: string) {
    return (typeof value === "string" && value.length > 0 && value.length <= MAX_ID_LENGTH) || this.fail(path, "must be a non-empty bounded ID");
  }

  boolean(value: unknown, path: string) {
    return typeof value === "boolean" || this.fail(path, "must be a boolean");
  }

  number(value: unknown, path: string, options: { integer?: boolean; min?: number; max?: number } = {}) {
    if (typeof value !== "number" || !Number.isFinite(value)) return this.fail(path, "must be a finite number");
    if (options.integer && !Number.isInteger(value)) return this.fail(path, "must be an integer");
    if (options.min !== undefined && value < options.min) return this.fail(path, `must be at least ${options.min}`);
    if (options.max !== undefined && value > options.max) return this.fail(path, `must be at most ${options.max}`);
    return true;
  }

  enumeration(value: unknown, allowed: readonly string[], path: string) {
    return (typeof value === "string" && allowed.includes(value)) || this.fail(path, `must be one of: ${allowed.join(", ")}`);
  }
}

function validateStringArray(value: unknown, path: string, checker: ShapeChecker, ids = true) {
  if (!checker.array(value, path)) return false;
  return value.every((entry, index) => ids ? checker.id(entry, `${path}[${index}]`) : checker.string(entry, `${path}[${index}]`, true));
}

function validateCoord(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["x", "y"], path)) return false;
  return checker.number(value.x, `${path}.x`, { integer: true }) && checker.number(value.y, `${path}.y`, { integer: true });
}

function validatePosition(value: unknown, path: string, checker: ShapeChecker): value is UnitPosition {
  if (!checker.record(value, path) || !checker.required(value, ["kind"], path) || !checker.string(value.kind, `${path}.kind`)) return false;
  switch (value.kind) {
    case "tile": case "water":
      return checker.required(value, ["x", "y"], path)
        && checker.number(value.x, `${path}.x`, { integer: true })
        && checker.number(value.y, `${path}.y`, { integer: true });
    case "base":
      return checker.required(value, ["baseId", "slotId"], path) && checker.id(value.baseId, `${path}.baseId`) && checker.id(value.slotId, `${path}.slotId`);
    case "bridge":
      return checker.required(value, ["bridgeId", "cellIndex"], path) && checker.id(value.bridgeId, `${path}.bridgeId`) && checker.number(value.cellIndex, `${path}.cellIndex`, { integer: true, min: 0 });
    case "removed":
      return checker.required(value, ["reason"], path) && checker.enumeration(value.reason, ["defeated", "water_trap", "king_defeat_reset", "team_defeat", "merged"], `${path}.reason`);
    default:
      return checker.fail(`${path}.kind`, "unknown unit position kind");
  }
}

function validateAttackIntent(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["teamId", "attackerUnitId", "pass"], path)) return false;
  if (!checker.id(value.teamId, `${path}.teamId`) || !checker.id(value.attackerUnitId, `${path}.attackerUnitId`) || !checker.boolean(value.pass, `${path}.pass`)) return false;
  if (value.target !== undefined) {
    const targetPath = `${path}.target`;
    if (!checker.record(value.target, targetPath) || !checker.required(value.target, ["kind", "unitId"], targetPath)) return false;
    if (value.target.kind !== "unit") return checker.fail(`${targetPath}.kind`, "must be unit");
    if (!checker.id(value.target.unitId, `${targetPath}.unitId`)) return false;
    for (const key of ["baseId", "slotId"] as const) if (value.target[key] !== undefined && !checker.id(value.target[key], `${targetPath}.${key}`)) return false;
    for (const key of ["baseSuccessDenominator", "finalSuccessDenominator"] as const) if (value.target[key] !== undefined && !checker.number(value.target[key], `${targetPath}.${key}`, { integer: true, min: 1 })) return false;
    if (value.target.encouraged !== undefined && !checker.boolean(value.target.encouraged, `${targetPath}.encouraged`)) return false;
  }
  return true;
}

function validateProductionChoice(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["teamId", "baseId", "unitType"], path)) return false;
  if (!checker.id(value.teamId, `${path}.teamId`) || !checker.id(value.baseId, `${path}.baseId`) || !checker.enumeration(value.unitType, UNIT_TYPES, `${path}.unitType`)) return false;
  return value.strategistRole === undefined || checker.enumeration(value.strategistRole, STRATEGIST_ROLES, `${path}.strategistRole`);
}

function validateUnit(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["id", "ownerTeamId", "type", "hp", "position", "statuses"], path)) return false;
  if (!checker.id(value.id, `${path}.id`) || !checker.id(value.ownerTeamId, `${path}.ownerTeamId`) || !checker.enumeration(value.type, UNIT_TYPES, `${path}.type`)
    || !checker.number(value.hp, `${path}.hp`, { min: 0 }) || !validatePosition(value.position, `${path}.position`, checker) || !checker.array(value.statuses, `${path}.statuses`)) return false;
  for (let index = 0; index < value.statuses.length; index += 1) {
    const status = value.statuses[index]; const statusPath = `${path}.statuses[${index}]`;
    if (!checker.record(status, statusPath) || !checker.required(status, ["kind"], statusPath) || !checker.enumeration(status.kind, ["retreating", "encouraged", "cannot_attack"], `${statusPath}.kind`)) return false;
    if (status.kind === "retreating" && (!checker.required(status, ["retreatTargetBaseId"], statusPath) || !checker.id(status.retreatTargetBaseId, `${statusPath}.retreatTargetBaseId`))) return false;
    if (status.remainingTurns !== undefined && !checker.number(status.remainingTurns, `${statusPath}.remainingTurns`, { integer: true, min: 0 })) return false;
    if (status.sourceId !== undefined && !checker.id(status.sourceId, `${statusPath}.sourceId`)) return false;
  }
  if (value.role !== undefined && !checker.enumeration(value.role, STRATEGIST_ROLES, `${path}.role`)) return false;
  return value.formation === undefined || value.formation === "heavy" || checker.fail(`${path}.formation`, "must be heavy");
}

function validateBase(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["id", "name", "type", "ownerTeamId", "coords", "slots"], path)) return false;
  if (!checker.id(value.id, `${path}.id`) || !checker.string(value.name, `${path}.name`) || !checker.enumeration(value.type, ["home", "neutral"], `${path}.type`)
    || !checker.id(value.ownerTeamId, `${path}.ownerTeamId`) || !checker.array(value.coords, `${path}.coords`) || !checker.array(value.slots, `${path}.slots`)) return false;
  if (!value.coords.every((coord, index) => validateCoord(coord, `${path}.coords[${index}]`, checker))) return false;
  for (let index = 0; index < value.slots.length; index += 1) {
    const slot = value.slots[index]; const slotPath = `${path}.slots[${index}]`;
    if (!checker.record(slot, slotPath) || !checker.required(slot, ["id", "baseId", "kind", "localRow", "localCol"], slotPath)) return false;
    if (!checker.id(slot.id, `${slotPath}.id`) || !checker.id(slot.baseId, `${slotPath}.baseId`) || !checker.enumeration(slot.kind, ["front", "protected"], `${slotPath}.kind`)
      || !checker.number(slot.localRow, `${slotPath}.localRow`, { integer: true, min: 0, max: 1 }) || !checker.number(slot.localCol, `${slotPath}.localCol`, { integer: true, min: 0, max: 1 })) return false;
    if (slot.unitId !== undefined && !checker.id(slot.unitId, `${slotPath}.unitId`)) return false;
  }
  if (value.protectedSlotId !== undefined && !checker.id(value.protectedSlotId, `${path}.protectedSlotId`)) return false;
  return value.occupationPriorityTeamId === undefined || checker.id(value.occupationPriorityTeamId, `${path}.occupationPriorityTeamId`);
}

function validateBoardMap(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["id", "name", "width", "height", "tiles", "bases"], path)) return false;
  if (!checker.id(value.id, `${path}.id`) || !checker.string(value.name, `${path}.name`) || !checker.number(value.width, `${path}.width`, { integer: true, min: 1 })
    || !checker.number(value.height, `${path}.height`, { integer: true, min: 1 }) || !checker.array(value.tiles, `${path}.tiles`) || !checker.array(value.bases, `${path}.bases`)) return false;
  for (let index = 0; index < value.tiles.length; index += 1) {
    const tile = value.tiles[index]; const tilePath = `${path}.tiles[${index}]`;
    if (!checker.record(tile, tilePath) || !checker.required(tile, ["x", "y", "symbol", "terrain"], tilePath)) return false;
    if (!checker.number(tile.x, `${tilePath}.x`, { integer: true }) || !checker.number(tile.y, `${tilePath}.y`, { integer: true })
      || !checker.string(tile.symbol, `${tilePath}.symbol`, true) || !checker.enumeration(tile.terrain, TERRAIN_TYPES, `${tilePath}.terrain`)) return false;
    if (tile.baseId !== undefined && !checker.id(tile.baseId, `${tilePath}.baseId`)) return false;
    if (tile.roadSectionId !== undefined && !checker.id(tile.roadSectionId, `${tilePath}.roadSectionId`)) return false;
  }
  return value.bases.every((base, index) => validateBase(base, `${path}.bases[${index}]`, checker));
}

function validateTeam(value: unknown, path: string, checker: ShapeChecker) {
  if (!checker.record(value, path) || !checker.required(value, ["id", "name", "color", "status", "controlledBaseIds"], path)) return false;
  if (!checker.id(value.id, `${path}.id`) || !checker.string(value.name, `${path}.name`) || !checker.string(value.color, `${path}.color`)
    || !checker.enumeration(value.status, TEAM_STATUSES, `${path}.status`) || !validateStringArray(value.controlledBaseIds, `${path}.controlledBaseIds`, checker)) return false;
  if (value.homeBaseId !== undefined && !checker.id(value.homeBaseId, `${path}.homeBaseId`)) return false;
  if (value.isNeutral !== undefined && !checker.boolean(value.isNeutral, `${path}.isNeutral`)) return false;
  if (value.defeatedUnitCount !== undefined && !checker.number(value.defeatedUnitCount, `${path}.defeatedUnitCount`, { integer: true, min: 0 })) return false;
  if (value.conqueredTeamIds !== undefined && !validateStringArray(value.conqueredTeamIds, `${path}.conqueredTeamIds`, checker)) return false;
  return value.constructionCapacityBonusStrategistId === undefined || checker.id(value.constructionCapacityBonusStrategistId, `${path}.constructionCapacityBonusStrategistId`);
}

function validateGameStateStructure(value: unknown, path: string, checker: ShapeChecker): value is GameState {
  if (!checker.record(value, path) || !checker.required(value, ["config", "map", "turnNumber", "phase", "teams", "units", "bases", "unitTurnFlags", "turnState", "logs", "siegeStates", "rewardPlacementRequests", "kingCampaignStates", "constructions", "strategistActionIntents", "strategistSubmittedTeamIds", "strategistCooldowns", "movementSeatOrderTeamIds", "movementOrderStartIndex", "movementOrderTeamIds", "movementCompletedTeamIds", "productionCompletedTeamIdsThisTurn", "teleportIntents", "teleportCooldowns", "movedUnitIdsThisMovementPhase"], path)) return false;
  if (!checker.record(value.config, `${path}.config`) || !checker.required(value.config, ["playerCount", "productionInterval", "mapId"], `${path}.config`)
    || !checker.number(value.config.playerCount, `${path}.config.playerCount`, { integer: true, min: 1 }) || !checker.number(value.config.productionInterval, `${path}.config.productionInterval`, { integer: true, min: 1 })
    || !checker.id(value.config.mapId, `${path}.config.mapId`) || !validateBoardMap(value.map, `${path}.map`, checker)
    || !checker.number(value.turnNumber, `${path}.turnNumber`, { integer: true, min: 1 }) || !checker.enumeration(value.phase, PHASES, `${path}.phase`)) return false;
  if (!checker.array(value.teams, `${path}.teams`) || !value.teams.every((team, index) => validateTeam(team, `${path}.teams[${index}]`, checker))) return false;
  if (!checker.array(value.units, `${path}.units`) || !value.units.every((unit, index) => validateUnit(unit, `${path}.units[${index}]`, checker))) return false;
  if (!checker.array(value.bases, `${path}.bases`) || !value.bases.every((base, index) => validateBase(base, `${path}.bases[${index}]`, checker))) return false;

  if (!checker.array(value.unitTurnFlags, `${path}.unitTurnFlags`)) return false;
  for (let index = 0; index < value.unitTurnFlags.length; index += 1) {
    const flag = value.unitTurnFlags[index]; const flagPath = `${path}.unitTurnFlags[${index}]`;
    if (!checker.record(flag, flagPath) || !checker.required(flag, ["unitId", "battleTurnNumber", "wasAliveAtBattleStart", "survivedPreviousBattle", "attackedInPreviousBattle", "wasTargetedInPreviousBattle", "retreatEligible"], flagPath)) return false;
    if (!checker.id(flag.unitId, `${flagPath}.unitId`) || !checker.number(flag.battleTurnNumber, `${flagPath}.battleTurnNumber`, { integer: true, min: 0 })) return false;
    for (const key of ["wasAliveAtBattleStart", "survivedPreviousBattle", "attackedInPreviousBattle", "wasTargetedInPreviousBattle", "retreatEligible"] as const) if (!checker.boolean(flag[key], `${flagPath}.${key}`)) return false;
    if (flag.positionAtBattleStart !== undefined && !validatePosition(flag.positionAtBattleStart, `${flagPath}.positionAtBattleStart`, checker)) return false;
    if (flag.enemyBaseDistanceAtBattleStart !== undefined && !checker.number(flag.enemyBaseDistanceAtBattleStart, `${flagPath}.enemyBaseDistanceAtBattleStart`, { min: 0 })) return false;
    if (flag.enemyBaseWithin3AtBattleStart !== undefined && !checker.boolean(flag.enemyBaseWithin3AtBattleStart, `${flagPath}.enemyBaseWithin3AtBattleStart`)) return false;
    for (const key of ["retreatFriendlyBaseIdsAtEligibility", "retreatHostileBaseIdsAtEligibility"] as const) if (flag[key] !== undefined && !validateStringArray(flag[key], `${flagPath}.${key}`, checker)) return false;
    if (flag.retreatEligibilityReason !== undefined && !checker.string(flag.retreatEligibilityReason, `${flagPath}.retreatEligibilityReason`, true)) return false;
  }

  if (!checker.record(value.turnState, `${path}.turnState`) || !checker.required(value.turnState, ["turnNumber", "phase", "actionIntents"], `${path}.turnState`)
    || !checker.number(value.turnState.turnNumber, `${path}.turnState.turnNumber`, { integer: true, min: 1 }) || !checker.enumeration(value.turnState.phase, PHASES, `${path}.turnState.phase`)
    || !checker.array(value.turnState.actionIntents, `${path}.turnState.actionIntents`)) return false;
  for (let index = 0; index < value.turnState.actionIntents.length; index += 1) {
    const group = value.turnState.actionIntents[index]; const groupPath = `${path}.turnState.actionIntents[${index}]`;
    if (!checker.record(group, groupPath) || !checker.required(group, ["teamId", "productionChoices", "movementIntents", "attackIntents"], groupPath)
      || !checker.id(group.teamId, `${groupPath}.teamId`) || !checker.array(group.productionChoices, `${groupPath}.productionChoices`) || !checker.array(group.movementIntents, `${groupPath}.movementIntents`) || !checker.array(group.attackIntents, `${groupPath}.attackIntents`)) return false;
    if (!group.productionChoices.every((choice, choiceIndex) => validateProductionChoice(choice, `${groupPath}.productionChoices[${choiceIndex}]`, checker))) return false;
    for (let movementIndex = 0; movementIndex < group.movementIntents.length; movementIndex += 1) {
      const intent = group.movementIntents[movementIndex]; const intentPath = `${groupPath}.movementIntents[${movementIndex}]`;
      if (!checker.record(intent, intentPath) || !checker.required(intent, ["teamId", "unitId", "from", "to", "stay"], intentPath)
        || !checker.id(intent.teamId, `${intentPath}.teamId`) || !checker.id(intent.unitId, `${intentPath}.unitId`) || !validatePosition(intent.from, `${intentPath}.from`, checker)
        || !validatePosition(intent.to, `${intentPath}.to`, checker) || !checker.boolean(intent.stay, `${intentPath}.stay`)) return false;
    }
    if (!group.attackIntents.every((intent, attackIndex) => validateAttackIntent(intent, `${groupPath}.attackIntents[${attackIndex}]`, checker))) return false;
  }

  if (!checker.array(value.logs, `${path}.logs`)) return false;
  for (let index = 0; index < value.logs.length; index += 1) {
    const log = value.logs[index]; const logPath = `${path}.logs[${index}]`;
    if (!checker.record(log, logPath) || !checker.required(log, ["id", "turnNumber", "type", "message"], logPath) || !checker.id(log.id, `${logPath}.id`)
      || !checker.number(log.turnNumber, `${logPath}.turnNumber`, { integer: true, min: 0 }) || !checker.enumeration(log.type, LOG_TYPES, `${logPath}.type`) || !checker.string(log.message, `${logPath}.message`, true)) return false;
    if (log.relatedIds !== undefined && !validateStringArray(log.relatedIds, `${logPath}.relatedIds`, checker)) return false;
  }

  if (!checker.array(value.siegeStates, `${path}.siegeStates`)) return false;
  for (let index = 0; index < value.siegeStates.length; index += 1) {
    const siege = value.siegeStates[index]; const siegePath = `${path}.siegeStates[${index}]`;
    if (!checker.record(siege, siegePath) || !checker.required(siege, ["baseId", "defendingTeamId", "teamRecords", "active", "defenderLossOccurred", "fallCandidateTeamIds"], siegePath)
      || !checker.id(siege.baseId, `${siegePath}.baseId`) || !checker.id(siege.defendingTeamId, `${siegePath}.defendingTeamId`) || !checker.array(siege.teamRecords, `${siegePath}.teamRecords`)
      || !checker.boolean(siege.active, `${siegePath}.active`) || !checker.boolean(siege.defenderLossOccurred, `${siegePath}.defenderLossOccurred`) || !validateStringArray(siege.fallCandidateTeamIds, `${siegePath}.fallCandidateTeamIds`, checker)) return false;
    if (siege.lastEffectiveAttackTurn !== undefined && !checker.number(siege.lastEffectiveAttackTurn, `${siegePath}.lastEffectiveAttackTurn`, { integer: true, min: 0 })) return false;
    for (let recordIndex = 0; recordIndex < siege.teamRecords.length; recordIndex += 1) {
      const record = siege.teamRecords[recordIndex]; const recordPath = `${siegePath}.teamRecords[${recordIndex}]`;
      if (!checker.record(record, recordPath) || !checker.required(record, ["teamId", "defenderKills", "effectiveAttackTurns"], recordPath) || !checker.id(record.teamId, `${recordPath}.teamId`)
        || !checker.number(record.defenderKills, `${recordPath}.defenderKills`, { integer: true, min: 0 }) || !checker.number(record.effectiveAttackTurns, `${recordPath}.effectiveAttackTurns`, { integer: true, min: 0 })) return false;
    }
  }

  if (!checker.array(value.rewardPlacementRequests, `${path}.rewardPlacementRequests`)) return false;
  for (let index = 0; index < value.rewardPlacementRequests.length; index += 1) {
    const request = value.rewardPlacementRequests[index]; const requestPath = `${path}.rewardPlacementRequests[${index}]`;
    if (!checker.record(request, requestPath) || !checker.required(request, ["id", "teamId", "rewardType", "sourceBaseId", "destinationKind", "eligibleBaseIds", "completed", "expired"], requestPath)
      || !checker.id(request.id, `${requestPath}.id`) || !checker.id(request.teamId, `${requestPath}.teamId`) || !checker.enumeration(request.rewardType, REWARD_TYPES, `${requestPath}.rewardType`)
      || !checker.id(request.sourceBaseId, `${requestPath}.sourceBaseId`) || !checker.enumeration(request.destinationKind, ["fixed", "selectable"], `${requestPath}.destinationKind`)
      || !validateStringArray(request.eligibleBaseIds, `${requestPath}.eligibleBaseIds`, checker) || !checker.boolean(request.completed, `${requestPath}.completed`) || !checker.boolean(request.expired, `${requestPath}.expired`)) return false;
    for (const key of ["sourceKingUnitId", "fixedBaseId"] as const) if (request[key] !== undefined && !checker.id(request[key], `${requestPath}.${key}`)) return false;
    if (request.selectedUnitType !== undefined && !checker.enumeration(request.selectedUnitType, UNIT_TYPES, `${requestPath}.selectedUnitType`)) return false;
    if (request.expirationReason !== undefined && !checker.string(request.expirationReason, `${requestPath}.expirationReason`, true)) return false;
  }

  if (!checker.array(value.kingCampaignStates, `${path}.kingCampaignStates`)) return false;
  for (let index = 0; index < value.kingCampaignStates.length; index += 1) {
    const campaign = value.kingCampaignStates[index]; const campaignPath = `${path}.kingCampaignStates[${index}]`;
    if (!checker.record(campaign, campaignPath) || !checker.required(campaign, ["kingUnitId", "kingTeamId", "contributions"], campaignPath) || !checker.id(campaign.kingUnitId, `${campaignPath}.kingUnitId`)
      || !checker.id(campaign.kingTeamId, `${campaignPath}.kingTeamId`) || !checker.array(campaign.contributions, `${campaignPath}.contributions`)) return false;
    for (let contributionIndex = 0; contributionIndex < campaign.contributions.length; contributionIndex += 1) {
      const contribution = campaign.contributions[contributionIndex]; const contributionPath = `${campaignPath}.contributions[${contributionIndex}]`;
      if (!checker.record(contribution, contributionPath) || !checker.required(contribution, ["teamId", "cumulativeDamage", "effectiveAttackTurns"], contributionPath)
        || !checker.id(contribution.teamId, `${contributionPath}.teamId`) || !checker.number(contribution.cumulativeDamage, `${contributionPath}.cumulativeDamage`, { min: 0 })
        || !checker.number(contribution.effectiveAttackTurns, `${contributionPath}.effectiveAttackTurns`, { integer: true, min: 0 })) return false;
    }
  }

  if (value.phaseAfterRewards !== undefined && !checker.enumeration(value.phaseAfterRewards, ["attack_input", "movement_input", "strategist_action_input"], `${path}.phaseAfterRewards`)) return false;
  if (!checker.array(value.constructions, `${path}.constructions`)) return false;
  for (let index = 0; index < value.constructions.length; index += 1) {
    const construction = value.constructions[index]; const constructionPath = `${path}.constructions[${index}]`;
    if (!checker.record(construction, constructionPath) || !checker.required(construction, ["id", "kind", "tiles", "placedTurn", "active"], constructionPath)
      || !checker.id(construction.id, `${constructionPath}.id`) || !checker.enumeration(construction.kind, CONSTRUCTION_KINDS, `${constructionPath}.kind`) || !checker.array(construction.tiles, `${constructionPath}.tiles`)
      || !construction.tiles.every((tile, tileIndex) => validateCoord(tile, `${constructionPath}.tiles[${tileIndex}]`, checker)) || !checker.number(construction.placedTurn, `${constructionPath}.placedTurn`, { integer: true, min: 0 })
      || !checker.boolean(construction.active, `${constructionPath}.active`)) return false;
    if (construction.ownerTeamId !== undefined && !checker.id(construction.ownerTeamId, `${constructionPath}.ownerTeamId`)) return false;
    if (construction.managerUnitId !== undefined && !checker.id(construction.managerUnitId, `${constructionPath}.managerUnitId`)) return false;
  }

  if (!checker.array(value.strategistActionIntents, `${path}.strategistActionIntents`)) return false;
  for (let index = 0; index < value.strategistActionIntents.length; index += 1) {
    const intent = value.strategistActionIntents[index]; const intentPath = `${path}.strategistActionIntents[${index}]`;
    if (!checker.record(intent, intentPath) || !checker.required(intent, ["teamId", "strategistUnitId", "action"], intentPath) || !checker.id(intent.teamId, `${intentPath}.teamId`)
      || !checker.id(intent.strategistUnitId, `${intentPath}.strategistUnitId`) || !checker.enumeration(intent.action, STRATEGIST_ACTIONS, `${intentPath}.action`)) return false;
    if (intent.tiles !== undefined && (!checker.array(intent.tiles, `${intentPath}.tiles`) || !intent.tiles.every((tile, tileIndex) => validateCoord(tile, `${intentPath}.tiles[${tileIndex}]`, checker)))) return false;
    if (intent.constructionId !== undefined && !checker.id(intent.constructionId, `${intentPath}.constructionId`)) return false;
  }
  if (!validateStringArray(value.strategistSubmittedTeamIds, `${path}.strategistSubmittedTeamIds`, checker)) return false;
  if (!checker.array(value.strategistCooldowns, `${path}.strategistCooldowns`)) return false;
  for (let index = 0; index < value.strategistCooldowns.length; index += 1) {
    const cooldown = value.strategistCooldowns[index]; const cooldownPath = `${path}.strategistCooldowns[${index}]`;
    if (!checker.record(cooldown, cooldownPath) || !checker.required(cooldown, ["strategistUnitId", "kind", "availableFromTurn"], cooldownPath) || !checker.id(cooldown.strategistUnitId, `${cooldownPath}.strategistUnitId`)
      || !checker.enumeration(cooldown.kind, CONSTRUCTION_KINDS, `${cooldownPath}.kind`) || !checker.number(cooldown.availableFromTurn, `${cooldownPath}.availableFromTurn`, { integer: true, min: 0 })) return false;
  }

  for (const key of ["movementSeatOrderTeamIds", "movementOrderTeamIds", "movementCompletedTeamIds", "productionCompletedTeamIdsThisTurn", "movedUnitIdsThisMovementPhase"] as const)
    if (!validateStringArray(value[key], `${path}.${key}`, checker)) return false;
  if (!checker.number(value.movementOrderStartIndex, `${path}.movementOrderStartIndex`, { integer: true, min: 0 })) return false;
  if (value.currentMovementTeamId !== undefined && !checker.id(value.currentMovementTeamId, `${path}.currentMovementTeamId`)) return false;
  if (value.movementDefendedBaseIdsAtTeamStart !== undefined && !validateStringArray(value.movementDefendedBaseIdsAtTeamStart, `${path}.movementDefendedBaseIdsAtTeamStart`, checker)) return false;

  if (!checker.array(value.teleportIntents, `${path}.teleportIntents`)) return false;
  for (let index = 0; index < value.teleportIntents.length; index += 1) {
    const intent = value.teleportIntents[index]; const intentPath = `${path}.teleportIntents[${index}]`;
    if (!checker.record(intent, intentPath) || !checker.required(intent, ["teamId", "strategistUnitId", "targetUnitId", "to"], intentPath) || !checker.id(intent.teamId, `${intentPath}.teamId`)
      || !checker.id(intent.strategistUnitId, `${intentPath}.strategistUnitId`) || !checker.id(intent.targetUnitId, `${intentPath}.targetUnitId`) || !validatePosition(intent.to, `${intentPath}.to`, checker)) return false;
  }
  if (!checker.array(value.teleportCooldowns, `${path}.teleportCooldowns`)) return false;
  for (let index = 0; index < value.teleportCooldowns.length; index += 1) {
    const cooldown = value.teleportCooldowns[index]; const cooldownPath = `${path}.teleportCooldowns[${index}]`;
    if (!checker.record(cooldown, cooldownPath) || !checker.required(cooldown, ["strategistUnitId", "availableFromTurn"], cooldownPath) || !checker.id(cooldown.strategistUnitId, `${cooldownPath}.strategistUnitId`)
      || !checker.number(cooldown.availableFromTurn, `${cooldownPath}.availableFromTurn`, { integer: true, min: 0 })) return false;
  }
  if (value.ninjaRevealStates !== undefined) {
    if (!checker.array(value.ninjaRevealStates, `${path}.ninjaRevealStates`)) return false;
    for (let index = 0; index < value.ninjaRevealStates.length; index += 1) {
      const reveal = value.ninjaRevealStates[index]; const revealPath = `${path}.ninjaRevealStates[${index}]`;
      if (!checker.record(reveal, revealPath) || !checker.required(reveal, ["ninjaUnitId", "visibleToTeamIds"], revealPath) || !checker.id(reveal.ninjaUnitId, `${revealPath}.ninjaUnitId`)
        || !validateStringArray(reveal.visibleToTeamIds, `${revealPath}.visibleToTeamIds`, checker)) return false;
    }
  }
  return true;
}

function validateCpuRuntimeStructure(value: unknown, path: string, checker: ShapeChecker): value is CpuRuntime {
  if (!checker.record(value, path) || !checker.required(value, ["seed", "rngState", "contextKey", "processedKeys", "completedProductionTeamIds", "completedAttackTeamIds", "hiddenAttackIntents", "logs", "appliedStepCount", "maxAppliedSteps"], path)) return false;
  if (!checker.number(value.seed, `${path}.seed`, { integer: true, min: 0, max: UINT32_MAX }) || !checker.number(value.rngState, `${path}.rngState`, { integer: true, min: 0, max: UINT32_MAX })
    || !checker.string(value.contextKey, `${path}.contextKey`, true) || !validateStringArray(value.processedKeys, `${path}.processedKeys`, checker, false)
    || !validateStringArray(value.completedProductionTeamIds, `${path}.completedProductionTeamIds`, checker) || !validateStringArray(value.completedAttackTeamIds, `${path}.completedAttackTeamIds`, checker)
    || !checker.array(value.hiddenAttackIntents, `${path}.hiddenAttackIntents`) || !value.hiddenAttackIntents.every((intent, index) => validateAttackIntent(intent, `${path}.hiddenAttackIntents[${index}]`, checker))
    || !checker.array(value.logs, `${path}.logs`) || !checker.number(value.appliedStepCount, `${path}.appliedStepCount`, { integer: true, min: 0 }) || !checker.number(value.maxAppliedSteps, `${path}.maxAppliedSteps`, { integer: true, min: 1 })) return false;
  for (let index = 0; index < value.logs.length; index += 1) {
    const log = value.logs[index]; const logPath = `${path}.logs[${index}]`;
    if (!checker.record(log, logPath) || !checker.required(log, ["id", "turnNumber", "phase", "action"], logPath) || !checker.id(log.id, `${logPath}.id`)
      || !checker.number(log.turnNumber, `${logPath}.turnNumber`, { integer: true, min: 0 }) || !checker.enumeration(log.phase, PHASES, `${logPath}.phase`) || !checker.string(log.action, `${logPath}.action`, true)) return false;
    for (const key of ["teamId", "detail", "error"] as const) if (log[key] !== undefined && !checker.string(log[key], `${logPath}.${key}`, key !== "teamId")) return false;
  }
  return value.stoppedReason === undefined || checker.string(value.stoppedReason, `${path}.stoppedReason`, true);
}

function validateHeuristicStructure(value: unknown, path: string, checker: ShapeChecker): value is HeuristicCpuPolicyState {
  if (!checker.record(value, path) || !checker.required(value, ["matches"], path) || !checker.array(value.matches, `${path}.matches`)) return false;
  for (let index = 0; index < value.matches.length; index += 1) {
    const match = value.matches[index]; const matchPath = `${path}.matches[${index}]`;
    if (!checker.record(match, matchPath) || !checker.required(match, ["seed", "lastTurn", "targetBaseIdByTeamId"], matchPath)
      || !checker.number(match.seed, `${matchPath}.seed`, { integer: true, min: 0, max: UINT32_MAX }) || !checker.number(match.lastTurn, `${matchPath}.lastTurn`, { integer: true, min: 0 })
      || !checker.record(match.targetBaseIdByTeamId, `${matchPath}.targetBaseIdByTeamId`)) return false;
    for (const [teamId, baseId] of Object.entries(match.targetBaseIdByTeamId)) if (!checker.id(teamId, `${matchPath}.targetBaseIdByTeamId key`) || !checker.id(baseId, `${matchPath}.targetBaseIdByTeamId.${teamId}`)) return false;
  }
  return true;
}

function validateSettingsStructure(value: unknown, path: string, checker: ShapeChecker): value is CpuTeamSettings {
  if (!checker.record(value, path)) return false;
  for (const [teamId, controller] of Object.entries(value)) if (!checker.id(teamId, `${path} key`) || !checker.enumeration(controller, CONTROLLERS, `${path}.${teamId}`)) return false;
  return true;
}

function structuralValidation(value: unknown): LocalMatchSaveResult<LocalMatchSaveV1> {
  const jsonIssue = inspectJsonValue(value);
  if (jsonIssue) return { ok: false, error: { code: "STRUCTURAL_ERROR", message: jsonIssue.message, path: jsonIssue.path } };
  const checker = new ShapeChecker();
  if (!checker.record(value, "$") || !checker.required(value, ["saveSchemaVersion", "saveId", "createdAt", "updatedAt", "compatibility", "metadata", "payload"], "$")) return shapeFailure(checker);
  if (!checker.number(value.saveSchemaVersion, "$.saveSchemaVersion", { integer: true, min: 0 }) || !checker.id(value.saveId, "$.saveId")
    || !checker.string(value.createdAt, "$.createdAt") || !checker.string(value.updatedAt, "$.updatedAt")) return shapeFailure(checker);
  if (!Number.isFinite(Date.parse(value.createdAt as string))) checker.fail("$.createdAt", "must be a valid timestamp");
  if (!Number.isFinite(Date.parse(value.updatedAt as string))) checker.fail("$.updatedAt", "must be a valid timestamp");
  if (checker.issue) return shapeFailure(checker);

  if (!checker.record(value.compatibility, "$.compatibility") || !checker.required(value.compatibility, ["gameRulesVersion", "mapId", "mapVersion"], "$.compatibility")
    || !checker.number(value.compatibility.gameRulesVersion, "$.compatibility.gameRulesVersion", { integer: true, min: 0 }) || !checker.id(value.compatibility.mapId, "$.compatibility.mapId")
    || !checker.number(value.compatibility.mapVersion, "$.compatibility.mapVersion", { integer: true, min: 0 })) return shapeFailure(checker);

  if (!checker.record(value.metadata, "$.metadata") || !checker.required(value.metadata, ["displayName", "turnNumber", "phase", "humanTeamIds", "controllersByTeamId", "activeTeamIds", "defeatedTeamIds", "preview"], "$.metadata")
    || !checker.string(value.metadata.displayName, "$.metadata.displayName") || !checker.number(value.metadata.turnNumber, "$.metadata.turnNumber", { integer: true, min: 1 })
    || !checker.enumeration(value.metadata.phase, PHASES, "$.metadata.phase") || !validateStringArray(value.metadata.humanTeamIds, "$.metadata.humanTeamIds", checker)
    || !validateSettingsStructure(value.metadata.controllersByTeamId, "$.metadata.controllersByTeamId", checker) || !validateStringArray(value.metadata.activeTeamIds, "$.metadata.activeTeamIds", checker)
    || !validateStringArray(value.metadata.defeatedTeamIds, "$.metadata.defeatedTeamIds", checker) || !checker.record(value.metadata.preview, "$.metadata.preview")
    || !checker.required(value.metadata.preview, ["mapName", "livingUnitCount", "ownedBaseCountByTeamId"], "$.metadata.preview") || !checker.string(value.metadata.preview.mapName, "$.metadata.preview.mapName")
    || !checker.number(value.metadata.preview.livingUnitCount, "$.metadata.preview.livingUnitCount", { integer: true, min: 0 }) || !checker.record(value.metadata.preview.ownedBaseCountByTeamId, "$.metadata.preview.ownedBaseCountByTeamId")) return shapeFailure(checker);
  for (const [teamId, count] of Object.entries(value.metadata.preview.ownedBaseCountByTeamId)) if (!checker.id(teamId, "$.metadata.preview.ownedBaseCountByTeamId key") || !checker.number(count, `$.metadata.preview.ownedBaseCountByTeamId.${teamId}`, { integer: true, min: 0 })) return shapeFailure(checker);

  if (!checker.record(value.payload, "$.payload") || !checker.required(value.payload, ["gameState", "cpuSettings", "cpuRuntime", "heuristicPolicyState"], "$.payload")
    || !validateGameStateStructure(value.payload.gameState, "$.payload.gameState", checker) || !validateSettingsStructure(value.payload.cpuSettings, "$.payload.cpuSettings", checker)
    || !validateCpuRuntimeStructure(value.payload.cpuRuntime, "$.payload.cpuRuntime", checker) || !validateHeuristicStructure(value.payload.heuristicPolicyState, "$.payload.heuristicPolicyState", checker)) return shapeFailure(checker);
  if (value.payload.resumeUi !== undefined) {
    if (!checker.record(value.payload.resumeUi, "$.payload.resumeUi")) return shapeFailure(checker);
    if (value.payload.resumeUi.viewerTeamId !== undefined && !checker.id(value.payload.resumeUi.viewerTeamId, "$.payload.resumeUi.viewerTeamId")) return shapeFailure(checker);
  }
  return { ok: true, value: value as unknown as LocalMatchSaveV1 };
}

function shapeFailure(checker: ShapeChecker): LocalMatchSaveResult<never> {
  const issue = checker.issue ?? { path: "$", message: "invalid save structure" };
  return { ok: false, error: { code: "STRUCTURAL_ERROR", message: issue.message, path: issue.path } };
}

function failure(code: LocalMatchSaveError["code"], message: string, path?: string): LocalMatchSaveResult<never> {
  return { ok: false, error: { code, message, path } };
}

function duplicates(values: string[]) {
  return values.filter((value, index) => values.indexOf(value) !== index);
}

function sameJson(left: unknown, right: unknown) {
  const canonicalize = (value: unknown): unknown => Array.isArray(value)
    ? value.map(canonicalize)
    : isRecord(value)
      ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]))
      : value;
  return JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right));
}

function validatePositionReference(position: UnitPosition, state: GameState, path: string, allowInactiveBridge = false): LocalMatchSaveError | undefined {
  const inBounds = (x: number, y: number) => x >= 0 && y >= 0 && x < state.map.width && y < state.map.height;
  if (position.kind === "tile" || position.kind === "water") {
    if (!inBounds(position.x, position.y)) return { code: "INVALID_GAME_STATE", message: "unit position is outside map bounds", path };
    const tile = state.map.tiles.find((entry) => entry.x === position.x && entry.y === position.y);
    if (!tile) return { code: "INVALID_GAME_STATE", message: "unit position has no map tile", path };
    if (position.kind === "water" && tile.terrain !== "lake") return { code: "INVALID_GAME_STATE", message: "water position is not a lake tile", path };
    if (position.kind === "tile" && !["road", "baseGate", "reorganize"].includes(tile.terrain)) return { code: "INVALID_GAME_STATE", message: "tile position is not traversable terrain", path };
  }
  if (position.kind === "base") {
    const base = state.bases.find((entry) => entry.id === position.baseId);
    if (!base?.slots.some((slot) => slot.id === position.slotId)) return { code: "INVALID_GAME_STATE", message: "base position references a missing base or slot", path };
  }
  if (position.kind === "bridge") {
    const bridge = state.constructions.find((entry) => entry.id === position.bridgeId && entry.kind === "bridge" && (allowInactiveBridge || entry.active));
    if (!bridge || position.cellIndex >= bridge.tiles.length) return { code: "INVALID_GAME_STATE", message: "bridge position references a missing bridge cell", path };
  }
  return undefined;
}

function validateGameStateInvariants(state: GameState): LocalMatchSaveError | undefined {
  const teamIds = state.teams.map((team) => team.id);
  const unitIds = state.units.map((unit) => unit.id);
  const baseIds = state.bases.map((base) => base.id);
  const constructionIds = state.constructions.map((entry) => entry.id);
  if (duplicates(teamIds).length) return { code: "INVALID_GAME_STATE", message: "duplicate team ID", path: "$.payload.gameState.teams" };
  if (duplicates(unitIds).length) return { code: "INVALID_GAME_STATE", message: "duplicate unit ID", path: "$.payload.gameState.units" };
  if (duplicates(baseIds).length) return { code: "INVALID_GAME_STATE", message: "duplicate base ID", path: "$.payload.gameState.bases" };
  if (duplicates(constructionIds).length) return { code: "INVALID_GAME_STATE", message: "duplicate construction ID", path: "$.payload.gameState.constructions" };
  const teamById = new Map(state.teams.map((team) => [team.id, team]));
  const unitById = new Map(state.units.map((unit) => [unit.id, unit]));
  const baseById = new Map(state.bases.map((base) => [base.id, base]));
  const constructionById = new Map(state.constructions.map((entry) => [entry.id, entry]));
  const validTeam = (teamId: string) => teamById.has(teamId);
  const validBase = (baseId: string) => baseById.has(baseId);
  const validUnit = (unitId: string) => unitById.has(unitId);

  if (state.config.playerCount !== state.teams.filter((team) => !team.isNeutral).length) return { code: "INVALID_GAME_STATE", message: "playerCount differs from non-neutral team count", path: "$.payload.gameState.config.playerCount" };
  if (state.bases.length !== state.map.bases.length || state.bases.some((base) => {
    const mapBase = state.map.bases.find((entry) => entry.id === base.id);
    if (!mapBase || base.name !== mapBase.name || base.type !== mapBase.type || base.protectedSlotId !== mapBase.protectedSlotId || !sameJson(base.coords, mapBase.coords)) return true;
    return !sameJson(base.slots.map(({ unitId: _unitId, ...slot }) => slot), mapBase.slots.map(({ unitId: _unitId, ...slot }) => slot));
  })) return { code: "INVALID_GAME_STATE", message: "dynamic base definitions differ from the current map", path: "$.payload.gameState.bases" };

  const neutralTeams = state.teams.filter((team) => team.isNeutral || team.status === "neutral");
  if (neutralTeams.length !== 1 || !neutralTeams.every((team) => team.isNeutral && team.status === "neutral")) return { code: "INVALID_GAME_STATE", message: "neutral team status is inconsistent", path: "$.payload.gameState.teams" };
  for (const team of state.teams) {
    if (!team.isNeutral && team.status === "neutral") return { code: "INVALID_GAME_STATE", message: "non-neutral team has neutral status", path: `$.payload.gameState.teams.${team.id}` };
    if (team.homeBaseId && (!validBase(team.homeBaseId) || baseById.get(team.homeBaseId)?.type !== "home")) return { code: "INVALID_GAME_STATE", message: "team home base reference is invalid", path: `$.payload.gameState.teams.${team.id}.homeBaseId` };
    if (team.controlledBaseIds.some((id) => !validBase(id) || baseById.get(id)?.ownerTeamId !== team.id)) return { code: "INVALID_GAME_STATE", message: "controlled base reference is invalid", path: `$.payload.gameState.teams.${team.id}.controlledBaseIds` };
    if ((team.status === "defeated" || team.status === "eliminated") && (team.controlledBaseIds.length || state.units.some((unit) => unit.ownerTeamId === team.id && unit.hp > 0 && unit.position.kind !== "removed"))) return { code: "INVALID_GAME_STATE", message: "inactive team retains controlled bases or living units", path: `$.payload.gameState.teams.${team.id}` };
    if (team.conqueredTeamIds?.some((id) => !validTeam(id))) return { code: "INVALID_GAME_STATE", message: "conquered team reference is invalid", path: `$.payload.gameState.teams.${team.id}.conqueredTeamIds` };
    if (team.constructionCapacityBonusStrategistId && !validUnit(team.constructionCapacityBonusStrategistId)) return { code: "INVALID_GAME_STATE", message: "construction bonus strategist reference is invalid", path: `$.payload.gameState.teams.${team.id}.constructionCapacityBonusStrategistId` };
  }
  for (const base of state.bases) {
    if (!validTeam(base.ownerTeamId)) return { code: "INVALID_GAME_STATE", message: "base owner team reference is invalid", path: `$.payload.gameState.bases.${base.id}.ownerTeamId` };
    if (base.ownerTeamId !== "neutral" && !teamById.get(base.ownerTeamId)?.controlledBaseIds.includes(base.id)) return { code: "INVALID_GAME_STATE", message: "base owner does not reciprocally control base", path: `$.payload.gameState.bases.${base.id}` };
    if (duplicates(base.slots.map((slot) => slot.id)).length || base.slots.some((slot) => slot.baseId !== base.id)) return { code: "INVALID_GAME_STATE", message: "base slot identity is invalid", path: `$.payload.gameState.bases.${base.id}.slots` };
    if (base.protectedSlotId && !base.slots.some((slot) => slot.id === base.protectedSlotId)) return { code: "INVALID_GAME_STATE", message: "protected slot reference is invalid", path: `$.payload.gameState.bases.${base.id}.protectedSlotId` };
    if (base.occupationPriorityTeamId && !validTeam(base.occupationPriorityTeamId)) return { code: "INVALID_GAME_STATE", message: "occupation priority team reference is invalid", path: `$.payload.gameState.bases.${base.id}.occupationPriorityTeamId` };
  }
  for (const unit of state.units) {
    if (!validTeam(unit.ownerTeamId)) return { code: "INVALID_GAME_STATE", message: "unit owner team reference is invalid", path: `$.payload.gameState.units.${unit.id}.ownerTeamId` };
    const positionError = validatePositionReference(unit.position, state, `$.payload.gameState.units.${unit.id}.position`);
    if (positionError) return positionError;
    if (unit.position.kind === "water" && unit.type !== "ninja") return { code: "INVALID_GAME_STATE", message: "non-ninja unit is on water", path: `$.payload.gameState.units.${unit.id}.position` };
    if ((unit.hp <= 0) !== (unit.position.kind === "removed")) return { code: "INVALID_GAME_STATE", message: "unit HP and removed position disagree", path: `$.payload.gameState.units.${unit.id}` };
    if (unit.type !== "strategist" && unit.role !== undefined) return { code: "INVALID_GAME_STATE", message: "non-strategist has strategist role", path: `$.payload.gameState.units.${unit.id}.role` };
    for (const status of unit.statuses) if (status.kind === "retreating" && !validBase(status.retreatTargetBaseId)) return { code: "INVALID_GAME_STATE", message: "retreat target base reference is invalid", path: `$.payload.gameState.units.${unit.id}.statuses` };
  }
  for (const base of state.bases) for (const slot of base.slots) {
    if (!slot.unitId) continue;
    const unit = unitById.get(slot.unitId);
    if (!unit || unit.position.kind !== "base" || unit.position.baseId !== base.id || unit.position.slotId !== slot.id || unit.hp <= 0) return { code: "INVALID_GAME_STATE", message: "base slot and unit position disagree", path: `$.payload.gameState.bases.${base.id}.slots.${slot.id}` };
  }
  for (const unit of state.units.filter((entry) => entry.position.kind === "base")) {
    const position = unit.position as Extract<UnitPosition, { kind: "base" }>;
    if (baseById.get(position.baseId)?.slots.find((slot) => slot.id === position.slotId)?.unitId !== unit.id) return { code: "INVALID_GAME_STATE", message: "base unit lacks reciprocal slot", path: `$.payload.gameState.units.${unit.id}.position` };
  }
  for (const construction of state.constructions) {
    if (construction.ownerTeamId && !validTeam(construction.ownerTeamId)) return { code: "INVALID_GAME_STATE", message: "construction owner reference is invalid", path: `$.payload.gameState.constructions.${construction.id}.ownerTeamId` };
    if (!construction.tiles.length || construction.tiles.some((tile) => tile.x < 0 || tile.y < 0 || tile.x >= state.map.width || tile.y >= state.map.height)) return { code: "INVALID_GAME_STATE", message: "construction tile list is empty or outside map bounds", path: `$.payload.gameState.constructions.${construction.id}.tiles` };
    if (construction.managerUnitId) {
      const manager = unitById.get(construction.managerUnitId);
      if (!manager || manager.type !== "strategist" || manager.role !== "builder" || manager.hp <= 0 || manager.position.kind === "removed" || manager.ownerTeamId !== construction.ownerTeamId) return { code: "INVALID_GAME_STATE", message: "construction manager reference is invalid", path: `$.payload.gameState.constructions.${construction.id}.managerUnitId` };
    }
  }
  for (const cooldown of state.strategistCooldowns) if (unitById.get(cooldown.strategistUnitId)?.type !== "strategist") return { code: "INVALID_GAME_STATE", message: "strategist cooldown reference is invalid", path: "$.payload.gameState.strategistCooldowns" };
  for (const cooldown of state.teleportCooldowns) if (unitById.get(cooldown.strategistUnitId)?.type !== "strategist") return { code: "INVALID_GAME_STATE", message: "teleport cooldown reference is invalid", path: "$.payload.gameState.teleportCooldowns" };
  for (const intent of state.teleportIntents) {
    const strategist = unitById.get(intent.strategistUnitId); const target = unitById.get(intent.targetUnitId);
    if (!validTeam(intent.teamId) || !strategist || !target || strategist.type !== "strategist" || strategist.role !== "teleporter" || strategist.ownerTeamId !== intent.teamId || target.ownerTeamId !== intent.teamId) return { code: "INVALID_GAME_STATE", message: "teleport intent reference is invalid", path: "$.payload.gameState.teleportIntents" };
    const destinationError = validatePositionReference(intent.to, state, "$.payload.gameState.teleportIntents.to");
    if (destinationError) return destinationError;
  }
  for (const intent of state.strategistActionIntents) {
    const strategist = unitById.get(intent.strategistUnitId);
    if (!validTeam(intent.teamId) || !strategist || strategist.type !== "strategist" || strategist.ownerTeamId !== intent.teamId || (intent.constructionId && !constructionById.has(intent.constructionId))) return { code: "INVALID_GAME_STATE", message: "strategist action reference is invalid", path: "$.payload.gameState.strategistActionIntents" };
    if (intent.tiles?.some((tile) => tile.x < 0 || tile.y < 0 || tile.x >= state.map.width || tile.y >= state.map.height)) return { code: "INVALID_GAME_STATE", message: "strategist action tile is outside map bounds", path: "$.payload.gameState.strategistActionIntents.tiles" };
  }
  for (const request of state.rewardPlacementRequests) {
    if (!validTeam(request.teamId) || !validBase(request.sourceBaseId) || (request.sourceKingUnitId && !validUnit(request.sourceKingUnitId)) || (request.fixedBaseId && !validBase(request.fixedBaseId)) || request.eligibleBaseIds.some((id) => !validBase(id))) return { code: "INVALID_GAME_STATE", message: "reward request reference is invalid", path: `$.payload.gameState.rewardPlacementRequests.${request.id}` };
    if (request.destinationKind === "fixed" && (!request.fixedBaseId || !request.eligibleBaseIds.includes(request.fixedBaseId))) return { code: "INVALID_GAME_STATE", message: "fixed reward destination is inconsistent", path: `$.payload.gameState.rewardPlacementRequests.${request.id}` };
  }
  for (const siege of state.siegeStates) {
    if (!validBase(siege.baseId) || !validTeam(siege.defendingTeamId) || siege.teamRecords.some((record) => !validTeam(record.teamId)) || siege.fallCandidateTeamIds.some((id) => !validTeam(id))) return { code: "INVALID_GAME_STATE", message: "siege state reference is invalid", path: `$.payload.gameState.siegeStates.${siege.baseId}` };
    if (siege.active && baseById.get(siege.baseId)?.ownerTeamId !== siege.defendingTeamId) return { code: "INVALID_GAME_STATE", message: "active siege defender differs from base owner", path: `$.payload.gameState.siegeStates.${siege.baseId}` };
  }
  for (const campaign of state.kingCampaignStates) {
    const king = unitById.get(campaign.kingUnitId);
    if (!king || king.type !== "king" || king.ownerTeamId !== campaign.kingTeamId || !validTeam(campaign.kingTeamId) || campaign.contributions.some((entry) => !validTeam(entry.teamId))) return { code: "INVALID_GAME_STATE", message: "king campaign reference is invalid", path: `$.payload.gameState.kingCampaignStates.${campaign.kingUnitId}` };
  }
  for (const flag of state.unitTurnFlags) {
    if (!validUnit(flag.unitId) || flag.retreatFriendlyBaseIdsAtEligibility?.some((id) => !validBase(id)) || flag.retreatHostileBaseIdsAtEligibility?.some((id) => !validBase(id))) return { code: "INVALID_GAME_STATE", message: "unit turn flag reference is invalid", path: `$.payload.gameState.unitTurnFlags.${flag.unitId}` };
    if (flag.positionAtBattleStart) { const error = validatePositionReference(flag.positionAtBattleStart, state, `$.payload.gameState.unitTurnFlags.${flag.unitId}.positionAtBattleStart`, true); if (error) return error; }
  }
  for (const reveal of state.ninjaRevealStates ?? []) if (unitById.get(reveal.ninjaUnitId)?.type !== "ninja" || reveal.visibleToTeamIds.some((id) => !validTeam(id))) return { code: "INVALID_GAME_STATE", message: "ninja reveal reference is invalid", path: "$.payload.gameState.ninjaRevealStates" };
  for (const group of state.turnState.actionIntents) {
    if (!validTeam(group.teamId)) return { code: "INVALID_GAME_STATE", message: "action intent team reference is invalid", path: "$.payload.gameState.turnState.actionIntents" };
    if (group.productionChoices.some((choice) => choice.teamId !== group.teamId || !validBase(choice.baseId))) return { code: "INVALID_GAME_STATE", message: "production intent reference is invalid", path: "$.payload.gameState.turnState.actionIntents" };
    for (const intent of group.movementIntents) {
      if (intent.teamId !== group.teamId || !validUnit(intent.unitId)) return { code: "INVALID_GAME_STATE", message: "movement intent reference is invalid", path: "$.payload.gameState.turnState.actionIntents" };
      const fromError = validatePositionReference(intent.from, state, "$.payload.gameState.turnState.actionIntents.movementIntents.from", true); if (fromError) return fromError;
      const toError = validatePositionReference(intent.to, state, "$.payload.gameState.turnState.actionIntents.movementIntents.to", true); if (toError) return toError;
    }
    for (const intent of group.attackIntents) {
      if (intent.teamId !== group.teamId || !validUnit(intent.attackerUnitId) || (intent.target && !validUnit(intent.target.unitId))) return { code: "INVALID_GAME_STATE", message: "attack intent reference is invalid", path: "$.payload.gameState.turnState.actionIntents" };
      if (intent.target?.baseId && !validBase(intent.target.baseId)) return { code: "INVALID_GAME_STATE", message: "attack target base reference is invalid", path: "$.payload.gameState.turnState.actionIntents.attackIntents.target.baseId" };
      if (intent.target?.slotId && (!intent.target.baseId || !baseById.get(intent.target.baseId)?.slots.some((slot) => slot.id === intent.target?.slotId))) return { code: "INVALID_GAME_STATE", message: "attack target slot reference is invalid", path: "$.payload.gameState.turnState.actionIntents.attackIntents.target.slotId" };
    }
  }
  const teamReferenceArrays = [state.movementSeatOrderTeamIds, state.movementOrderTeamIds, state.movementCompletedTeamIds, state.productionCompletedTeamIdsThisTurn, state.strategistSubmittedTeamIds];
  if (teamReferenceArrays.some((values) => values.some((id) => !validTeam(id) || Boolean(teamById.get(id)?.isNeutral)) || duplicates(values).length)) return { code: "INVALID_GAME_STATE", message: "team order/completion reference is invalid", path: "$.payload.gameState" };
  if (state.movementSeatOrderTeamIds.length && state.movementOrderStartIndex >= state.movementSeatOrderTeamIds.length) return { code: "INVALID_GAME_STATE", message: "movement order start index is out of range", path: "$.payload.gameState.movementOrderStartIndex" };
  if (state.movedUnitIdsThisMovementPhase.some((id) => !validUnit(id)) || state.movementDefendedBaseIdsAtTeamStart?.some((id) => !validBase(id))) return { code: "INVALID_GAME_STATE", message: "movement state reference is invalid", path: "$.payload.gameState" };
  return validatePhaseInvariants(state);
}

function validatePhaseInvariants(state: GameState): LocalMatchSaveError | undefined {
  if (state.phase !== state.turnState.phase || state.turnNumber !== state.turnState.turnNumber) return { code: "INVALID_GAME_STATE", message: "state and turnState phase/turn disagree", path: "$.payload.gameState.turnState" };
  const rewardInterruptsMovement = state.phase === "reward_placement" && state.phaseAfterRewards === "movement_input";
  if (state.phase === "reward_placement") {
    if (!state.phaseAfterRewards || !state.rewardPlacementRequests.some((request) => !request.completed && !request.expired)) return { code: "INVALID_GAME_STATE", message: "reward phase lacks pending request or resume phase", path: "$.payload.gameState.phaseAfterRewards" };
  } else if (state.phaseAfterRewards !== undefined) return { code: "INVALID_GAME_STATE", message: "phaseAfterRewards exists outside reward phase", path: "$.payload.gameState.phaseAfterRewards" };
  if (state.phase === "movement_input" || rewardInterruptsMovement) {
    const team = state.teams.find((entry) => entry.id === state.currentMovementTeamId);
    if (!team || team.status !== "active" || !state.movementOrderTeamIds.includes(team.id) || state.movementCompletedTeamIds.includes(team.id)) return { code: "INVALID_GAME_STATE", message: "current movement team is inconsistent", path: "$.payload.gameState.currentMovementTeamId" };
  } else if (state.currentMovementTeamId !== undefined) return { code: "INVALID_GAME_STATE", message: "current movement team exists outside movement phase", path: "$.payload.gameState.currentMovementTeamId" };
  if (state.teleportIntents.length && state.phase !== "movement_input" && !rewardInterruptsMovement) return { code: "INVALID_GAME_STATE", message: "teleport intents exist outside movement phase", path: "$.payload.gameState.teleportIntents" };
  if (state.movementDefendedBaseIdsAtTeamStart !== undefined && state.phase !== "movement_input" && !rewardInterruptsMovement) return { code: "INVALID_GAME_STATE", message: "movement defended-base snapshot exists outside movement phase", path: "$.payload.gameState.movementDefendedBaseIdsAtTeamStart" };
  if ((state.strategistActionIntents.length || state.strategistSubmittedTeamIds.length) && state.phase !== "strategist_action_input" && state.phase !== "strategist_action_resolution") return { code: "INVALID_GAME_STATE", message: "strategist action state exists outside strategist phases", path: "$.payload.gameState.strategistActionIntents" };
  if (state.phase === "strategist_action_resolution" && !state.teams.filter((team) => team.status === "active").every((team) => state.strategistSubmittedTeamIds.includes(team.id))) return { code: "INVALID_GAME_STATE", message: "strategist resolution started before all active teams submitted", path: "$.payload.gameState.strategistSubmittedTeamIds" };
  return undefined;
}

function validateCpuRuntimeInvariants(runtime: CpuRuntime, state: GameState): LocalMatchSaveError | undefined {
  const teamIds = new Set(state.teams.map((team) => team.id));
  const units = new Map(state.units.map((unit) => [unit.id, unit]));
  if (runtime.appliedStepCount > runtime.maxAppliedSteps) return { code: "INVALID_CPU_RUNTIME", message: "appliedStepCount exceeds maxAppliedSteps", path: "$.payload.cpuRuntime.appliedStepCount" };
  if ([runtime.completedProductionTeamIds, runtime.completedAttackTeamIds].some((values) => values.some((id) => !teamIds.has(id)) || duplicates(values).length)) return { code: "INVALID_CPU_RUNTIME", message: "CPU completion list references an invalid team", path: "$.payload.cpuRuntime" };
  for (const intent of runtime.hiddenAttackIntents) {
    const attacker = units.get(intent.attackerUnitId); const target = intent.target ? units.get(intent.target.unitId) : undefined;
    if (!teamIds.has(intent.teamId) || !attacker || attacker.ownerTeamId !== intent.teamId || (intent.target && !target)) return { code: "INVALID_CPU_RUNTIME", message: "hidden attack intent reference is invalid", path: "$.payload.cpuRuntime.hiddenAttackIntents" };
  }
  return undefined;
}

function validateHeuristicInvariants(policyState: HeuristicCpuPolicyState, state: GameState): LocalMatchSaveError | undefined {
  const teamIds = new Set(state.teams.map((team) => team.id));
  const baseIds = new Set(state.bases.map((base) => base.id));
  if (duplicates(policyState.matches.map((match) => String(match.seed))).length) return { code: "INVALID_HEURISTIC_STATE", message: "duplicate heuristic match seed", path: "$.payload.heuristicPolicyState.matches" };
  for (const match of policyState.matches) for (const [teamId, baseId] of Object.entries(match.targetBaseIdByTeamId)) {
    if (!teamIds.has(teamId) || !baseIds.has(baseId)) return { code: "INVALID_HEURISTIC_STATE", message: "heuristic target references a missing team or base", path: `$.payload.heuristicPolicyState.matches.${match.seed}` };
  }
  return undefined;
}

function validateSettings(settings: CpuTeamSettings, state: GameState): LocalMatchSaveError | undefined {
  const expected = state.teams.filter((team) => !team.isNeutral).map((team) => team.id).sort();
  const actual = Object.keys(settings).sort();
  if (!sameJson(actual, expected)) return { code: "INVALID_CPU_RUNTIME", message: "CPU settings must contain exactly every non-neutral team", path: "$.payload.cpuSettings" };
  if (Object.values(settings).includes("bc_cpu")) return { code: "UNSUPPORTED_CPU_CONTROLLER", message: "bc_cpu LOCAL matches cannot be saved in schema version 1", path: "$.payload.cpuSettings" };
  return undefined;
}

function validateMetadata(save: LocalMatchSaveV1): LocalMatchSaveError | undefined {
  const { gameState, cpuSettings } = save.payload;
  const expected = {
    turnNumber: gameState.turnNumber,
    phase: gameState.phase,
    humanTeamIds: gameState.teams.filter((team) => !team.isNeutral && cpuSettings[team.id] === "human").map((team) => team.id),
    controllersByTeamId: cpuSettings,
    activeTeamIds: gameState.teams.filter((team) => !team.isNeutral && team.status === "active").map((team) => team.id),
    defeatedTeamIds: gameState.teams.filter((team) => !team.isNeutral && team.status !== "active").map((team) => team.id),
    preview: {
      mapName: gameState.map.name,
      livingUnitCount: gameState.units.filter((unit) => unit.hp > 0 && unit.position.kind !== "removed").length,
      ownedBaseCountByTeamId: Object.fromEntries(gameState.teams.filter((team) => !team.isNeutral).map((team) => [team.id, gameState.bases.filter((base) => base.ownerTeamId === team.id).length])),
    },
  };
  const actual = { ...save.metadata }; delete (actual as Partial<typeof actual>).displayName;
  if (!sameJson(actual, expected)) return { code: "INVALID_GAME_STATE", message: "metadata does not match the payload", path: "$.metadata" };
  return undefined;
}

export function isLocalMatchContinuable(state: GameState) {
  return state.teams.filter((team) => !team.isNeutral && team.status === "active").length > 1;
}

export function validateLocalMatchSave(value: unknown): LocalMatchSaveResult<LocalMatchSaveV1> {
  const structural = structuralValidation(value);
  if (!structural.ok) return structural;
  const save = structural.value;
  if (save.saveSchemaVersion !== LOCAL_MATCH_SAVE_SCHEMA_VERSION) return failure("UNSUPPORTED_SCHEMA_VERSION", `Unsupported save schema version: ${save.saveSchemaVersion}`, "$.saveSchemaVersion");
  if (save.compatibility.gameRulesVersion !== LOCAL_GAME_RULES_VERSION) return failure("UNSUPPORTED_RULES_VERSION", `Unsupported game rules version: ${save.compatibility.gameRulesVersion}`, "$.compatibility.gameRulesVersion");
  if (save.compatibility.mapId !== LOCAL_MAP_ID || save.payload.gameState.config.mapId !== LOCAL_MAP_ID || save.payload.gameState.map.id !== LOCAL_MAP_ID) return failure("MAP_MISMATCH", "Save map ID does not match the current LOCAL map", "$.compatibility.mapId");
  if (save.compatibility.mapVersion !== LOCAL_MAP_VERSION || !sameJson(save.payload.gameState.map, testMap4p)) return failure("MAP_MISMATCH", "Save map version or map definition does not match", "$.compatibility.mapVersion");
  const settingsError = validateSettings(save.payload.cpuSettings, save.payload.gameState); if (settingsError) return { ok: false, error: settingsError };
  if (!sameJson(save.metadata.controllersByTeamId, save.payload.cpuSettings)) return failure("INVALID_CPU_RUNTIME", "metadata controller settings differ from payload", "$.metadata.controllersByTeamId");
  const gameError = validateGameStateInvariants(save.payload.gameState); if (gameError) return { ok: false, error: gameError };
  const runtimeError = validateCpuRuntimeInvariants(save.payload.cpuRuntime, save.payload.gameState); if (runtimeError) return { ok: false, error: runtimeError };
  const heuristicError = validateHeuristicInvariants(save.payload.heuristicPolicyState, save.payload.gameState); if (heuristicError) return { ok: false, error: heuristicError };
  if (save.payload.resumeUi?.viewerTeamId && !save.payload.gameState.teams.some((team) => team.id === save.payload.resumeUi?.viewerTeamId && !team.isNeutral)) return failure("INVALID_GAME_STATE", "resume viewer team is invalid", "$.payload.resumeUi.viewerTeamId");
  const metadataError = validateMetadata(save); if (metadataError) return { ok: false, error: metadataError };
  if (!isLocalMatchContinuable(save.payload.gameState)) return failure("MATCH_ALREADY_FINISHED", "Finished LOCAL matches are not resumable", "$.payload.gameState.teams");
  return { ok: true, value: save };
}
