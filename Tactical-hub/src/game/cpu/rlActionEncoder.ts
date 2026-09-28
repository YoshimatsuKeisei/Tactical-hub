import type { Base, Construction, Tile, Unit, UnitPosition } from "../types";
import type { RlLegalAction, RlObservation } from "./rlEnvironment";
import {
  RL_POSITION_KINDS,
  RL_REWARD_TYPES,
  RL_STRATEGIST_ACTIONS,
  RL_STRATEGIST_ROLES,
  RL_TERRAIN_TYPES,
  RL_UNIT_TYPES,
} from "./rlObservationEncoder";

export const RL_ACTION_ENCODER_VERSION = 1;
export const RL_ACTION_ENCODER_VERSION_V2 = 2;
export const RL_ACTION_TYPES = [
  "production",
  "movement",
  "teleport",
  "submit_movement",
  "submit_team_production",
  "attack",
  "complete_attack_team",
  "reward",
  "strategist",
  "submit_strategist",
  "resolve_production",
  "resolve_battle",
  "resolve_strategists",
] as const;
export const RL_ACTION_TYPES_V2 = [...RL_ACTION_TYPES, "merge_infantry"] as const;

export type EncodedLegalActions = {
  schemaVersion: 1;
  actions: number[][];
  actionKeys: string[];
};
export type EncodedLegalActionsV2 = Omit<EncodedLegalActions, "schemaVersion"> & { schemaVersion: 2 };

export type SparseActionRow = {
  indices: number[];
  values: number[];
  width: number;
};

export type EncodedLegalActionsSparseV2 = {
  schemaVersion: 2;
  sparseActions: SparseActionRow[];
  actionKeys: string[];
};

export const RL_ACTION_SCHEMA_BLOCKS = [
  "actionType",
  "isPass",
  "actorTeam",
  "actorUnit",
  "targetUnit",
  "destination",
  "base",
  "baseSlot",
  "slotOccupant",
  "productionUnitType",
  "strategistRole",
  "strategistActionKind",
  "rewardRequest",
  "construction",
  "targetTileCount",
  "targetTileMask",
] as const;
export const RL_ACTION_SCHEMA_BLOCKS_V2 = [
  ...RL_ACTION_SCHEMA_BLOCKS.slice(0, 5),
  "partnerUnit",
  ...RL_ACTION_SCHEMA_BLOCKS.slice(5),
] as const;

const oneHot = (value: string | undefined, values: readonly string[]) => values.map((entry) => Number(value === entry));
const normalized = (value: number, size: number) => size <= 1 ? 0 : Math.max(0, Math.min(1, value / (size - 1)));
const coordKey = (x: number, y: number) => `${x},${y}`;

class SparseActionWriter {
  readonly indices: number[] = [];
  readonly values: number[] = [];
  cursor = 0;

  emit(value: number) {
    if (value !== 0) {
      this.indices.push(this.cursor);
      this.values.push(value);
    }
    this.cursor += 1;
  }

  skip(width: number) {
    this.cursor += width;
  }

  oneHot(value: string | undefined, values: readonly string[]) {
    for (const entry of values) this.emit(Number(value === entry));
  }

  oneAt(index: number | undefined, width: number) {
    if (index !== undefined && index >= 0 && index < width) {
      this.indices.push(this.cursor + index);
      this.values.push(1);
    }
    this.cursor += width;
  }

  binaryMask(indices: Iterable<number>, width: number) {
    const unique = [...new Set(indices)]
      .filter((index) => Number.isInteger(index) && index >= 0 && index < width)
      .sort((left, right) => left - right);
    for (const index of unique) {
      this.indices.push(this.cursor + index);
      this.values.push(1);
    }
    this.cursor += width;
  }

  finish(): SparseActionRow {
    return {
      indices: this.indices,
      values: this.values,
      width: this.cursor,
    };
  }
}

type EncodingContext = {
  observation: RlObservation;
  teams: RlObservation["teams"];
  teamIndex: Map<string, number>;
  units: Unit[];
  unitIndex: Map<string, number>;
  maxUnits: number;
  bases: Base[];
  baseIndex: Map<string, number>;
  constructions: Construction[];
  constructionIndex: Map<string, number>;
  tiles: Map<string, Tile>;
  unitSemanticWidth: number;
  baseSemanticWidth: number;
  constructionSemanticWidth: number;
  tileSemanticWidth: number;
  rewardSemanticWidth: number;
};

function positionCoord(observation: RlObservation, position: UnitPosition): { x: number; y: number } | undefined {
  if (position.kind === "tile" || position.kind === "water") return position;
  if (position.kind === "bridge") return observation.constructions.find((entry) => entry.id === position.bridgeId)?.tiles[position.cellIndex];
  if (position.kind === "base") {
    const base = observation.bases.find((entry) => entry.id === position.baseId);
    if (!base?.coords.length) return undefined;
    return {
      x: base.coords.reduce((sum, coord) => sum + coord.x, 0) / base.coords.length,
      y: base.coords.reduce((sum, coord) => sum + coord.y, 0) / base.coords.length,
    };
  }
  return undefined;
}

function orderedTeams(observation: RlObservation) {
  const observer = observation.teams.find((team) => team.id === observation.observingTeamId);
  return observer ? [observer, ...observation.teams.filter((team) => team.id !== observer.id)] : [...observation.teams];
}

function createContext(observation: RlObservation): EncodingContext {
  const teams = orderedTeams(observation);
  const teamIndex = new Map(teams.map((team, index) => [team.id, index]));
  const units = observation.units.filter((unit) => unit.position.kind !== "removed").sort((left, right) => {
    const leftCoord = positionCoord(observation, left.position), rightCoord = positionCoord(observation, right.position);
    return (teamIndex.get(left.ownerTeamId) ?? teams.length) - (teamIndex.get(right.ownerTeamId) ?? teams.length)
      || RL_UNIT_TYPES.indexOf(left.type) - RL_UNIT_TYPES.indexOf(right.type)
      || (leftCoord?.y ?? 999) - (rightCoord?.y ?? 999)
      || (leftCoord?.x ?? 999) - (rightCoord?.x ?? 999)
      || left.id.localeCompare(right.id);
  });
  const bases = [...observation.bases].sort((left, right) =>
    (left.coords[0]?.y ?? 999) - (right.coords[0]?.y ?? 999)
    || (left.coords[0]?.x ?? 999) - (right.coords[0]?.x ?? 999)
    || left.id.localeCompare(right.id));
  const constructions = [...observation.constructions].sort((left, right) =>
    (left.tiles[0]?.y ?? 999) - (right.tiles[0]?.y ?? 999)
    || (left.tiles[0]?.x ?? 999) - (right.tiles[0]?.x ?? 999)
    || left.kind.localeCompare(right.kind)
    || left.id.localeCompare(right.id));
  const maxUnits = observation.map.tiles.length + observation.bases.reduce((sum, base) => sum + base.slots.length, 0);
  const teamWidth = teams.length + 1;
  const unitSemanticWidth = 1 + maxUnits + teamWidth + RL_UNIT_TYPES.length + 1 + RL_POSITION_KINDS.length + 3 + 2 + 3 + RL_STRATEGIST_ROLES.length;
  const baseSemanticWidth = 1 + bases.length + 2 + teamWidth + 3 + 4 + teamWidth;
  const constructionSemanticWidth = 1 + observation.map.tiles.length + 2 + 1 + teamWidth + 1 + 3;
  const tileSemanticWidth = 1 + RL_POSITION_KINDS.length + 3 + RL_TERRAIN_TYPES.length + 2 + teamWidth + 2 + teamWidth;
  const rewardSemanticWidth = 1 + RL_REWARD_TYPES.length + teamWidth + baseSemanticWidth * 3 + unitSemanticWidth + bases.length + 2;
  return {
    observation,
    teams,
    teamIndex,
    units,
    unitIndex: new Map(units.map((unit, index) => [unit.id, index])),
    maxUnits,
    bases,
    baseIndex: new Map(bases.map((base, index) => [base.id, index])),
    constructions,
    constructionIndex: new Map(constructions.map((construction, index) => [construction.id, index])),
    tiles: new Map(observation.map.tiles.map((tile) => [coordKey(tile.x, tile.y), tile])),
    unitSemanticWidth,
    baseSemanticWidth,
    constructionSemanticWidth,
    tileSemanticWidth,
    rewardSemanticWidth,
  };
}

function teamRef(context: EncodingContext, teamId: string | undefined) {
  const result = Array(context.teams.length + 1).fill(0);
  result[context.teamIndex.get(teamId ?? "") ?? context.teams.length] = 1;
  return result;
}

function unitRef(context: EncodingContext, unitId: string | undefined): number[] {
  const unit = unitId ? context.observation.units.find((entry) => entry.id === unitId && entry.position.kind !== "removed") : undefined;
  if (!unit) return Array(context.unitSemanticWidth).fill(0);
  const slotRef = Array(context.maxUnits).fill(0);
  const slot = context.unitIndex.get(unit.id);
  if (slot !== undefined) slotRef[slot] = 1;
  const coord = positionCoord(context.observation, unit.position);
  const position = unit.position;
  const baseSlot = position.kind === "base"
    ? context.bases.find((base) => base.id === position.baseId)?.slots.find((entry) => entry.id === position.slotId)
    : undefined;
  return [
    1,
    ...slotRef,
    ...teamRef(context, unit.ownerTeamId),
    ...oneHot(unit.type, RL_UNIT_TYPES),
    unit.hp,
    ...oneHot(unit.position.kind, RL_POSITION_KINDS),
    Number(Boolean(coord)),
    coord ? normalized(coord.x, context.observation.map.width) : 0,
    coord ? normalized(coord.y, context.observation.map.height) : 0,
    baseSlot ? baseSlot.localRow : 0,
    baseSlot ? baseSlot.localCol : 0,
    Number(unit.statuses.some((status) => status.kind === "retreating")),
    Number(unit.statuses.some((status) => status.kind === "encouraged")),
    Number(unit.statuses.some((status) => status.kind === "cannot_attack")),
    ...oneHot(unit.role, RL_STRATEGIST_ROLES),
  ];
}

function baseRef(context: EncodingContext, baseId: string | undefined): number[] {
  const base = baseId ? context.bases.find((entry) => entry.id === baseId) : undefined;
  if (!base) return Array(context.baseSemanticWidth).fill(0);
  const identity = Array(context.bases.length).fill(0);
  identity[context.baseIndex.get(base.id)!] = 1;
  const centroid = base.coords.length ? {
    x: base.coords.reduce((sum, coord) => sum + coord.x, 0) / base.coords.length,
    y: base.coords.reduce((sum, coord) => sum + coord.y, 0) / base.coords.length,
  } : undefined;
  return [
    1, ...identity,
    Number(base.type === "home"), Number(base.type === "neutral"),
    ...teamRef(context, base.ownerTeamId),
    Number(Boolean(centroid)),
    centroid ? normalized(centroid.x, context.observation.map.width) : 0,
    centroid ? normalized(centroid.y, context.observation.map.height) : 0,
    base.slots.length,
    base.slots.filter((slot) => Boolean(slot.unitId)).length,
    base.slots.filter((slot) => slot.kind === "front" && Boolean(slot.unitId)).length,
    base.slots.filter((slot) => slot.kind === "protected" && Boolean(slot.unitId)).length,
    ...teamRef(context, base.occupationPriorityTeamId),
  ];
}

function constructionRef(context: EncodingContext, constructionId: string | undefined): number[] {
  const construction = constructionId ? context.constructions.find((entry) => entry.id === constructionId) : undefined;
  if (!construction) return Array(context.constructionSemanticWidth).fill(0);
  const identity = Array(context.observation.map.tiles.length).fill(0);
  const index = context.constructionIndex.get(construction.id);
  if (index !== undefined) identity[index] = 1;
  const centroid = construction.tiles.length ? {
    x: construction.tiles.reduce((sum, tile) => sum + tile.x, 0) / construction.tiles.length,
    y: construction.tiles.reduce((sum, tile) => sum + tile.y, 0) / construction.tiles.length,
  } : undefined;
  return [
    1, ...identity,
    Number(construction.kind === "bridge"), Number(construction.kind === "obstacle"),
    Number(construction.active), ...teamRef(context, construction.ownerTeamId), construction.tiles.length,
    Number(Boolean(centroid)),
    centroid ? normalized(centroid.x, context.observation.map.width) : 0,
    centroid ? normalized(centroid.y, context.observation.map.height) : 0,
  ];
}

function parseDestination(context: EncodingContext, tileId: string | undefined) {
  if (!tileId) return undefined;
  const coordinate = /^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/.exec(tileId);
  if (coordinate) return { kind: "tile" as const, x: Number(coordinate[1]), y: Number(coordinate[2]) };
  if (tileId.startsWith("base:")) {
    const [, baseId, slotId] = tileId.split(":");
    const base = context.bases.find((entry) => entry.id === baseId);
    const coord = base?.coords[0];
    return coord ? { kind: "base" as const, x: coord.x, y: coord.y, baseId, slotId } : { kind: "base" as const, baseId, slotId };
  }
  if (tileId.startsWith("bridge:")) {
    const [, bridgeId, rawIndex] = tileId.split(":");
    const construction = context.constructions.find((entry) => entry.id === bridgeId);
    const coord = construction?.tiles[Number(rawIndex)];
    return coord ? { kind: "bridge" as const, x: coord.x, y: coord.y, construction } : { kind: "bridge" as const, construction };
  }
  return undefined;
}

function tileRef(context: EncodingContext, tileId: string | undefined): number[] {
  const destination = parseDestination(context, tileId);
  if (!destination) return Array(context.tileSemanticWidth).fill(0);
  const coord = "x" in destination && typeof destination.x === "number" && typeof destination.y === "number"
    ? { x: destination.x, y: destination.y }
    : undefined;
  const tile = coord ? context.tiles.get(coordKey(coord.x, coord.y)) : undefined;
  const baseId = destination.kind === "base" ? destination.baseId : tile?.baseId;
  const base = baseId ? context.bases.find((entry) => entry.id === baseId) : undefined;
  const construction = destination.kind === "bridge"
    ? destination.construction
    : coord ? context.constructions.find((entry) => entry.active && entry.tiles.some((cell) => cell.x === coord.x && cell.y === coord.y)) : undefined;
  return [
    1,
    ...oneHot(destination.kind, RL_POSITION_KINDS),
    Number(Boolean(coord)),
    coord ? normalized(coord.x, context.observation.map.width) : 0,
    coord ? normalized(coord.y, context.observation.map.height) : 0,
    ...oneHot(tile?.terrain, RL_TERRAIN_TYPES),
    Number(Boolean(tile?.roadSectionId)), Number(Boolean(base)),
    ...teamRef(context, base?.ownerTeamId),
    Number(construction?.kind === "bridge"), Number(construction?.kind === "obstacle"),
    ...teamRef(context, construction?.ownerTeamId),
  ];
}

function rewardRef(context: EncodingContext, requestId: string | undefined): number[] {
  const request = requestId ? context.observation.rewardPlacementRequests.find((entry) => entry.id === requestId) : undefined;
  if (!request) return Array(context.rewardSemanticWidth).fill(0);
  return [
    1,
    ...oneHot(request.rewardType, RL_REWARD_TYPES),
    ...teamRef(context, request.teamId),
    ...baseRef(context, request.sourceBaseId),
    ...baseRef(context, request.fixedBaseId),
    ...baseRef(context, request.eligibleBaseIds.length === 1 ? request.eligibleBaseIds[0] : undefined),
    ...unitRef(context, request.sourceKingUnitId),
    ...context.bases.map((base) => Number(request.eligibleBaseIds.includes(base.id))),
    Number(request.destinationKind === "fixed"), Number(request.destinationKind === "selectable"),
  ];
}

function tileSetMask(context: EncodingContext, tileIds: string[] | undefined) {
  const mask = Array(context.observation.map.width * context.observation.map.height).fill(0);
  for (const tileId of tileIds ?? []) {
    const match = /^(-?\d+),(-?\d+)$/.exec(tileId);
    if (!match) continue;
    const x = Number(match[1]), y = Number(match[2]);
    if (x >= 0 && x < context.observation.map.width && y >= 0 && y < context.observation.map.height) mask[y * context.observation.map.width + x] = 1;
  }
  return mask;
}

function assertSparseWidth(
  writer: SparseActionWriter,
  start: number,
  expected: number,
  label: string,
) {
  const actual = writer.cursor - start;
  if (actual !== expected) {
    throw new Error(`${label} sparse width mismatch: ${actual} != ${expected}`);
  }
}

function writeTeamRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  teamId: string | undefined,
) {
  writer.oneAt(
    context.teamIndex.get(teamId ?? "") ?? context.teams.length,
    context.teams.length + 1,
  );
}

function writeUnitRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  unitId: string | undefined,
) {
  const start = writer.cursor;
  const unit = unitId
    ? context.observation.units.find(
      (entry) => entry.id === unitId && entry.position.kind !== "removed",
    )
    : undefined;
  if (!unit) {
    writer.skip(context.unitSemanticWidth);
    return;
  }

  writer.emit(1);
  writer.oneAt(context.unitIndex.get(unit.id), context.maxUnits);
  writeTeamRefSparse(writer, context, unit.ownerTeamId);
  writer.oneHot(unit.type, RL_UNIT_TYPES);
  writer.emit(unit.hp);
  writer.oneHot(unit.position.kind, RL_POSITION_KINDS);

  const coord = positionCoord(context.observation, unit.position);
  writer.emit(Number(Boolean(coord)));
  writer.emit(coord ? normalized(coord.x, context.observation.map.width) : 0);
  writer.emit(coord ? normalized(coord.y, context.observation.map.height) : 0);

  const position = unit.position;
  const baseSlot = position.kind === "base"
    ? context.bases
      .find((base) => base.id === position.baseId)
      ?.slots.find((entry) => entry.id === position.slotId)
    : undefined;
  writer.emit(baseSlot ? baseSlot.localRow : 0);
  writer.emit(baseSlot ? baseSlot.localCol : 0);
  writer.emit(Number(unit.statuses.some((status) => status.kind === "retreating")));
  writer.emit(Number(unit.statuses.some((status) => status.kind === "encouraged")));
  writer.emit(Number(unit.statuses.some((status) => status.kind === "cannot_attack")));
  writer.oneHot(unit.role, RL_STRATEGIST_ROLES);

  assertSparseWidth(writer, start, context.unitSemanticWidth, "unitRef");
}

function writeBaseRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  baseId: string | undefined,
) {
  const start = writer.cursor;
  const base = baseId
    ? context.bases.find((entry) => entry.id === baseId)
    : undefined;
  if (!base) {
    writer.skip(context.baseSemanticWidth);
    return;
  }

  writer.emit(1);
  writer.oneAt(context.baseIndex.get(base.id), context.bases.length);
  writer.emit(Number(base.type === "home"));
  writer.emit(Number(base.type === "neutral"));
  writeTeamRefSparse(writer, context, base.ownerTeamId);

  const centroid = base.coords.length
    ? {
      x: base.coords.reduce((sum, coord) => sum + coord.x, 0) / base.coords.length,
      y: base.coords.reduce((sum, coord) => sum + coord.y, 0) / base.coords.length,
    }
    : undefined;
  writer.emit(Number(Boolean(centroid)));
  writer.emit(centroid ? normalized(centroid.x, context.observation.map.width) : 0);
  writer.emit(centroid ? normalized(centroid.y, context.observation.map.height) : 0);
  writer.emit(base.slots.length);
  writer.emit(base.slots.filter((slot) => Boolean(slot.unitId)).length);
  writer.emit(base.slots.filter((slot) => slot.kind === "front" && Boolean(slot.unitId)).length);
  writer.emit(base.slots.filter((slot) => slot.kind === "protected" && Boolean(slot.unitId)).length);
  writeTeamRefSparse(writer, context, base.occupationPriorityTeamId);

  assertSparseWidth(writer, start, context.baseSemanticWidth, "baseRef");
}

function writeConstructionRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  constructionId: string | undefined,
) {
  const start = writer.cursor;
  const construction = constructionId
    ? context.constructions.find((entry) => entry.id === constructionId)
    : undefined;
  if (!construction) {
    writer.skip(context.constructionSemanticWidth);
    return;
  }

  writer.emit(1);
  writer.oneAt(
    context.constructionIndex.get(construction.id),
    context.observation.map.tiles.length,
  );
  writer.emit(Number(construction.kind === "bridge"));
  writer.emit(Number(construction.kind === "obstacle"));
  writer.emit(Number(construction.active));
  writeTeamRefSparse(writer, context, construction.ownerTeamId);
  writer.emit(construction.tiles.length);

  const centroid = construction.tiles.length
    ? {
      x: construction.tiles.reduce((sum, tile) => sum + tile.x, 0) / construction.tiles.length,
      y: construction.tiles.reduce((sum, tile) => sum + tile.y, 0) / construction.tiles.length,
    }
    : undefined;
  writer.emit(Number(Boolean(centroid)));
  writer.emit(centroid ? normalized(centroid.x, context.observation.map.width) : 0);
  writer.emit(centroid ? normalized(centroid.y, context.observation.map.height) : 0);

  assertSparseWidth(
    writer,
    start,
    context.constructionSemanticWidth,
    "constructionRef",
  );
}

function writeTileRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  tileId: string | undefined,
) {
  const start = writer.cursor;
  const destination = parseDestination(context, tileId);
  if (!destination) {
    writer.skip(context.tileSemanticWidth);
    return;
  }

  const coord = "x" in destination
    && typeof destination.x === "number"
    && typeof destination.y === "number"
    ? { x: destination.x, y: destination.y }
    : undefined;
  const tile = coord ? context.tiles.get(coordKey(coord.x, coord.y)) : undefined;
  const baseId = destination.kind === "base" ? destination.baseId : tile?.baseId;
  const base = baseId
    ? context.bases.find((entry) => entry.id === baseId)
    : undefined;
  const construction = destination.kind === "bridge"
    ? destination.construction
    : coord
      ? context.constructions.find(
        (entry) => entry.active
          && entry.tiles.some((cell) => cell.x === coord.x && cell.y === coord.y),
      )
      : undefined;

  writer.emit(1);
  writer.oneHot(destination.kind, RL_POSITION_KINDS);
  writer.emit(Number(Boolean(coord)));
  writer.emit(coord ? normalized(coord.x, context.observation.map.width) : 0);
  writer.emit(coord ? normalized(coord.y, context.observation.map.height) : 0);
  writer.oneHot(tile?.terrain, RL_TERRAIN_TYPES);
  writer.emit(Number(Boolean(tile?.roadSectionId)));
  writer.emit(Number(Boolean(base)));
  writeTeamRefSparse(writer, context, base?.ownerTeamId);
  writer.emit(Number(construction?.kind === "bridge"));
  writer.emit(Number(construction?.kind === "obstacle"));
  writeTeamRefSparse(writer, context, construction?.ownerTeamId);

  assertSparseWidth(writer, start, context.tileSemanticWidth, "tileRef");
}

function writeRewardRefSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  requestId: string | undefined,
) {
  const start = writer.cursor;
  const request = requestId
    ? context.observation.rewardPlacementRequests.find(
      (entry) => entry.id === requestId,
    )
    : undefined;
  if (!request) {
    writer.skip(context.rewardSemanticWidth);
    return;
  }

  writer.emit(1);
  writer.oneHot(request.rewardType, RL_REWARD_TYPES);
  writeTeamRefSparse(writer, context, request.teamId);
  writeBaseRefSparse(writer, context, request.sourceBaseId);
  writeBaseRefSparse(writer, context, request.fixedBaseId);
  writeBaseRefSparse(
    writer,
    context,
    request.eligibleBaseIds.length === 1
      ? request.eligibleBaseIds[0]
      : undefined,
  );
  writeUnitRefSparse(writer, context, request.sourceKingUnitId);
  writer.binaryMask(
    context.bases.flatMap((base, index) =>
      request.eligibleBaseIds.includes(base.id) ? [index] : []
    ),
    context.bases.length,
  );
  writer.emit(Number(request.destinationKind === "fixed"));
  writer.emit(Number(request.destinationKind === "selectable"));

  assertSparseWidth(writer, start, context.rewardSemanticWidth, "rewardRef");
}

function writeTileSetMaskSparse(
  writer: SparseActionWriter,
  context: EncodingContext,
  tileIds: string[] | undefined,
) {
  const width = context.observation.map.width * context.observation.map.height;
  const indices: number[] = [];
  for (const tileId of tileIds ?? []) {
    const match = /^(-?\d+),(-?\d+)$/.exec(tileId);
    if (!match) continue;
    const x = Number(match[1]);
    const y = Number(match[2]);
    if (
      x >= 0
      && x < context.observation.map.width
      && y >= 0
      && y < context.observation.map.height
    ) {
      indices.push(y * context.observation.map.width + x);
    }
  }
  writer.binaryMask(indices, width);
}

function encodeSparseAction(
  context: EncodingContext,
  action: RlLegalAction,
  schemaVersion: 1 | 2,
): SparseActionRow {
  const writer = new SparseActionWriter();
  const base = action.baseId
    ? context.bases.find((entry) => entry.id === action.baseId)
    : undefined;
  const slot = base?.slots.find((entry) => entry.id === action.slotId);

  writer.oneHot(
    action.actionType,
    schemaVersion === 1 ? RL_ACTION_TYPES : RL_ACTION_TYPES_V2,
  );
  writer.emit(Number(action.isPass));
  writeTeamRefSparse(writer, context, action.actorTeamId);
  writeUnitRefSparse(writer, context, action.unitId);
  writeUnitRefSparse(writer, context, action.targetId);
  if (schemaVersion === 2) {
    writeUnitRefSparse(writer, context, action.partnerUnitId);
  }
  writeTileRefSparse(writer, context, action.tileId);
  writeBaseRefSparse(writer, context, action.baseId);
  writer.emit(Number(Boolean(slot)));
  writer.emit(Number(slot?.kind === "front"));
  writer.emit(Number(slot?.kind === "protected"));
  writeUnitRefSparse(writer, context, slot?.unitId);
  writer.oneHot(action.unitType, RL_UNIT_TYPES);
  writer.oneHot(action.strategistRole, RL_STRATEGIST_ROLES);
  writer.oneHot(action.strategistActionKind, RL_STRATEGIST_ACTIONS);
  writeRewardRefSparse(writer, context, action.requestId);
  writeConstructionRefSparse(writer, context, action.constructionId);
  writer.emit(action.tileIds?.length ?? 0);
  writeTileSetMaskSparse(writer, context, action.tileIds);

  return writer.finish();
}

function encodeAction(context: EncodingContext, action: RlLegalAction, schemaVersion: 1 | 2) {
  const base = action.baseId ? context.bases.find((entry) => entry.id === action.baseId) : undefined;
  const slot = base?.slots.find((entry) => entry.id === action.slotId);
  return [
    ...oneHot(action.actionType, schemaVersion === 1 ? RL_ACTION_TYPES : RL_ACTION_TYPES_V2),
    Number(action.isPass),
    ...teamRef(context, action.actorTeamId),
    ...unitRef(context, action.unitId),
    ...unitRef(context, action.targetId),
    ...(schemaVersion === 2 ? unitRef(context, action.partnerUnitId) : []),
    ...tileRef(context, action.tileId),
    ...baseRef(context, action.baseId),
    Number(Boolean(slot)), Number(slot?.kind === "front"), Number(slot?.kind === "protected"), ...unitRef(context, slot?.unitId),
    ...oneHot(action.unitType, RL_UNIT_TYPES),
    ...oneHot(action.strategistRole, RL_STRATEGIST_ROLES),
    ...oneHot(action.strategistActionKind, RL_STRATEGIST_ACTIONS),
    ...rewardRef(context, action.requestId),
    ...constructionRef(context, action.constructionId),
    action.tileIds?.length ?? 0,
    ...tileSetMask(context, action.tileIds),
  ];
}

export function encodeRlLegalActions(observation: RlObservation, legalActions: readonly RlLegalAction[]): EncodedLegalActions {
  const context = createContext(observation);
  return {
    schemaVersion: RL_ACTION_ENCODER_VERSION,
    actions: legalActions.map((action) => encodeAction(context, action, 1)),
    actionKeys: legalActions.map((action) => action.actionKey),
  };
}
export const encodeRlLegalActionsV1 = encodeRlLegalActions;

export function encodeRlLegalActionsV2(observation: RlObservation, legalActions: readonly RlLegalAction[]): EncodedLegalActionsV2 {
  const context = createContext(observation);
  return {
    schemaVersion: 2,
    actions: legalActions.map((action) => encodeAction(context, action, 2)),
    actionKeys: legalActions.map((action) => action.actionKey),
  };
}

export function encodeRlLegalActionsSparseV2(
  observation: RlObservation,
  legalActions: readonly RlLegalAction[],
): EncodedLegalActionsSparseV2 {
  const context = createContext(observation);
  return {
    schemaVersion: 2,
    sparseActions: legalActions.map((action) =>
      encodeSparseAction(context, action, 2)
    ),
    actionKeys: legalActions.map((action) => action.actionKey),
  };
}

export function getRlActionFeatureWidth(observation: RlObservation) {
  const context = createContext(observation);
  return encodeAction(context, {
    actionKey: "<schema-width>",
    actionType: "submit_movement",
    actorTeamId: observation.observingTeamId,
    isPass: false,
  }, 1).length;
}

export const getRlActionFeatureWidthV1 = getRlActionFeatureWidth;

export function getRlActionFeatureWidthV2(observation: RlObservation) {
  const context = createContext(observation);
  return encodeAction(context, {
    actionKey: "<schema-width-v2>",
    actionType: "submit_movement",
    actorTeamId: observation.observingTeamId,
    isPass: false,
  }, 2).length;
}
