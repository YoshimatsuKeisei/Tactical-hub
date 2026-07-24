import type {
  ActionIntent,
  BoardCoord,
  Construction,
  RewardPlacementRequest,
  StrategistActionIntent,
  TeleportIntent,
  Unit,
  UnitPosition,
} from "../types";
import type { RlObservation } from "./rlEnvironment";

export const RL_OBSERVATION_ENCODER_VERSION = 1;

export const RL_PHASES = ["production", "movement_input", "movement_resolution", "attack_input", "battle_resolution", "capture_resolution", "reward_placement", "strategist_action_input", "strategist_action_resolution"] as const;
const TEAM_STATUSES = ["active", "defeated", "eliminated", "neutral"] as const;
export const RL_UNIT_TYPES = ["king", "infantry", "cavalry", "archer", "engineer", "ninja", "apprentice_ninja", "strategist"] as const;
export const RL_POSITION_KINDS = ["tile", "water", "base", "bridge"] as const;
export const RL_TERRAIN_TYPES = ["outside", "road", "lake", "base", "baseGate", "reorganize"] as const;
export const RL_STRATEGIST_ROLES = ["encourage", "builder", "teleporter"] as const;
export const RL_REWARD_TYPES = ["capture_reward", "contribution_compensation", "king_conquest_reward", "king_contribution_compensation", "overridden_capture_compensation"] as const;
export const RL_STRATEGIST_ACTIONS = ["place_bridge", "reset_bridge", "place_obstacle", "reset_obstacle", "pass"] as const;

const PHASES = RL_PHASES;
const UNIT_TYPES = RL_UNIT_TYPES;
const POSITION_KINDS = RL_POSITION_KINDS;
const TERRAIN_TYPES = RL_TERRAIN_TYPES;
const STRATEGIST_ROLES = RL_STRATEGIST_ROLES;
const REWARD_TYPES = RL_REWARD_TYPES;
const STRATEGIST_ACTIONS = RL_STRATEGIST_ACTIONS;

const oneHot = <T extends string>(value: string | undefined, values: readonly T[]) => values.map((entry) => Number(value === entry));
const finite = (value: number | undefined) => Number.isFinite(value) ? value! : 0;
const normalized = (value: number, size: number) => size <= 1 ? 0 : Math.max(0, Math.min(1, value / (size - 1)));
const coordKey = (coord: BoardCoord) => `${coord.x},${coord.y}`;

export const RL_OBSERVATION_SCHEMA = {
  global: ["turnNumber", "productionInterval", "mapWidth", "mapHeight", "actorPresent", "observerIsActor", ...PHASES.map((phase) => `phase:${phase}`)],
  teamBase: ["isObserver", "isNeutral", ...TEAM_STATUSES.map((status) => `status:${status}`), "controlledBaseCount", "defeatedUnitCount", "conqueredTeamCount", "hasHomeBase", "movementCompleted", "strategistSubmitted", "productionCompleted", "movementOrderIndex", "seatOrderIndex"],
  unitBase: ["hp", ...UNIT_TYPES.map((type) => `type:${type}`), ...POSITION_KINDS.map((kind) => `position:${kind}`), "hasCoordinate", "xNormalized", "yNormalized", ...STRATEGIST_ROLES.map((role) => `role:${role}`), "retreating", "encouraged", "cannotAttack", "statusRemainingTurns", "movedThisMovementPhase", "battleFlagPresent", "survivedPreviousBattle", "attackedInPreviousBattle", "wasTargetedInPreviousBattle", "retreatEligible", "enemyBaseDistanceAtBattleStart", "enemyBaseWithin3AtBattleStart"],
  mapBase: ["xNormalized", "yNormalized", ...TERRAIN_TYPES.map((terrain) => `terrain:${terrain}`), "hasBase", "hasRoadSection", "roadSection:N", "roadSection:NE", "roadSection:E", "roadSection:SE", "roadSection:S", "roadSection:SW", "roadSection:W", "roadSection:NW", "activeBridge", "activeObstacle"],
  constructionBase: ["active", "placedTurn", "tileCount", "hasCoordinate", "centroidXNormalized", "centroidYNormalized", "kind:bridge", "kind:obstacle", "hasOwner", "hasManager"],
} as const;

export type EncodedStrategicState = {
  global: number[];
  siegeStates: number[][];
  kingCampaignStates: number[][];
  rewardPlacementRequests: number[][];
  strategistCooldowns: number[][];
  teleportCooldowns: number[][];
  productionIntents: number[][];
  movementIntents: number[][];
  attackIntents: number[][];
  strategistActionIntents: number[][];
  teleportIntents: number[][];
};

export type EncodedObservation = {
  schemaVersion: number;
  global: number[];
  teams: number[][];
  teamMask: number[];
  units: number[][];
  unitMask: number[];
  map: number[][][];
  bases: number[][];
  baseMask: number[];
  constructions: number[][];
  constructionMask: number[];
  strategicState: EncodedStrategicState;
};

type Context = {
  observation: RlObservation;
  teams: RlObservation["teams"];
  teamIndex: Map<string, number>;
  units: Unit[];
  unitById: Map<string, Unit>;
  bases: RlObservation["bases"];
  baseIndex: Map<string, number>;
  constructionById: Map<string, Construction>;
};

function orderedTeams(observation: RlObservation) {
  const observer = observation.teams.find((team) => team.id === observation.observingTeamId);
  return observer ? [observer, ...observation.teams.filter((team) => team.id !== observer.id)] : [...observation.teams];
}

function teamVector(context: Context, teamId: string | undefined) {
  const vector = Array(context.teams.length + 1).fill(0);
  const index = teamId === undefined ? undefined : context.teamIndex.get(teamId);
  vector[index === undefined ? context.teams.length : index] = 1;
  return vector;
}

function baseVector(context: Context, baseId: string | undefined) {
  const vector = Array(context.bases.length + 1).fill(0);
  const index = baseId === undefined ? undefined : context.baseIndex.get(baseId);
  vector[index === undefined ? context.bases.length : index] = 1;
  return vector;
}

function unitReference(context: Context, unitId: string | undefined) {
  const unit = unitId ? context.unitById.get(unitId) : undefined;
  return [Number(Boolean(unit)), ...teamVector(context, unit?.ownerTeamId), ...oneHot(unit?.type, UNIT_TYPES)];
}

function positionCoordinate(context: Context, position: UnitPosition): BoardCoord | undefined {
  if (position.kind === "tile" || position.kind === "water") return position;
  if (position.kind === "bridge") return context.constructionById.get(position.bridgeId)?.tiles[position.cellIndex];
  if (position.kind === "base") {
    const coords = context.bases.find((base) => base.id === position.baseId)?.coords ?? [];
    if (!coords.length) return undefined;
    return {
      x: coords.reduce((sum, coord) => sum + coord.x, 0) / coords.length,
      y: coords.reduce((sum, coord) => sum + coord.y, 0) / coords.length,
    };
  }
  return undefined;
}

function positionVector(context: Context, position: UnitPosition | undefined) {
  if (!position || position.kind === "removed") return [0, ...Array(POSITION_KINDS.length).fill(0), 0, 0, 0, ...baseVector(context, undefined)];
  const coord = positionCoordinate(context, position);
  return [
    1,
    ...oneHot(position.kind, POSITION_KINDS),
    Number(Boolean(coord)),
    coord ? normalized(coord.x, context.observation.map.width) : 0,
    coord ? normalized(coord.y, context.observation.map.height) : 0,
    ...baseVector(context, position.kind === "base" ? position.baseId : undefined),
  ];
}

function unitSortKey(context: Context, unit: Unit) {
  const team = context.teamIndex.get(unit.ownerTeamId) ?? context.teams.length;
  const type = UNIT_TYPES.indexOf(unit.type);
  const coord = positionCoordinate(context, unit.position);
  return [team, type, coord?.y ?? Number.MAX_SAFE_INTEGER, coord?.x ?? Number.MAX_SAFE_INTEGER, unit.id] as const;
}

function compareKeys(left: readonly (number | string)[], right: readonly (number | string)[]) {
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const a = left[index], b = right[index];
    if (a === b) continue;
    return typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
  }
  return 0;
}

function encodeUnit(context: Context, unit: Unit) {
  const flag = context.observation.unitTurnFlags.find((entry) => entry.unitId === unit.id);
  const remainingTurns = unit.statuses.reduce((maximum, status) => Math.max(maximum, status.remainingTurns ?? 0), 0);
  return [
    unit.hp,
    ...oneHot(unit.type, UNIT_TYPES),
    ...oneHot(unit.position.kind, POSITION_KINDS),
    ...positionVector(context, unit.position).slice(1 + POSITION_KINDS.length),
    ...oneHot(unit.role, STRATEGIST_ROLES),
    Number(unit.statuses.some((status) => status.kind === "retreating")),
    Number(unit.statuses.some((status) => status.kind === "encouraged")),
    Number(unit.statuses.some((status) => status.kind === "cannot_attack")),
    remainingTurns,
    Number(context.observation.movedUnitIdsThisMovementPhase.includes(unit.id)),
    Number(Boolean(flag)),
    Number(flag?.survivedPreviousBattle ?? false),
    Number(flag?.attackedInPreviousBattle ?? false),
    Number(flag?.wasTargetedInPreviousBattle ?? false),
    Number(flag?.retreatEligible ?? false),
    finite(flag?.enemyBaseDistanceAtBattleStart),
    Number(flag?.enemyBaseWithin3AtBattleStart ?? false),
    ...teamVector(context, unit.ownerTeamId),
    ...baseVector(context, unit.position.kind === "base" ? unit.position.baseId : undefined),
  ];
}

function encodeMap(context: Context) {
  const tileByCoord = new Map(context.observation.map.tiles.map((tile) => [coordKey(tile), tile]));
  const constructionAt = new Map<string, Construction[]>();
  for (const construction of context.observation.constructions.filter((entry) => entry.active)) {
    for (const tile of construction.tiles) constructionAt.set(coordKey(tile), [...(constructionAt.get(coordKey(tile)) ?? []), construction]);
  }
  const directions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]] as const;
  return Array.from({ length: context.observation.map.height }, (_, y) =>
    Array.from({ length: context.observation.map.width }, (_, x) => {
      const tile = tileByCoord.get(`${x},${y}`);
      const constructions = constructionAt.get(`${x},${y}`) ?? [];
      const base = tile?.baseId ? context.bases.find((entry) => entry.id === tile.baseId) : undefined;
      return [
        normalized(x, context.observation.map.width),
        normalized(y, context.observation.map.height),
        ...oneHot(tile?.terrain, TERRAIN_TYPES),
        Number(Boolean(base)),
        Number(Boolean(tile?.roadSectionId)),
        ...directions.map(([dx, dy]) => Number(Boolean(tile?.roadSectionId) && tileByCoord.get(`${x + dx},${y + dy}`)?.roadSectionId === tile?.roadSectionId)),
        Number(constructions.some((entry) => entry.kind === "bridge")),
        Number(constructions.some((entry) => entry.kind === "obstacle")),
        ...teamVector(context, base?.ownerTeamId),
        ...teamVector(context, constructions.find((entry) => entry.kind === "bridge")?.ownerTeamId),
        ...teamVector(context, constructions.find((entry) => entry.kind === "obstacle")?.ownerTeamId),
      ];
    }),
  );
}

function encodeBase(context: Context, base: RlObservation["bases"][number], maxSlots: number) {
  const centroid = base.coords.length ? {
    x: base.coords.reduce((sum, coord) => sum + coord.x, 0) / base.coords.length,
    y: base.coords.reduce((sum, coord) => sum + coord.y, 0) / base.coords.length,
  } : undefined;
  const slots = [...base.slots].sort((left, right) => left.localRow - right.localRow || left.localCol - right.localCol);
  const slotWidth = 3 + context.teams.length + 1 + UNIT_TYPES.length;
  const slotFeatures = slots.flatMap((slot) => {
    const occupant = slot.unitId ? context.unitById.get(slot.unitId) : undefined;
    return [Number(slot.kind === "front"), Number(slot.kind === "protected"), Number(Boolean(occupant)), ...teamVector(context, occupant?.ownerTeamId), ...oneHot(occupant?.type, UNIT_TYPES)];
  });
  slotFeatures.push(...Array((maxSlots - slots.length) * slotWidth).fill(0));
  return [
    Number(base.type === "home"), Number(base.type === "neutral"),
    ...teamVector(context, base.ownerTeamId),
    Number(Boolean(centroid)),
    centroid ? normalized(centroid.x, context.observation.map.width) : 0,
    centroid ? normalized(centroid.y, context.observation.map.height) : 0,
    base.coords.length,
    base.slots.length,
    ...teamVector(context, base.occupationPriorityTeamId),
    ...slotFeatures,
  ];
}

function encodeConstruction(context: Context, construction: Construction) {
  const centroid = construction.tiles.length ? {
    x: construction.tiles.reduce((sum, coord) => sum + coord.x, 0) / construction.tiles.length,
    y: construction.tiles.reduce((sum, coord) => sum + coord.y, 0) / construction.tiles.length,
  } : undefined;
  const manager = construction.managerUnitId ? context.unitById.get(construction.managerUnitId) : undefined;
  return [
    Number(construction.active), construction.placedTurn, construction.tiles.length, Number(Boolean(centroid)),
    centroid ? normalized(centroid.x, context.observation.map.width) : 0,
    centroid ? normalized(centroid.y, context.observation.map.height) : 0,
    Number(construction.kind === "bridge"), Number(construction.kind === "obstacle"),
    Number(Boolean(construction.ownerTeamId)), Number(Boolean(manager)),
    ...teamVector(context, construction.ownerTeamId),
    ...teamVector(context, manager?.ownerTeamId),
  ];
}

function encodeReward(context: Context, request: RewardPlacementRequest) {
  return [
    ...teamVector(context, request.teamId),
    ...oneHot(request.rewardType, REWARD_TYPES),
    ...baseVector(context, request.sourceBaseId),
    ...unitReference(context, request.sourceKingUnitId),
    Number(request.destinationKind === "fixed"), Number(request.destinationKind === "selectable"),
    ...baseVector(context, request.fixedBaseId),
    ...context.bases.map((base) => Number(request.eligibleBaseIds.includes(base.id))),
    ...oneHot(request.selectedUnitType, UNIT_TYPES),
    Number(request.completed), Number(request.expired),
  ];
}

function encodeActionIntents(context: Context, intents: ActionIntent[]) {
  return {
    production: intents.flatMap((group) => group.productionChoices.map((choice) => [
      ...teamVector(context, choice.teamId), ...baseVector(context, choice.baseId), ...oneHot(choice.unitType, UNIT_TYPES), ...oneHot(choice.strategistRole, STRATEGIST_ROLES),
    ])),
    movement: intents.flatMap((group) => group.movementIntents.map((intent) => [
      ...teamVector(context, intent.teamId), ...unitReference(context, intent.unitId), ...positionVector(context, intent.from), ...positionVector(context, intent.to), Number(intent.stay),
    ])),
    attack: intents.flatMap((group) => group.attackIntents.map((intent) => [
      ...teamVector(context, intent.teamId), ...unitReference(context, intent.attackerUnitId), ...unitReference(context, intent.target?.unitId), Number(intent.pass),
    ])),
  };
}

function encodeStrategistIntent(context: Context, intent: StrategistActionIntent) {
  const tiles = intent.tiles ?? [];
  const centroid = tiles.length ? {
    x: tiles.reduce((sum, coord) => sum + coord.x, 0) / tiles.length,
    y: tiles.reduce((sum, coord) => sum + coord.y, 0) / tiles.length,
  } : undefined;
  const construction = intent.constructionId ? context.constructionById.get(intent.constructionId) : undefined;
  return [
    ...teamVector(context, intent.teamId), ...unitReference(context, intent.strategistUnitId),
    ...oneHot(intent.action, STRATEGIST_ACTIONS),
    tiles.length, Number(Boolean(centroid)),
    centroid ? normalized(centroid.x, context.observation.map.width) : 0,
    centroid ? normalized(centroid.y, context.observation.map.height) : 0,
    Number(Boolean(construction)), Number(construction?.kind === "bridge"), Number(construction?.kind === "obstacle"),
  ];
}

function encodeTeleportIntent(context: Context, intent: TeleportIntent) {
  return [...teamVector(context, intent.teamId), ...unitReference(context, intent.strategistUnitId), ...unitReference(context, intent.targetUnitId), ...positionVector(context, intent.to)];
}

function encodeStrategicState(context: Context): EncodedStrategicState {
  const observation = context.observation;
  const intents = encodeActionIntents(context, observation.actionIntents);
  return {
    global: [
      ...oneHot(observation.phaseAfterRewards, PHASES),
      ...teamVector(context, observation.currentMovementTeamId),
      observation.movementOrderStartIndex,
      ...context.teams.map((team) => observation.movementOrderTeamIds.indexOf(team.id)),
      ...context.teams.map((team) => Number(observation.movementCompletedTeamIds.includes(team.id))),
      ...context.teams.map((team) => Number(observation.strategistSubmittedTeamIds.includes(team.id))),
      ...context.teams.map((team) => Number(observation.productionCompletedTeamIdsThisTurn.includes(team.id))),
    ],
    siegeStates: [...observation.siegeStates].sort((a, b) => (context.baseIndex.get(a.baseId) ?? 999) - (context.baseIndex.get(b.baseId) ?? 999)).map((siege) => [
      ...baseVector(context, siege.baseId), ...teamVector(context, siege.defendingTeamId),
      Number(siege.active), Number(siege.defenderLossOccurred), Number(siege.lastEffectiveAttackTurn !== undefined), finite(siege.lastEffectiveAttackTurn),
      ...context.teams.map((team) => Number(siege.fallCandidateTeamIds.includes(team.id))),
      ...context.teams.flatMap((team) => {
        const record = siege.teamRecords.find((entry) => entry.teamId === team.id);
        return [finite(record?.defenderKills), finite(record?.effectiveAttackTurns)];
      }),
    ]),
    kingCampaignStates: [...observation.kingCampaignStates].sort((a, b) => a.kingUnitId.localeCompare(b.kingUnitId)).map((campaign) => [
      ...unitReference(context, campaign.kingUnitId), ...teamVector(context, campaign.kingTeamId),
      ...context.teams.flatMap((team) => {
        const contribution = campaign.contributions.find((entry) => entry.teamId === team.id);
        return [finite(contribution?.cumulativeDamage), finite(contribution?.effectiveAttackTurns)];
      }),
    ]),
    rewardPlacementRequests: [...observation.rewardPlacementRequests].sort((a, b) => a.id.localeCompare(b.id)).map((request) => encodeReward(context, request)),
    strategistCooldowns: [...observation.strategistCooldowns].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId) || a.kind.localeCompare(b.kind)).map((cooldown) => [
      ...unitReference(context, cooldown.strategistUnitId), Number(cooldown.kind === "bridge"), Number(cooldown.kind === "obstacle"), cooldown.availableFromTurn - observation.turnNumber,
    ]),
    teleportCooldowns: [...observation.teleportCooldowns].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId)).map((cooldown) => [
      ...unitReference(context, cooldown.strategistUnitId), cooldown.availableFromTurn - observation.turnNumber,
    ]),
    productionIntents: intents.production,
    movementIntents: intents.movement,
    attackIntents: intents.attack,
    strategistActionIntents: [...observation.strategistActionIntents].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId)).map((intent) => encodeStrategistIntent(context, intent)),
    teleportIntents: [...observation.teleportIntents].sort((a, b) => a.strategistUnitId.localeCompare(b.strategistUnitId)).map((intent) => encodeTeleportIntent(context, intent)),
  };
}

export function encodeRlObservation(observation: RlObservation): EncodedObservation {
  const teams = orderedTeams(observation);
  const bases = [...observation.bases].sort((left, right) => {
    const leftCoord = left.coords[0], rightCoord = right.coords[0];
    return (leftCoord?.y ?? 999) - (rightCoord?.y ?? 999) || (leftCoord?.x ?? 999) - (rightCoord?.x ?? 999) || left.id.localeCompare(right.id);
  });
  const context: Context = {
    observation,
    teams,
    teamIndex: new Map(teams.map((team, index) => [team.id, index])),
    units: [],
    unitById: new Map(observation.units.map((unit) => [unit.id, unit])),
    bases,
    baseIndex: new Map(bases.map((base, index) => [base.id, index])),
    constructionById: new Map(observation.constructions.map((construction) => [construction.id, construction])),
  };
  const units = observation.units.filter((unit) => unit.position.kind !== "removed").sort((left, right) => compareKeys(unitSortKey(context, left), unitSortKey(context, right)));
  context.units = units;
  const maxUnits = observation.map.tiles.length + observation.bases.reduce((sum, base) => sum + base.slots.length, 0);
  if (units.length > maxUnits) throw new Error(`RL Observation contains ${units.length} board units, exceeding safe capacity ${maxUnits}`);
  const encodedUnits = units.map((unit) => encodeUnit(context, unit));
  const unitWidth = encodedUnits[0]?.length ?? encodeUnit(context, { id: "", ownerTeamId: "", type: "infantry", hp: 0, position: { kind: "tile", x: 0, y: 0 }, statuses: [] }).length;
  const maxBases = Math.max(observation.map.bases.length, observation.bases.length);
  const maxBaseSlots = Math.max(0, ...observation.bases.map((base) => base.slots.length), ...observation.map.bases.map((base) => base.slots.length));
  const encodedBases = bases.map((base) => encodeBase(context, base, maxBaseSlots));
  const baseWidth = encodedBases[0]?.length ?? 0;
  const maxConstructions = observation.map.tiles.length;
  if (observation.constructions.length > maxConstructions) throw new Error(`RL Observation contains ${observation.constructions.length} constructions, exceeding safe capacity ${maxConstructions}`);
  const sortedConstructions = [...observation.constructions].sort((left, right) => {
    const a = left.tiles[0], b = right.tiles[0];
    return (a?.y ?? 999) - (b?.y ?? 999) || (a?.x ?? 999) - (b?.x ?? 999) || left.kind.localeCompare(right.kind) || left.id.localeCompare(right.id);
  });
  const encodedConstructions = sortedConstructions.map((construction) => encodeConstruction(context, construction));
  const constructionWidth = encodedConstructions[0]?.length ?? encodeConstruction(context, { id: "", kind: "bridge", tiles: [], placedTurn: 0, active: false }).length;

  return {
    schemaVersion: RL_OBSERVATION_ENCODER_VERSION,
    global: [
      observation.turnNumber,
      observation.config.productionInterval,
      observation.map.width,
      observation.map.height,
      Number(Boolean(observation.actorTeamId)),
      Number(observation.actorTeamId === observation.observingTeamId),
      ...oneHot(observation.phase, PHASES),
    ],
    teams: teams.map((team, index) => [
      Number(index === 0),
      Number(Boolean(team.isNeutral)),
      ...oneHot(team.status, TEAM_STATUSES),
      team.controlledBaseIds.length,
      finite(team.defeatedUnitCount),
      team.conqueredTeamIds?.length ?? 0,
      Number(Boolean(team.homeBaseId)),
      Number(observation.movementCompletedTeamIds.includes(team.id)),
      Number(observation.strategistSubmittedTeamIds.includes(team.id)),
      Number(observation.productionCompletedTeamIdsThisTurn.includes(team.id)),
      observation.movementOrderTeamIds.indexOf(team.id),
      observation.movementSeatOrderTeamIds.indexOf(team.id),
    ]),
    teamMask: teams.map(() => 1),
    units: [...encodedUnits, ...Array.from({ length: maxUnits - encodedUnits.length }, () => Array(unitWidth).fill(0))],
    unitMask: [...units.map(() => 1), ...Array(maxUnits - units.length).fill(0)],
    map: encodeMap(context),
    bases: [...encodedBases, ...Array.from({ length: maxBases - encodedBases.length }, () => Array(baseWidth).fill(0))],
    baseMask: [...bases.map(() => 1), ...Array(maxBases - bases.length).fill(0)],
    constructions: [...encodedConstructions, ...Array.from({ length: maxConstructions - encodedConstructions.length }, () => Array(constructionWidth).fill(0))],
    constructionMask: [...encodedConstructions.map(() => 1), ...Array(maxConstructions - encodedConstructions.length).fill(0)],
    strategicState: encodeStrategicState(context),
  };
}
