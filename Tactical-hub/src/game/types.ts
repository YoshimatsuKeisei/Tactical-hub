export type TeamStatus = "active" | "defeated" | "eliminated" | "neutral";

export type Team = {
  id: string;
  name: string;
  color: string;
  status: TeamStatus;
  homeBaseId?: string;
  controlledBaseIds: string[];
  isNeutral?: boolean;
};

export type UnitType =
  | "king"
  | "infantry"
  | "cavalry"
  | "archer"
  | "engineer"
  | "ninja"
  | "apprentice_ninja"
  | "strategist";

export type StrategistRole = "encourage" | "builder" | "teleporter";

export type UnitStatus =
  | { kind: "retreating"; remainingTurns?: number; sourceId?: string }
  | { kind: "encouraged"; remainingTurns?: number; sourceId?: string }
  | { kind: "cannot_attack"; remainingTurns?: number; sourceId?: string };

export type UnitPosition =
  | { kind: "tile"; x: number; y: number }
  | { kind: "water"; x: number; y: number }
  | { kind: "base"; baseId: string; slotId: string }
  | { kind: "bridge"; bridgeId: string; cellIndex: number }
  | {
      kind: "removed";
      reason: "defeated" | "water_trap" | "king_defeat_reset";
    };

export type Unit = {
  id: string;
  ownerTeamId: string;
  type: UnitType;
  hp: number;
  position: UnitPosition;
  statuses: UnitStatus[];
  role?: StrategistRole;
};

export type BaseSlot = {
  id: string;
  baseId: string;
  kind: "front" | "protected";
  localRow: 0 | 1;
  localCol: 0 | 1;
  unitId?: string;
};

export type Base = {
  id: string;
  name: string;
  type: "home" | "neutral";
  ownerTeamId: string;
  coords: { x: number; y: number }[];
  slots: BaseSlot[];
  protectedSlotId?: string;
  occupationPriorityTeamId?: string;
};

export type TerrainType =
  | "outside"
  | "road"
  | "lake"
  | "base"
  | "baseGate"
  | "reorganize";

export type Tile = {
  x: number;
  y: number;
  symbol: string;
  terrain: TerrainType;
  baseId?: string;

  /**
   * 両端を拠点に囲まれた1本の道区間を表すID。
   *
   * road / baseGate / reorganize に設定する。
   * base / lake / outside には設定しない。
   */
  roadSectionId?: string;
};
export type BoardMap = {
  id: string;
  name: string;
  width: number;
  height: number;
  tiles: Tile[];
  bases: Base[];
};

export type ProductionChoice = {
  teamId: string;
  baseId: string;
  unitType: UnitType;
};

export type MovementIntent = {
  teamId: string;
  unitId: string;
  from: UnitPosition;
  to: UnitPosition;
  stay: boolean;
};

export type AttackTarget = {
  kind: "unit";
  unitId: string;
  baseId?: string;
  slotId?: string;
  baseSuccessDenominator?: number;
  finalSuccessDenominator?: number;
  encouraged?: boolean;
};

export type AttackIntent = {
  teamId: string;
  attackerUnitId: string;
  target?: AttackTarget;
  pass: boolean;
};

export type HitResult = "hit" | "miss" | "invalid";

export type BattleResult = {
  hit: HitResult;
  damage: number;
  defeated: boolean;
};

export type BattleEvent = {
  id: string;
  attackerUnitId: string;
  target: AttackTarget;
  baseSuccessDenominator: number;
  finalSuccessDenominator: number;
  encouraged: boolean;
  result?: BattleResult;
};

export type UnitTurnFlags = {
  unitId: string;
  battleTurnNumber: number;
  positionAtBattleStart?: UnitPosition;
  enemyBaseDistanceAtBattleStart?: number;
  enemyBaseWithin3AtBattleStart?: boolean;
  wasAliveAtBattleStart: boolean;
  survivedPreviousBattle: boolean;
  attackedInPreviousBattle: boolean;
  wasTargetedInPreviousBattle: boolean;
  retreatEligible: boolean;
  retreatEligibilityReason?: string;
};

export type ActionIntent = {
  teamId: string;
  productionChoices: ProductionChoice[];
  movementIntents: MovementIntent[];
  attackIntents: AttackIntent[];
};

export type TurnState = {
  turnNumber: number;
  phase:
    | "production"
    | "movement_input"
    | "movement_resolution"
    | "attack_input"
    | "battle_resolution"
    | "capture_resolution"
    | "reward_placement";
  actionIntents: ActionIntent[];
};

export type GameLog = {
  id: string;
  turnNumber: number;
  type: "setup" | "production" | "movement" | "battle" | "siege" | "capture" | "reward";
  message: string;
  relatedIds?: string[];
};

export type SiegeTeamRecord = {
  teamId: string;
  defenderKills: number;
  effectiveAttackTurns: number;
};

export type SiegeState = {
  baseId: string;
  defendingTeamId: string;
  teamRecords: SiegeTeamRecord[];
  lastEffectiveAttackTurn?: number;
  active: boolean;
  defenderLossOccurred: boolean;
  fallCandidateTeamIds: string[];
};

export type RewardType = "capture_reward" | "contribution_compensation";
export type RewardPlacementRequest = {
  id: string;
  teamId: string;
  rewardType: RewardType;
  sourceBaseId: string;
  destinationKind: "fixed" | "selectable";
  fixedBaseId?: string;
  eligibleBaseIds: string[];
  selectedUnitType?: UnitType;
  completed: boolean;
  expired: boolean;
  expirationReason?: string;
};

export type GameConfig = {
  playerCount: number;
  productionInterval: number;
  mapId: string;
};

export type GameState = {
  config: GameConfig;
  map: BoardMap;
  turnNumber: number;
  phase: TurnState["phase"];
  teams: Team[];
  units: Unit[];
  bases: Base[];
  unitTurnFlags: UnitTurnFlags[];
  turnState: TurnState;
  logs: GameLog[];
  siegeStates: SiegeState[];
  rewardPlacementRequests: RewardPlacementRequest[];
  phaseAfterRewards?: "attack_input" | "movement_input";
};
