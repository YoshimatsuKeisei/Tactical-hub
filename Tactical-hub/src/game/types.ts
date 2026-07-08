export type TeamStatus = "active" | "eliminated" | "neutral";

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
  | { kind: "removed"; reason: "defeated" | "water_trap" | "king_defeat_reset" };

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

export type TerrainType = "outside" | "road" | "lake" | "base" | "baseGate" | "reorganize";

export type Tile = {
  x: number;
  y: number;
  symbol: string;
  terrain: TerrainType;
  baseId?: string;
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

export type ActionIntent = {
  teamId: string;
  productionChoices: ProductionChoice[];
  movementIntents: MovementIntent[];
};

export type TurnState = {
  turnNumber: number;
  phase: "production" | "movement_input" | "movement_resolution";
  actionIntents: ActionIntent[];
};

export type GameLog = {
  id: string;
  turnNumber: number;
  type: "setup" | "production" | "movement";
  message: string;
  relatedIds?: string[];
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
  turnState: TurnState;
  logs: GameLog[];
};
