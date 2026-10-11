import type { GameState, Tile, Unit } from "../game/types";
import { getConstructionAt } from "../game/engine/construction";
import type { MoveHighlightSettings, TileAssetKey } from "./tileAssets";

export const TERRAIN_COLORS = { road: 0x999c96, base: 0xc6a253, lake: 0x338ac0 };
export function pixiTerrainKind(terrain: Tile["terrain"]): keyof typeof TERRAIN_COLORS | null {
  if (terrain === "outside") return null;
  if (terrain === "base" || terrain === "lake") return terrain;
  return "road";
}

// Appearance and elevation are intentionally separate: bases retain their gold
// top while sharing one raised surface with road/baseGate/reorganize.
export function isRaisedRoadTerrain(terrain: Tile["terrain"]): boolean {
  return terrain === "road" || terrain === "base" || terrain === "baseGate" || terrain === "reorganize";
}

export function shouldShowRoadSouthFace(
  terrain: Tile["terrain"],
  southTerrain?: Tile["terrain"],
): boolean {
  return isRaisedRoadTerrain(terrain) && (southTerrain === undefined || !isRaisedRoadTerrain(southTerrain));
}

export interface PixiBaseGroup {
  id: string;
  anchorX: number;
  anchorY: number;
  coords: readonly { x: number; y: number }[];
  ownerTeamId: string;
  thirdParty: boolean;
  coreAssetKey: "base" | "base_core_third_party";
  flagAssetKey?: Extract<TileAssetKey, `base_flag_team_${number}`>;
}

export function isTwoByTwoBase(coords: readonly { x: number; y: number }[]) {
  if (coords.length !== 4) return false;
  const xs = [...new Set(coords.map((coord) => coord.x))].sort((a, b) => a - b);
  const ys = [...new Set(coords.map((coord) => coord.y))].sort((a, b) => a - b);
  if (xs.length !== 2 || ys.length !== 2 || xs[1] !== xs[0] + 1 || ys[1] !== ys[0] + 1) return false;
  const keys = new Set(coords.map((coord) => `${coord.x},${coord.y}`));
  return xs.every((x) => ys.every((y) => keys.has(`${x},${y}`)));
}

export function baseFlagAssetKey(state: Pick<GameState, "teams">, ownerTeamId: string): PixiBaseGroup["flagAssetKey"] {
  const playableTeams = state.teams.filter((team) => !team.isNeutral && team.status !== "neutral");
  const index = playableTeams.findIndex((team) => team.id === ownerTeamId);
  return index >= 0 && index < 4 ? `base_flag_team_${index + 1}` as PixiBaseGroup["flagAssetKey"] : undefined;
}

export function buildPixiBaseGroups(state: Pick<GameState, "bases" | "teams">): PixiBaseGroup[] {
  return state.bases.flatMap((base) => {
    if (!isTwoByTwoBase(base.coords)) return [];
    const owner = state.teams.find((team) => team.id === base.ownerTeamId);
    const thirdParty = !owner || Boolean(owner.isNeutral) || owner.status === "neutral";
    const coords = [...base.coords].sort((a, b) => a.y - b.y || a.x - b.x);
    return [{
      id: base.id,
      anchorX: Math.min(...coords.map((coord) => coord.x)),
      anchorY: Math.min(...coords.map((coord) => coord.y)),
      coords,
      ownerTeamId: base.ownerTeamId,
      thirdParty,
      coreAssetKey: thirdParty ? "base_core_third_party" : "base",
      flagAssetKey: thirdParty ? undefined : baseFlagAssetKey(state, base.ownerTeamId),
    }];
  });
}

export function buildPixiBoardTerrain(state: GameState) {
  const baseGroups = buildPixiBaseGroups(state);
  const groupByCell = new Map<string, PixiBaseGroup>();
  for (const group of baseGroups) for (const coord of group.coords) groupByCell.set(`${coord.x},${coord.y}`, group);
  return state.map.tiles.map((tile) => {
    const bridge = getConstructionAt(state, tile.x, tile.y, "bridge");
    const obstacle = getConstructionAt(state, tile.x, tile.y, "obstacle");
    const ordered = bridge ? [...bridge.tiles].sort((a, b) => a.y - b.y || a.x - b.x) : [];
    const index = ordered.findIndex((cell) => cell.x === tile.x && cell.y === tile.y);
    const south = state.map.tiles.find((cell) => cell.x === tile.x && cell.y === tile.y + 1);
    const kind = pixiTerrainKind(tile.terrain);
    const baseGroup = groupByCell.get(`${tile.x},${tile.y}`);
    return {
      x: tile.x, y: tile.y, kind,
      baseGroupId: baseGroup?.id,
      baseGroup: baseGroup && tile.x === baseGroup.anchorX && tile.y === baseGroup.anchorY ? baseGroup : undefined,
      southFace: shouldShowRoadSouthFace(tile.terrain, south?.terrain),
      bridge: bridge ? {
        direction: ordered[0]?.y === ordered.at(-1)?.y ? "horizontal" as const : "vertical" as const,
        segment: index === 0 ? "start" as const : index === ordered.length - 1 ? "end" as const : "middle" as const,
      } : null,
      obstacle: Boolean(obstacle),
    };
  });
}
export interface BuiltPixiBoardCell {
  x: number;
  y: number;
  kind: keyof typeof TERRAIN_COLORS | null;
  baseGroupId?: string;
  baseGroup?: PixiBaseGroup;
  southFace: boolean;
  bridge: { direction: "horizontal" | "vertical"; segment: "start" | "middle" | "end" } | null;
  obstacle: boolean;
}
export interface PixiMoveHighlight {
  enabled: boolean;
  color: number;
  cssColor: string;
  topOpacity: number;
  southFaceOpacity: number;
  outlineEnabled: boolean;
  outlineColor: number;
  outlineOpacity: number;
}
export type PixiBoardCell = BuiltPixiBoardCell & { moveHighlight?: PixiMoveHighlight };
export type PixiBoardTerrain = PixiBoardCell[];

function parseHexColor(value: string | undefined): number {
  if (!value || !/^#[0-9a-f]{6}$/i.test(value)) return 0x777777;
  return Number.parseInt(value.slice(1), 16);
}

function toCssColor(color: number) {
  return `#${color.toString(16).padStart(6, "0")}`;
}

export function lightenColor(color: number, amount: number) {
  const factor = Math.min(1, Math.max(0, amount));
  return [16, 8, 0].reduce((result, shift) => {
    const channel = (color >> shift) & 0xff;
    return result | Math.round(channel + (255 - channel) * factor) << shift;
  }, 0);
}

export function getMoveHighlightStyle(
  unit: Pick<Unit, "type">,
  teamColor: string | undefined,
  settings: MoveHighlightSettings,
): PixiMoveHighlight {
  if (unit.type === "ninja") {
    const gray = Math.round(Math.min(160, Math.max(0, settings.ninja.gray)));
    const color = gray * 0x010101;
    return {
      enabled: settings.ninja.enabled, color, cssColor: toCssColor(color),
      topOpacity: settings.ninja.topOpacity, southFaceOpacity: settings.ninja.southFaceOpacity,
      outlineEnabled: settings.ninja.outlineEnabled,
      outlineColor: lightenColor(color, settings.ninja.outlineLighten),
      outlineOpacity: settings.ninja.outlineOpacity,
    };
  }
  const base = parseHexColor(teamColor), factor = 1 - Math.min(0.85, Math.max(0, settings.normal.darken));
  const color = [16, 8, 0].reduce((result, shift) => result | Math.round(((base >> shift) & 0xff) * factor) << shift, 0);
  return {
    enabled: settings.normal.enabled, color, cssColor: toCssColor(color),
    topOpacity: settings.normal.topOpacity, southFaceOpacity: settings.normal.southFaceOpacity,
    outlineEnabled: settings.normal.outlineEnabled,
    outlineColor: lightenColor(color, settings.normal.outlineLighten),
    outlineOpacity: settings.normal.outlineOpacity,
  };
}

export function applyMoveHighlights(
  cells: readonly BuiltPixiBoardCell[],
  highlightedKeys: ReadonlySet<string>,
  style?: PixiMoveHighlight,
): PixiBoardTerrain {
  if (!style?.enabled || highlightedKeys.size === 0) return cells as PixiBoardTerrain;
  return cells.map((cell) => highlightedKeys.has(`${cell.x},${cell.y}`) ? { ...cell, moveHighlight: style } : cell);
}
export interface CellBounds { x: number; y: number; width: number; height: number }
export function baseGroupBounds(group: PixiBaseGroup, bounds: ReadonlyMap<string, CellBounds>): CellBounds | undefined {
  const rects = group.coords.map((coord) => bounds.get(`${coord.x},${coord.y}`));
  if (rects.some((rect) => !rect)) return undefined;
  const resolved = rects as CellBounds[];
  const x = Math.min(...resolved.map((rect) => rect.x));
  const y = Math.min(...resolved.map((rect) => rect.y));
  const right = Math.max(...resolved.map((rect) => rect.x + rect.width));
  const bottom = Math.max(...resolved.map((rect) => rect.y + rect.height));
  return { x, y, width: right - x, height: bottom - y };
}
export function relativeCellBounds(cell: CellBounds, board: Pick<CellBounds, "x" | "y">): CellBounds {
  return { x: cell.x - board.x, y: cell.y - board.y, width: cell.width, height: cell.height };
}
