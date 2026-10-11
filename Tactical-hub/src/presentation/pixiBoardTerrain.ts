import type { GameState, Tile, Unit } from "../game/types";
import { getConstructionAt } from "../game/engine/construction";
import type { MoveHighlightSettings } from "./tileAssets";

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

export function buildPixiBoardTerrain(state: GameState) {
  return state.map.tiles.map((tile) => {
    const bridge = getConstructionAt(state, tile.x, tile.y, "bridge");
    const obstacle = getConstructionAt(state, tile.x, tile.y, "obstacle");
    const ordered = bridge ? [...bridge.tiles].sort((a, b) => a.y - b.y || a.x - b.x) : [];
    const index = ordered.findIndex((cell) => cell.x === tile.x && cell.y === tile.y);
    const south = state.map.tiles.find((cell) => cell.x === tile.x && cell.y === tile.y + 1);
    const kind = pixiTerrainKind(tile.terrain);
    return {
      x: tile.x, y: tile.y, kind,
      southFace: shouldShowRoadSouthFace(tile.terrain, south?.terrain),
      bridge: bridge ? {
        direction: ordered[0]?.y === ordered.at(-1)?.y ? "horizontal" as const : "vertical" as const,
        segment: index === 0 ? "start" as const : index === ordered.length - 1 ? "end" as const : "middle" as const,
      } : null,
      obstacle: Boolean(obstacle),
    };
  });
}
type BuiltPixiBoardCell = ReturnType<typeof buildPixiBoardTerrain>[number];
export interface PixiMoveHighlight {
  enabled: boolean;
  color: number;
  cssColor: string;
  topOpacity: number;
  southFaceOpacity: number;
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

export function getMoveHighlightStyle(
  unit: Pick<Unit, "type">,
  teamColor: string | undefined,
  settings: MoveHighlightSettings,
): PixiMoveHighlight {
  if (unit.type === "ninja") {
    const gray = Math.round(Math.min(160, Math.max(0, settings.ninja.gray)));
    const color = gray * 0x010101;
    return { enabled: settings.ninja.enabled, color, cssColor: toCssColor(color), topOpacity: settings.ninja.topOpacity, southFaceOpacity: settings.ninja.southFaceOpacity };
  }
  const base = parseHexColor(teamColor), factor = 1 - Math.min(0.85, Math.max(0, settings.normal.darken));
  const color = [16, 8, 0].reduce((result, shift) => result | Math.round(((base >> shift) & 0xff) * factor) << shift, 0);
  return { enabled: settings.normal.enabled, color, cssColor: toCssColor(color), topOpacity: settings.normal.topOpacity, southFaceOpacity: settings.normal.southFaceOpacity };
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
export function relativeCellBounds(cell: CellBounds, board: Pick<CellBounds, "x" | "y">): CellBounds {
  return { x: cell.x - board.x, y: cell.y - board.y, width: cell.width, height: cell.height };
}
