import type { GameState, Tile } from "../game/types";
import { getConstructionAt } from "../game/engine/construction";

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
export type PixiBoardTerrain = ReturnType<typeof buildPixiBoardTerrain>;
export interface CellBounds { x: number; y: number; width: number; height: number }
export function relativeCellBounds(cell: CellBounds, board: Pick<CellBounds, "x" | "y">): CellBounds {
  return { x: cell.x - board.x, y: cell.y - board.y, width: cell.width, height: cell.height };
}
