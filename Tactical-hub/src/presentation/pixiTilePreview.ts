// Synthetic DEV fixture only. No imports from the game model or map.
export type PreviewTerrain = "road" | "base" | "lake";
export interface PreviewCell { x: number; y: number }
export interface PreviewBridge extends PreviewCell {
  direction: "horizontal" | "vertical";
  segment: "start" | "middle" | "end";
}
export const PREVIEW_CELL_SIZE = 48;
export const PREVIEW_DEPTH = 10;
export const PREVIEW_TERRAIN: readonly (readonly PreviewTerrain[])[] = [
  ["road", "road", "road", "road", "road", "road", "road"],
  ["road", "lake", "lake", "lake", "road", "road", "road"],
  ["base", "base", "road", "road", "road", "lake", "road"],
  ["base", "base", "lake", "lake", "road", "lake", "road"],
  ["road", "road", "base", "base", "road", "lake", "road"],
  ["road", "road", "base", "base", "road", "road", "road"],
];
export const PREVIEW_BRIDGES: readonly PreviewBridge[] = [
  { x: 1, y: 1, direction: "horizontal", segment: "start" },
  { x: 2, y: 1, direction: "horizontal", segment: "middle" },
  { x: 3, y: 1, direction: "horizontal", segment: "end" },
  { x: 5, y: 2, direction: "vertical", segment: "start" },
  { x: 5, y: 3, direction: "vertical", segment: "middle" },
  { x: 5, y: 4, direction: "vertical", segment: "end" },
];
export const PREVIEW_OBSTACLES: readonly PreviewCell[] = [{ x: 1, y: 4 }, { x: 5, y: 3 }];

export function previewCellAt(
  x: number, y: number, origin: PreviewCell, columns: number, rows: number,
  cellSize = PREVIEW_CELL_SIZE,
): PreviewCell | null {
  if (!Number.isFinite(x) || !Number.isFinite(y) || cellSize <= 0) return null;
  const cell = { x: Math.floor((x - origin.x) / cellSize), y: Math.floor((y - origin.y) / cellSize) };
  return cell.x >= 0 && cell.y >= 0 && cell.x < columns && cell.y < rows ? cell : null;
}

export function isPreviewRaisedTerrain(terrain: PreviewTerrain | undefined): boolean {
  return terrain === "road" || terrain === "base";
}

// The only visible vertical face in this front-facing square projection is south.
// Road and base tops share one elevation; zIndex alone cannot remove internal faces.
export function hasPreviewRoadSouthFace(
  terrain: readonly (readonly PreviewTerrain[])[], x: number, y: number,
): boolean {
  return isPreviewRaisedTerrain(terrain[y]?.[x]) && !isPreviewRaisedTerrain(terrain[y + 1]?.[x]);
}

export function previewRowDepth(y: number): number { return y; }
