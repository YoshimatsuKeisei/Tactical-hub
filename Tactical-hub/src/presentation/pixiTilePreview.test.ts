import { describe, expect, it } from "vitest";
import {
  hasPreviewRoadSouthFace, previewCellAt, previewRowDepth,
  PREVIEW_BRIDGES, PREVIEW_TERRAIN, PREVIEW_OBSTACLES,
} from "./pixiTilePreview";

describe("independent square Pixi preview", () => {
  it("maps coordinates relative to each panel origin", () => {
    expect(previewCellAt(117, 169, { x: 20, y: 24 }, 7, 6)).toEqual({ x: 2, y: 3 });
    expect(previewCellAt(517, 169, { x: 420, y: 24 }, 7, 6)).toEqual({ x: 2, y: 3 });
  });
  it("uses logical cell boundaries even when a sidewall overhangs", () => {
    expect(previewCellAt(25, 50, { x: 0, y: 0 }, 7, 6)).toEqual({ x: 0, y: 1 });
    expect(previewCellAt(48, 48, { x: 0, y: 0 }, 7, 6)).toEqual({ x: 1, y: 1 });
  });
  it("rejects margins and bottom/right edges", () => {
    for (const [x, y] of [[-1, 0], [0, -1], [336, 0], [0, 288], [NaN, 0]]) {
      expect(previewCellAt(x, y, { x: 0, y: 0 }, 7, 6)).toBeNull();
    }
  });
  it("suppresses internal south walls, keeps exposed and bottom walls", () => {
    const terrain = [["road", "road"], ["road", "lake"]] as const;
    expect(hasPreviewRoadSouthFace(terrain, 0, 0)).toBe(false);
    expect(hasPreviewRoadSouthFace(terrain, 1, 0)).toBe(true);
    expect(hasPreviewRoadSouthFace(terrain, 0, 1)).toBe(true);
    expect(hasPreviewRoadSouthFace(terrain, 1, 1)).toBe(false);
  });
  it("orders lower rows in front", () => {
    expect(previewRowDepth(4)).toBeGreaterThan(previewRowDepth(3));
  });
  it("preserves lakes under both bridge directions and distinguishes segments", () => {
    for (const bridge of PREVIEW_BRIDGES) expect(PREVIEW_TERRAIN[bridge.y][bridge.x]).toBe("lake");
    for (const direction of ["horizontal", "vertical"]) {
      expect(PREVIEW_BRIDGES.filter((b) => b.direction === direction).map((b) => b.segment))
        .toEqual(["start", "middle", "end"]);
    }
  });
  it("places separate obstacles on road and bridge", () => {
    expect(PREVIEW_TERRAIN[PREVIEW_OBSTACLES[0].y][PREVIEW_OBSTACLES[0].x]).toBe("road");
    expect(PREVIEW_BRIDGES.some((b) => b.x === PREVIEW_OBSTACLES[1].x && b.y === PREVIEW_OBSTACLES[1].y)).toBe(true);
  });
});
