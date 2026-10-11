import { Application, Container, Graphics } from "pixi.js";
import { Tilemap } from "@pixi/tilemap";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { TERRAIN_COLORS } from "./pixiBoardTerrain";
import { ATLAS_KINDS, ATLAS_SIZE, createTerrainAtlas } from "./pixiTerrainAtlas";

export async function mountPixiBoard(host: HTMLDivElement, signal: AbortSignal) {
  const app = new Application();
  let initialized = false;
  let disposed = false;
  let atlas: ReturnType<typeof createTerrainAtlas> | undefined;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    if (initialized) app.destroy(true, { children: true });
    else app.stage.destroy({ children: true });
    atlas?.destroy(true);
  };
  try {
    await app.init({ width: 1, height: 1, backgroundAlpha: 0, preference: "webgl", autoStart: false, antialias: false, resolution: 1 });
    initialized = true;
    if (signal.aborted) { dispose(); return { dispose, update: () => {} }; }
    atlas = createTerrainAtlas();
    host.appendChild(app.canvas);
    app.canvas.setAttribute("aria-hidden", "true");
    const update = (cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, width: number, height: number) => {
      if (disposed || signal.aborted || !atlas || width <= 0 || height <= 0) return;
      for (const child of app.stage.removeChildren()) child.destroy({ children: true });
      app.renderer.resize(width, height);
      const ground = new Container();
      // Atlas cells retain their source size; each tile is transformed to its DOM rectangle.
      // Different row/column rounding is handled per cell, not by a guessed common stride.
      for (const cell of cells) {
        const rect = bounds.get(`${cell.x},${cell.y}`);
        if (!rect || !cell.kind) continue;
        const tile = new Tilemap([atlas.source]);
        tile.tile(0, 0, 0, { u: ATLAS_KINDS.indexOf(cell.kind) * ATLAS_SIZE, tileWidth: ATLAS_SIZE, tileHeight: ATLAS_SIZE });
        tile.position.set(rect.x, rect.y);
        tile.scale.set(rect.width / ATLAS_SIZE, rect.height / ATLAS_SIZE);
        ground.addChild(tile);
      }
      app.stage.addChild(ground);
      const depth = new Container(); depth.sortableChildren = true; app.stage.addChild(depth);
      for (const cell of cells) {
        const rect = bounds.get(`${cell.x},${cell.y}`);
        if (!rect || !cell.kind) continue;
        const layer = new Container(); layer.zIndex = cell.y; depth.addChild(layer);
        const g = new Graphics(); layer.addChild(g);
        const { x, y, width: w, height: h } = rect;
        const d = Math.min(8, h * 0.18);
        if (cell.kind === "road") {
          g.rect(x, y, w, h).fill(TERRAIN_COLORS.road).stroke({ color: 0x647066, width: 1 });
          if (cell.southFace) g.rect(x, y + h, w, d).fill(0x4e544f);
        }
        if (cell.bridge) {
          const horizontal = cell.bridge.direction === "horizontal";
          const bx = x + (horizontal ? 0 : w * 0.2), by = y + (horizontal ? h * 0.2 : 0);
          const bw = horizontal ? w : w * 0.6, bh = horizontal ? h * 0.6 : h;
          g.rect(bx, by + bh, bw, d * 0.6).fill(0x624530);
          g.rect(bx, by, bw, bh).fill(0xba8958).stroke({ color: 0x61432a, width: 1 });
          for (let p = 1; p < 6; p++) {
            if (horizontal) g.moveTo(bx + w * p / 6, by).lineTo(bx + w * p / 6, by + bh);
            else g.moveTo(bx, by + h * p / 6).lineTo(bx + bw, by + h * p / 6);
          }
          g.stroke({ color: 0x735133, width: 1 });
          if (cell.bridge.segment !== "middle") {
            const end = cell.bridge.segment === "end";
            g.rect(horizontal ? bx + (end ? bw - 3 : 0) : bx, horizontal ? by : by + (end ? bh - 3 : 0), horizontal ? 3 : bw, horizontal ? bh : 3).fill(0xffd281);
          }
        }
        if (cell.obstacle) {
          g.rect(x + w * 0.3, y - h * 0.2, w * 0.45, h * 0.7).fill(0xc27060).stroke({ color: 0x40242d, width: 1 });
        }
      }
      app.stage.eventMode = "none";
      app.render();
    };
    return { dispose, update };
  } catch (error) {
    initialized = Boolean(app.renderer); dispose(); throw error;
  }
}
