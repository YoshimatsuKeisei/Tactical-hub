import { Application, Graphics, Rectangle, Text } from "pixi.js";
import { PREVIEW_CELL_SIZE as S, PREVIEW_TERRAIN, PREVIEW_BRIDGES, PREVIEW_OBSTACLES, hasPreviewRoadSouthFace, previewCellAt } from "./pixiTilePreview";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { EMPTY_ASSETS, type AssetSnapshot } from "./tileAssets";
import { PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
const WIDTH = PREVIEW_TERRAIN[0].length * S, HEIGHT = PREVIEW_TERRAIN.length * S;
const ORIGINS = [{ x: 16, y: 44 }, { x: WIDTH + 48, y: 44 }];
const cells: PixiBoardTerrain = PREVIEW_TERRAIN.flatMap((row, y) => row.map((kind, x) => ({
  x, y, kind, southFace: hasPreviewRoadSouthFace(PREVIEW_TERRAIN, x, y),
  bridge: PREVIEW_BRIDGES.find((b) => b.x === x && b.y === y) ?? null,
  obstacle: PREVIEW_OBSTACLES.some((o) => o.x === x && o.y === y),
})));
export async function mountPixiTilePreview(host: HTMLDivElement, signal: AbortSignal, onCell: (text: string) => void) {
  const app = new Application(), cache = new TerrainTextureCache();
  const scenes = [new PersistentTerrainScene(cache), new PersistentTerrainScene(cache)];
  let initialized = false, disposed = false;
  let assets = EMPTY_ASSETS;
  let selected: { panel: number; x: number; y: number } | undefined;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    scenes.forEach((scene) => scene.destroy());
    if (initialized) app.destroy(true, { children: true, texture: false, textureSource: false });
    else app.stage.destroy({ children: true, texture: false, textureSource: false });
    cache.dispose();
  };
  try {
    await app.init({ width: WIDTH * 2 + 64, height: HEIGHT + 72, background: "#182332", preference: "webgl", antialias: false, autoStart: false, resolution: 1 });
    initialized = true;
    if (signal.aborted) { dispose(); return { dispose, update: () => {} }; }
    host.appendChild(app.canvas);
    app.canvas.setAttribute("aria-label", "Flat and raised square tile comparison; click a logical cell");
    const bounds = new Map<string, CellBounds>(cells.map((cell) => [`${cell.x},${cell.y}`, { x: cell.x * S, y: cell.y * S, width: S, height: S }]));
    const highlights = ORIGINS.map(() => new Graphics());
    ORIGINS.forEach((origin, index) => {
      const label = new Text({ text: index === 0 ? "Flat" : "Raised / exposed south faces", style: { fill: 0xffffff, fontSize: 15 } });
      label.position.set(origin.x, 12);
      scenes[index].root.position.set(origin.x, origin.y);
      app.stage.addChild(label, scenes[index].root, highlights[index]);
    });
    const draw = () => {
      if (disposed || signal.aborted) return;
      ORIGINS.forEach((origin, index) => {
        scenes[index].update(cells, bounds, assets, index === 1);
        highlights[index].clear();
        if (selected?.panel === index) highlights[index].rect(origin.x + selected.x * S + 2, origin.y + selected.y * S + 2, S - 4, S - 4).stroke({ color: 0xfff15a, width: 3 });
      });
      app.render();
    };
    app.stage.eventMode = "static";
    app.stage.hitArea = new Rectangle(0, 0, WIDTH * 2 + 64, HEIGHT + 72);
    app.stage.on("pointerdown", (event) => {
      selected = undefined;
      let label = "Outside logical grid";
      ORIGINS.forEach((origin, panel) => {
        const cell = previewCellAt(event.global.x, event.global.y, origin, PREVIEW_TERRAIN[0].length, PREVIEW_TERRAIN.length);
        if (!cell) return;
        selected = { ...cell, panel };
        label = `${panel === 0 ? "Flat" : "Raised"}: cell (${cell.x}, ${cell.y}) — ${PREVIEW_TERRAIN[cell.y][cell.x]}`;
      });
      onCell(label); draw();
    });
    const update = (next: AssetSnapshot) => { assets = next; draw(); };
    draw(); return { dispose, update };
  } catch (error) { initialized = Boolean(app.renderer); dispose(); throw error; }
}
