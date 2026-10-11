import { Application, Graphics, Rectangle, Text } from "pixi.js";
import { PREVIEW_CELL_SIZE as S, PREVIEW_TERRAIN, PREVIEW_BRIDGES, PREVIEW_OBSTACLES, hasPreviewRoadSouthFace, previewCellAt } from "./pixiTilePreview";
import type { CellBounds, PixiBaseGroup, PixiBoardTerrain } from "./pixiBoardTerrain";
import { EMPTY_ASSETS, type AssetSnapshot } from "./tileAssets";
import { PersistentBaseForegroundScene, PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
const WIDTH = PREVIEW_TERRAIN[0].length * S, HEIGHT = PREVIEW_TERRAIN.length * S;
const ORIGINS = [{ x: 16, y: 44 }, { x: WIDTH + 48, y: 44 }];
const previewBaseGroups: PixiBaseGroup[] = [
  {
    id: "preview-owned", anchorX: 0, anchorY: 2, ownerTeamId: "preview-team-1", thirdParty: false,
    coreAssetKey: "base", flagAssetKey: "base_flag_team_1",
    coords: [{ x: 0, y: 2 }, { x: 1, y: 2 }, { x: 0, y: 3 }, { x: 1, y: 3 }],
  },
  {
    id: "preview-third-party", anchorX: 2, anchorY: 4, ownerTeamId: "neutral", thirdParty: true,
    coreAssetKey: "base_core_third_party",
    coords: [{ x: 2, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 5 }, { x: 3, y: 5 }],
  },
];
const cells: PixiBoardTerrain = PREVIEW_TERRAIN.flatMap((row, y) => row.map((kind, x) => {
  const baseGroup = previewBaseGroups.find((group) => group.coords.some((coord) => coord.x === x && coord.y === y));
  return {
    x, y, kind, southFace: hasPreviewRoadSouthFace(PREVIEW_TERRAIN, x, y),
    baseGroupId: baseGroup?.id,
    baseGroup: baseGroup && baseGroup.anchorX === x && baseGroup.anchorY === y ? baseGroup : undefined,
    bridge: PREVIEW_BRIDGES.find((b) => b.x === x && b.y === y) ?? null,
    obstacle: PREVIEW_OBSTACLES.some((o) => o.x === x && o.y === y),
  };
}));
export async function mountPixiTilePreview(host: HTMLDivElement, signal: AbortSignal, onCell: (text: string) => void) {
  const app = new Application(), cache = new TerrainTextureCache(), foregroundCache = new TerrainTextureCache();
  const scenes = [new PersistentTerrainScene(cache), new PersistentTerrainScene(cache)];
  const foregroundScenes = [new PersistentBaseForegroundScene(foregroundCache), new PersistentBaseForegroundScene(foregroundCache)];
  let initialized = false, disposed = false;
  let assets = EMPTY_ASSETS;
  let selected: { panel: number; x: number; y: number } | undefined;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    scenes.forEach((scene) => scene.destroy());
    foregroundScenes.forEach((scene) => scene.destroy());
    if (initialized) app.destroy(true, { children: true, texture: false, textureSource: false });
    else app.stage.destroy({ children: true, texture: false, textureSource: false });
    cache.dispose();
    foregroundCache.dispose();
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
      foregroundScenes[index].root.position.set(origin.x, origin.y);
      app.stage.addChild(label, scenes[index].root, foregroundScenes[index].root, highlights[index]);
    });
    const draw = () => {
      if (disposed || signal.aborted) return;
      ORIGINS.forEach((origin, index) => {
        scenes[index].update(cells, bounds, assets, index === 1);
        foregroundScenes[index].update(previewBaseGroups, bounds, assets);
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
