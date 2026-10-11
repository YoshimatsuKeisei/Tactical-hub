import { Application } from "pixi.js";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { EMPTY_ASSETS, type AssetSnapshot } from "./tileAssets";
import { createTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
export async function mountPixiBoard(host: HTMLDivElement, signal: AbortSignal) {
  const app = new Application(), cache = new TerrainTextureCache();
  let initialized = false, disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    if (initialized) app.destroy(true, { children: true });
    else app.stage.destroy({ children: true });
    cache.dispose();
  };
  try {
    await app.init({ width: 1, height: 1, backgroundAlpha: 0, preference: "webgl", autoStart: false, antialias: false, resolution: 1 });
    initialized = true;
    if (signal.aborted) { dispose(); return { dispose, update: () => {} }; }
    host.appendChild(app.canvas);
    app.canvas.setAttribute("aria-hidden", "true");
    const update = (cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, width: number, height: number, assets: AssetSnapshot = EMPTY_ASSETS) => {
      if (disposed || signal.aborted || width <= 0 || height <= 0) return;
      for (const child of app.stage.removeChildren()) child.destroy({ children: true });
      cache.sync(assets);
      app.renderer.resize(width, height);
      app.stage.addChild(createTerrainScene(cells, bounds, assets, cache));
      app.render();
    };
    return { dispose, update };
  } catch (error) { initialized = Boolean(app.renderer); dispose(); throw error; }
}
