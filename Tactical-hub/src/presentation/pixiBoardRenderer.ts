import { Application } from "pixi.js";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { EMPTY_ASSETS, type AssetSnapshot } from "./tileAssets";
import { PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
export async function mountPixiBoard(host: HTMLDivElement, signal: AbortSignal) {
  const app = new Application(), cache = new TerrainTextureCache();
  const terrain = new PersistentTerrainScene(cache);
  let initialized = false, disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    terrain.destroy();
    if (initialized) app.destroy(true, { children: false });
    else app.stage.destroy({ children: false });
    cache.dispose();
  };
  try {
    await app.init({ width: 1, height: 1, backgroundAlpha: 0, preference: "webgl", autoStart: false, antialias: false, resolution: 1 });
    initialized = true;
    if (signal.aborted) { dispose(); return { dispose, update: () => {} }; }
    host.appendChild(app.canvas);
    app.canvas.setAttribute("aria-hidden", "true");
    app.stage.addChild(terrain.root);
    let rendererWidth = 0, rendererHeight = 0;
    const update = (cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, width: number, height: number, assets: AssetSnapshot = EMPTY_ASSETS) => {
      if (disposed || signal.aborted || width <= 0 || height <= 0) return;
      if (rendererWidth !== width || rendererHeight !== height) {
        app.renderer.resize(width, height);
        rendererWidth = width;
        rendererHeight = height;
      }
      terrain.update(cells, bounds, assets);
      app.render();
    };
    return { dispose, update };
  } catch (error) { initialized = Boolean(app.renderer); dispose(); throw error; }
}
