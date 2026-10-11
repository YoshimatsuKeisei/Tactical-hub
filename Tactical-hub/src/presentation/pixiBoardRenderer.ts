import { Application } from "pixi.js";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { EMPTY_ASSETS, lakeAnimationCanRun, type AssetSnapshot } from "./tileAssets";
import { PersistentBaseForegroundScene, PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
import { applyPixiBoardExtent, pixiBoardExtent } from "./pixiBoardExtent";
import { LakeAnimationPlayer } from "./pixiLakeAnimation";
export async function mountPixiBoard(host: HTMLDivElement, foregroundHost: HTMLDivElement, signal: AbortSignal) {
  const app = new Application(), foregroundApp = new Application();
  const cache = new TerrainTextureCache(), foregroundCache = new TerrainTextureCache();
  const terrain = new PersistentTerrainScene(cache);
  const foreground = new PersistentBaseForegroundScene(foregroundCache);
  let initialized = false, foregroundInitialized = false, disposed = false;
  let lakeAnimation: LakeAnimationPlayer | undefined;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    signal.removeEventListener("abort", dispose);
    lakeAnimation?.dispose();
    terrain.destroy();
    foreground.destroy();
    if (initialized) app.destroy(true, { children: false });
    else app.stage.destroy({ children: false });
    if (foregroundInitialized) foregroundApp.destroy(true, { children: false });
    else foregroundApp.stage.destroy({ children: false });
    cache.dispose();
    foregroundCache.dispose();
  };
  try {
    const options = { width: 1, height: 1, backgroundAlpha: 0, preference: "webgl" as const, autoStart: false, antialias: false, resolution: 1 };
    const initialization = await Promise.allSettled([
      app.init({ ...options }).then(() => { initialized = true; }),
      foregroundApp.init({ ...options }).then(() => { foregroundInitialized = true; }),
    ]);
    const failure = initialization.find((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failure) throw failure.reason;
    if (signal.aborted) { dispose(); return { dispose, update: () => {} }; }
    signal.addEventListener("abort", dispose, { once: true });
    host.appendChild(app.canvas);
    foregroundHost.appendChild(foregroundApp.canvas);
    app.canvas.setAttribute("aria-hidden", "true");
    foregroundApp.canvas.setAttribute("aria-hidden", "true");
    app.stage.addChild(terrain.root);
    foregroundApp.stage.addChild(foreground.root);
    let latestAssets: AssetSnapshot = EMPTY_ASSETS;
    lakeAnimation = new LakeAnimationPlayer((state) => {
      if (disposed || signal.aborted) return;
      terrain.applyLakeAnimation(state, latestAssets);
      app.render();
    });
    let rendererWidth = 0, rendererHeight = 0;
    const update = (cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, width: number, height: number, assets: AssetSnapshot = EMPTY_ASSETS) => {
      if (disposed || signal.aborted || width <= 0 || height <= 0) return;
      const extent = pixiBoardExtent(cells, bounds, width, height, assets);
      applyPixiBoardExtent(host, foregroundHost, extent);
      app.stage.position.set(-extent.left, -extent.top);
      foregroundApp.stage.position.set(-extent.left, -extent.top);
      if (rendererWidth !== extent.width || rendererHeight !== extent.height) {
        app.renderer.resize(extent.width, extent.height);
        foregroundApp.renderer.resize(extent.width, extent.height);
        rendererWidth = extent.width;
        rendererHeight = extent.height;
      }
      latestAssets = assets;
      terrain.update(cells, bounds, assets);
      const lakeState = lakeAnimation?.configure(lakeAnimationCanRun(assets) && cells.some((cell) => cell.kind === "lake"));
      if (lakeState) terrain.applyLakeAnimation(lakeState, assets);
      foreground.update(cells.flatMap((cell) => cell.baseGroup ? [cell.baseGroup] : []), bounds, assets);
      app.render();
      foregroundApp.render();
    };
    return { dispose, update };
  } catch (error) {
    initialized = Boolean(app.renderer);
    foregroundInitialized = Boolean(foregroundApp.renderer);
    dispose();
    throw error;
  }
}
