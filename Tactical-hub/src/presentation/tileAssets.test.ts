import { afterEach, describe, expect, it, vi } from "vitest";
import {
  assetPlacement, bridgeAssetKey, decodeTilePng, DEFAULT_ADJUSTMENT, EMPTY_ASSETS,
  DEFAULT_ROAD_SOUTH_SHADOW, roadSouthShadowSettings, TILE_ASSET_KEYS, TILE_ASSET_LABELS,
  TileAssetStore, type TileAsset,
} from "./tileAssets";
const asset = (name = "road.png"): TileAsset => ({ image: {} as HTMLImageElement, name, width: 128, height: 64, ...DEFAULT_ADJUSTMENT });
const file = { name: "test.png", type: "image/png", slice: () => ({ arrayBuffer: async () => new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]).buffer }) } as unknown as File;
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (error: unknown) => void; const promise = new Promise<T>((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }
afterEach(() => vi.unstubAllGlobals());
describe("shared browser-only tile assets", () => {
  it("defaults to shapes and imports without enabling images", async () => {
    const store = new TileAssetStore(async () => asset());
    expect(store.getSnapshot()).toBe(EMPTY_ASSETS);
    await store.importFile("road", file);
    expect(store.getSnapshot().assets.road?.name).toBe("road.png");
    expect(store.getSnapshot().enabled).toBe(false);
  });
  it("keeps previous good image after decode failure", async () => {
    const decode = vi.fn().mockResolvedValueOnce(asset()).mockRejectedValueOnce(new Error("bad PNG"));
    const store = new TileAssetStore(decode);
    await store.importFile("road", file); const old = store.getSnapshot().assets.road;
    await store.importFile("road", file);
    expect(store.getSnapshot().assets.road).toBe(old);
    expect(store.getSnapshot().statuses.road).toContain("bad PNG");
  });
  it("reports rendering failure and returns to shapes without discarding decoded images", async () => {
    const store = new TileAssetStore(async () => asset());
    await store.importFile("road", file); store.setEnabled(true);
    store.fallbackShapes("GPU allocation failed");
    expect(store.getSnapshot().enabled).toBe(false);
    expect(store.getSnapshot().renderError).toBe("GPU allocation failed");
    expect(store.getSnapshot().assets.road).toBeDefined();
    store.setEnabled(true); expect(store.getSnapshot().renderError).toBeUndefined();
  });
  it("ignores stale completion and cancels the previous request", async () => {
    const first = deferred<TileAsset>(), second = deferred<TileAsset>();
    const signals: AbortSignal[] = [];
    const store = new TileAssetStore((_, signal) => { signals.push(signal!); return signals.length === 1 ? first.promise : second.promise; });
    const a = store.importFile("road", file), b = store.importFile("road", file);
    expect(signals[0].aborted).toBe(true);
    second.resolve(asset("new.png")); await b;
    first.resolve(asset("old.png")); await a;
    expect(store.getSnapshot().assets.road?.name).toBe("new.png");
  });
  it("individual reset cancels pending work while preserving other assets", async () => {
    const pending = deferred<TileAsset>(); let signal: AbortSignal | undefined;
    const store = new TileAssetStore((_, next) => { signal = next; return pending.promise; });
    const operation = store.importFile("road", file);
    store.reset("road"); expect(signal?.aborted).toBe(true);
    pending.resolve(asset()); await operation;
    expect(store.getSnapshot().assets.road).toBeUndefined();
  });
  it("supports adjustment reset, scale clamp, individual and full reset", async () => {
    const store = new TileAssetStore(async () => asset());
    await store.importFile("road", file); await store.importFile("lake", file);
    store.adjust("road", { scale: 8, offsetX: 3, offsetY: -4 });
    expect(store.getSnapshot().assets.road).toMatchObject({ scale: 4, offsetX: 3, offsetY: -4 });
    store.adjust("road", DEFAULT_ADJUSTMENT);
    expect(store.getSnapshot().assets.road).toMatchObject(DEFAULT_ADJUSTMENT);
    store.reset("road"); expect(store.getSnapshot().assets.lake).toBeDefined();
    store.setEnabled(true); store.resetAll(); expect(store.getSnapshot()).toBe(EMPTY_ASSETS);
  });
  it("keeps road top and south-face images as independent shared state", async () => {
    const store = new TileAssetStore(async (_, signal) => {
      expect(signal?.aborted).toBe(false);
      return asset("selected.png");
    });
    expect(TILE_ASSET_KEYS.slice(0, 3)).toEqual(["road", "road_south_face", "road_south_shadow"]);
    expect(TILE_ASSET_LABELS.road).toContain("道路上面");
    expect(TILE_ASSET_LABELS.road_south_face).toContain("道路南側面");
    await store.importFile("road", file);
    await store.importFile("road_south_face", file);
    store.adjust("road_south_face", { scale: 1.5, offsetX: 2, offsetY: 6 });
    store.reset("road");
    expect(store.getSnapshot().assets.road).toBeUndefined();
    expect(store.getSnapshot().assets.road_south_face).toMatchObject({ scale: 1.5, offsetX: 2, offsetY: 6 });
    store.resetAll();
    expect(store.getSnapshot()).toBe(EMPTY_ASSETS);
  });
  it("keeps shadow enable, opacity, and placement settings without a PNG", () => {
    const store = new TileAssetStore(async () => asset());
    expect(roadSouthShadowSettings(store.getSnapshot())).toEqual(DEFAULT_ROAD_SOUTH_SHADOW);
    store.setRoadSouthShadowEnabled(false);
    store.setRoadSouthShadowOpacity(0.35);
    store.adjust("road_south_shadow", { scale: 1.25, offsetX: -2, offsetY: 5 });
    expect(roadSouthShadowSettings(store.getSnapshot())).toEqual({
      enabled: false, opacity: 0.35, scale: 1.25, offsetX: -2, offsetY: 5,
    });
    store.setRoadSouthShadowOpacity(2);
    expect(roadSouthShadowSettings(store.getSnapshot()).opacity).toBe(1);
    store.resetRoadSouthShadowSettings();
    expect(roadSouthShadowSettings(store.getSnapshot())).toEqual(DEFAULT_ROAD_SOUTH_SHADOW);
  });
  it("imports and resets the shadow PNG independently from its fallback settings", async () => {
    const store = new TileAssetStore(async () => asset("shadow.png"));
    store.setRoadSouthShadowOpacity(0.3);
    await store.importFile("road_south_shadow", file);
    expect(store.getSnapshot().assets.road_south_shadow?.name).toBe("shadow.png");
    store.reset("road_south_shadow");
    expect(store.getSnapshot().assets.road_south_shadow).toBeUndefined();
    expect(roadSouthShadowSettings(store.getSnapshot()).opacity).toBe(0.3);
    store.resetAll();
    expect(store.getSnapshot()).toBe(EMPTY_ASSETS);
    expect(roadSouthShadowSettings(store.getSnapshot())).toEqual(DEFAULT_ROAD_SOUTH_SHADOW);
  });
  it("session disposal cancels pending work and prevents late state resurrection", async () => {
    const pending = deferred<TileAsset>(); let signal: AbortSignal | undefined;
    const store = new TileAssetStore((_, next) => { signal = next; return pending.promise; });
    const operation = store.importFile("road", file);
    store.dispose(); expect(signal?.aborted).toBe(true);
    pending.resolve(asset()); await operation;
    expect(store.getSnapshot()).toBe(EMPTY_ASSETS);
    store.activate();
  });
  it("notifies both subscribers and leaves state intact when one unsubscribes", async () => {
    const store = new TileAssetStore(async () => asset());
    const first = vi.fn(), second = vi.fn(), unsubscribe = store.subscribe(first);
    store.subscribe(second); await store.importFile("road", file); unsubscribe();
    const count = first.mock.calls.length; store.setEnabled(true);
    expect(first).toHaveBeenCalledTimes(count); expect(second.mock.calls.length).toBeGreaterThan(count);
    expect(store.getSnapshot().assets.road).toBeDefined();
  });
  it("maps all six bridge slots and keeps aspect ratio/offsets relative to cell size", () => {
    for (const direction of ["horizontal", "vertical"] as const) for (const segment of ["start", "middle", "end"] as const) expect(bridgeAssetKey(direction, segment)).toBe(`bridge_${direction}_${segment}`);
    expect(assetPlacement({ ...asset(), scale: 2, offsetX: 4, offsetY: -2 }, { x: 100, y: 200, width: 24 })).toEqual({ x: 102, y: 199, width: 48, height: 24 });
  });
});
describe("PNG decode and Object URL cleanup", () => {
  const install = (decode: () => Promise<void>) => {
    const revoke = vi.fn(), create = vi.fn(() => "blob:test");
    vi.stubGlobal("URL", { createObjectURL: create, revokeObjectURL: revoke });
    vi.stubGlobal("Image", class { src = ""; naturalWidth = 128; naturalHeight = 64; decode = decode; });
    return { revoke, create };
  };
  it("validates PNG signature, decodes dimensions, and revokes success URL", async () => {
    const { revoke } = install(async () => {});
    expect(await decodeTilePng(file)).toMatchObject({ width: 128, height: 64, name: "test.png" });
    expect(revoke).toHaveBeenCalledOnce(); expect(revoke).toHaveBeenCalledWith("blob:test");
  });
  it("rejects non-PNG before allocating an Object URL", async () => {
    const { create } = install(async () => {});
    await expect(decodeTilePng({ ...file, name: "test.jpg", type: "image/jpeg" } as File)).rejects.toThrow("PNG");
    await expect(decodeTilePng({ ...file, slice: () => ({ arrayBuffer: async () => new Uint8Array(8).buffer }) } as unknown as File)).rejects.toThrow("PNG");
    expect(create).not.toHaveBeenCalled();
  });
  it("revokes URL on decode failure", async () => {
    const { revoke } = install(async () => { throw new Error("invalid image"); });
    await expect(decodeTilePng(file)).rejects.toThrow("decode");
    expect(revoke).toHaveBeenCalledOnce();
  });
  it("aborting an unresolved decode revokes URL without waiting for decoder completion", async () => {
    const decode = deferred<void>(); const { revoke } = install(() => decode.promise);
    const controller = new AbortController(), operation = decodeTilePng(file, controller.signal);
    await Promise.resolve(); await Promise.resolve(); controller.abort();
    await expect(operation).rejects.toThrow(); expect(revoke).toHaveBeenCalledOnce();
  });
});
