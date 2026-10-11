export const TILE_ASSET_KEYS = [
  "road", "base", "lake", "obstacle", "road_south_face",
  "bridge_horizontal_start", "bridge_horizontal_middle", "bridge_horizontal_end",
  "bridge_vertical_start", "bridge_vertical_middle", "bridge_vertical_end",
] as const;
export type TileAssetKey = typeof TILE_ASSET_KEYS[number];
export interface AssetAdjustment { scale: number; offsetX: number; offsetY: number }
export interface TileAsset extends AssetAdjustment {
  image: HTMLImageElement; name: string; width: number; height: number;
}
export interface AssetSnapshot {
  enabled: boolean;
  renderError?: string;
  assets: Partial<Record<TileAssetKey, TileAsset>>;
  statuses: Partial<Record<TileAssetKey, string>>;
}
export const EMPTY_ASSETS: AssetSnapshot = { enabled: false, assets: {}, statuses: {} };
export const DEFAULT_ADJUSTMENT: AssetAdjustment = { scale: 1, offsetX: 0, offsetY: 0 };
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
export async function decodeTilePng(file: File, signal?: AbortSignal): Promise<TileAsset> {
  const header = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  if (signal?.aborted) throw new Error("読み込みをキャンセルしました。");
  if (!/\.png$/i.test(file.name) || (file.type && file.type !== "image/png") ||
    !PNG_SIGNATURE.every((value, index) => header[index] === value)) {
    throw new Error("PNGファイルを選択してください（拡張子・形式を確認）。");
  }
  const url = URL.createObjectURL(file);
  let abort: (() => void) | undefined;
  try {
    const image = new Image();
    image.src = url;
    const cancelled = new Promise<never>((_, reject) => {
      abort = () => { image.src = ""; reject(new Error("読み込みをキャンセルしました。")); };
      signal?.addEventListener("abort", abort, { once: true });
    });
    await Promise.race([image.decode(), cancelled]);
    if (!image.naturalWidth || !image.naturalHeight) throw new Error("画像サイズを取得できません。");
    return { image, name: file.name, width: image.naturalWidth, height: image.naturalHeight, ...DEFAULT_ADJUSTMENT };
  } catch { throw new Error("PNGをdecodeできません。直前の正常な素材を維持します。"); }
  finally { if (abort) signal?.removeEventListener("abort", abort); URL.revokeObjectURL(url); }
}

// CPU images only. Each renderer independently owns its GPU textures.
export class TileAssetStore {
  private snapshot: AssetSnapshot = EMPTY_ASSETS;
  private listeners = new Set<() => void>();
  private versions: Partial<Record<TileAssetKey, number>> = {};
  private alive = true;
  private pending = new Map<TileAssetKey, AbortController>();
  constructor(private decode: (file: File, signal?: AbortSignal) => Promise<TileAsset> = decodeTilePng) {}
  getSnapshot = () => this.snapshot;
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  private publish(snapshot: AssetSnapshot) { this.snapshot = snapshot; this.listeners.forEach((listener) => listener()); }
  activate() { this.alive = true; } // React StrictMode effect replay
  setEnabled(enabled: boolean) { this.publish({ ...this.snapshot, enabled, renderError: undefined }); }
  fallbackShapes(renderError: string) { this.publish({ ...this.snapshot, enabled: false, renderError }); }
  async importFile(key: TileAssetKey, file: File) {
    const version = this.versions[key] = (this.versions[key] ?? 0) + 1;
    this.pending.get(key)?.abort();
    const controller = new AbortController(); this.pending.set(key, controller);
    this.publish({ ...this.snapshot, statuses: { ...this.snapshot.statuses, [key]: "読み込み中…" } });
    try {
      const asset = await this.decode(file, controller.signal);
      if (!this.alive || this.versions[key] !== version) return;
      this.publish({ ...this.snapshot, assets: { ...this.snapshot.assets, [key]: asset }, statuses: { ...this.snapshot.statuses, [key]: "読み込み済み" } });
    } catch (error) {
      if (!this.alive || this.versions[key] !== version) return;
      this.publish({ ...this.snapshot, statuses: { ...this.snapshot.statuses, [key]: String(error) } });
    } finally { if (this.pending.get(key) === controller) this.pending.delete(key); }
  }
  adjust(key: TileAssetKey, change: Partial<AssetAdjustment>) {
    const asset = this.snapshot.assets[key];
    if (!asset) return;
    const next = { ...asset, ...change };
    if (![next.scale, next.offsetX, next.offsetY].every(Number.isFinite)) return;
    next.scale = Math.min(4, Math.max(0.1, next.scale));
    this.publish({ ...this.snapshot, assets: { ...this.snapshot.assets, [key]: next } });
  }
  reset(key: TileAssetKey) {
    this.versions[key] = (this.versions[key] ?? 0) + 1;
    this.pending.get(key)?.abort(); this.pending.delete(key);
    const assets = { ...this.snapshot.assets }, statuses = { ...this.snapshot.statuses };
    delete assets[key]; delete statuses[key];
    this.publish({ ...this.snapshot, assets, statuses });
  }
  resetAll() { TILE_ASSET_KEYS.forEach((key) => { this.versions[key] = (this.versions[key] ?? 0) + 1; }); this.pending.forEach((controller) => controller.abort()); this.pending.clear(); this.publish(EMPTY_ASSETS); }
  dispose() { this.alive = false; this.resetAll(); }
}

export function bridgeAssetKey(direction: "horizontal" | "vertical", segment: "start" | "middle" | "end"): TileAssetKey {
  return `bridge_${direction}_${segment}`;
}
export function assetPlacement(asset: Pick<TileAsset, "width" | "height" | keyof AssetAdjustment>, rect: { x: number; y: number; width: number }) {
  const unit = rect.width / 48;
  const width = rect.width * asset.scale;
  return { x: rect.x + asset.offsetX * unit, y: rect.y + asset.offsetY * unit, width, height: width * asset.height / asset.width };
}
