export const TILE_ASSET_KEYS = [
  "road", "road_south_face", "road_south_shadow", "base", "base_core_third_party", "base_front_wall",
  "base_front_wall_third_party",
  "base_flag_team_1", "base_flag_team_2", "base_flag_team_3", "base_flag_team_4",
  "lake", "lake_frame_2", "lake_frame_3", "lake_frame_4", "obstacle",
  "bridge_top", "bridge_pillar", "bridge_shadow",
  "bridge_horizontal_start", "bridge_horizontal_middle", "bridge_horizontal_end",
  "bridge_vertical_start", "bridge_vertical_middle", "bridge_vertical_end",
] as const;
export type TileAssetKey = typeof TILE_ASSET_KEYS[number];
export const TILE_ASSET_LABELS: Record<TileAssetKey, string> = {
  road: "Road top / 道路上面",
  road_south_face: "Road south face / 道路南側面",
  road_south_shadow: "Road south shadow / 道路南側の落ち影",
  base: "Base core - Team owned / 所属拠点本体",
  base_core_third_party: "Base core - Third party / 第三者拠点本体",
  base_front_wall: "Base front wall / 拠点前側城壁",
  base_front_wall_third_party: "Base front wall - Third party / 第三者拠点の前側城壁",
  base_flag_team_1: "Team 1 flag / チーム1旗",
  base_flag_team_2: "Team 2 flag / チーム2旗",
  base_flag_team_3: "Team 3 flag / チーム3旗",
  base_flag_team_4: "Team 4 flag / チーム4旗",
  lake: "Frame 1: Lake (existing) / 湖",
  lake_frame_2: "Frame 2 / Lake animation 2",
  lake_frame_3: "Frame 3 / Lake animation 3",
  lake_frame_4: "Frame 4 / Lake animation 4",
  obstacle: "Obstacle / 障害物",
  bridge_top: "Bridge top / 橋本体上面",
  bridge_pillar: "Bridge pillar / 橋支柱",
  bridge_shadow: "Bridge shadow / 橋陰影",
  bridge_horizontal_start: "Bridge horizontal start",
  bridge_horizontal_middle: "Bridge horizontal middle",
  bridge_horizontal_end: "Bridge horizontal end",
  bridge_vertical_start: "Bridge vertical start",
  bridge_vertical_middle: "Bridge vertical middle",
  bridge_vertical_end: "Bridge vertical end",
};
export interface AssetAdjustment { scale: number; offsetX: number; offsetY: number }
export interface AxisAssetAdjustment { scaleX: number; scaleY: number; offsetX: number; offsetY: number }
export interface RoadSouthShadowSettings {
  enabled: boolean;
  opacity: number;
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}
export interface MoveHighlightSettings {
  normal: {
    enabled: boolean;
    topOpacity: number;
    southFaceOpacity: number;
    darken: number;
    outlineEnabled: boolean;
    outlineOpacity: number;
    outlineLighten: number;
  };
  ninja: {
    enabled: boolean;
    topOpacity: number;
    southFaceOpacity: number;
    gray: number;
    outlineEnabled: boolean;
    outlineOpacity: number;
    outlineLighten: number;
  };
}
interface LegacyRoadSouthShadowSettings extends AssetAdjustment { enabled: boolean; opacity: number }
export interface TileAsset extends AssetAdjustment {
  scaleX?: number;
  scaleY?: number;
  image: HTMLImageElement; name: string; width: number; height: number;
}
export interface AssetSnapshot {
  enabled: boolean;
  lakeAnimationEnabled?: boolean;
  renderError?: string;
  assets: Partial<Record<TileAssetKey, TileAsset>>;
  statuses: Partial<Record<TileAssetKey, string>>;
  roadSouthShadow?: RoadSouthShadowSettings | LegacyRoadSouthShadowSettings;
  moveHighlight?: MoveHighlightSettings;
}
export const DEFAULT_ADJUSTMENT: AssetAdjustment = { scale: 1, offsetX: 0, offsetY: 0 };
export const DEFAULT_AXIS_ADJUSTMENT: AxisAssetAdjustment = { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };
export const BRIDGE_COMPONENT_ASSET_KEYS = ["bridge_top", "bridge_pillar", "bridge_shadow"] as const;
export type BridgeComponentAssetKey = typeof BRIDGE_COMPONENT_ASSET_KEYS[number];
export const DEFAULT_ROAD_SOUTH_SHADOW: RoadSouthShadowSettings = {
  enabled: true,
  opacity: 0.24,
  scaleX: 1,
  scaleY: 1,
  offsetX: 0,
  offsetY: 0,
};
export const DEFAULT_MOVE_HIGHLIGHT: MoveHighlightSettings = {
  normal: {
    enabled: true, topOpacity: 0.28, southFaceOpacity: 0.36, darken: 0.28,
    outlineEnabled: true, outlineOpacity: 0.6, outlineLighten: 0.35,
  },
  ninja: {
    enabled: true, topOpacity: 0.26, southFaceOpacity: 0.34, gray: 48,
    outlineEnabled: true, outlineOpacity: 0.6, outlineLighten: 0.35,
  },
};
export const EMPTY_ASSETS: AssetSnapshot = {
  enabled: false,
  assets: {},
  statuses: {},
  roadSouthShadow: DEFAULT_ROAD_SOUTH_SHADOW,
  moveHighlight: DEFAULT_MOVE_HIGHLIGHT,
};
export function roadSouthShadowSettings(snapshot: AssetSnapshot): RoadSouthShadowSettings {
  const settings = snapshot.roadSouthShadow;
  if (!settings) return DEFAULT_ROAD_SOUTH_SHADOW;
  if ("scale" in settings) {
    const { scale, ...rest } = settings;
    return { ...rest, scaleX: scale, scaleY: scale };
  }
  return settings;
}
export function moveHighlightSettings(snapshot: AssetSnapshot): MoveHighlightSettings {
  return snapshot.moveHighlight ?? DEFAULT_MOVE_HIGHLIGHT;
}
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
  setLakeAnimationEnabled(enabled: boolean) {
    this.publish({ ...this.snapshot, lakeAnimationEnabled: enabled });
  }
  fallbackShapes(renderError: string) { this.publish({ ...this.snapshot, enabled: false, renderError }); }
  setRoadSouthShadowEnabled(enabled: boolean) {
    this.publish({ ...this.snapshot, roadSouthShadow: { ...roadSouthShadowSettings(this.snapshot), enabled } });
  }
  setRoadSouthShadowOpacity(opacity: number) {
    if (!Number.isFinite(opacity)) return;
    this.publish({
      ...this.snapshot,
      roadSouthShadow: { ...roadSouthShadowSettings(this.snapshot), opacity: Math.min(1, Math.max(0, opacity)) },
    });
  }
  resetRoadSouthShadowSettings() {
    this.publish({ ...this.snapshot, roadSouthShadow: DEFAULT_ROAD_SOUTH_SHADOW });
  }
  adjustMoveHighlight<K extends keyof MoveHighlightSettings>(kind: K, change: Partial<MoveHighlightSettings[K]>) {
    const current = moveHighlightSettings(this.snapshot);
    const next = { ...current[kind], ...change };
    next.topOpacity = Math.min(1, Math.max(0, next.topOpacity));
    next.southFaceOpacity = Math.min(1, Math.max(0, next.southFaceOpacity));
    next.outlineOpacity = Math.min(1, Math.max(0, next.outlineOpacity));
    next.outlineLighten = Math.min(1, Math.max(0, next.outlineLighten));
    if (kind === "normal") {
      (next as MoveHighlightSettings["normal"]).darken = Math.min(0.85, Math.max(0, (next as MoveHighlightSettings["normal"]).darken));
    } else {
      (next as MoveHighlightSettings["ninja"]).gray = Math.round(Math.min(160, Math.max(0, (next as MoveHighlightSettings["ninja"]).gray)));
    }
    this.publish({ ...this.snapshot, moveHighlight: { ...current, [kind]: next } as MoveHighlightSettings });
  }
  resetMoveHighlightSettings(kind?: keyof MoveHighlightSettings) {
    if (!kind) {
      this.publish({ ...this.snapshot, moveHighlight: DEFAULT_MOVE_HIGHLIGHT });
      return;
    }
    this.publish({
      ...this.snapshot,
      moveHighlight: { ...moveHighlightSettings(this.snapshot), [kind]: DEFAULT_MOVE_HIGHLIGHT[kind] },
    });
  }
  adjustRoadSouthShadow(change: Partial<RoadSouthShadowSettings>) {
    const next = { ...roadSouthShadowSettings(this.snapshot), ...change };
    if (![next.opacity, next.scaleX, next.scaleY, next.offsetX, next.offsetY].every(Number.isFinite)) return;
    next.opacity = Math.min(1, Math.max(0, next.opacity));
    next.scaleX = Math.min(4, Math.max(0.1, next.scaleX));
    next.scaleY = Math.min(4, Math.max(0.1, next.scaleY));
    this.publish({ ...this.snapshot, roadSouthShadow: next });
  }
  async importFile(key: TileAssetKey, file: File) {
    const version = this.versions[key] = (this.versions[key] ?? 0) + 1;
    this.pending.get(key)?.abort();
    const controller = new AbortController(); this.pending.set(key, controller);
    this.publish({ ...this.snapshot, statuses: { ...this.snapshot.statuses, [key]: "読み込み中…" } });
    try {
      const decoded = await this.decode(file, controller.signal);
      if (!this.alive || this.versions[key] !== version) return;
      const asset = isBridgeComponentAssetKey(key) ? { ...decoded, ...DEFAULT_AXIS_ADJUSTMENT } : decoded;
      this.publish({ ...this.snapshot, assets: { ...this.snapshot.assets, [key]: asset }, statuses: { ...this.snapshot.statuses, [key]: "読み込み済み" } });
    } catch (error) {
      if (!this.alive || this.versions[key] !== version) return;
      this.publish({ ...this.snapshot, statuses: { ...this.snapshot.statuses, [key]: String(error) } });
    } finally { if (this.pending.get(key) === controller) this.pending.delete(key); }
  }
  adjust(key: TileAssetKey, change: Partial<AssetAdjustment>) {
    if (key === "road_south_shadow") {
      const { scale, offsetX, offsetY } = change;
      this.adjustRoadSouthShadow({
        ...(scale === undefined ? {} : { scaleX: scale, scaleY: scale }),
        ...(offsetX === undefined ? {} : { offsetX }),
        ...(offsetY === undefined ? {} : { offsetY }),
      });
      return;
    }
    const asset = this.snapshot.assets[key];
    if (!asset) return;
    const next = { ...asset, ...change };
    if (![next.scale, next.offsetX, next.offsetY].every(Number.isFinite)) return;
    next.scale = Math.min(4, Math.max(0.1, next.scale));
    this.publish({ ...this.snapshot, assets: { ...this.snapshot.assets, [key]: next } });
  }
  adjustBridgeAsset(key: BridgeComponentAssetKey, change: Partial<AxisAssetAdjustment>) {
    const asset = this.snapshot.assets[key];
    if (!asset) return;
    const next = { ...asset, ...bridgeAssetAdjustment(asset), ...change };
    if (![next.scaleX, next.scaleY, next.offsetX, next.offsetY].every(Number.isFinite)) return;
    next.scaleX = Math.min(4, Math.max(0.1, next.scaleX));
    next.scaleY = Math.min(4, Math.max(0.1, next.scaleY));
    this.publish({ ...this.snapshot, assets: { ...this.snapshot.assets, [key]: next } });
  }
  resetBridgeAssetAdjustment(key: BridgeComponentAssetKey) {
    this.adjustBridgeAsset(key, DEFAULT_AXIS_ADJUSTMENT);
  }
  reset(key: TileAssetKey) {
    this.versions[key] = (this.versions[key] ?? 0) + 1;
    this.pending.get(key)?.abort(); this.pending.delete(key);
    const assets = { ...this.snapshot.assets }, statuses = { ...this.snapshot.statuses };
    delete assets[key]; delete statuses[key];
    this.publish({
      ...this.snapshot,
      assets,
      statuses,
      lakeAnimationEnabled: LAKE_FRAME_KEYS.includes(key as typeof LAKE_FRAME_KEYS[number])
        ? false
        : this.snapshot.lakeAnimationEnabled,
    });
  }
  resetAll() { TILE_ASSET_KEYS.forEach((key) => { this.versions[key] = (this.versions[key] ?? 0) + 1; }); this.pending.forEach((controller) => controller.abort()); this.pending.clear(); this.publish(EMPTY_ASSETS); }
  dispose() { this.alive = false; this.resetAll(); }
}

export const LAKE_FRAME_KEYS = ["lake", "lake_frame_2", "lake_frame_3", "lake_frame_4"] as const;
export type LakeFrameAssetKey = typeof LAKE_FRAME_KEYS[number];

export function missingLakeAnimationAssets(snapshot: AssetSnapshot): LakeFrameAssetKey[] {
  return LAKE_FRAME_KEYS.filter((key) => !snapshot.assets[key]);
}

export function lakeAnimationCanRun(snapshot: AssetSnapshot) {
  return snapshot.enabled && snapshot.lakeAnimationEnabled === true && missingLakeAnimationAssets(snapshot).length === 0;
}

export function isBridgeComponentAssetKey(key: TileAssetKey): key is BridgeComponentAssetKey {
  return BRIDGE_COMPONENT_ASSET_KEYS.some((bridgeKey) => bridgeKey === key);
}

export function bridgeAssetAdjustment(asset?: Pick<TileAsset, "scale" | "scaleX" | "scaleY" | "offsetX" | "offsetY">): AxisAssetAdjustment {
  return {
    scaleX: asset?.scaleX ?? asset?.scale ?? 1,
    scaleY: asset?.scaleY ?? asset?.scale ?? 1,
    offsetX: asset?.offsetX ?? 0,
    offsetY: asset?.offsetY ?? 0,
  };
}

export function bridgeComponentPlacement(asset: TileAsset, rect: { x: number; y: number; width: number; height: number }) {
  const adjustment = bridgeAssetAdjustment(asset);
  return {
    x: rect.x + adjustment.offsetX * rect.width / 48,
    y: rect.y + adjustment.offsetY * rect.height / 48,
    width: rect.width * adjustment.scaleX,
    height: rect.height * adjustment.scaleY,
  };
}

export function bridgeAssetKey(direction: "horizontal" | "vertical", segment: "start" | "middle" | "end"): TileAssetKey {
  return `bridge_${direction}_${segment}`;
}
export function assetPlacement(asset: Pick<TileAsset, "width" | "height" | keyof AssetAdjustment>, rect: { x: number; y: number; width: number }) {
  const unit = rect.width / 48;
  const width = rect.width * asset.scale;
  return { x: rect.x + asset.offsetX * unit, y: rect.y + asset.offsetY * unit, width, height: width * asset.height / asset.width };
}
