import { baseGroupBounds, roadDecorationVisibility, type CellBounds, type PixiBoardTerrain } from "./pixiBoardTerrain";
import { baseAssetPlacement } from "./pixiTerrainScene";
import {
  assetPlacement, bridgeComponentBounds, hasCompleteModularBridgeAssets, modularBridgeAssetKeys, ROAD_DECORATION_ASSET_KEYS,
  type AssetSnapshot, type TileAssetKey,
} from "./tileAssets";

export function pixiBoardExtent(cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, width: number, height: number, snapshot: AssetSnapshot) {
  let left = 0, top = 0, right = width, bottom = height;
  const include = (placed: CellBounds) => {
    left = Math.min(left, placed.x);
    top = Math.min(top, placed.y);
    right = Math.max(right, placed.x + placed.width);
    bottom = Math.max(bottom, placed.y + placed.height);
  };
  const roadKeys = new Set(cells.filter((cell) => cell.kind === "road").map((cell) => `${cell.x},${cell.y}`));
  if (snapshot.enabled) for (const cell of cells) {
    const group = cell.baseGroup;
    if (group) {
      const rect = baseGroupBounds(group, bounds);
      if (rect) {
        const wall: TileAssetKey = group.thirdParty && snapshot.assets.base_front_wall_third_party
          ? "base_front_wall_third_party" : "base_front_wall";
        const keys = [group.coreAssetKey, wall, group.flagAssetKey];
        for (const key of keys) {
          const asset = key ? snapshot.assets[key] : undefined;
          if (asset) include(baseAssetPlacement(asset, rect));
        }
      }
    }
    if (cell.bridge) {
      const rect = bounds.get(`${cell.x},${cell.y}`);
      if (!rect) continue;
      if (!hasCompleteModularBridgeAssets(snapshot, cell.bridge.ownerAssetSlot)) continue;
      for (const key of Object.values(modularBridgeAssetKeys(cell.bridge.ownerAssetSlot))) {
        const asset = snapshot.assets[key];
        if (asset) include(bridgeComponentBounds(asset, rect, cell.bridge.direction));
      }
    }
    if (cell.kind === "road") {
      const rect = bounds.get(`${cell.x},${cell.y}`);
      if (!rect) continue;
      const visibility = roadDecorationVisibility(roadKeys, cell.x, cell.y);
      for (const key of ROAD_DECORATION_ASSET_KEYS) {
        const asset = visibility[key] ? snapshot.assets[key] : undefined;
        if (asset) include(assetPlacement(asset, rect));
      }
    }
  }
  // Outward rounding ensures fractional CSS pixels fit both render targets.
  left = Math.floor(left); top = Math.floor(top);
  right = Math.ceil(right); bottom = Math.ceil(bottom);
  return { left, top, width: right - left, height: bottom - top,
    paddingLeft: Math.max(0, -left), paddingTop: Math.max(0, -top),
    paddingRight: right - width, paddingBottom: bottom - height };
}

export function applyPixiBoardExtent(host: HTMLDivElement, foregroundHost: HTMLDivElement, extent: ReturnType<typeof pixiBoardExtent>) {
  for (const element of [host, foregroundHost]) {
    Object.assign(element.style, { left: `${extent.left}px`, top: `${extent.top}px`,
      right: "auto", bottom: "auto", width: `${extent.width}px`, height: `${extent.height}px` });
  }
  const viewport = host.parentElement?.parentElement;
  if (viewport?.classList.contains("board-pixi-viewport")) {
    for (const [side, value] of Object.entries({ left: extent.paddingLeft, top: extent.paddingTop,
      right: extent.paddingRight, bottom: extent.paddingBottom })) {
      viewport.style.setProperty(`--pixi-${side}`, `${value}px`);
    }
  }
}
