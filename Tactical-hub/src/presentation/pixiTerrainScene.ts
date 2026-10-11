import { Container, Graphics, ImageSource, Sprite, Texture } from "pixi.js";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { TERRAIN_COLORS } from "./pixiBoardTerrain";
import { assetPlacement, bridgeAssetKey, type AssetSnapshot, type TileAssetKey } from "./tileAssets";

// Each Canvas owns its GPU textures; only decoded CPU images are shared.
export class TerrainTextureCache {
  private entries = new Map<TileAssetKey, { image: HTMLImageElement; texture: Texture }>();
  sync(snapshot: AssetSnapshot) {
    for (const [key, entry] of this.entries) {
      if (!snapshot.enabled || snapshot.assets[key]?.image !== entry.image) {
        entry.texture.destroy(true); this.entries.delete(key);
      }
    }
    if (!snapshot.enabled) return;
    for (const [key, asset] of Object.entries(snapshot.assets)) {
      if (!asset || this.entries.has(key as TileAssetKey)) continue;
      try {
        const texture = new Texture({ source: new ImageSource({ resource: asset.image, scaleMode: "nearest" }) });
        this.entries.set(key as TileAssetKey, { image: asset.image, texture });
      } catch { /* Individual allocation failure uses Graphics instead. */ }
    }
  }
  draw(parent: Container, key: TileAssetKey, rect: CellBounds, snapshot: AssetSnapshot) {
    const asset = snapshot.enabled ? snapshot.assets[key] : undefined;
    const texture = this.entries.get(key)?.texture;
    if (!asset || !texture) return false;
    const placement = assetPlacement(asset, rect);
    const sprite = new Sprite(texture);
    sprite.position.set(placement.x, placement.y);
    sprite.width = placement.width; sprite.height = placement.height;
    parent.addChild(sprite);
    return true;
  }
  dispose() { this.entries.forEach((entry) => entry.texture.destroy(true)); this.entries.clear(); }
}

export function createTerrainScene(cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, snapshot: AssetSnapshot, cache: TerrainTextureCache, raised = true) {
  const scene = new Container(); scene.eventMode = "none";
  const ground = new Container(); scene.addChild(ground);
  const depth = new Container(); depth.sortableChildren = true; scene.addChild(depth);
  for (const cell of cells) {
    const rect = bounds.get(`${cell.x},${cell.y}`);
    if (!rect || !cell.kind) continue;
    const { x, y, width: w, height: h } = rect;
    const d = raised ? Math.min(10, h * 0.2) : 0;
    const layer = new Container(); layer.zIndex = cell.y; depth.addChild(layer);
    const surface = cell.kind === "road" ? layer : ground;
    if (!cache.draw(surface, cell.kind, rect, snapshot)) surface.addChild(new Graphics().rect(x, y, w, h).fill(TERRAIN_COLORS[cell.kind]).stroke({ color: 0x647066, width: 1 }));
    if (raised && cell.southFace) {
      const face = { ...rect, y: y + h, height: d };
      if (!cache.draw(layer, "road_south_face", face, snapshot)) layer.addChild(new Graphics().rect(x, y + h, w, d).fill(0x4e544f));
    }
    if (cell.bridge && !cache.draw(layer, bridgeAssetKey(cell.bridge.direction, cell.bridge.segment), rect, snapshot)) {
      const g = new Graphics(); layer.addChild(g);
      const horizontal = cell.bridge.direction === "horizontal";
      const bx = x + (horizontal ? 0 : w * 0.2), by = y + (horizontal ? h * 0.2 : 0);
      const bw = horizontal ? w : w * 0.6, bh = horizontal ? h * 0.6 : h;
      g.rect(bx, by + bh, bw, raised ? d * 0.6 : 2).fill(0x624530);
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
    if (cell.obstacle && !cache.draw(layer, "obstacle", rect, snapshot)) layer.addChild(new Graphics().rect(x + w * 0.3, y - h * 0.2, w * 0.45, h * 0.7).fill(0xc27060).stroke({ color: 0x40242d, width: 1 }));
  }
  return scene;
}
