import { Container, Graphics, ImageSource, Sprite, Texture } from "pixi.js";
import type { CellBounds, PixiBoardTerrain } from "./pixiBoardTerrain";
import { TERRAIN_COLORS } from "./pixiBoardTerrain";
import { assetPlacement, bridgeAssetKey, type AssetSnapshot, type TileAssetKey } from "./tileAssets";

// Each Canvas owns its GPU textures; only decoded CPU images are shared.
export class TerrainTextureCache {
  private entries = new Map<TileAssetKey, { image: HTMLImageElement; texture: Texture }>();
  private enabled = false;
  private images = new Map<TileAssetKey, HTMLImageElement>();

  sync(snapshot: AssetSnapshot) {
    const nextImages = new Map<TileAssetKey, HTMLImageElement>();
    if (snapshot.enabled) {
      for (const [key, asset] of Object.entries(snapshot.assets)) {
        if (asset) nextImages.set(key as TileAssetKey, asset.image);
      }
    }
    const unchanged = this.enabled === snapshot.enabled && this.images.size === nextImages.size &&
      [...nextImages].every(([key, image]) => this.images.get(key) === image);
    if (unchanged) return false;

    for (const [key, entry] of this.entries) {
      if (!snapshot.enabled || nextImages.get(key) !== entry.image) {
        entry.texture.destroy(true);
        this.entries.delete(key);
      }
    }
    if (snapshot.enabled) {
      for (const [key, image] of nextImages) {
        if (this.entries.has(key)) continue;
        try {
          const texture = new Texture({ source: new ImageSource({ resource: image, scaleMode: "nearest" }) });
          this.entries.set(key, { image, texture });
        } catch { /* Individual allocation failure uses Graphics instead. */ }
      }
    }
    this.enabled = snapshot.enabled;
    this.images = nextImages;
    return true;
  }

  get(key: TileAssetKey, snapshot: AssetSnapshot) {
    return snapshot.enabled && snapshot.assets[key] ? this.entries.get(key)?.texture : undefined;
  }

  dispose() {
    this.entries.forEach((entry) => entry.texture.destroy(true));
    this.entries.clear();
    this.images.clear();
    this.enabled = false;
  }
}

type FallbackDraw = (graphics: Graphics) => void;

class PersistentVisual {
  readonly graphics = new Graphics();
  readonly sprite = new Sprite(Texture.EMPTY);
  private fallbackSignature = "";

  constructor(parent: Container) {
    parent.addChild(this.graphics, this.sprite);
    this.hide();
  }

  update(key: TileAssetKey, rect: CellBounds, snapshot: AssetSnapshot, cache: TerrainTextureCache, fallbackSignature: string, drawFallback: FallbackDraw) {
    const asset = snapshot.enabled ? snapshot.assets[key] : undefined;
    const texture = cache.get(key, snapshot);
    if (asset && texture) {
      const placement = assetPlacement(asset, rect);
      this.sprite.texture = texture;
      this.sprite.position.set(placement.x, placement.y);
      this.sprite.width = placement.width;
      this.sprite.height = placement.height;
      this.sprite.visible = true;
      this.graphics.visible = false;
      return;
    }

    this.sprite.visible = false;
    this.graphics.visible = true;
    if (this.fallbackSignature !== fallbackSignature) {
      this.graphics.clear();
      drawFallback(this.graphics);
      this.fallbackSignature = fallbackSignature;
    }
  }

  hide() {
    this.sprite.visible = false;
    this.graphics.visible = false;
  }

  destroy() {
    this.sprite.destroy({ texture: false, textureSource: false });
    this.graphics.destroy();
  }
}

interface PersistentCell {
  ground: PersistentVisual;
  road: PersistentVisual;
  face: PersistentVisual;
  bridge: PersistentVisual;
  obstacle: PersistentVisual;
  layer: Container;
}

const rectSignature = (rect: CellBounds) => `${rect.x},${rect.y},${rect.width},${rect.height}`;

export class PersistentTerrainScene {
  readonly root = new Container();
  private readonly groundLayer = new Container();
  private readonly depthLayer = new Container();
  private readonly nodes = new Map<string, PersistentCell>();

  constructor(private readonly cache: TerrainTextureCache) {
    this.root.eventMode = "none";
    this.depthLayer.sortableChildren = true;
    this.root.addChild(this.groundLayer, this.depthLayer);
  }

  private makeCell(key: string, row: number) {
    const layer = new Container();
    layer.label = `terrain-cell-${key}`;
    layer.zIndex = row;
    this.depthLayer.addChild(layer);
    const node = {
      ground: new PersistentVisual(this.groundLayer),
      road: new PersistentVisual(layer),
      face: new PersistentVisual(layer),
      bridge: new PersistentVisual(layer),
      obstacle: new PersistentVisual(layer),
      layer,
    };
    this.nodes.set(key, node);
    return node;
  }

  update(cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, snapshot: AssetSnapshot, raised = true) {
    this.cache.sync(snapshot);
    const seen = new Set<string>();
    for (const cell of cells) {
      const key = `${cell.x},${cell.y}`;
      const rect = bounds.get(key);
      let node = this.nodes.get(key);
      if (!rect || !cell.kind) {
        if (node) this.hideCell(node);
        continue;
      }
      seen.add(key);
      node ??= this.makeCell(key, cell.y);
      node.layer.visible = true;
      node.layer.zIndex = cell.y;
      const { x, y, width: w, height: h } = rect;
      const signature = rectSignature(rect);

      if (cell.kind === "road") {
        node.ground.hide();
        node.road.update("road", rect, snapshot, this.cache, `road:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS.road).stroke({ color: 0x647066, width: 1 }));
      } else {
        node.road.hide();
        node.ground.update(cell.kind, rect, snapshot, this.cache, `${cell.kind}:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS[cell.kind!]).stroke({ color: 0x647066, width: 1 }));
      }

      const depth = raised ? Math.min(10, h * 0.2) : 0;
      if (raised && cell.southFace) {
        const faceRect = { ...rect, y: y + h, height: depth };
        node.face.update("road_south_face", faceRect, snapshot, this.cache, `face:${rectSignature(faceRect)}`,
          (g) => g.rect(x, y + h, w, depth).fill(0x4e544f));
      } else node.face.hide();

      if (cell.bridge) {
        const bridgeKey = bridgeAssetKey(cell.bridge.direction, cell.bridge.segment);
        node.bridge.update(bridgeKey, rect, snapshot, this.cache,
          `bridge:${cell.bridge.direction}:${cell.bridge.segment}:${raised}:${signature}`,
          (g) => drawBridge(g, rect, cell.bridge!, raised, depth));
      } else node.bridge.hide();

      if (cell.obstacle) {
        node.obstacle.update("obstacle", rect, snapshot, this.cache, `obstacle:${signature}`,
          (g) => g.rect(x + w * 0.3, y - h * 0.2, w * 0.45, h * 0.7).fill(0xc27060).stroke({ color: 0x40242d, width: 1 }));
      } else node.obstacle.hide();
    }

    for (const [key, node] of this.nodes) if (!seen.has(key)) this.hideCell(node);
  }

  getCellDisplayObjects(key: string) {
    const node = this.nodes.get(key);
    return node ? {
      groundGraphics: node.ground.graphics, groundSprite: node.ground.sprite,
      roadGraphics: node.road.graphics, roadSprite: node.road.sprite,
      faceGraphics: node.face.graphics, faceSprite: node.face.sprite,
      bridgeGraphics: node.bridge.graphics, bridgeSprite: node.bridge.sprite,
      obstacleGraphics: node.obstacle.graphics, obstacleSprite: node.obstacle.sprite,
    } : undefined;
  }

  get cellCount() { return this.nodes.size; }

  private hideCell(node: PersistentCell) {
    node.ground.hide();
    node.road.hide();
    node.face.hide();
    node.bridge.hide();
    node.obstacle.hide();
    node.layer.visible = false;
  }

  destroy() {
    for (const node of this.nodes.values()) {
      node.ground.destroy();
      node.road.destroy();
      node.face.destroy();
      node.bridge.destroy();
      node.obstacle.destroy();
      node.layer.destroy({ children: false });
    }
    this.nodes.clear();
    this.groundLayer.destroy({ children: false });
    this.depthLayer.destroy({ children: false });
    this.root.destroy({ children: false });
  }
}

function drawBridge(graphics: Graphics, rect: CellBounds, bridge: NonNullable<PixiBoardTerrain[number]["bridge"]>, raised: boolean, depth: number) {
  const { x, y, width: w, height: h } = rect;
  const horizontal = bridge.direction === "horizontal";
  const bx = x + (horizontal ? 0 : w * 0.2), by = y + (horizontal ? h * 0.2 : 0);
  const bw = horizontal ? w : w * 0.6, bh = horizontal ? h * 0.6 : h;
  graphics.rect(bx, by + bh, bw, raised ? depth * 0.6 : 2).fill(0x624530);
  graphics.rect(bx, by, bw, bh).fill(0xba8958).stroke({ color: 0x61432a, width: 1 });
  for (let p = 1; p < 6; p++) {
    if (horizontal) graphics.moveTo(bx + w * p / 6, by).lineTo(bx + w * p / 6, by + bh);
    else graphics.moveTo(bx, by + h * p / 6).lineTo(bx + bw, by + h * p / 6);
  }
  graphics.stroke({ color: 0x735133, width: 1 });
  if (bridge.segment !== "middle") {
    const end = bridge.segment === "end";
    graphics.rect(horizontal ? bx + (end ? bw - 3 : 0) : bx, horizontal ? by : by + (end ? bh - 3 : 0), horizontal ? 3 : bw, horizontal ? bh : 3).fill(0xffd281);
  }
}
