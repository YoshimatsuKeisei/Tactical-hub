import { Container, Graphics, ImageSource, Sprite, Texture } from "pixi.js";
import type { CellBounds, PixiBaseGroup, PixiBoardTerrain } from "./pixiBoardTerrain";
import { baseGroupBounds, TERRAIN_COLORS } from "./pixiBoardTerrain";
import {
  assetPlacement, bridgeAssetKey, bridgeComponentPlacement, hasCompleteModularBridgeAssets,
  LEGACY_BRIDGE_ASSET_KEYS, modularBridgeAssetKeys, MODULAR_BRIDGE_ASSET_KEYS, roadSouthShadowSettings, type AssetSnapshot,
  type RoadSouthShadowSettings, type TileAsset, type TileAssetKey,
} from "./tileAssets";
import type { LakeAnimationState } from "./pixiLakeAnimation";

// Each Canvas owns its GPU textures; only decoded CPU images are shared.
export class TerrainTextureCache {
  private entries = new Map<TileAssetKey, { image: HTMLImageElement; texture: Texture }>();
  private enabled = false;
  private images = new Map<TileAssetKey, HTMLImageElement>();

  sync(snapshot: AssetSnapshot, keys?: ReadonlySet<TileAssetKey>) {
    const nextImages = new Map<TileAssetKey, HTMLImageElement>();
    if (snapshot.enabled) {
      for (const [key, asset] of Object.entries(snapshot.assets)) {
        if (asset && (!keys || keys.has(key as TileAssetKey))) nextImages.set(key as TileAssetKey, asset.image);
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

const BASE_FOREGROUND_ASSET_KEYS = new Set<TileAssetKey>([
  "base_front_wall", "base_front_wall_third_party",
  "base_flag_team_1", "base_flag_team_2", "base_flag_team_3", "base_flag_team_4",
]);
const TERRAIN_ASSET_KEYS = new Set<TileAssetKey>([
  "road", "road_south_face", "road_south_shadow", "base", "base_core_third_party",
  "lake", "lake_frame_2", "lake_frame_3", "lake_frame_4", "obstacle",
  ...MODULAR_BRIDGE_ASSET_KEYS,
  ...LEGACY_BRIDGE_ASSET_KEYS,
]);

type FallbackDraw = (graphics: Graphics) => void;
type VisualPlacement = CellBounds & { rotation?: number; anchorX?: number; anchorY?: number };
interface VisualOptions {
  opacity?: number;
  placement?: (asset: TileAsset, rect: CellBounds) => VisualPlacement;
  tint?: number;
  fallback?: boolean;
}

class PersistentVisual {
  readonly graphics = new Graphics();
  readonly sprite = new Sprite(Texture.EMPTY);
  private fallbackSignature = "";

  constructor(parent: Container) {
    parent.addChild(this.graphics, this.sprite);
    this.hide();
  }

  update(
    key: TileAssetKey,
    rect: CellBounds,
    snapshot: AssetSnapshot,
    cache: TerrainTextureCache,
    fallbackSignature: string,
    drawFallback: FallbackDraw,
    options: VisualOptions = {},
  ): CellBounds {
    const asset = snapshot.enabled ? snapshot.assets[key] : undefined;
    const texture = cache.get(key, snapshot);
    const opacity = Math.min(1, Math.max(0, options.opacity ?? 1));
    this.sprite.alpha = opacity;
    this.sprite.tint = options.tint ?? 0xffffff;
    this.graphics.alpha = opacity;
    if (asset && texture) {
      const placement: VisualPlacement = options.placement ? options.placement(asset, rect) : assetPlacement(asset, rect);
      this.sprite.texture = texture;
      this.sprite.position.set(placement.x, placement.y);
      this.sprite.width = placement.width;
      this.sprite.height = placement.height;
      this.sprite.anchor.set(placement.anchorX ?? 0, placement.anchorY ?? 0);
      this.sprite.rotation = placement.rotation ?? 0;
      this.sprite.visible = true;
      this.graphics.visible = false;
      return placement;
    }

    this.sprite.visible = false;
    this.graphics.visible = options.fallback !== false;
    if (options.fallback === false) return rect;
    if (this.fallbackSignature !== fallbackSignature) {
      this.graphics.clear();
      drawFallback(this.graphics);
      this.fallbackSignature = fallbackSignature;
    }
    return rect;
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

class PersistentOverlay {
  readonly graphics = new Graphics();
  private signature = "";

  constructor(parent: Container) {
    parent.addChild(this.graphics);
    this.graphics.visible = false;
  }

  update(rect: CellBounds, color: number, opacity: number) {
    this.graphics.alpha = Math.min(1, Math.max(0, opacity));
    this.graphics.visible = this.graphics.alpha > 0;
    const signature = `${rectSignature(rect)}:${color}`;
    if (this.signature !== signature) {
      this.graphics.clear();
      this.graphics.rect(rect.x, rect.y, rect.width, rect.height).fill(color);
      this.signature = signature;
    }
  }

  hide() { this.graphics.visible = false; }
  destroy() { this.graphics.destroy(); }
}

export interface MovementOutlineEdges {
  top: boolean;
  right: boolean;
  bottom: boolean;
  left: boolean;
}

export function movementOutlineEdges(x: number, y: number, outlinedKeys: ReadonlySet<string>): MovementOutlineEdges {
  return {
    // Top and left own every shared edge. Right and bottom close only the
    // exterior, so adjacent cells never paint the same edge twice.
    top: true,
    left: true,
    right: !outlinedKeys.has(`${x + 1},${y}`),
    bottom: !outlinedKeys.has(`${x},${y + 1}`),
  };
}

class PersistentOutline {
  readonly graphics = new Graphics();
  private signature = "";

  constructor(parent: Container) {
    parent.addChild(this.graphics);
    this.graphics.visible = false;
  }

  update(rect: CellBounds, color: number, opacity: number, edges: MovementOutlineEdges) {
    this.graphics.alpha = Math.min(1, Math.max(0, opacity));
    this.graphics.visible = this.graphics.alpha > 0 && Object.values(edges).some(Boolean);
    const signature = `${rectSignature(rect)}:${color}:${Number(edges.top)}${Number(edges.right)}${Number(edges.bottom)}${Number(edges.left)}`;
    if (this.signature === signature) return;
    this.graphics.clear();
    const x2 = rect.x + rect.width, y2 = rect.y + rect.height;
    if (edges.top) this.graphics.moveTo(rect.x, rect.y).lineTo(x2, rect.y);
    if (edges.right) this.graphics.moveTo(x2, rect.y).lineTo(x2, y2);
    if (edges.bottom) this.graphics.moveTo(x2, y2).lineTo(rect.x, y2);
    if (edges.left) this.graphics.moveTo(rect.x, y2).lineTo(rect.x, rect.y);
    this.graphics.stroke({ color, width: 1 });
    this.signature = signature;
  }

  hide() { this.graphics.visible = false; }
  destroy() { this.graphics.destroy(); }
}

interface PersistentCell {
  outside: PersistentOverlay;
  ground: PersistentVisual;
  lakeBlend: Sprite;
  isLake: boolean;
  lakePlacement?: CellBounds;
  road: PersistentVisual;
  face: PersistentVisual;
  shadow: PersistentVisual;
  topHighlight: PersistentOverlay;
  faceHighlight: PersistentOverlay;
  topOutline: PersistentOutline;
  bridgeShadow: PersistentVisual;
  bridgePillar: PersistentVisual;
  bridgeTop: PersistentVisual;
  bridgeShadowLayer: Container;
  bridgePillarLayer: Container;
  bridgeTopLayer: Container;
  obstacle: PersistentVisual;
  layer: Container;
}

interface PersistentBaseCore {
  core: PersistentVisual;
  layer: Container;
}

interface PersistentBaseForeground {
  wall: PersistentVisual;
  flag: PersistentVisual;
  layer: Container;
}

const rectSignature = (rect: CellBounds) => `${rect.x},${rect.y},${rect.width},${rect.height}`;

export function baseAssetPlacement(asset: TileAsset, rect: CellBounds): CellBounds {
  // Base components use a transparent 2x2 canvas. Offsets remain expressed in
  // the existing 48px-per-cell unit rather than doubling with the group width.
  const unit = rect.width / 96;
  const width = rect.width * asset.scale;
  return {
    x: rect.x + asset.offsetX * unit,
    y: rect.y + asset.offsetY * unit,
    width,
    height: width * asset.height / asset.width,
  };
}

export class PersistentTerrainScene {
  readonly root = new Container();
  private readonly groundLayer = new Container();
  private readonly roadLayer = new Container();
  private readonly baseLayer = new Container();
  private readonly shadowLayer = new Container();
  private readonly depthLayer = new Container();
  private readonly nodes = new Map<string, PersistentCell>();
  private readonly baseNodes = new Map<string, PersistentBaseCore>();

  constructor(private readonly cache: TerrainTextureCache) {
    this.root.eventMode = "none";
    this.groundLayer.sortableChildren = true;
    this.depthLayer.sortableChildren = true;
    this.roadLayer.sortableChildren = true;
    this.root.addChild(this.groundLayer, this.roadLayer, this.baseLayer, this.shadowLayer, this.depthLayer);
  }

  private makeBase(id: string) {
    const layer = new Container();
    layer.label = `terrain-base-${id}`;
    this.baseLayer.addChild(layer);
    const node = { core: new PersistentVisual(layer), layer };
    this.baseNodes.set(id, node);
    return node;
  }

  private makeCell(key: string, row: number) {
    const layer = new Container();
    layer.label = `terrain-cell-${key}`;
    layer.zIndex = row;
    this.depthLayer.addChild(layer);
    const ground = new PersistentVisual(this.groundLayer);
    const lakeBlend = new Sprite(Texture.EMPTY);
    lakeBlend.visible = false;
    lakeBlend.zIndex = 1;
    this.groundLayer.addChild(lakeBlend);
    const bridgeShadowLayer = new Container();
    const bridgePillarLayer = new Container();
    const bridgeTopLayer = new Container();
    bridgeShadowLayer.label = `bridge-shadow-${key}`;
    bridgePillarLayer.label = `bridge-pillar-${key}`;
    bridgeTopLayer.label = `bridge-top-${key}`;
    const face = new PersistentVisual(layer);
    const topHighlight = new PersistentOverlay(layer);
    const faceHighlight = new PersistentOverlay(layer);
    const topOutline = new PersistentOutline(layer);
    const bridgeShadow = new PersistentVisual(bridgeShadowLayer);
    const bridgePillar = new PersistentVisual(bridgePillarLayer);
    const bridgeTop = new PersistentVisual(bridgeTopLayer);
    layer.addChild(bridgeShadowLayer, bridgePillarLayer, bridgeTopLayer);
    const obstacle = new PersistentVisual(layer);
    const node = {
      outside: new PersistentOverlay(this.groundLayer),
      ground,
      lakeBlend,
      isLake: false,
      lakePlacement: undefined,
      shadow: new PersistentVisual(this.shadowLayer),
      road: new PersistentVisual(this.roadLayer),
      face,
      topHighlight,
      faceHighlight,
      topOutline,
      bridgeShadow,
      bridgePillar,
      bridgeTop,
      bridgeShadowLayer,
      bridgePillarLayer,
      bridgeTopLayer,
      obstacle,
      layer,
    };
    this.nodes.set(key, node);
    return node;
  }

  update(cells: PixiBoardTerrain, bounds: ReadonlyMap<string, CellBounds>, snapshot: AssetSnapshot, raised = true) {
    this.cache.sync(snapshot, TERRAIN_ASSET_KEYS);
    this.updateBases(cells.flatMap((cell) => cell.baseGroup ? [cell.baseGroup] : []), bounds, snapshot);
    const seen = new Set<string>();
    const outlinedKeys = new Set(cells
      .filter((cell) => cell.kind && bounds.has(`${cell.x},${cell.y}`) && cell.moveHighlight?.outlineEnabled && cell.moveHighlight.outlineOpacity > 0)
      .map((cell) => `${cell.x},${cell.y}`));
    for (const cell of cells) {
      const key = `${cell.x},${cell.y}`;
      const rect = bounds.get(key);
      let node = this.nodes.get(key);
      if (!rect) {
        if (node) this.hideCell(node);
        continue;
      }
      seen.add(key);
      node ??= this.makeCell(key, cell.y);
      node.layer.visible = true;
      node.layer.zIndex = cell.y;
      node.road.sprite.zIndex = cell.y;
      node.road.graphics.zIndex = cell.y;
      const { x, y, width: w, height: h } = rect;
      const signature = rectSignature(rect);

      node.outside.hide();
      node.lakeBlend.visible = false;
      node.isLake = false;
      node.lakePlacement = undefined;
      if (!cell.kind) {
        // Outside remains logically absent; only its background participates.
        this.hideCell(node);
        node.outside.update(rect, 0xd8dde2, 1);
        continue;
      } else if (cell.kind === "road") {
        node.ground.hide();
        node.road.update("road", rect, snapshot, this.cache, `road:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS.road));
      } else if (cell.kind === "base" && cell.baseGroupId) {
        // Keep the logical base untouched while using the existing road visual
        // as a four-cell cobblestone underlay behind the one grouped core.
        node.road.hide();
        node.ground.update("road", rect, snapshot, this.cache, `base-road-underlay:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS.road));
      } else if (cell.kind === "lake") {
        node.road.hide();
        node.isLake = true;
        node.lakePlacement = node.ground.update("lake", rect, snapshot, this.cache, `lake:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS.lake));
      } else {
        node.road.hide();
        node.ground.update(cell.kind, rect, snapshot, this.cache, `${cell.kind}:${signature}`,
          (g) => g.rect(x, y, w, h).fill(TERRAIN_COLORS[cell.kind!]));
      }

      const depth = raised ? Math.min(10, h * 0.2) : 0;
      let displayedFace: CellBounds | undefined;
      if (raised && cell.southFace) {
        const faceRect = { ...rect, y: y + h, height: depth };
        displayedFace = node.face.update("road_south_face", faceRect, snapshot, this.cache, `face:${rectSignature(faceRect)}`,
          (g) => g.rect(x, y + h, w, depth).fill(0x4e544f));
        const shadow = roadSouthShadowSettings(snapshot);
        if (shadow.enabled) {
          const anchor = roadSouthShadowAnchor(displayedFace, h);
          const fallbackRect = adjustedShadowRect(anchor, shadow);
          node.shadow.update(
            "road_south_shadow",
            anchor,
            snapshot,
            this.cache,
            `shadow:${rectSignature(fallbackRect)}`,
            (g) => drawRoadSouthShadow(g, fallbackRect),
            { placement: (asset, rect) => roadSouthShadowAssetPlacement(asset, rect, shadow), opacity: shadow.opacity },
          );
        } else node.shadow.hide();
      } else {
        node.face.hide();
        node.shadow.hide();
      }

      if (cell.moveHighlight) {
        node.topHighlight.update(rect, cell.moveHighlight.color, cell.moveHighlight.topOpacity);
        if (displayedFace) node.faceHighlight.update(displayedFace, cell.moveHighlight.color, cell.moveHighlight.southFaceOpacity);
        else node.faceHighlight.hide();
        if (cell.moveHighlight.outlineEnabled) {
          node.topOutline.update(rect, cell.moveHighlight.outlineColor, cell.moveHighlight.outlineOpacity,
            movementOutlineEdges(cell.x, cell.y, outlinedKeys));
        } else node.topOutline.hide();
      } else {
        node.topHighlight.hide();
        node.faceHighlight.hide();
        node.topOutline.hide();
      }

      if (cell.bridge) {
        const modularKeys = modularBridgeAssetKeys(cell.bridge.ownerAssetSlot);
        const modularReady = snapshot.enabled && hasCompleteModularBridgeAssets(snapshot, cell.bridge.ownerAssetSlot) &&
          Object.values(modularKeys).every((assetKey) => this.cache.get(assetKey, snapshot));
        if (modularReady) {
          const placement = (asset: TileAsset, target: CellBounds) =>
            bridgeComponentPlacement(asset, target, cell.bridge!.direction);
          node.bridgeShadow.update(modularKeys.shadow, rect, snapshot, this.cache,
            `bridge-shadow:${cell.bridge.direction}:${signature}`, () => {}, { placement, fallback: false });
          node.bridgePillar.update(modularKeys.pillar, rect, snapshot, this.cache,
            `bridge-pillar:${cell.bridge.direction}:${signature}`, () => {}, { placement, fallback: false });
          node.bridgeTop.update(modularKeys.top, rect, snapshot, this.cache,
            `bridge-top:${cell.bridge.direction}:${signature}`, () => {}, { placement, fallback: false });
        } else {
          node.bridgeShadow.hide();
          node.bridgePillar.hide();
          const legacyBridgeKey = bridgeAssetKey(cell.bridge.direction, cell.bridge.segment);
          node.bridgeTop.update(legacyBridgeKey, rect, snapshot, this.cache,
            `bridge-top:${cell.bridge.direction}:${cell.bridge.segment}:${raised}:${signature}`,
            (g) => drawBridge(g, rect, cell.bridge!, raised, depth));
        }
      } else {
        node.bridgeShadow.hide();
        node.bridgePillar.hide();
        node.bridgeTop.hide();
      }

      if (cell.obstacle) {
        node.obstacle.update("obstacle", rect, snapshot, this.cache, `obstacle:${signature}`,
          (g) => g.rect(x + w * 0.3, y - h * 0.2, w * 0.45, h * 0.7).fill(0xc27060).stroke({ color: 0x40242d, width: 1 }));
      } else node.obstacle.hide();
    }

    for (const [key, node] of this.nodes) if (!seen.has(key)) this.hideCell(node);
  }

  applyLakeAnimation(state: LakeAnimationState, snapshot: AssetSnapshot) {
    const currentTexture = this.cache.get(state.current, snapshot);
    const nextTexture = this.cache.get(state.next, snapshot);
    if (!currentTexture || !nextTexture || !snapshot.assets.lake) return;
    for (const node of this.nodes.values()) {
      if (!node.isLake || !node.lakePlacement || !node.ground.sprite.visible) continue;
      node.ground.sprite.texture = currentTexture;
      node.ground.graphics.visible = false;
      node.lakeBlend.texture = nextTexture;
      node.lakeBlend.position.set(node.lakePlacement.x, node.lakePlacement.y);
      node.lakeBlend.width = node.lakePlacement.width;
      node.lakeBlend.height = node.lakePlacement.height;
      node.lakeBlend.alpha = state.alpha;
      node.lakeBlend.visible = state.alpha > 0;
    }
  }

  private updateBases(groups: readonly PixiBaseGroup[], bounds: ReadonlyMap<string, CellBounds>, snapshot: AssetSnapshot) {
    const seen = new Set<string>();
    for (const group of groups) {
      const rect = baseGroupBounds(group, bounds);
      let node = this.baseNodes.get(group.id);
      if (!rect) {
        if (node) { node.core.hide(); node.layer.visible = false; }
        continue;
      }
      seen.add(group.id);
      node ??= this.makeBase(group.id);
      node.layer.visible = true;
      node.core.update(group.coreAssetKey, rect, snapshot, this.cache,
        `base-core:${group.coreAssetKey}:${rectSignature(rect)}`,
        (g) => g.rect(rect.x, rect.y, rect.width, rect.height).fill(TERRAIN_COLORS.base),
        { placement: baseAssetPlacement });
    }
    for (const [id, node] of this.baseNodes) if (!seen.has(id)) {
      node.core.hide();
      node.layer.visible = false;
    }
  }

  getCellDisplayObjects(key: string) {
    const node = this.nodes.get(key);
    return node ? {
      outsideGraphics: node.outside.graphics,
      groundGraphics: node.ground.graphics, groundSprite: node.ground.sprite,
      lakeBlendSprite: node.lakeBlend,
      roadGraphics: node.road.graphics, roadSprite: node.road.sprite,
      faceGraphics: node.face.graphics, faceSprite: node.face.sprite,
      shadowGraphics: node.shadow.graphics, shadowSprite: node.shadow.sprite,
      topHighlightGraphics: node.topHighlight.graphics,
      faceHighlightGraphics: node.faceHighlight.graphics,
      topOutlineGraphics: node.topOutline.graphics,
      bridgeGraphics: node.bridgeTop.graphics, bridgeSprite: node.bridgeTop.sprite,
      bridgeShadowGraphics: node.bridgeShadow.graphics, bridgeShadowSprite: node.bridgeShadow.sprite,
      bridgePillarGraphics: node.bridgePillar.graphics, bridgePillarSprite: node.bridgePillar.sprite,
      bridgeTopGraphics: node.bridgeTop.graphics, bridgeTopSprite: node.bridgeTop.sprite,
      bridgeShadowLayer: node.bridgeShadowLayer,
      bridgePillarLayer: node.bridgePillarLayer,
      bridgeTopLayer: node.bridgeTopLayer,
      obstacleGraphics: node.obstacle.graphics, obstacleSprite: node.obstacle.sprite,
    } : undefined;
  }

  get cellCount() { return this.nodes.size; }
  get baseCount() { return [...this.baseNodes.values()].filter((node) => node.layer.visible).length; }

  getBaseDisplayObjects(id: string) {
    const node = this.baseNodes.get(id);
    return node ? { coreGraphics: node.core.graphics, coreSprite: node.core.sprite } : undefined;
  }

  private hideCell(node: PersistentCell) {
    node.outside.hide();
    node.ground.hide();
    node.lakeBlend.visible = false;
    node.isLake = false;
    node.lakePlacement = undefined;
    node.road.hide();
    node.face.hide();
    node.shadow.hide();
    node.topHighlight.hide();
    node.faceHighlight.hide();
    node.topOutline.hide();
    node.bridgeShadow.hide();
    node.bridgePillar.hide();
    node.bridgeTop.hide();
    node.obstacle.hide();
    node.layer.visible = false;
  }

  destroy() {
    for (const node of this.baseNodes.values()) {
      node.core.destroy();
      node.layer.destroy({ children: false });
    }
    this.baseNodes.clear();
    for (const node of this.nodes.values()) {
      node.outside.destroy();
      node.ground.destroy();
      node.lakeBlend.destroy({ texture: false, textureSource: false });
      node.road.destroy();
      node.face.destroy();
      node.shadow.destroy();
      node.topHighlight.destroy();
      node.faceHighlight.destroy();
      node.topOutline.destroy();
      node.bridgeShadow.destroy();
      node.bridgePillar.destroy();
      node.bridgeTop.destroy();
      node.bridgeShadowLayer.destroy({ children: false });
      node.bridgePillarLayer.destroy({ children: false });
      node.bridgeTopLayer.destroy({ children: false });
      node.obstacle.destroy();
      node.layer.destroy({ children: false });
    }
    this.nodes.clear();
    this.groundLayer.destroy({ children: false });
    this.roadLayer.destroy({ children: false });
    this.baseLayer.destroy({ children: false });
    this.shadowLayer.destroy({ children: false });
    this.depthLayer.destroy({ children: false });
    this.root.destroy({ children: false });
  }
}

export class PersistentBaseForegroundScene {
  readonly root = new Container();
  private readonly nodes = new Map<string, PersistentBaseForeground>();

  constructor(private readonly cache: TerrainTextureCache) {
    this.root.eventMode = "none";
    this.root.sortableChildren = true;
  }

  private makeBase(group: PixiBaseGroup) {
    const layer = new Container();
    layer.label = `base-foreground-${group.id}`;
    layer.zIndex = group.anchorY;
    this.root.addChild(layer);
    const node = {
      wall: new PersistentVisual(layer),
      flag: new PersistentVisual(layer),
      layer,
    };
    this.nodes.set(group.id, node);
    return node;
  }

  update(groups: readonly PixiBaseGroup[], bounds: ReadonlyMap<string, CellBounds>, snapshot: AssetSnapshot) {
    this.cache.sync(snapshot, BASE_FOREGROUND_ASSET_KEYS);
    const seen = new Set<string>();
    for (const group of groups) {
      const rect = baseGroupBounds(group, bounds);
      let node = this.nodes.get(group.id);
      if (!rect) {
        if (node) this.hideBase(node);
        continue;
      }
      seen.add(group.id);
      node ??= this.makeBase(group);
      node.layer.visible = true;
      node.layer.zIndex = group.anchorY;
      const thirdPartyWall = group.thirdParty &&
        this.cache.get("base_front_wall_third_party", snapshot) !== undefined;
      const wallAssetKey = thirdPartyWall ? "base_front_wall_third_party" : "base_front_wall";
      node.wall.update(wallAssetKey, rect, snapshot, this.cache,
        `base-wall:${wallAssetKey}:${rectSignature(rect)}`,
        () => {}, {
          placement: baseAssetPlacement,
          fallback: false,
        });
      if (group.flagAssetKey) {
        node.flag.update(group.flagAssetKey, rect, snapshot, this.cache,
          `base-flag:${group.flagAssetKey}:${rectSignature(rect)}`,
          () => {}, { placement: baseAssetPlacement, fallback: false });
      } else node.flag.hide();
    }
    for (const [id, node] of this.nodes) if (!seen.has(id)) this.hideBase(node);
  }

  private hideBase(node: PersistentBaseForeground) {
    node.wall.hide();
    node.flag.hide();
    node.layer.visible = false;
  }

  getBaseDisplayObjects(id: string) {
    const node = this.nodes.get(id);
    return node ? {
      wallGraphics: node.wall.graphics,
      wallSprite: node.wall.sprite,
      flagGraphics: node.flag.graphics,
      flagSprite: node.flag.sprite,
      layer: node.layer,
    } : undefined;
  }

  get baseCount() { return [...this.nodes.values()].filter((node) => node.layer.visible).length; }

  destroy() {
    for (const node of this.nodes.values()) {
      node.wall.destroy();
      node.flag.destroy();
      node.layer.destroy({ children: false });
    }
    this.nodes.clear();
    this.root.destroy({ children: false });
  }
}

export function roadSouthShadowAnchor(face: CellBounds, cellHeight: number): CellBounds {
  return {
    x: face.x,
    y: face.y + face.height,
    width: face.width,
    height: Math.min(8, Math.max(4, cellHeight * 0.125)),
  };
}

export function adjustedShadowRect(anchor: CellBounds, settings: RoadSouthShadowSettings): CellBounds {
  const unit = anchor.width / 48;
  return {
    x: anchor.x + settings.offsetX * unit,
    y: anchor.y + settings.offsetY * unit,
    width: anchor.width * settings.scaleX,
    height: anchor.height * settings.scaleY,
  };
}

export function roadSouthShadowAssetPlacement(
  asset: Pick<TileAsset, "width" | "height">,
  anchor: CellBounds,
  settings: RoadSouthShadowSettings,
): CellBounds {
  const unit = anchor.width / 48;
  return {
    x: anchor.x + settings.offsetX * unit,
    y: anchor.y + settings.offsetY * unit,
    width: anchor.width * settings.scaleX,
    height: anchor.width * asset.height / asset.width * settings.scaleY,
  };
}

function drawRoadSouthShadow(graphics: Graphics, rect: CellBounds) {
  const band = rect.height / 3;
  graphics.rect(rect.x, rect.y, rect.width, band).fill({ color: 0x24170f, alpha: 0.72 });
  graphics.rect(rect.x, rect.y + band, rect.width, band).fill({ color: 0x24170f, alpha: 0.42 });
  graphics.rect(rect.x, rect.y + band * 2, rect.width, rect.height - band * 2).fill({ color: 0x24170f, alpha: 0.16 });
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
