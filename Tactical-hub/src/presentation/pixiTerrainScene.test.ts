import { describe, expect, it, vi } from "vitest";
import { DEFAULT_ADJUSTMENT, EMPTY_ASSETS, type AssetSnapshot, type TileAssetKey } from "./tileAssets";

vi.mock("pixi.js", () => {
  class Container {
    children: unknown[] = [];
    eventMode = "";
    sortableChildren = false;
    zIndex = 0;
    visible = true;
    label = "";
    position = { set: vi.fn() };
    addChild(...children: unknown[]) { this.children.push(...children); return children[0]; }
    destroy = vi.fn();
  }
  class Graphics {
    visible = true;
    alpha = 1;
    clear = vi.fn(() => this);
    destroy = vi.fn();
    rect = vi.fn(() => this);
    fill = vi.fn(() => this);
    stroke = vi.fn(() => this);
    moveTo = vi.fn(() => this);
    lineTo = vi.fn(() => this);
  }
  class ImageSource { constructor(public options: unknown) {} }
  class Texture {
    static EMPTY: Texture;
    destroy = vi.fn();
    constructor(public options: unknown) {}
  }
  Texture.EMPTY = new Texture({ empty: true });
  class Sprite {
    position = { set: vi.fn() };
    width = 0;
    height = 0;
    visible = true;
    alpha = 1;
    tint = 0xffffff;
    destroy = vi.fn();
    constructor(public texture: Texture) {}
  }
  class Application {
    static instances: Application[] = [];
    stage = new Container();
    canvas = { setAttribute: vi.fn() };
    renderer = { resize: vi.fn() };
    init = vi.fn(async () => {});
    render = vi.fn();
    destroy = vi.fn();
    constructor() { Application.instances.push(this); }
  }
  return { Container, Graphics, ImageSource, Texture, Sprite, Application };
});

import { baseAssetPlacement, movementOutlineEdges, PersistentBaseForegroundScene, PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
import { relativeCellBounds, type PixiBaseGroup, type PixiBoardTerrain } from "./pixiBoardTerrain";
import { Application, Sprite, Texture } from "pixi.js";
import { pixiBoardExtent } from "./pixiBoardExtent";
import { mountPixiBoard } from "./pixiBoardRenderer";
import { lakeAnimationStateAt } from "./pixiLakeAnimation";

const image = {} as HTMLImageElement;
const imported: AssetSnapshot = {
  enabled: true,
  statuses: {},
  assets: { road: { image, width: 128, height: 128, name: "road.png", ...DEFAULT_ADJUSTMENT } },
};
const cell = { x: 0, y: 0, kind: "road" as const, southFace: true, bridge: null, obstacle: false };
const bounds = new Map([["0,0", { x: 0, y: 0, width: 48, height: 48 }]]);
const ownedBase: PixiBaseGroup = {
  id: "base-a", anchorX: 0, anchorY: 0, ownerTeamId: "alpha", thirdParty: false,
  coreAssetKey: "base", flagAssetKey: "base_flag_team_1",
  coords: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }],
};
const baseCells: PixiBoardTerrain = ownedBase.coords.map((coord, index) => ({
  ...coord, kind: "base", southFace: coord.y === 1, bridge: null, obstacle: false,
  baseGroupId: ownedBase.id,
  baseGroup: index === 0 ? ownedBase : undefined,
}));
const baseBounds = new Map([
  ["0,0", { x: 10, y: 20, width: 40, height: 36 }],
  ["1,0", { x: 50, y: 20, width: 40, height: 36 }],
  ["0,1", { x: 10, y: 56, width: 40, height: 36 }],
  ["1,1", { x: 50, y: 56, width: 40, height: 36 }],
]);

describe("persistent Pixi terrain scene", () => {
  it("draws normal fallback terrain without permanent white grid strokes", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([
      { ...cell, southFace: false },
      { ...cell, x: 1, kind: "lake" as const, southFace: false },
    ], new Map([
      ["0,0", { x: 0, y: 0, width: 48, height: 48 }],
      ["1,0", { x: 48, y: 0, width: 48, height: 48 }],
    ]), EMPTY_ASSETS);
    expect(scene.getCellDisplayObjects("0,0")!.roadGraphics.stroke).not.toHaveBeenCalled();
    expect(scene.getCellDisplayObjects("1,0")!.groundGraphics.stroke).not.toHaveBeenCalled();
  });

  it("updates legal-move top and south-face overlays in place and clears them with selection", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    const highlighted = {
      ...cell,
      moveHighlight: {
        enabled: true, color: 0x772929, cssColor: "#772929", topOpacity: 0.25, southFaceOpacity: 0.4,
        outlineEnabled: true, outlineColor: 0xb56f6f, outlineOpacity: 0.6,
      },
    };
    scene.update([highlighted], bounds, EMPTY_ASSETS, true);
    const objects = scene.getCellDisplayObjects("0,0")!;
    const top = objects.topHighlightGraphics, face = objects.faceHighlightGraphics, outline = objects.topOutlineGraphics;
    expect(top.visible).toBe(true);
    expect(top.alpha).toBe(0.25);
    expect(top.rect).toHaveBeenLastCalledWith(0, 0, 48, 48);
    expect(face.visible).toBe(true);
    expect(face.alpha).toBe(0.4);
    expect(face.rect).toHaveBeenLastCalledWith(0, 48, 48, 9.600000000000001);
    expect(outline.visible).toBe(true);
    expect(outline.alpha).toBe(0.6);
    expect(outline.stroke).toHaveBeenLastCalledWith({ color: 0xb56f6f, width: 1 });
    expect(outline.moveTo).toHaveBeenCalledTimes(4);
    expect(face.stroke).not.toHaveBeenCalled();
    expect(objects.shadowGraphics.stroke).not.toHaveBeenCalled();
    const topClears = vi.mocked(top.clear).mock.calls.length;
    const outlineClears = vi.mocked(outline.clear).mock.calls.length;

    scene.update([{ ...highlighted, moveHighlight: { ...highlighted.moveHighlight, topOpacity: 0.55, outlineOpacity: 0.25 } }], bounds, EMPTY_ASSETS, true);
    expect(scene.getCellDisplayObjects("0,0")!.topHighlightGraphics).toBe(top);
    expect(scene.getCellDisplayObjects("0,0")!.topOutlineGraphics).toBe(outline);
    expect(top.alpha).toBe(0.55);
    expect(top.clear).toHaveBeenCalledTimes(topClears);
    expect(outline.alpha).toBe(0.25);
    expect(outline.clear).toHaveBeenCalledTimes(outlineClears);

    scene.update([{ ...highlighted, moveHighlight: { ...highlighted.moveHighlight, outlineEnabled: false } }], bounds, EMPTY_ASSETS, true);
    expect(top.visible).toBe(true);
    expect(outline.visible).toBe(false);

    scene.update([{ ...cell, southFace: false, moveHighlight: highlighted.moveHighlight }], bounds, EMPTY_ASSETS, true);
    expect(top.visible).toBe(true);
    expect(face.visible).toBe(false);
    scene.update([cell], bounds, EMPTY_ASSETS, true);
    expect(top.visible).toBe(false);
    expect(face.visible).toBe(false);
    expect(outline.visible).toBe(false);
  });

  it("assigns every shared candidate edge to one cell without losing internal boundaries", () => {
    const keys = new Set(["0,0", "1,0", "0,1", "1,1"]);
    expect(movementOutlineEdges(0, 0, keys)).toEqual({ top: true, right: false, bottom: false, left: true });
    expect(movementOutlineEdges(1, 0, keys)).toEqual({ top: true, right: true, bottom: false, left: true });
    expect(movementOutlineEdges(0, 1, keys)).toEqual({ top: true, right: false, bottom: true, left: true });
    expect(movementOutlineEdges(1, 1, keys)).toEqual({ top: true, right: true, bottom: true, left: true });

    // The internal vertical edge is owned by the right cell's left edge; the
    // internal horizontal edge is owned by the lower cell's top edge.
    expect(movementOutlineEdges(0, 0, keys).right).toBe(false);
    expect(movementOutlineEdges(1, 0, keys).left).toBe(true);
    expect(movementOutlineEdges(0, 0, keys).bottom).toBe(false);
    expect(movementOutlineEdges(0, 1, keys).top).toBe(true);
  });
  it("keeps the same scene, containers, and Sprite across unchanged updates", () => {
    const cache = new TerrainTextureCache();
    const scene = new PersistentTerrainScene(cache);
    scene.update([cell], bounds, imported);
    const first = scene.getCellDisplayObjects("0,0")!;
    const texture = first.roadSprite.texture;
    scene.update([cell], bounds, imported);
    const second = scene.getCellDisplayObjects("0,0")!;
    expect(second.roadSprite).toBe(first.roadSprite);
    expect(second.roadGraphics).toBe(first.roadGraphics);
    expect(second.roadSprite.texture).toBe(texture);
    expect(scene.cellCount).toBe(1);
  });

  it("updates bounds in place without recreating Sprite or Texture", () => {
    const cache = new TerrainTextureCache();
    const scene = new PersistentTerrainScene(cache);
    scene.update([cell], bounds, imported);
    const sprite = scene.getCellDisplayObjects("0,0")!.roadSprite;
    const texture = sprite.texture;
    const resized = new Map([["0,0", { x: 12, y: 20, width: 72, height: 72 }]]);
    scene.update([cell], resized, imported);
    expect(scene.getCellDisplayObjects("0,0")!.roadSprite).toBe(sprite);
    expect(sprite.texture).toBe(texture);
    expect(sprite.position.set).toHaveBeenLastCalledWith(12, 20);
    expect(sprite.width).toBe(72);
    expect(sprite.height).toBe(72);
  });

  it("switches imported asset off/on using the same Sprite and Graphics fallback", () => {
    const cache = new TerrainTextureCache();
    const scene = new PersistentTerrainScene(cache);
    scene.update([cell], bounds, imported);
    const objects = scene.getCellDisplayObjects("0,0")!;
    expect(objects.roadSprite.visible).toBe(true);
    expect(objects.roadGraphics.visible).toBe(false);
    scene.update([cell], bounds, EMPTY_ASSETS);
    expect(scene.getCellDisplayObjects("0,0")!.roadSprite).toBe(objects.roadSprite);
    expect(objects.roadSprite.visible).toBe(false);
    expect(objects.roadGraphics.visible).toBe(true);
    scene.update([cell], bounds, imported);
    expect(scene.getCellDisplayObjects("0,0")!.roadSprite).toBe(objects.roadSprite);
    expect(objects.roadSprite.visible).toBe(true);
  });

  it("uses the separate south-face asset only for exposed faces in raised mode", () => {
    const withFace: AssetSnapshot = {
      ...imported,
      assets: {
        ...imported.assets,
        road_south_face: { image, width: 128, height: 32, name: "road-face.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, withFace, true);
    const objects = scene.getCellDisplayObjects("0,0")!;
    expect(objects.faceSprite.visible).toBe(true);
    expect(objects.faceGraphics.visible).toBe(false);

    scene.update([{ ...cell, southFace: false }], bounds, withFace, true);
    expect(objects.faceSprite.visible).toBe(false);
    expect(objects.shadowSprite.visible).toBe(false);
    expect(objects.shadowGraphics.visible).toBe(false);
    scene.update([cell], bounds, withFace, false);
    expect(objects.faceSprite.visible).toBe(false);
    expect(objects.shadowSprite.visible).toBe(false);
    expect(objects.shadowGraphics.visible).toBe(false);
  });

  it("draws a soft Graphics shadow below the displayed face without moving the face", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, EMPTY_ASSETS, true);
    const objects = scene.getCellDisplayObjects("0,0")!;
    expect(objects.faceGraphics.visible).toBe(true);
    expect(objects.shadowGraphics.visible).toBe(true);
    expect(objects.shadowSprite.visible).toBe(false);
    expect(objects.shadowGraphics.alpha).toBe(0.24);
    expect((scene.root.children[3] as { children: unknown[] }).children).toContain(objects.shadowGraphics);
    const shadowRects = vi.mocked(objects.shadowGraphics.rect).mock.calls;
    expect(shadowRects).toHaveLength(3);
    expect(shadowRects[0][0]).toBe(0);
    expect(shadowRects[0][1]).toBeCloseTo(57.6);
    expect(shadowRects[0][2]).toBe(48);
    expect(shadowRects[0][3]).toBeCloseTo(2);
    expect(shadowRects[1][1]).toBeCloseTo(59.6);
    expect(shadowRects[2][1]).toBeCloseTo(61.6);
    expect(objects.faceGraphics.rect).toHaveBeenCalledWith(0, 48, 48, 9.600000000000001);

    scene.update([cell], bounds, {
      ...EMPTY_ASSETS,
      roadSouthShadow: { enabled: true, opacity: 0.24, scaleX: 2, scaleY: 0.5, offsetX: 0, offsetY: 0 },
    }, true);
    const resizedRects = vi.mocked(objects.shadowGraphics.rect).mock.calls.slice(-3);
    expect(resizedRects[0][2]).toBe(96);
    expect(resizedRects[0][3]).toBeCloseTo(1);
    expect(resizedRects[2][1]).toBeCloseTo(59.6);
  });

  it("reuses the shadow Sprite and Texture across opacity, position, and enable changes", () => {
    const shadowImage = {} as HTMLImageElement;
    const snapshot: AssetSnapshot = {
      enabled: true,
      statuses: {},
      assets: {
        road_south_face: { image, width: 48, height: 10, name: "face.png", ...DEFAULT_ADJUSTMENT },
        road_south_shadow: { image: shadowImage, width: 48, height: 6, name: "shadow.png", ...DEFAULT_ADJUSTMENT },
      },
      roadSouthShadow: { enabled: true, opacity: 0.3, scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
    };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, snapshot, true);
    const objects = scene.getCellDisplayObjects("0,0")!;
    const shadowSprite = objects.shadowSprite;
    const shadowTexture = shadowSprite.texture;
    const facePosition = vi.mocked(objects.faceSprite.position.set).mock.calls.at(-1);
    const faceSize = { width: objects.faceSprite.width, height: objects.faceSprite.height };
    expect(shadowSprite.visible).toBe(true);
    expect(shadowSprite.alpha).toBe(0.3);
    expect(shadowSprite.position.set).toHaveBeenLastCalledWith(0, 58);

    const adjusted: AssetSnapshot = {
      ...snapshot,
      roadSouthShadow: { enabled: true, opacity: 0.4, scaleX: 1.5, scaleY: 2, offsetX: 2, offsetY: 3 },
    };
    scene.update([cell], bounds, adjusted, true);
    expect(scene.getCellDisplayObjects("0,0")!.shadowSprite).toBe(shadowSprite);
    expect(shadowSprite.texture).toBe(shadowTexture);
    expect(shadowSprite.alpha).toBe(0.4);
    expect(shadowSprite.position.set).toHaveBeenLastCalledWith(2, 61);
    expect(shadowSprite.width).toBe(72);
    expect(shadowSprite.height).toBe(12);
    expect(vi.mocked(objects.faceSprite.position.set).mock.calls.at(-1)).toEqual(facePosition);
    expect({ width: objects.faceSprite.width, height: objects.faceSprite.height }).toEqual(faceSize);

    scene.update([cell], bounds, { ...adjusted, roadSouthShadow: { ...adjusted.roadSouthShadow!, enabled: false } }, true);
    expect(objects.shadowSprite.visible).toBe(false);
    expect(objects.shadowGraphics.visible).toBe(false);
    expect(objects.faceSprite.visible).toBe(true);
  });

  it("redraws fallback only when its geometry changes", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, EMPTY_ASSETS);
    const graphics = scene.getCellDisplayObjects("0,0")!.roadGraphics;
    const initialClears = vi.mocked(graphics.clear).mock.calls.length;
    scene.update([cell], bounds, EMPTY_ASSETS);
    expect(graphics.clear).toHaveBeenCalledTimes(initialClears);
    scene.update([cell], new Map([["0,0", { x: 0, y: 0, width: 72, height: 72 }]]), EMPTY_ASSETS);
    expect(graphics.clear).toHaveBeenCalledTimes(initialClears + 1);
  });

  it("updates bridge and obstacle visibility without replacing their display objects", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, EMPTY_ASSETS);
    const objects = scene.getCellDisplayObjects("0,0")!;
    const constructed = [{ ...cell, bridge: { direction: "horizontal" as const, segment: "middle" as const }, obstacle: true }];
    scene.update(constructed, bounds, EMPTY_ASSETS);
    expect(scene.getCellDisplayObjects("0,0")!.bridgeGraphics).toBe(objects.bridgeGraphics);
    expect(objects.bridgeGraphics.visible).toBe(true);
    expect(objects.obstacleGraphics.visible).toBe(true);
    scene.update([cell], bounds, EMPTY_ASSETS);
    expect(objects.bridgeGraphics.visible).toBe(false);
    expect(objects.obstacleGraphics.visible).toBe(false);
  });

  it("retains lake below imported bridge and obstacle Sprites", () => {
    const allImported: AssetSnapshot = {
      enabled: true, statuses: {},
      assets: {
        lake: { ...imported.assets.road! },
        bridge_vertical_end: { ...imported.assets.road! },
        obstacle: { ...imported.assets.road! },
      },
    };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([{ ...cell, kind: "lake", southFace: false, bridge: { direction: "vertical", segment: "end" }, obstacle: true }], bounds, allImported);
    const objects = scene.getCellDisplayObjects("0,0")!;
    expect(objects.groundSprite.visible).toBe(true);
    expect(objects.bridgeSprite.visible).toBe(true);
    expect(objects.obstacleSprite.visible).toBe(true);
  });

  it("layers independently adjusted bridge shadow, pillar and top without recreating display objects", () => {
    const topImage = {} as HTMLImageElement, pillarImage = {} as HTMLImageElement;
    const shadowImage = {} as HTMLImageElement, legacyImage = {} as HTMLImageElement;
    const bridgeAssets: AssetSnapshot = {
      enabled: true, statuses: {}, assets: {
        bridge_top: { image: topImage, width: 96, height: 24, name: "top.png", ...DEFAULT_ADJUSTMENT,
          scaleX: 1.5, scaleY: 0.75, offsetX: 2, offsetY: -1 },
        bridge_pillar: { image: pillarImage, width: 32, height: 128, name: "pillar.png", ...DEFAULT_ADJUSTMENT,
          scaleX: 0.5, scaleY: 2, offsetX: 12, offsetY: 8 },
        bridge_shadow: { image: shadowImage, width: 64, height: 32, name: "shadow.png", ...DEFAULT_ADJUSTMENT,
          scaleX: 2, scaleY: 1.25, offsetX: -4, offsetY: 6 },
        bridge_horizontal_end: { image: legacyImage, width: 48, height: 48, name: "legacy.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const bridgeCell = { ...cell, kind: "lake" as const, southFace: false,
      bridge: { direction: "horizontal" as const, segment: "end" as const } };
    const original = { ...bridgeCell, bridge: { ...bridgeCell.bridge } };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([bridgeCell], bounds, bridgeAssets);
    const objects = scene.getCellDisplayObjects("0,0")!;

    expect(objects.bridgeTopSprite.visible).toBe(true);
    expect(objects.bridgeTopGraphics.visible).toBe(false);
    expect(objects.bridgeTopSprite.position.set).toHaveBeenLastCalledWith(2, -1);
    expect(objects.bridgeTopSprite).toMatchObject({ width: 72, height: 36 });
    expect(objects.bridgePillarSprite.position.set).toHaveBeenLastCalledWith(12, 8);
    expect(objects.bridgePillarSprite).toMatchObject({ visible: true, width: 24, height: 96 });
    expect(objects.bridgeShadowSprite.position.set).toHaveBeenLastCalledWith(-4, 6);
    expect(objects.bridgeShadowSprite).toMatchObject({ visible: true, width: 96, height: 60 });
    const depthLayer = scene.root.children[4] as { children: { children: unknown[] }[] };
    const cellLayer = depthLayer.children[0];
    expect(cellLayer.children.indexOf(objects.bridgeShadowLayer)).toBeLessThan(cellLayer.children.indexOf(objects.bridgePillarLayer));
    expect(cellLayer.children.indexOf(objects.bridgePillarLayer)).toBeLessThan(cellLayer.children.indexOf(objects.bridgeTopLayer));
    expect(bridgeCell).toEqual(original);

    const topSprite = objects.bridgeTopSprite, topTexture = topSprite.texture;
    const pillarSprite = objects.bridgePillarSprite, shadowSprite = objects.bridgeShadowSprite;
    const adjusted = { ...bridgeAssets, assets: { ...bridgeAssets.assets,
      bridge_top: { ...bridgeAssets.assets.bridge_top!, scaleX: 2, scaleY: 1.25, offsetX: -3, offsetY: 5 },
    } };
    scene.update([bridgeCell], bounds, adjusted);
    expect(objects.bridgeTopSprite).toBe(topSprite);
    expect(objects.bridgeTopSprite.texture).toBe(topTexture);
    expect(objects.bridgeTopSprite.position.set).toHaveBeenLastCalledWith(-3, 5);
    expect(objects.bridgeTopSprite).toMatchObject({ width: 96, height: 60 });
    expect(objects.bridgePillarSprite).toBe(pillarSprite);
    expect(objects.bridgeShadowSprite).toBe(shadowSprite);

    const legacyOnly = { ...adjusted, assets: { ...adjusted.assets, bridge_top: undefined } };
    scene.update([bridgeCell], bounds, legacyOnly);
    expect(objects.bridgeTopSprite).toBe(topSprite);
    expect(objects.bridgeTopSprite.texture).not.toBe(topTexture);
    expect(objects.bridgeTopSprite).toMatchObject({ visible: true, width: 48, height: 48 });
    scene.update([bridgeCell], bounds, EMPTY_ASSETS);
    expect(objects.bridgeTopSprite.visible).toBe(false);
    expect(objects.bridgeTopGraphics.visible).toBe(true);
    expect(objects.bridgePillarSprite.visible).toBe(false);
    expect(objects.bridgePillarGraphics.visible).toBe(false);
    expect(objects.bridgeShadowSprite.visible).toBe(false);
    expect(objects.bridgeShadowGraphics.visible).toBe(false);
  });

  it("shows pillar and shadow only for horizontal bridges and reuses their Sprites", () => {
    const component = { image, width: 48, height: 48, name: "component.png", ...DEFAULT_ADJUSTMENT,
      scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 12 };
    const snapshot: AssetSnapshot = { enabled: true, statuses: {}, assets: {
      bridge_top: component, bridge_pillar: { ...component }, bridge_shadow: { ...component },
    } };
    const horizontal = { ...cell, kind: "lake" as const, southFace: false,
      bridge: { direction: "horizontal" as const, segment: "middle" as const } };
    const vertical = { ...horizontal,
      bridge: { direction: "vertical" as const, segment: "middle" as const } };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([horizontal], bounds, snapshot);
    const objects = scene.getCellDisplayObjects("0,0")!;
    const pillarSprite = objects.bridgePillarSprite, shadowSprite = objects.bridgeShadowSprite;
    expect(pillarSprite.visible).toBe(true);
    expect(shadowSprite.visible).toBe(true);

    scene.update([vertical], bounds, snapshot);
    expect(objects.bridgeTopSprite.visible).toBe(true);
    expect(pillarSprite.visible).toBe(false);
    expect(shadowSprite.visible).toBe(false);
    expect(objects.bridgePillarGraphics.visible).toBe(false);
    expect(objects.bridgeShadowGraphics.visible).toBe(false);

    scene.update([horizontal], bounds, snapshot);
    expect(objects.bridgePillarSprite).toBe(pillarSprite);
    expect(objects.bridgeShadowSprite).toBe(shadowSprite);
    expect(pillarSprite.visible).toBe(true);
    expect(shadowSprite.visible).toBe(true);
  });

  it("preserves all six direction and segment legacy bridge top fallbacks", () => {
    const variants = [
      ["horizontal", "start", "bridge_horizontal_start"],
      ["horizontal", "middle", "bridge_horizontal_middle"],
      ["horizontal", "end", "bridge_horizontal_end"],
      ["vertical", "start", "bridge_vertical_start"],
      ["vertical", "middle", "bridge_vertical_middle"],
      ["vertical", "end", "bridge_vertical_end"],
    ] as const;
    for (const [direction, segment, key] of variants) {
      const legacy = { image, width: 48, height: 48, name: `${key}.png`, ...DEFAULT_ADJUSTMENT };
      const snapshot: AssetSnapshot = { enabled: true, statuses: {}, assets: { [key]: legacy } };
      const bridgeCell = { ...cell, kind: "lake" as const, southFace: false, bridge: { direction, segment } };
      const scene = new PersistentTerrainScene(new TerrainTextureCache());
      scene.update([bridgeCell], bounds, snapshot);
      const objects = scene.getCellDisplayObjects("0,0")!;
      expect(objects.bridgeTopSprite.visible, key).toBe(true);
      expect(objects.bridgeTopGraphics.visible, key).toBe(false);
    }
  });

  it("applies the same bridge component snapshot to preview and board scenes", () => {
    const component = { image, width: 48, height: 48, name: "component.png", ...DEFAULT_ADJUSTMENT,
      scaleX: 1.25, scaleY: 1.75, offsetX: 3, offsetY: 9 };
    const snapshot: AssetSnapshot = { enabled: true, statuses: {}, assets: {
      bridge_top: component, bridge_pillar: { ...component }, bridge_shadow: { ...component },
    } };
    const bridgeCell = { ...cell, kind: "lake" as const, southFace: false,
      bridge: { direction: "horizontal" as const, segment: "start" as const } };
    const previewScene = new PersistentTerrainScene(new TerrainTextureCache());
    const boardScene = new PersistentTerrainScene(new TerrainTextureCache());
    previewScene.update([bridgeCell], bounds, snapshot, false);
    boardScene.update([bridgeCell], bounds, snapshot, true);
    for (const objects of [previewScene.getCellDisplayObjects("0,0")!, boardScene.getCellDisplayObjects("0,0")!]) {
      for (const sprite of [objects.bridgeShadowSprite, objects.bridgePillarSprite, objects.bridgeTopSprite]) {
        expect(sprite.position.set).toHaveBeenLastCalledWith(3, 9);
        expect(sprite).toMatchObject({ visible: true, width: 60, height: 84 });
      }
    }
  });

  it("draws four persistent road underlays behind one measured 2x2 base core", () => {
    const baseImage = {} as HTMLImageElement, roadImage = {} as HTMLImageElement;
    const snapshot: AssetSnapshot = {
      enabled: true, statuses: {},
      assets: {
        base: { image: baseImage, width: 192, height: 192, name: "base.png", ...DEFAULT_ADJUSTMENT },
        road: { image: roadImage, width: 48, height: 48, name: "road.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update(baseCells, baseBounds, snapshot);
    expect(scene.baseCount).toBe(1);
    const core = scene.getBaseDisplayObjects(ownedBase.id)!;
    expect(core.coreSprite.visible).toBe(true);
    expect(core.coreSprite.position.set).toHaveBeenLastCalledWith(10, 20);
    expect(core.coreSprite.width).toBe(80);
    expect(core.coreSprite.height).toBe(80);
    for (const coord of ownedBase.coords) {
      const cellObjects = scene.getCellDisplayObjects(`${coord.x},${coord.y}`)!;
      expect(cellObjects.groundSprite.visible).toBe(true);
      expect(cellObjects.groundGraphics.visible).toBe(false);
      expect(cellObjects.roadSprite.visible).toBe(false);
      expect(cellObjects.roadGraphics.visible).toBe(false);
      expect(cellObjects.faceSprite.visible).toBe(false);
      expect(cellObjects.faceGraphics.visible).toBe(coord.y === 1);
      expect(cellObjects.shadowGraphics.visible).toBe(coord.y === 1);
    }
    expect((scene.root.children[0] as { children: unknown[] }).children).toContain(
      scene.getCellDisplayObjects("0,0")!.groundSprite,
    );
    const baseLayer = scene.root.children[2] as { children: { children: unknown[] }[] };
    expect(baseLayer.children[0].children).toContain(core.coreSprite);
    const underlaySprite = scene.getCellDisplayObjects("0,0")!.groundSprite;
    const underlayTexture = underlaySprite.texture;
    const changedRoad: AssetSnapshot = {
      ...snapshot,
      assets: { ...snapshot.assets, road: { ...snapshot.assets.road!, image: {} as HTMLImageElement, name: "road-2.png" } },
    };
    scene.update(baseCells, baseBounds, changedRoad);
    expect(scene.getCellDisplayObjects("0,0")!.groundSprite).toBe(underlaySprite);
    expect(underlaySprite.texture).not.toBe(underlayTexture);

    const sprite = core.coreSprite, texture = sprite.texture;
    scene.update(baseCells, baseBounds, snapshot);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite).toBe(sprite);
    expect(sprite.texture).toBe(texture);
    const resized = new Map([...baseBounds].map(([key, rect]) => [key, {
      x: 5 + (rect.x - 10) * 1.5,
      y: 8 + (rect.y - 20) * 1.5,
      width: rect.width * 1.5,
      height: rect.height * 1.5,
    }]));
    scene.update(baseCells, resized, snapshot);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite).toBe(sprite);
    expect(sprite.position.set).toHaveBeenLastCalledWith(5, 8);
    expect(sprite.width).toBe(120);
  });

  it("uses the road Graphics fallback below a grouped base without changing its core count", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update(baseCells, baseBounds, EMPTY_ASSETS);
    expect(scene.baseCount).toBe(1);
    for (const coord of ownedBase.coords) {
      const objects = scene.getCellDisplayObjects(`${coord.x},${coord.y}`)!;
      expect(objects.groundGraphics.visible).toBe(true);
      expect(objects.groundGraphics.rect).toHaveBeenCalledWith(
        baseBounds.get(`${coord.x},${coord.y}`)!.x,
        baseBounds.get(`${coord.x},${coord.y}`)!.y,
        baseBounds.get(`${coord.x},${coord.y}`)!.width,
        baseBounds.get(`${coord.x},${coord.y}`)!.height,
      );
    }
  });

  it("uses 48px-per-cell offsets for independently adjustable 2x2 base parts", () => {
    expect(baseAssetPlacement({
      image, name: "part.png", width: 192, height: 96, scale: 1.5, offsetX: 4, offsetY: -2,
    }, { x: 10, y: 20, width: 96, height: 96 })).toEqual({ x: 14, y: 18, width: 144, height: 72 });
  });

  it("switches owned and third-party core textures on the same persistent Sprite", () => {
    const snapshot: AssetSnapshot = {
      enabled: true, statuses: {}, assets: {
        base: { image: {} as HTMLImageElement, width: 192, height: 192, name: "owned.png", ...DEFAULT_ADJUSTMENT },
        base_core_third_party: { image: {} as HTMLImageElement, width: 192, height: 192, name: "third.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update(baseCells, baseBounds, snapshot);
    const sprite = scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite;
    const ownedTexture = sprite.texture;
    const third = { ...ownedBase, ownerTeamId: "neutral", thirdParty: true, coreAssetKey: "base_core_third_party" as const, flagAssetKey: undefined };
    const changedCells = baseCells.map((cell, index) => ({ ...cell, baseGroup: index === 0 ? third : undefined }));
    scene.update(changedCells, baseBounds, snapshot);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite).toBe(sprite);
    expect(sprite.texture).not.toBe(ownedTexture);
  });

  it("switches owner/third-party walls and flags in place without tinting or changing source images", () => {
    const wallImage = {} as HTMLImageElement, thirdWallImage = {} as HTMLImageElement;
    const flagOne = {} as HTMLImageElement, flagTwo = {} as HTMLImageElement;
    const snapshot: AssetSnapshot = {
      enabled: true, statuses: {}, assets: {
        base_front_wall: { image: wallImage, width: 192, height: 192, name: "wall.png", scale: 1.25, offsetX: -3, offsetY: 2 },
        base_front_wall_third_party: { image: thirdWallImage, width: 192, height: 192, name: "third-wall.png", scale: 0.75, offsetX: 5, offsetY: -4 },
        base_flag_team_1: { image: flagOne, width: 192, height: 192, name: "flag-1.png", scale: 0.5, offsetX: 4, offsetY: -5 },
        base_flag_team_2: { image: flagTwo, width: 192, height: 192, name: "flag-2.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const scene = new PersistentBaseForegroundScene(new TerrainTextureCache());
    scene.update([ownedBase], baseBounds, snapshot);
    const objects = scene.getBaseDisplayObjects(ownedBase.id)!;
    const wall = objects.wallSprite, flag = objects.flagSprite, wallTexture = wall.texture;
    expect(wall.visible).toBe(true);
    expect(wall.tint).toBe(0xffffff);
    expect(wall.position.set).toHaveBeenLastCalledWith(7.5, 21.666666666666668);
    expect(wall.width).toBe(100);
    expect(flag.visible).toBe(true);
    expect(flag.position.set).toHaveBeenLastCalledWith(13.333333333333334, 15.833333333333332);
    expect(flag.width).toBe(40);

    const changedOwner = { ...ownedBase, ownerTeamId: "beta", flagAssetKey: "base_flag_team_2" as const };
    scene.update([changedOwner], baseBounds, snapshot);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.wallSprite).toBe(wall);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.flagSprite).toBe(flag);
    expect(wall.texture).toBe(wallTexture);
    expect(flag.texture).not.toBe(Texture.EMPTY);

    const neutral = { ...changedOwner, ownerTeamId: "neutral", thirdParty: true, coreAssetKey: "base_core_third_party" as const, flagAssetKey: undefined };
    scene.update([neutral], baseBounds, snapshot);
    expect(wall.texture).not.toBe(wallTexture);
    expect(wall.tint).toBe(0xffffff);
    expect(wall.position.set).toHaveBeenLastCalledWith(14.166666666666668, 16.666666666666668);
    expect(wall.width).toBe(60);
    expect(flag.visible).toBe(false);
    expect(snapshot.assets.base_front_wall?.image).toBe(wallImage);
    expect(snapshot.assets.base_front_wall_third_party?.image).toBe(thirdWallImage);

    const { base_front_wall_third_party: _removed, ...fallbackAssets } = snapshot.assets;
    scene.update([neutral], baseBounds, { ...snapshot, assets: fallbackAssets });
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.wallSprite).toBe(wall);
    expect(wall.texture).toBe(wallTexture);
    expect(wall.tint).toBe(0xffffff);
    expect(wall.position.set).toHaveBeenLastCalledWith(7.5, 21.666666666666668);

    scene.update([changedOwner], baseBounds, snapshot);
    expect(wall.texture).toBe(wallTexture);
    expect(flag.visible).toBe(true);
  });

  it("keeps optional wall and flag hidden when their PNGs are unset", () => {
    const scene = new PersistentBaseForegroundScene(new TerrainTextureCache());
    scene.update([ownedBase], baseBounds, EMPTY_ASSETS);
    const objects = scene.getBaseDisplayObjects(ownedBase.id)!;
    expect(objects.wallSprite.visible).toBe(false);
    expect(objects.wallGraphics.visible).toBe(false);
    expect(objects.flagSprite.visible).toBe(false);
    expect(objects.flagGraphics.visible).toBe(false);
  });

  it("keeps renderer-local textures and destroys only changed images", () => {
    const a = new TerrainTextureCache(), b = new TerrainTextureCache();
    a.sync(imported); b.sync(imported);
    const sceneA = new PersistentTerrainScene(a), sceneB = new PersistentTerrainScene(b);
    sceneA.update([cell], bounds, imported); sceneB.update([cell], bounds, imported);
    const textureA = sceneA.getCellDisplayObjects("0,0")!.roadSprite.texture as Texture;
    const textureB = sceneB.getCellDisplayObjects("0,0")!.roadSprite.texture as Texture;
    expect(textureA).not.toBe(textureB);
    expect(a.sync(imported)).toBe(false);
    a.dispose();
    expect(textureA.destroy).toHaveBeenCalledOnce();
    expect(textureB.destroy).not.toHaveBeenCalled();
    b.dispose();
  });

  it("reuses one Sprite when only placement adjustment changes", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([cell], bounds, imported);
    const sprite = scene.getCellDisplayObjects("0,0")!.roadSprite as Sprite;
    const texture = sprite.texture;
    const adjusted = { ...imported, assets: { road: { ...imported.assets.road!, scale: 2 } } };
    scene.update([cell], bounds, adjusted);
    expect(scene.getCellDisplayObjects("0,0")!.roadSprite).toBe(sprite);
    expect(sprite.texture).toBe(texture);
    expect(sprite.width).toBe(96);
  });

  it("shares four lake textures, synchronizes alpha, and reuses two Sprites per lake cell", () => {
    const images = [{}, {}, {}, {}] as HTMLImageElement[];
    const lakeAssets: AssetSnapshot = {
      enabled: true,
      lakeAnimationEnabled: true,
      statuses: {},
      assets: {
        lake: { image: images[0], width: 96, height: 48, name: "lake-1.png", scale: 1.25, offsetX: 2, offsetY: -3 },
        lake_frame_2: { image: images[1], width: 32, height: 128, name: "lake-2.png", ...DEFAULT_ADJUSTMENT },
        lake_frame_3: { image: images[2], width: 128, height: 32, name: "lake-3.png", ...DEFAULT_ADJUSTMENT },
        lake_frame_4: { image: images[3], width: 64, height: 64, name: "lake-4.png", ...DEFAULT_ADJUSTMENT },
      },
    };
    const cells = [
      { ...cell, kind: "lake" as const, southFace: false },
      { ...cell, x: 1, kind: "lake" as const, southFace: false },
    ];
    const lakeBounds = new Map([
      ["0,0", { x: 0, y: 0, width: 48, height: 48 }],
      ["1,0", { x: 48, y: 0, width: 48, height: 48 }],
    ]);
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update(cells, lakeBounds, lakeAssets);
    const first = scene.getCellDisplayObjects("0,0")!;
    const second = scene.getCellDisplayObjects("1,0")!;
    const firstBaseSprite = first.groundSprite, firstBlendSprite = first.lakeBlendSprite;
    scene.applyLakeAnimation(lakeAnimationStateAt(700), lakeAssets);

    expect(first.groundSprite.texture).toBe(second.groundSprite.texture);
    expect(first.lakeBlendSprite.texture).toBe(second.lakeBlendSprite.texture);
    expect(first.lakeBlendSprite.alpha).toBe(0.5);
    expect(second.lakeBlendSprite.alpha).toBe(0.5);
    expect(first.groundSprite).not.toBe(first.lakeBlendSprite);
    expect(first.lakeBlendSprite.position.set).toHaveBeenLastCalledWith(2, -3);
    expect(first.lakeBlendSprite.width).toBe(60);
    expect(first.lakeBlendSprite.height).toBe(30);
    expect(second.lakeBlendSprite.width).toBe(60);
    expect(second.lakeBlendSprite.height).toBe(30);
    const frame2Texture = first.lakeBlendSprite.texture;

    scene.applyLakeAnimation(lakeAnimationStateAt(900), lakeAssets);
    expect(first.groundSprite).toBe(firstBaseSprite);
    expect(first.lakeBlendSprite).toBe(firstBlendSprite);
    expect(first.groundSprite.texture).toBe(frame2Texture);
    const frame3Texture = first.lakeBlendSprite.texture as Texture;

    const withoutFrame3 = { ...lakeAssets, assets: { ...lakeAssets.assets, lake_frame_3: undefined } };
    scene.update(cells, lakeBounds, withoutFrame3);
    expect(first.lakeBlendSprite.visible).toBe(false);
    expect(first.groundSprite.texture).not.toBe(frame2Texture);
    expect(frame3Texture.destroy).toHaveBeenCalledOnce();

    const groundLayer = scene.root.children[0] as { children: unknown[] };
    expect(groundLayer.children).toContain(first.lakeBlendSprite);
    expect(groundLayer.children).toContain(second.lakeBlendSprite);
  });
});


describe("Phase 2 fortress overflow", () => {
  it.each(["road", "lake", null] as const)("keeps %s behind an expanded core on all four sides", (kind) => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    const snapshot: AssetSnapshot = { ...imported, assets: { ...imported.assets,
      base: { image, width: 96, height: 96, name: "core", scale: 3, offsetX: -96, offsetY: -96 },
    } };
    const neighbors = [{ x: -1, y: 0 }, { x: 2, y: 0 }, { x: 0, y: -1 }, { x: 0, y: 2 }];
    const rects = new Map(baseBounds);
    for (const coord of neighbors) rects.set(`${coord.x},${coord.y}`, { x: 10 + coord.x * 40, y: 20 + coord.y * 36, width: 40, height: 36 });
    const cells: PixiBoardTerrain = [...baseCells, ...neighbors.map((coord) => ({ ...cell, ...coord, kind, southFace: kind === "road" }))];
    for (const assets of [snapshot, { ...snapshot, assets: { base: snapshot.assets.base } }]) {
      scene.update(cells, rects, assets);
      const core = scene.getBaseDisplayObjects(ownedBase.id)!;
      expect(core.coreSprite.position.set).toHaveBeenLastCalledWith(-70, -60);
      expect(core.coreSprite.width).toBe(240);
      expect(core.coreSprite.height).toBe(240);
      expect(scene.baseCount).toBe(1);
      const layers = scene.root.children as unknown as { children: unknown[] }[];
      expect(layers[2].children.some((node) => (node as { children: unknown[] }).children.includes(core.coreSprite))).toBe(true);
      for (const coord of neighbors) {
        const objects = scene.getCellDisplayObjects(`${coord.x},${coord.y}`)!;
        if (kind === "road") {
          expect(layers[1].children).toContain(objects.roadSprite);
          expect(layers[1].children).toContain(objects.roadGraphics);
          expect(objects.roadSprite.visible).toBe(Boolean(assets.assets.road));
          expect(objects.roadGraphics.visible).toBe(!assets.assets.road);
          expect(layers[4].children.some((node) => (node as { children: unknown[] }).children.includes(objects.faceGraphics))).toBe(true);
        } else if (kind === "lake") {
          expect(layers[0].children).toContain(objects.groundGraphics);
          expect(objects.groundGraphics.visible).toBe(true);
        } else {
          expect(layers[0].children).toContain(objects.outsideGraphics);
          expect(objects.outsideGraphics.visible).toBe(true);
          expect(objects.outsideGraphics.fill).toHaveBeenLastCalledWith(0xd8dde2);
        }
      }
    }
    const core = scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite;
    scene.update(cells, rects, snapshot);
    expect(scene.getBaseDisplayObjects(ownedBase.id)!.coreSprite).toBe(core);
    expect(ownedBase.coords).toHaveLength(4);
  });
});


const overflowAssets: AssetSnapshot = { enabled: true, statuses: {}, assets: {
  base: { image, width: 96, height: 96, name: "core", scale: 3, offsetX: -96, offsetY: -96 },
  base_front_wall: { image, width: 96, height: 192, name: "wall", scale: 2, offsetX: 48, offsetY: 48 },
  base_flag_team_1: { image, width: 96, height: 96, name: "flag", scale: 1, offsetX: -120, offsetY: -120 },
  base_core_third_party: { image, width: 96, height: 96, name: "neutral", scale: 1, offsetX: 0, offsetY: 0 },
} };

describe("shared Canvas extent", () => {
  it("includes independently scaled and offset bridge layers outside the board", () => {
    const bridgeCell = [{ ...cell, kind: "lake" as const, southFace: false,
      bridge: { direction: "horizontal" as const, segment: "middle" as const } }];
    const bridgeBounds = new Map([["0,0", { x: 0, y: 0, width: 40, height: 40 }]]);
    const baseAsset = { image, width: 48, height: 48, name: "bridge.png", ...DEFAULT_ADJUSTMENT };
    const bridgeAssets: AssetSnapshot = { enabled: true, statuses: {}, assets: {
      bridge_top: { ...baseAsset, scaleX: 2, scaleY: 1, offsetX: 48, offsetY: 0 },
      bridge_pillar: { ...baseAsset, scaleX: 1, scaleY: 2, offsetX: 0, offsetY: 48 },
      bridge_shadow: { ...baseAsset, scaleX: 1, scaleY: 1, offsetX: -24, offsetY: 0 },
    } };
    expect(pixiBoardExtent(bridgeCell, bridgeBounds, 100, 100, bridgeAssets)).toEqual({
      left: -20, top: 0, width: 140, height: 120,
      paddingLeft: 20, paddingTop: 0, paddingRight: 20, paddingBottom: 20,
    });
    const verticalBridge = [{ ...bridgeCell[0],
      bridge: { direction: "vertical" as const, segment: "middle" as const } }];
    expect(pixiBoardExtent(verticalBridge, bridgeBounds, 100, 100, bridgeAssets)).toEqual({
      left: 0, top: 0, width: 120, height: 100,
      paddingLeft: 0, paddingTop: 0, paddingRight: 20, paddingBottom: 0,
    });
    expect(pixiBoardExtent(bridgeCell, bridgeBounds, 100, 100, { ...bridgeAssets, enabled: false })).toEqual({
      left: 0, top: 0, width: 100, height: 100,
      paddingLeft: 0, paddingTop: 0, paddingRight: 0, paddingBottom: 0,
    });
  });

  it("unions the actual core, wall and flag rectangles and resets without changing bounds", () => {
    const extent = pixiBoardExtent(baseCells, baseBounds, 100, 100, overflowAssets);
    expect(extent).toEqual({ left: -90, top: -80, width: 300, height: 460,
      paddingLeft: 90, paddingTop: 80, paddingRight: 110, paddingBottom: 280 });
    expect(pixiBoardExtent(baseCells, baseBounds, 100, 100, EMPTY_ASSETS)).toEqual({
      left: 0, top: 0, width: 100, height: 100, paddingLeft: 0, paddingTop: 0, paddingRight: 0, paddingBottom: 0,
    });
    const neutral = { ...ownedBase, thirdParty: true, coreAssetKey: "base_core_third_party" as const, flagAssetKey: undefined };
    const neutralCells = baseCells.map((cell) => ({ ...cell, baseGroup: cell.baseGroup ? neutral : undefined }));
    // Unset third-party wall continues using the normal wall.
    expect(pixiBoardExtent(neutralCells, baseBounds, 100, 100, overflowAssets).paddingBottom).toBe(280);
    const assets = { ...overflowAssets, assets: { ...overflowAssets.assets, base_front_wall_third_party: overflowAssets.assets.base_core_third_party } };
    expect(pixiBoardExtent(neutralCells, baseBounds, 100, 100, assets).paddingBottom).toBe(0);
    expect(baseBounds.get("0,0")).toEqual({ x: 10, y: 20, width: 40, height: 36 });
    expect(baseCells).toHaveLength(4);
  });

  it("tracks cell resize, scale and offsets while ignoring screen/scroll origin", () => {
    const small = new Map([...baseBounds].map(([key, rect]) => [key, {
      x: rect.x / 2, y: rect.y / 2, width: rect.width / 2, height: rect.height / 2,
    }]));
    const extent = pixiBoardExtent(baseCells, small, 50, 50, overflowAssets);
    expect(extent.width).toBe(150);
    expect(extent.height).toBe(230);
    for (const origin of [{ x: 200, y: 300 }, { x: -400, y: -500 }]) {
      const scrolled = new Map([...baseBounds].map(([key, rect]) => [key,
        relativeCellBounds({ ...rect, x: rect.x + origin.x, y: rect.y + origin.y }, origin),
      ]));
      expect(pixiBoardExtent(baseCells, scrolled, 100, 100, overflowAssets))
        .toEqual(pixiBoardExtent(baseCells, baseBounds, 100, 100, overflowAssets));
    }
    const changed = { ...overflowAssets, assets: { base: { ...overflowAssets.assets.base!, scale: 1, offsetX: 0, offsetY: 0 } } };
    expect(pixiBoardExtent(baseCells, baseBounds, 100, 100, changed).width).toBe(100);
  });

  it("reuses both Applications and canvases and applies equal stage/CSS compensation", async () => {
    const viewport = { classList: { contains: () => true }, style: { setProperty: vi.fn() } };
    const board = { parentElement: viewport };
    const host = { style: {}, appendChild: vi.fn(), parentElement: board } as unknown as HTMLDivElement;
    const foregroundHost = { style: {}, appendChild: vi.fn(), parentElement: board } as unknown as HTMLDivElement;
    const instances = (Application as unknown as { instances: Application[] }).instances;
    const before = instances.length;
    const mounted = await mountPixiBoard(host, foregroundHost, new AbortController().signal);
    const [background, foreground] = instances.slice(before);
    mounted.update(baseCells, baseBounds, 100, 100, overflowAssets);
    for (const app of [background, foreground]) {
      expect(app.stage.position.set).toHaveBeenLastCalledWith(90, 80);
      expect(app.renderer.resize).toHaveBeenLastCalledWith(300, 460);
    }
    expect(host.style).toEqual(foregroundHost.style);
    expect(host.style.left).toBe("-90px");
    expect(host.style.top).toBe("-80px");
    expect(viewport.style.setProperty).toHaveBeenCalledWith("--pixi-left", "90px");
    mounted.update(baseCells, baseBounds, 100, 100, overflowAssets);
    expect(background.renderer.resize).toHaveBeenCalledTimes(1);
    mounted.update(baseCells, baseBounds, 100, 100, EMPTY_ASSETS);
    expect(background.renderer.resize).toHaveBeenLastCalledWith(100, 100);
    expect(background.stage.position.set).toHaveBeenLastCalledWith(-0, -0);
    expect(instances).toHaveLength(before + 2);
    expect(host.appendChild).toHaveBeenCalledTimes(1);
    expect(foregroundHost.appendChild).toHaveBeenCalledTimes(1);
    mounted.dispose();
  });

  it("redraws only the background Canvas during lake fades and stops when switched off", async () => {
    vi.useFakeTimers();
    try {
      const host = { style: {}, appendChild: vi.fn() } as unknown as HTMLDivElement;
      const foregroundHost = { style: {}, appendChild: vi.fn() } as unknown as HTMLDivElement;
      const instances = (Application as unknown as { instances: Application[] }).instances;
      const before = instances.length;
      const mounted = await mountPixiBoard(host, foregroundHost, new AbortController().signal);
      const [background, foreground] = instances.slice(before);
      const lakeAsset = { image, width: 48, height: 48, name: "lake.png", ...DEFAULT_ADJUSTMENT };
      const animated: AssetSnapshot = {
        enabled: true,
        lakeAnimationEnabled: true,
        statuses: {},
        assets: {
          lake: lakeAsset,
          lake_frame_2: { ...lakeAsset, name: "lake-2.png" },
          lake_frame_3: { ...lakeAsset, name: "lake-3.png" },
          lake_frame_4: { ...lakeAsset, name: "lake-4.png" },
        },
      };
      const lakeCell = { ...cell, kind: "lake" as const, southFace: false };
      mounted.update([lakeCell], bounds, 48, 48, animated);
      expect(background.render).toHaveBeenCalledTimes(1);
      expect(foreground.render).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(700);
      expect(vi.mocked(background.render).mock.calls.length).toBeGreaterThan(1);
      expect(foreground.render).toHaveBeenCalledTimes(1);

      mounted.update([lakeCell], bounds, 48, 48, { ...animated, lakeAnimationEnabled: false });
      const backgroundAfterOff = vi.mocked(background.render).mock.calls.length;
      const foregroundAfterOff = vi.mocked(foreground.render).mock.calls.length;
      await vi.advanceTimersByTimeAsync(2000);
      expect(background.render).toHaveBeenCalledTimes(backgroundAfterOff);
      expect(foreground.render).toHaveBeenCalledTimes(foregroundAfterOff);
      mounted.dispose();
    } finally {
      vi.useRealTimers();
    }
  });
});


describe("individual fortress assets and outside transitions", () => {
  it.each(["base", "base_core_third_party", "base_front_wall", "base_front_wall_third_party",
    "base_flag_team_1", "base_flag_team_2", "base_flag_team_3", "base_flag_team_4"] as const)(
    "tracks %s import, independent Reset and replacement dimensions", (key) => {
      const thirdParty = key === "base_core_third_party" || key === "base_front_wall_third_party";
      const group: PixiBaseGroup = { ...ownedBase, thirdParty,
        coreAssetKey: thirdParty ? "base_core_third_party" : "base",
        flagAssetKey: key.startsWith("base_flag") ? key as PixiBaseGroup["flagAssetKey"] : undefined };
      const cells = baseCells.map((cell) => ({ ...cell, baseGroup: cell.baseGroup ? group : undefined }));
      const asset = { image, width: 96, height: 192, name: key, scale: 3, offsetX: -96, offsetY: -96 };
      const snapshot: AssetSnapshot = { enabled: true, statuses: {}, assets: { [key as TileAssetKey]: asset } };
      expect(pixiBoardExtent(cells, baseBounds, 100, 100, snapshot)).toMatchObject({ left: -70, top: -60, width: 240, height: 480 });
      expect(pixiBoardExtent(cells, baseBounds, 100, 100, { ...snapshot, assets: {} })).toMatchObject({ width: 100, height: 100 });
      const replaced = { ...snapshot, assets: { [key]: { ...asset, height: 96 } } };
      expect(pixiBoardExtent(cells, baseBounds, 100, 100, replaced).height).toBe(240);
    },
  );

  it("reuses outside Graphics and restores lake/road visuals when terrain changes", () => {
    const scene = new PersistentTerrainScene(new TerrainTextureCache());
    scene.update([{ ...cell, kind: null }], bounds, EMPTY_ASSETS);
    const objects = scene.getCellDisplayObjects("0,0")!;
    expect(objects.outsideGraphics.visible).toBe(true);
    expect(objects.faceGraphics.visible).toBe(false);
    scene.update([{ ...cell, kind: "lake", southFace: false }], bounds, EMPTY_ASSETS);
    expect(objects.outsideGraphics.visible).toBe(false);
    expect(objects.groundGraphics.visible).toBe(true);
    scene.update([cell], bounds, imported);
    expect(objects.roadSprite.visible).toBe(true);
    scene.update([{ ...cell, kind: null }], bounds, EMPTY_ASSETS);
    expect(objects.outsideGraphics.visible).toBe(true);
    expect(objects.roadSprite.visible).toBe(false);
    expect(scene.getCellDisplayObjects("0,0")!.outsideGraphics).toBe(objects.outsideGraphics);
    expect(objects.outsideGraphics.clear).toHaveBeenCalledTimes(1);
    scene.update([], bounds, EMPTY_ASSETS);
    expect(objects.outsideGraphics.visible).toBe(false);
  });
});
