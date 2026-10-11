import { describe, expect, it, vi } from "vitest";
import { DEFAULT_ADJUSTMENT, EMPTY_ASSETS, type AssetSnapshot } from "./tileAssets";

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
    fill() { return this; }
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
    destroy = vi.fn();
    constructor(public texture: Texture) {}
  }
  return { Container, Graphics, ImageSource, Texture, Sprite };
});

import { movementOutlineEdges, PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
import { Sprite, Texture } from "pixi.js";

const image = {} as HTMLImageElement;
const imported: AssetSnapshot = {
  enabled: true,
  statuses: {},
  assets: { road: { image, width: 128, height: 128, name: "road.png", ...DEFAULT_ADJUSTMENT } },
};
const cell = { x: 0, y: 0, kind: "road" as const, southFace: true, bridge: null, obstacle: false };
const bounds = new Map([["0,0", { x: 0, y: 0, width: 48, height: 48 }]]);

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
    expect((scene.root.children[1] as { children: unknown[] }).children).toContain(objects.shadowGraphics);
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
});
