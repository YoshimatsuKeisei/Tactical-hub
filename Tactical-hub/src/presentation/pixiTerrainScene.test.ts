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
    clear = vi.fn(() => this);
    destroy = vi.fn();
    rect() { return this; }
    fill() { return this; }
    stroke() { return this; }
    moveTo() { return this; }
    lineTo() { return this; }
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
    destroy = vi.fn();
    constructor(public texture: Texture) {}
  }
  return { Container, Graphics, ImageSource, Texture, Sprite };
});

import { PersistentTerrainScene, TerrainTextureCache } from "./pixiTerrainScene";
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
