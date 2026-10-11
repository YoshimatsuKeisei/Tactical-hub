import { describe, expect, it, vi } from "vitest";
import { TerrainTextureCache, createTerrainScene } from "./pixiTerrainScene";
import { EMPTY_ASSETS, DEFAULT_ADJUSTMENT, type AssetSnapshot } from "./tileAssets";
import type { PixiBoardTerrain } from "./pixiBoardTerrain";
// Test scene contracts without a DOM/WebGL renderer.
vi.mock("pixi.js", () => {
  class Container { children: unknown[] = []; eventMode = ""; sortableChildren = false; zIndex = 0; addChild(child: unknown) { this.children.push(child); return child; } }
  class Graphics { rect() { return this; } fill() { return this; } stroke() { return this; } moveTo() { return this; } lineTo() { return this; } }
  class ImageSource { constructor(public options: unknown) {} }
  class Texture { destroy = vi.fn(); constructor(public options: unknown) {} }
  class Sprite { position = { set: vi.fn() }; width = 0; height = 0; constructor(public texture: Texture) {} }
  return { Container, Graphics, ImageSource, Texture, Sprite };
});
import { Container, Sprite, Graphics } from "pixi.js";
const image = {} as HTMLImageElement;
const snapshot: AssetSnapshot = { enabled: true, statuses: {}, assets: { road: { image, width: 128, height: 128, name: "r.png", ...DEFAULT_ADJUSTMENT } } };
const cell = { x: 0, y: 0, kind: "road" as const, southFace: true, bridge: null, obstacle: false };
const bounds = new Map([["0,0", { x: 0, y: 0, width: 48, height: 48 }]]);
function descendants(container: Container): unknown[] { return container.children.flatMap((child) => child instanceof Container ? [child, ...descendants(child)] : [child]); }
describe("Graphics fallback and renderer-local Sprite sources", () => {
  it("uses Graphics for unset assets and never draws sprites in shapes mode", () => {
    const cache = new TerrainTextureCache(); cache.sync(snapshot);
    const scene = createTerrainScene([cell], bounds, EMPTY_ASSETS, cache);
    expect(descendants(scene).filter((child) => child instanceof Graphics).length).toBeGreaterThan(0);
    expect(descendants(scene).some((child) => child instanceof Sprite)).toBe(false);
  });
  it("draws an imported road, retains optional Graphics face, and suppresses internal faces", () => {
    const cache = new TerrainTextureCache(); cache.sync(snapshot);
    expect(descendants(createTerrainScene([cell], bounds, snapshot, cache)).filter((child) => child instanceof Sprite)).toHaveLength(1);
    const scene = createTerrainScene([{ ...cell, southFace: false }], bounds, snapshot, cache);
    expect(descendants(scene).filter((child) => child instanceof Graphics)).toHaveLength(0);
    cache.dispose();
  });
  it("keeps the lake underneath an imported bridge and obstacle", () => {
    const current = { ...snapshot, assets: { lake: snapshot.assets.road, bridge_vertical_end: snapshot.assets.road, obstacle: snapshot.assets.road } };
    const cache = new TerrainTextureCache(); cache.sync(current);
    const scene = createTerrainScene([{ ...cell, kind: "lake", southFace: false, bridge: { direction: "vertical", segment: "end" }, obstacle: true }] as PixiBoardTerrain, bounds, current, cache);
    expect(descendants(scene).filter((child) => child instanceof Sprite)).toHaveLength(3);
    const ground = scene.children[0] as Container;
    expect(ground.children[0]).toBeInstanceOf(Sprite);
  });
  it("uses the exact selected bridge key, with fallback when only another segment is loaded", () => {
    const current = { ...snapshot, assets: { bridge_horizontal_start: snapshot.assets.road } };
    const cache = new TerrainTextureCache(); cache.sync(current);
    const cells: PixiBoardTerrain = [{ ...cell, kind: "lake", bridge: { direction: "horizontal", segment: "middle" }, southFace: false }];
    expect(descendants(createTerrainScene(cells, bounds, current, cache)).some((child) => child instanceof Sprite)).toBe(false);
  });
  it("retains Flat/Raised behavior: separate south-face sprite appears only when exposed and raised", () => {
    const current = { ...snapshot, assets: { ...snapshot.assets, road_south_face: snapshot.assets.road } };
    const cache = new TerrainTextureCache(); cache.sync(current);
    expect(descendants(createTerrainScene([cell], bounds, current, cache, false)).filter((child) => child instanceof Sprite)).toHaveLength(1);
    expect(descendants(createTerrainScene([cell], bounds, current, cache, true)).filter((child) => child instanceof Sprite)).toHaveLength(2);
    expect(descendants(createTerrainScene([{ ...cell, southFace: false }], bounds, current, cache, true)).filter((child) => child instanceof Sprite)).toHaveLength(1);
  });
  it("owns separate textures per Canvas; replacement/reset/disposal destroys only local textures", () => {
    const a = new TerrainTextureCache(), b = new TerrainTextureCache(); a.sync(snapshot); b.sync(snapshot);
    const pa = new Container(), pb = new Container();
    a.draw(pa, "road", bounds.get("0,0")!, snapshot); b.draw(pb, "road", bounds.get("0,0")!, snapshot);
    const ta = (pa.children[0] as Sprite).texture, tb = (pb.children[0] as Sprite).texture;
    expect(ta).not.toBe(tb); a.dispose(); expect(ta.destroy).toHaveBeenCalledOnce(); expect(tb.destroy).not.toHaveBeenCalled();
    b.sync({ ...snapshot, assets: { road: { ...snapshot.assets.road!, image: {} as HTMLImageElement } } });
    expect(tb.destroy).toHaveBeenCalledOnce(); b.sync(EMPTY_ASSETS); b.dispose();
  });
  it("does not allocate a new texture when only placement changes", () => {
    const cache = new TerrainTextureCache(); cache.sync(snapshot); const parent = new Container();
    cache.draw(parent, "road", bounds.get("0,0")!, snapshot); const texture = (parent.children[0] as Sprite).texture;
    const adjusted = { ...snapshot, assets: { road: { ...snapshot.assets.road!, scale: 2 } } };
    cache.sync(adjusted); cache.draw(parent, "road", bounds.get("0,0")!, adjusted);
    expect((parent.children[1] as Sprite).texture).toBe(texture);
    expect((parent.children[1] as Sprite).width).toBe(96);
  });
});
