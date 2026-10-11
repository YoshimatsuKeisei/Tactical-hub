import { afterEach, describe, expect, it, vi } from "vitest";
import { ATLAS_KINDS, ATLAS_SIZE, createTerrainAtlas } from "./pixiTerrainAtlas";

vi.mock("pixi.js", () => ({ Texture: { from: (resource: unknown) => ({ source: { resource, scaleMode: "linear" } }) } }));
afterEach(() => vi.unstubAllGlobals());

describe("single-source diagnostic terrain atlas", () => {
  it("writes three distinct color regions at explicit atlas offsets", () => {
    const fills: unknown[] = [];
    const context = {
      fillStyle: "", strokeStyle: "",
      fillRect(x: number, y: number, width: number, height: number) { fills.push([this.fillStyle, x, y, width, height]); },
      strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    };
    const canvas = { width: 0, height: 0, getContext: () => context };
    vi.stubGlobal("document", { createElement: () => canvas });
    const atlas = createTerrainAtlas();
    expect(ATLAS_KINDS).toEqual(["road", "base", "lake"]);
    expect(fills).toEqual([
      ["#999c96", 0, 0, 48, 48], ["#c6a253", 48, 0, 48, 48], ["#338ac0", 96, 0, 48, 48],
    ]);
    expect(canvas.width).toBe(ATLAS_SIZE * 3);
    expect(atlas.source.resource).toBe(canvas);
    expect(atlas.source.scaleMode).toBe("nearest");
  });
  it("reports a missing 2D context instead of silently rendering wrong colors", () => {
    vi.stubGlobal("document", { createElement: () => ({ getContext: () => null }) });
    expect(() => createTerrainAtlas()).toThrow("Cannot create terrain atlas canvas");
  });
});
