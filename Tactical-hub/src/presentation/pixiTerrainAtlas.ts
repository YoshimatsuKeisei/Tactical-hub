import { Texture } from "pixi.js";
import { TERRAIN_COLORS } from "./pixiBoardTerrain";

export const ATLAS_SIZE = 48;
export const ATLAS_KINDS = ["road", "base", "lake"] as const;
// One CPU canvas source: explicit atlas offsets avoid multi-render-target sampling.
// Disposable diagnostic shapes, never exported as game assets.
export function createTerrainAtlas() {
  const canvas = document.createElement("canvas");
  canvas.width = ATLAS_SIZE * ATLAS_KINDS.length;
  canvas.height = ATLAS_SIZE;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Cannot create terrain atlas canvas");
  ATLAS_KINDS.forEach((kind, index) => {
    const x = index * ATLAS_SIZE;
    context.fillStyle = `#${TERRAIN_COLORS[kind].toString(16).padStart(6, "0")}`;
    context.fillRect(x, 0, ATLAS_SIZE, ATLAS_SIZE);
    context.strokeStyle = "#243546";
    context.strokeRect(x + 0.5, 0.5, ATLAS_SIZE - 1, ATLAS_SIZE - 1);
    if (kind === "lake") {
      context.strokeStyle = "#83c6e5";
      context.beginPath(); context.moveTo(x + 8, 24); context.lineTo(x + 40, 24); context.stroke();
    }
    if (kind === "base") {
      context.strokeStyle = "#ffe7a2";
      context.strokeRect(x + 12, 12, 24, 24);
    }
  });
  const texture = Texture.from(canvas);
  texture.source.scaleMode = "nearest";
  return texture;
}
