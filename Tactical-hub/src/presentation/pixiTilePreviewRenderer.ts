import { Application, Container, Graphics, Rectangle, Text, type Texture } from "pixi.js";
import { Tilemap } from "@pixi/tilemap";
import {
  PREVIEW_CELL_SIZE as S, PREVIEW_DEPTH, PREVIEW_TERRAIN, PREVIEW_BRIDGES,
  PREVIEW_OBSTACLES, hasPreviewRoadSouthFace, previewCellAt, previewRowDepth,
  type PreviewTerrain,
} from "./pixiTilePreview";

const WIDTH = PREVIEW_TERRAIN[0].length * S;
const HEIGHT = PREVIEW_TERRAIN.length * S;
const ORIGINS = [{ x: 16, y: 44 }, { x: WIDTH + 48, y: 44 }];

// Imported only while the DEV panel is open. All textures are disposable test shapes.
export async function mountPixiTilePreview(
  host: HTMLDivElement, signal: AbortSignal, onCell: (text: string) => void,
): Promise<() => void> {
  const app = new Application();
  const textures: Texture[] = [];
  let initialized = false;
  let destroyed = false;
  const dispose = () => {
    if (destroyed) return;
    destroyed = true;
    if (initialized) app.destroy(true, { children: true });
    else app.stage.destroy({ children: true });
    for (const texture of textures) texture.destroy(true);
  };
  try {
    await app.init({
      width: WIDTH * 2 + 64, height: HEIGHT + 72,
      background: "#182332", preference: "webgl", antialias: false,
      autoStart: false, resolution: 1,
    });
    initialized = true;
    if (signal.aborted) { dispose(); return dispose; }
    host.appendChild(app.canvas);
    app.canvas.setAttribute("aria-label", "Flat and raised square tile comparison; click a logical cell");
    const colors: Record<PreviewTerrain, number> = { road: 0x999c96, base: 0xc6a253, lake: 0x338ac0 };
    const tiles = {} as Record<PreviewTerrain, Texture>;
    for (const kind of ["road", "base", "lake"] as const) {
      const shape = new Graphics().rect(0, 0, S, S).fill(colors[kind]);
      shape.rect(0.5, 0.5, S - 1, S - 1).stroke({ color: 0x243546, width: 1 });
      if (kind === "lake") shape.moveTo(8, 24).lineTo(40, 24).stroke({ color: 0x83c6e5, width: 2 });
      if (kind === "base") shape.rect(12, 12, 24, 24).stroke({ color: 0xffe7a2, width: 3 });
      tiles[kind] = app.renderer.generateTexture(shape);
      tiles[kind].source.scaleMode = "nearest";
      textures.push(tiles[kind]);
      shape.destroy();
    }
    const highlights: Graphics[] = [];
    ORIGINS.forEach((origin, panelIndex) => {
      const raised = panelIndex === 1;
      app.stage.addChild(new Text({ text: raised ? "Raised / exposed south faces" : "Flat", style: { fill: 0xffffff, fontSize: 15 } })).position.set(origin.x, 12);
      const panel = new Container();
      panel.position.set(origin.x, origin.y);
      panel.eventMode = "none";
      app.stage.addChild(panel);
      const ground = new Tilemap(Object.values(tiles).map((texture) => texture.source));
      PREVIEW_TERRAIN.forEach((row, y) => row.forEach((kind, x) => ground.tile(tiles[kind], x * S, y * S)));
      panel.addChild(ground);
      const depth = new Container();
      depth.sortableChildren = true;
      panel.addChild(depth);
      PREVIEW_TERRAIN.forEach((row, y) => {
        const rowLayer = new Container();
        rowLayer.zIndex = previewRowDepth(y);
        depth.addChild(rowLayer);
        // Explicit layers per row: road surfaces/faces, bridges, then obstacles.
        const roadLayer = new Graphics();
        row.forEach((kind, x) => {
          if (raised && kind === "road") {
            roadLayer.rect(x * S, y * S, S, S).fill(colors.road);
            roadLayer.rect(x * S + 0.5, y * S + 0.5, S - 1, S - 1).stroke({ color: 0x647066, width: 1 });
            if (hasPreviewRoadSouthFace(PREVIEW_TERRAIN, x, y)) {
              roadLayer.rect(x * S, (y + 1) * S, S, PREVIEW_DEPTH).fill(0x4e544f);
              roadLayer.moveTo(x * S, (y + 1) * S).lineTo((x + 1) * S, (y + 1) * S).stroke({ color: 0xd7d9cc, width: 2 });
            }
          }
        });
        rowLayer.addChild(roadLayer);
        const bridges = new Graphics();
        PREVIEW_BRIDGES.filter((bridge) => bridge.y === y).forEach((bridge) => {
          const horizontal = bridge.direction === "horizontal";
          const x = bridge.x * S + (horizontal ? 0 : 10);
          const top = y * S + (horizontal ? 10 : 0);
          const w = horizontal ? S : 28;
          const h = horizontal ? 28 : S;
          bridges.rect(x, top + h, w, raised ? 6 : 2).fill(0x624530);
          bridges.rect(x, top, w, h).fill(0xba8958).stroke({ color: 0x61432a, width: 2 });
          for (let p = 8; p < S; p += 8) {
            if (horizontal) bridges.moveTo(x + p, top).lineTo(x + p, top + h);
            else bridges.moveTo(x, top + p).lineTo(x + w, top + p);
          }
          bridges.stroke({ color: 0x735133, width: 1 });
          if (bridge.segment !== "middle") {
            const cap = bridge.segment === "start" ? 2 : S - 5;
            bridges.rect(horizontal ? x + cap : x, horizontal ? top : top + cap, horizontal ? 3 : w, horizontal ? h : 3).fill(0xffd281);
          }
        });
        rowLayer.addChild(bridges);
        const obstacles = new Graphics();
        PREVIEW_OBSTACLES.filter((cell) => cell.y === y).forEach((cell) => {
          // Tall silhouette overhangs the preceding row; never defines the hit area.
          const x = cell.x * S + 12;
          const top = y * S - 14;
          obstacles.rect(x + 3, top + 7, 26, 38).fill(0x4b3031);
          obstacles.rect(x, top, 26, 38).fill(0xc27060).stroke({ color: 0x40242d, width: 2 });
        });
        rowLayer.addChild(obstacles);
      });
      const highlight = new Graphics();
      app.stage.addChild(highlight);
      highlights.push(highlight);
    });
    // A single logical hit plane, independent of Tilemap/shape bounds and z-order.
    app.stage.eventMode = "static";
    app.stage.hitArea = new Rectangle(0, 0, WIDTH * 2 + 64, HEIGHT + 72);
    app.stage.on("pointerdown", (event) => {
      for (const highlight of highlights) highlight.clear();
      let label = "Outside logical grid";
      ORIGINS.forEach((origin, index) => {
        const cell = previewCellAt(event.global.x, event.global.y, origin, PREVIEW_TERRAIN[0].length, PREVIEW_TERRAIN.length);
        if (!cell) return;
        highlights[index].rect(origin.x + cell.x * S + 2, origin.y + cell.y * S + 2, S - 4, S - 4).stroke({ color: 0xfff15a, width: 3 });
        label = `${index === 0 ? "Flat" : "Raised"}: cell (${cell.x}, ${cell.y}) — ${PREVIEW_TERRAIN[cell.y][cell.x]}`;
      });
      onCell(label);
      app.render();
    });
    app.render();
    return dispose;
  } catch (error) {
    // init may have allocated a renderer before failing.
    initialized = Boolean(app.renderer);
    dispose();
    throw error;
  }
}
