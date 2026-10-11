import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createInitialGameState } from "../game/initialState";
import { getConstructionAt } from "../game/engine/construction";
import { buildPixiBoardTerrain, pixiTerrainKind, relativeCellBounds, TERRAIN_COLORS } from "./pixiBoardTerrain";
import { TileView } from "../components/TileView";
import { BoardView } from "../components/BoardView";

describe("real-map Pixi presentation adapter", () => {
  it("preserves every real-map coordinate and terrain without mutating state", () => {
    const state = createInitialGameState(), before = JSON.stringify(state);
    const cells = buildPixiBoardTerrain(state);
    expect(cells.map(({ x, y }) => ({ x, y }))).toEqual(state.map.tiles.map(({ x, y }) => ({ x, y })));
    for (const [i, cell] of cells.entries()) expect(cell.kind).toBe(pixiTerrainKind(state.map.tiles[i].terrain));
    expect(JSON.stringify(state)).toBe(before);
  });
  it("shares road appearance for gates/reorganize; keeps base, lake and outside distinct", () => {
    expect(["road", "baseGate", "reorganize"].map((kind) => pixiTerrainKind(kind as "road"))).toEqual(["road", "road", "road"]);
    expect(pixiTerrainKind("base")).toBe("base");
    expect(pixiTerrainKind("lake")).toBe("lake");
    expect(pixiTerrainKind("outside")).toBeNull();
    expect(new Set(Object.values(TERRAIN_COLORS)).size).toBe(3);
  });
  it("matches BoardView active construction lookup, including reset constructions", () => {
    const state = createInitialGameState();
    const horizontal = [{ x: 4, y: 3 }, { x: 5, y: 3 }, { x: 6, y: 3 }];
    const vertical = [{ x: 9, y: 4 }, { x: 9, y: 3 }, { x: 9, y: 2 }];
    state.constructions = [
      { id: "h", kind: "bridge", active: true, placedTurn: 1, tiles: horizontal },
      { id: "v", kind: "bridge", active: true, placedTurn: 1, tiles: vertical },
      { id: "o", kind: "obstacle", active: true, placedTurn: 1, tiles: [horizontal[1]] },
      { id: "reset", kind: "obstacle", active: false, placedTurn: 1, tiles: [horizontal[0]] },
    ];
    const cells = buildPixiBoardTerrain(state);
    for (const cell of cells) {
      expect(Boolean(cell.bridge)).toBe(Boolean(getConstructionAt(state, cell.x, cell.y, "bridge")));
      expect(cell.obstacle).toBe(Boolean(getConstructionAt(state, cell.x, cell.y, "obstacle")));
    }
    expect(horizontal.map((p) => cells.find((c) => c.x === p.x && c.y === p.y)?.bridge)).toEqual([
      { direction: "horizontal", segment: "start" }, { direction: "horizontal", segment: "middle" }, { direction: "horizontal", segment: "end" },
    ]);
    expect([...vertical].reverse().map((p) => cells.find((c) => c.x === p.x && c.y === p.y)?.bridge)).toEqual([
      { direction: "vertical", segment: "start" }, { direction: "vertical", segment: "middle" }, { direction: "vertical", segment: "end" },
    ]);
  });
  it("retains lake terrain below bridges and suppresses shared road-height faces", () => {
    const state = createInitialGameState();
    const tiles = state.map.tiles;
    tiles[0] = { ...tiles[0], x: 0, y: 0, terrain: "road" };
    tiles[1] = { ...tiles[1], x: 0, y: 1, terrain: "baseGate" };
    tiles[2] = { ...tiles[2], x: 0, y: 2, terrain: "lake" };
    state.map.tiles = tiles.slice(0, 3);
    state.constructions = [{ id: "bridge", kind: "bridge", active: true, placedTurn: 1, tiles: [{ x: 0, y: 2 }] }];
    const cells = buildPixiBoardTerrain(state);
    expect(cells[0].southFace).toBe(false);
    expect(cells[1].southFace).toBe(true);
    expect(cells[2].kind).toBe("lake");
    expect(cells[2].bridge).not.toBeNull();
  });
  it("synchronizes fractional rectangles and cancels scroll offsets", () => {
    const cell = { x: 119.25, y: 208.5, width: 30.125, height: 30.125 };
    const board = { x: 20.5, y: 40.25 };
    const expected = { x: 98.75, y: 168.25, width: 30.125, height: 30.125 };
    expect(relativeCellBounds(cell, board)).toEqual(expected);
    expect(relativeCellBounds({ ...cell, x: cell.x - 50, y: cell.y - 90 }, { x: board.x - 50, y: board.y - 90 })).toEqual(expected);
  });
  it("keeps logical button coordinates, callbacks, and interaction highlights", () => {
    const tile = createInitialGameState().map.tiles[0];
    let clicks = 0;
    const element = TileView({ tile, highlighted: true, attackHighlighted: true, constructionPreview: true, bridgeCandidateMarker: true, onClick: () => clicks++ });
    element.props.onClick();
    expect(clicks).toBe(1);
    const html = renderToStaticMarkup(element);
    expect(html).toContain(`data-board-x="${tile.x}"`);
    expect(html).toContain(`data-board-y="${tile.y}"`);
    for (const name of ["highlighted", "attack-highlighted", "construction-preview", "bridge-candidate-marker"]) expect(html).toContain(name);
  });
  it("starts BoardView in CSS mode with no Canvas mounted", () => {
    const html = renderToStaticMarkup(<BoardView state={createInitialGameState()} manualTeamId="team-1"
      onSelectUnit={() => {}} onChooseDestination={() => {}} onChooseAttackTarget={() => {}} onChooseConstruction={() => {}} />);
    expect(html).not.toContain("board-pixi");
    expect(html).not.toContain("pixi-board-canvas");
    expect(html).toContain('class="board"');
  });
});
