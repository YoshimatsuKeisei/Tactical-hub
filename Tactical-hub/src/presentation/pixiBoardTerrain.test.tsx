import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createInitialGameState } from "../game/initialState";
import { getConstructionAt } from "../game/engine/construction";
import {
  applyMoveHighlights, baseFlagAssetKey, baseGroupBounds, buildPixiBaseGroups, buildPixiBoardTerrain, getMoveHighlightStyle,
  isRaisedRoadTerrain, isTwoByTwoBase, pixiTerrainKind, relativeCellBounds, shouldShowRoadSouthFace, TERRAIN_COLORS,
} from "./pixiBoardTerrain";
import { DEFAULT_MOVE_HIGHLIGHT } from "./tileAssets";
import { TileView } from "../components/TileView";
import { BoardView } from "../components/BoardView";
import { UnitToken } from "../components/UnitToken";
import { PixiBoardTerrain } from "../components/PixiBoardTerrain";

describe("real-map Pixi presentation adapter", () => {
  it("preserves every real-map coordinate and terrain without mutating state", () => {
    const state = createInitialGameState(), before = JSON.stringify(state);
    const cells = buildPixiBoardTerrain(state);
    expect(cells.map(({ x, y }) => ({ x, y }))).toEqual(state.map.tiles.map(({ x, y }) => ({ x, y })));
    for (const [i, cell] of cells.entries()) expect(cell.kind).toBe(pixiTerrainKind(state.map.tiles[i].terrain));
    expect(JSON.stringify(state)).toBe(before);
  });
  it("groups every current 2x2 base once at its top-left cell", () => {
    const state = createInitialGameState();
    const groups = buildPixiBaseGroups(state);
    const cells = buildPixiBoardTerrain(state);
    expect(groups).toHaveLength(state.bases.length);
    for (const base of state.bases) {
      expect(isTwoByTwoBase(base.coords)).toBe(true);
      const group = groups.find((candidate) => candidate.id === base.id)!;
      expect(group.anchorX).toBe(Math.min(...base.coords.map((coord) => coord.x)));
      expect(group.anchorY).toBe(Math.min(...base.coords.map((coord) => coord.y)));
      expect(cells.filter((cell) => cell.baseGroup?.id === base.id)).toHaveLength(1);
      const groupedCells = cells.filter((cell) => cell.baseGroupId === base.id);
      expect(groupedCells).toHaveLength(4);
      expect(groupedCells.every((cell) => cell.kind === "base")).toBe(true);
      expect(base.coords.every((coord) => state.map.tiles.find((tile) => tile.x === coord.x && tile.y === coord.y)?.terrain === "base")).toBe(true);
    }
  });
  it("maps flags from the actual non-neutral team order and switches neutral ownership presentation", () => {
    const state = createInitialGameState();
    const reordered = { teams: [state.teams[2], state.teams[0], state.teams[3], state.teams[1], state.teams[4]] };
    expect(baseFlagAssetKey(reordered, "team-3")).toBe("base_flag_team_1");
    expect(baseFlagAssetKey(reordered, "team-1")).toBe("base_flag_team_2");
    expect(baseFlagAssetKey(reordered, "neutral")).toBeUndefined();

    const neutral = state.bases.find((base) => base.type === "neutral")!;
    let group = buildPixiBaseGroups(state).find((candidate) => candidate.id === neutral.id)!;
    expect(group).toMatchObject({ thirdParty: true, coreAssetKey: "base_core_third_party", flagAssetKey: undefined });
    neutral.ownerTeamId = "team-4";
    group = buildPixiBaseGroups(state).find((candidate) => candidate.id === neutral.id)!;
    expect(group).toMatchObject({ thirdParty: false, coreAssetKey: "base", flagAssetKey: "base_flag_team_4" });
  });
  it("leaves non-2x2 bases on the compatible per-cell rendering path", () => {
    const state = createInitialGameState();
    const base = state.bases[0];
    base.coords = base.coords.slice(0, 3);
    expect(isTwoByTwoBase(base.coords)).toBe(false);
    expect(buildPixiBaseGroups(state).some((group) => group.id === base.id)).toBe(false);
    expect(buildPixiBoardTerrain(state).filter((cell) => cell.baseGroupId === base.id)).toHaveLength(0);
  });
  it("measures a grouped base from all four live HTML cell rectangles", () => {
    const group = buildPixiBaseGroups(createInitialGameState())[0];
    const bounds = new Map(group.coords.map((coord) => [
      `${coord.x},${coord.y}`,
      { x: 10 + (coord.x - group.anchorX) * 31.5, y: 20 + (coord.y - group.anchorY) * 29.25, width: 31.5, height: 29.25 },
    ]));
    expect(baseGroupBounds(group, bounds)).toEqual({ x: 10, y: 20, width: 63, height: 58.5 });
    bounds.delete(`${group.anchorX + 1},${group.anchorY + 1}`);
    expect(baseGroupBounds(group, bounds)).toBeUndefined();
  });
  it("shares road appearance for gates/reorganize; keeps base, lake and outside distinct", () => {
    expect(["road", "baseGate", "reorganize"].map((kind) => pixiTerrainKind(kind as "road"))).toEqual(["road", "road", "road"]);
    expect(pixiTerrainKind("base")).toBe("base");
    expect(pixiTerrainKind("lake")).toBe("lake");
    expect(pixiTerrainKind("outside")).toBeNull();
    expect(new Set(Object.values(TERRAIN_COLORS)).size).toBe(3);
  });
  it("shows a south face only where the shared raised terrain drops", () => {
    for (const terrain of ["road", "base", "baseGate", "reorganize"] as const) {
      expect(isRaisedRoadTerrain(terrain)).toBe(true);
      expect(shouldShowRoadSouthFace("road", terrain)).toBe(false);
    }
    expect(isRaisedRoadTerrain("lake")).toBe(false);
    expect(shouldShowRoadSouthFace("road", "lake")).toBe(true);
    expect(shouldShowRoadSouthFace("base", "lake")).toBe(true);
    expect(shouldShowRoadSouthFace("road", "outside")).toBe(true);
    expect(shouldShowRoadSouthFace("road")).toBe(true);
    expect(shouldShowRoadSouthFace("lake", "outside")).toBe(false);
  });
  it("uses a darkened team color for normal units and team-independent gray for ninja", () => {
    const normal = getMoveHighlightStyle({ type: "cavalry" }, "#d94a4a", DEFAULT_MOVE_HIGHLIGHT);
    expect(normal.color).toBe(0x9c3535);
    expect(normal.cssColor).toBe("#9c3535");
    expect(normal.outlineColor).toBe(0xbf7c7c);
    expect(normal.outlineOpacity).toBe(0.6);
    const ninjaA = getMoveHighlightStyle({ type: "ninja" }, "#d94a4a", DEFAULT_MOVE_HIGHLIGHT);
    const ninjaB = getMoveHighlightStyle({ type: "ninja" }, "#3e7bd8", DEFAULT_MOVE_HIGHLIGHT);
    expect(ninjaA.color).toBe(0x303030);
    expect(ninjaA.outlineColor).toBe(0x787878);
    expect(ninjaB).toEqual(ninjaA);
    const apprentice = getMoveHighlightStyle({ type: "apprentice_ninja" }, "#d94a4a", DEFAULT_MOVE_HIGHLIGHT);
    expect(apprentice.color).toBe(normal.color);
    expect(apprentice.outlineColor).toBe(normal.outlineColor);
    const adjusted = getMoveHighlightStyle({ type: "cavalry" }, "#d94a4a", {
      ...DEFAULT_MOVE_HIGHLIGHT,
      normal: { ...DEFAULT_MOVE_HIGHLIGHT.normal, outlineEnabled: false, outlineOpacity: 0.2, outlineLighten: 0 },
    });
    expect(adjusted).toMatchObject({ outlineEnabled: false, outlineOpacity: 0.2, outlineColor: adjusted.color });
  });
  it("adds overlays only to legal keys without mutating terrain and removes them when disabled", () => {
    const state = createInitialGameState(), original = buildPixiBoardTerrain(state);
    const style = getMoveHighlightStyle({ type: "cavalry" }, "#d94a4a", DEFAULT_MOVE_HIGHLIGHT);
    const key = `${original[0].x},${original[0].y}`;
    const highlighted = applyMoveHighlights(original, new Set([key]), style);
    expect(highlighted[0].moveHighlight).toBe(style);
    expect(highlighted.slice(1).every((cell) => cell.moveHighlight === undefined)).toBe(true);
    expect(original[0]).not.toHaveProperty("moveHighlight");
    const disabled = applyMoveHighlights(original, new Set([key]), { ...style, enabled: false });
    expect(disabled).toBe(original);
    expect(applyMoveHighlights(original, new Set(), style)).toBe(original);
  });
  it("marks a movement-selected unit with the resolved presentation color", () => {
    const html = renderToStaticMarkup(<UnitToken
      unit={{ id: "u", ownerTeamId: "team-1", type: "ninja", hp: 1, position: { kind: "tile", x: 0, y: 0 }, statuses: [] }}
      team={{ id: "team-1", name: "Red", color: "#d94a4a", status: "active", controlledBaseIds: [] }}
      selected movementSelectionColor="#303030"
    />);
    expect(html).toContain("movement-selected");
    expect(html).toContain("--movement-selection-color:#303030");
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
  it("retains lake terrain below bridges and applies south faces from the map adapter", () => {
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
    expect(cells[2].southFace).toBe(false);
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
    expect(html).not.toContain("pixi-board-foreground");
    expect(html).toContain('class="board"');
  });
  it("mounts separate click-through terrain and base-foreground hosts", () => {
    const html = renderToStaticMarkup(<PixiBoardTerrain
      cells={[]}
      boardRef={{ current: null }}
      onError={() => {}}
    />);
    expect(html).toContain("pixi-board-canvas");
    expect(html).toContain("pixi-board-foreground");
    expect(html.match(/aria-hidden="true"/g)).toHaveLength(2);
  });
});
