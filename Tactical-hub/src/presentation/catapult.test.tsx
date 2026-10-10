import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { CatapultAssetFallback, handleCatapultImageError } from "../components/CatapultUnitSprite";
import { UnitToken } from "../components/UnitToken";
import { createInitialGameState } from "../game/initialState";
import type { AttackIntent, Unit } from "../game/types";
import {
  CATAPULT_BREAK_FRAME_COUNT,
  CATAPULT_LOAD_FRAME_COUNT,
  CATAPULT_THROW_FRAME_COUNT,
  catapultDirectionFromDelta,
  createCatapultVisualEvents,
  getCatapultAttackFrames,
  getCatapultBreakFrames,
  getCatapultIdleFrame,
  isBuilderCatapultUnit,
  loadCatapultManifest,
} from "./catapult";

function unit(overrides: Partial<Unit> = {}): Unit {
  return {
    id: "builder",
    ownerTeamId: "team-1",
    type: "strategist",
    role: "builder",
    hp: 1,
    position: { kind: "tile", x: 5, y: 5 },
    statuses: [],
    ...overrides,
  };
}

describe("Catapult animation descriptions", () => {
  it("maps idle to move frame 0000 for every direction", () => {
    for (let direction = 0; direction < 8; direction += 1) {
      const frame = getCatapultIdleFrame(direction as 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7);
      expect(frame).toMatchObject({ animation: "move", frame: 0, direction });
      expect(frame.src).toBe(`/local-assets/catapult/idle/dir${direction}.png`);
    }
  });

  it("keeps all load frames before all throw frames", () => {
    const frames = getCatapultAttackFrames(3);
    expect(frames).toHaveLength(CATAPULT_LOAD_FRAME_COUNT + CATAPULT_THROW_FRAME_COUNT);
    expect(frames.slice(0, 31).map((frame) => frame.animation)).toEqual(
      Array(31).fill("load"),
    );
    expect(frames.slice(31).map((frame) => frame.animation)).toEqual(
      Array(16).fill("throw"),
    );
    expect(frames[30].frame).toBe(30);
    expect(frames[31].frame).toBe(0);
    expect(frames.at(-1)?.frame).toBe(15);
  });

  it("exposes every break frame without affecting game state", () => {
    const frames = getCatapultBreakFrames(7);
    expect(frames).toHaveLength(CATAPULT_BREAK_FRAME_COUNT);
    expect(frames[0].frame).toBe(0);
    expect(frames.at(-1)?.frame).toBe(30);
  });

  it("maps board deltas to the documented asset indices", () => {
    expect([
      catapultDirectionFromDelta(0, 1),
      catapultDirectionFromDelta(1, 1),
      catapultDirectionFromDelta(1, 0),
      catapultDirectionFromDelta(1, -1),
      catapultDirectionFromDelta(0, -1),
      catapultDirectionFromDelta(-1, -1),
      catapultDirectionFromDelta(-1, 0),
      catapultDirectionFromDelta(-1, 1),
    ]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
});

describe("UnitToken Catapult routing", () => {
  it("uses the Catapult component only for builder strategists", () => {
    const builder = renderToStaticMarkup(<UnitToken unit={unit()} />);
    const encourager = renderToStaticMarkup(
      <UnitToken unit={unit({ role: "encourage" })} />,
    );
    const infantry = renderToStaticMarkup(
      <UnitToken unit={unit({ type: "infantry", role: undefined })} />,
    );
    expect(builder).toContain("catapult-sprite-image");
    expect(builder).toContain("/local-assets/catapult/idle/dir0.png");
    expect(encourager).not.toContain("catapult-sprite-image");
    expect(encourager).toContain("帥");
    expect(infantry).not.toContain("catapult-sprite-image");
    expect(infantry).toContain("歩");
    expect(isBuilderCatapultUnit(unit())).toBe(true);
    expect(isBuilderCatapultUnit(unit({ role: "teleporter" }))).toBe(false);
  });

  it("provides a visible fallback and warning when local assets are absent", async () => {
    const warning = vi.fn();
    expect(handleCatapultImageError("/missing.png", warning)).toBe(false);
    expect(warning).toHaveBeenCalledWith("Catapult asset unavailable: /missing.png");
    expect(renderToStaticMarkup(<CatapultAssetFallback>帥</CatapultAssetFallback>))
      .toContain("Catapult asset unavailable");

    const consoleWarning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const missing = await loadCatapultManifest(async () => new Response(null, { status: 404 }));
    expect(missing).toBeUndefined();
    expect(consoleWarning).toHaveBeenCalled();
    consoleWarning.mockRestore();
  });
});

describe("Catapult battle presentation events", () => {
  it("emits attack only for a builder attacker and derives direction", () => {
    const before = createInitialGameState();
    const builder = unit();
    const infantry = unit({ id: "infantry", type: "infantry", role: undefined, position: { kind: "tile", x: 3, y: 3 } });
    const target = unit({ id: "target", type: "infantry", role: undefined, ownerTeamId: "team-2", position: { kind: "tile", x: 7, y: 3 } });
    before.units = [builder, infantry, target];
    const intents: AttackIntent[] = [
      { teamId: "team-1", attackerUnitId: builder.id, target: { kind: "unit", unitId: target.id }, pass: false },
      { teamId: "team-1", attackerUnitId: infantry.id, target: { kind: "unit", unitId: target.id }, pass: false },
    ];
    const events = createCatapultVisualEvents(before, structuredClone(before), intents);
    expect(events).toEqual([{
      unitId: builder.id,
      kind: "attack",
      direction: 3,
      eventId: `battle-${before.turnNumber}:attack:${builder.id}`,
    }]);
  });

  it("emits hit for surviving HP loss and break for removal", () => {
    const before = createInitialGameState();
    before.units = [unit({ hp: 3 })];
    const hitAfter = structuredClone(before);
    hitAfter.units[0].hp = 2;
    expect(createCatapultVisualEvents(before, hitAfter, [])).toMatchObject([
      { unitId: "builder", kind: "hit" },
    ]);

    const breakAfter = structuredClone(before);
    breakAfter.units[0].position = { kind: "removed", reason: "defeated" };
    expect(createCatapultVisualEvents(before, breakAfter, [])).toMatchObject([
      { unitId: "builder", kind: "break" },
    ]);
  });

  it("does not mutate battle states while deriving visual events", () => {
    const before = createInitialGameState();
    before.units = [unit({ hp: 3 })];
    const after = structuredClone(before);
    after.units[0].hp = 2;
    const beforeSnapshot = structuredClone(before);
    const afterSnapshot = structuredClone(after);
    createCatapultVisualEvents(before, after, []);
    expect(before).toEqual(beforeSnapshot);
    expect(after).toEqual(afterSnapshot);
  });
});
