import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  HdEnemyAssetFallback,
  HdEnemySpriteFrame,
  handleHdEnemyAssetError,
} from "../components/HdEnemyUnitSprite";
import { UnitToken } from "../components/UnitToken";
import type { Unit } from "../game/types";
import {
  HD_ENEMY_ATTACK_FRAME_MS,
  HD_ENEMY_DIRECTION_ROWS,
  HD_ENEMY_DIE_FRAME_MS,
  HD_ENEMY_HIT_FRAME_MS,
  HD_ENEMY_IDLE_FRAME_MS,
  advanceHdEnemyFrame,
  getHdEnemyAnimationQueue,
  getHdEnemyBackgroundPosition,
  getHdEnemyCharacterForUnit,
  getHdEnemyDirectionRow,
  getHdEnemyFrameDuration,
  getHdEnemySheetUrl,
  loadHdEnemyManifest,
} from "./hdEnemy";

function unit(type: Unit["type"], role?: Unit["role"]): Unit {
  return {
    id: `${type}-${role ?? "unit"}`,
    ownerTeamId: "team-1",
    type,
    role,
    hp: 1,
    position: { kind: "tile", x: 2, y: 2 },
    statuses: [],
  };
}

describe("HD Enemy sprite descriptions", () => {
  it("maps all game directions to the documented asset rows", () => {
    expect(HD_ENEMY_DIRECTION_ROWS).toEqual([6, 7, 0, 1, 2, 3, 4, 5]);
    expect(Array.from({ length: 8 }, (_, direction) => getHdEnemyDirectionRow(direction as 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7)))
      .toEqual([6, 7, 0, 1, 2, 3, 4, 5]);
  });

  it("computes exact first/last background positions for a 15x8 sheet", () => {
    expect(getHdEnemyBackgroundPosition(0, 2)).toEqual({ xPercent: 0, yPercent: 0 });
    expect(getHdEnemyBackgroundPosition(14, 1)).toEqual({ xPercent: 100, yPercent: 100 });
    expect(() => getHdEnemyBackgroundPosition(15, 0)).toThrow("frame must be 0-14");
  });

  it("uses the four exact purchased shadow-sheet filenames", () => {
    expect(getHdEnemySheetUrl("6Crusader", "idle")).toBe("/local-assets/hd-enemy/6Crusader/Idle.png");
    expect(getHdEnemySheetUrl("6Crusader", "attack")).toBe("/local-assets/hd-enemy/6Crusader/Attack1.png");
    expect(getHdEnemySheetUrl("10Caster", "hit")).toBe("/local-assets/hd-enemy/10Caster/TakeDamage.png");
    expect(getHdEnemySheetUrl("10Caster", "die")).toBe("/local-assets/hd-enemy/10Caster/Die.png");
  });

  it("keeps provisional timings named and state-specific", () => {
    expect(getHdEnemyFrameDuration("idle")).toBe(HD_ENEMY_IDLE_FRAME_MS);
    expect(getHdEnemyFrameDuration("attack")).toBe(HD_ENEMY_ATTACK_FRAME_MS);
    expect(getHdEnemyFrameDuration("hit")).toBe(HD_ENEMY_HIT_FRAME_MS);
    expect(getHdEnemyFrameDuration("die")).toBe(HD_ENEMY_DIE_FRAME_MS);
  });

  it("loops all 15 Idle frames", () => {
    expect(advanceHdEnemyFrame("idle", 13)).toEqual({ frame: 14, completed: false });
    expect(advanceHdEnemyFrame("idle", 14)).toEqual({ frame: 0, completed: false });
  });

  it("completes Attack, TakeDamage, and Die once after frame 14", () => {
    for (const animation of ["attack", "hit", "die"] as const) {
      expect(advanceHdEnemyFrame(animation, 13)).toEqual({ frame: 14, completed: false });
      expect(advanceHdEnemyFrame(animation, 14)).toEqual({ frame: 0, completed: true });
    }
  });

  it("queues attack/hit once and expands death to TakeDamage then Die", () => {
    const queue = getHdEnemyAnimationQueue([
      { unitId: "u", eventId: "attack", kind: "attack", direction: 3 },
      { unitId: "u", eventId: "hit", kind: "hit", direction: 4 },
      { unitId: "u", eventId: "death", kind: "death", direction: 5 },
    ]);
    expect(queue.map((entry) => entry.animation)).toEqual(["attack", "hit", "hit", "die"]);
    expect(queue.slice(2).map((entry) => entry.eventId)).toEqual(["death:damage", "death:die"]);
  });

  it("renders one 128px-frame window via CSS background positioning", () => {
    const markup = renderToStaticMarkup(
      <HdEnemySpriteFrame character="6Crusader" animation="attack" direction={7} frame={14} />,
    );
    expect(markup).toContain("data-character=\"6Crusader\"");
    expect(markup).toContain("background-size:1500% 800%");
    expect(markup).toContain("background-position:100% 71.42857142857143%");
  });
});

describe("HD Enemy UnitToken routing", () => {
  it("maps king to 6Crusader", () => {
    const king = unit("king");
    expect(getHdEnemyCharacterForUnit(king)).toBe("6Crusader");
    const markup = renderToStaticMarkup(<UnitToken unit={king} />);
    expect(markup).toContain("data-character=\"6Crusader\"");
    expect(markup).not.toContain(">王<");
  });

  it("maps all strategist roles to 10Caster", () => {
    for (const role of ["builder", "encourage", "teleporter"] as const) {
      const strategist = unit("strategist", role);
      expect(getHdEnemyCharacterForUnit(strategist)).toBe("10Caster");
      const markup = renderToStaticMarkup(<UnitToken unit={strategist} />);
      expect(markup).toContain("data-character=\"10Caster\"");
      expect(markup).not.toContain("catapult-sprite-image");
    }
  });

  it("keeps engineer on Catapult and other units on text", () => {
    expect(renderToStaticMarkup(<UnitToken unit={unit("engineer")} />)).toContain("catapult-sprite-image");
    for (const type of ["infantry", "cavalry", "archer", "ninja", "apprentice_ninja"] as const) {
      const markup = renderToStaticMarkup(<UnitToken unit={unit(type)} />);
      expect(markup).not.toContain("hd-enemy-sprite-image");
      expect(markup).not.toContain("catapult-sprite-image");
    }
  });

  it("retains title, aria label, team badge, HP badge, and click target", () => {
    const king = unit("king");
    king.hp = 3;
    const markup = renderToStaticMarkup(<UnitToken unit={king} team={{
      id: "team-1", name: "Blue", color: "#123456", status: "active", controlledBaseIds: [],
    }} />);
    expect(markup).toContain("title=\"Blue king HP:3\"");
    expect(markup).toContain("aria-label=\"Blue king HP:3\"");
    expect(markup).toContain("unit-sprite-team-badge");
    expect(markup).toContain("hp-badge");
    expect(markup).toContain("<button");
  });

  it("provides a visible fallback and a deduplicated asset warning", async () => {
    const warning = vi.fn();
    expect(handleHdEnemyAssetError("/missing-hd.png", warning)).toBe(false);
    expect(handleHdEnemyAssetError("/missing-hd.png", warning)).toBe(false);
    expect(warning).toHaveBeenCalledTimes(1);
    expect(renderToStaticMarkup(<HdEnemyAssetFallback>王</HdEnemyAssetFallback>))
      .toContain("HD Enemy asset unavailable");

    const consoleWarning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const missing = await loadHdEnemyManifest(async () => new Response(null, { status: 404 }));
    expect(missing).toBeUndefined();
    expect(consoleWarning).toHaveBeenCalled();
    consoleWarning.mockRestore();
  });
});
