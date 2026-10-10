import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  HdCharacterAssetFallback,
  HdCharacterSpriteFrame,
  handleHdCharacterAssetError,
} from "../components/HdCharacterUnitSprite";
import { UnitToken } from "../components/UnitToken";
import { mergeHeavyInfantry } from "../game/engine/heavyInfantry";
import { createInitialGameState } from "../game/initialState";
import type { Unit } from "../game/types";
import {
  DIRECTIONAL_SPRITE_DIRECTION_ROWS,
  advanceDirectionalSpriteFrame,
  getDirectionalAnimationQueue,
} from "./directionalSpriteSheet";
import {
  HD_CHARACTER_CHARACTERS,
  HD_CHARACTER_STATE_SHEETS,
  getHdCharacterForUnit,
  getHdCharacterSheetUrl,
  loadHdCharacterManifest,
} from "./hdCharacter";
import { createUnitVisualPresentation, getUnitSpriteKind } from "./unitVisualEvents";

function unit(type: Unit["type"], overrides: Partial<Unit> = {}): Unit {
  return {
    id: `${type}-unit`,
    ownerTeamId: "team-1",
    type,
    hp: 1,
    position: { kind: "tile", x: 3, y: 3 },
    statuses: [],
    ...overrides,
  };
}

describe("HD Character mapping", () => {
  it("prioritizes heavy infantry before normal infantry", () => {
    expect(getHdCharacterForUnit(unit("infantry"))).toBe("1Knight");
    expect(getHdCharacterForUnit(unit("infantry", { formation: "heavy", hp: 2 }))).toBe("4Paladin");
  });

  it("maps archer and ninja but not cavalry or apprentice ninja", () => {
    expect(getHdCharacterForUnit(unit("archer"))).toBe("2Archer");
    expect(getHdCharacterForUnit(unit("ninja"))).toBe("7DeathKnight");
    expect(getHdCharacterForUnit(unit("cavalry"))).toBeUndefined();
    expect(getHdCharacterForUnit(unit("apprentice_ninja"))).toBeUndefined();
  });

  it("keeps existing king, strategist, and engineer sprite dispatch", () => {
    expect(getUnitSpriteKind(unit("king"))).toBe("6Crusader");
    expect(getUnitSpriteKind(unit("strategist", { role: "builder" }))).toBe("10Caster");
    expect(getUnitSpriteKind(unit("engineer"))).toBe("catapult");
  });

  it("switches the retained infantry visual to Paladin after the existing merge", () => {
    const state = createInitialGameState();
    const base = state.bases.find((candidate) => candidate.id === "home-1")!;
    const firstSlot = base.slots.find((slot) => slot.id === "slot_0_1")!;
    const secondSlot = base.slots.find((slot) => slot.id === "slot_1_1")!;
    const first = unit("infantry", { id: "visual-merge-a", position: { kind: "base", baseId: base.id, slotId: firstSlot.id } });
    const second = unit("infantry", { id: "visual-merge-b", position: { kind: "base", baseId: base.id, slotId: secondSlot.id } });
    firstSlot.unitId = first.id;
    secondSlot.unitId = second.id;
    state.units.push(first, second);
    expect(getUnitSpriteKind(first)).toBe("1Knight");
    const merged = mergeHeavyInfantry(state, first.id, second.id);
    const retained = merged.units.find((candidate) => candidate.id === first.id)!;
    expect(retained).toMatchObject({ type: "infantry", formation: "heavy", hp: 2 });
    expect(getUnitSpriteKind(retained)).toBe("4Paladin");
    expect(state.units.find((candidate) => candidate.id === first.id)?.formation).toBeUndefined();
  });
});

describe("HD Character animation config", () => {
  it("uses Melee for Knight, Paladin, DeathKnight and Attack1 for Archer", () => {
    expect(HD_CHARACTER_STATE_SHEETS["1Knight"].attack).toBe("Melee.png");
    expect(HD_CHARACTER_STATE_SHEETS["2Archer"].attack).toBe("Attack1.png");
    expect(HD_CHARACTER_STATE_SHEETS["4Paladin"].attack).toBe("Melee.png");
    expect(HD_CHARACTER_STATE_SHEETS["7DeathKnight"].attack).toBe("Melee.png");
  });

  it("uses the shared direction mapping and 15-frame loop/one-shot behavior", () => {
    expect(DIRECTIONAL_SPRITE_DIRECTION_ROWS).toEqual([6, 7, 0, 1, 2, 3, 4, 5]);
    expect(advanceDirectionalSpriteFrame("idle", 14)).toEqual({ frame: 0, completed: false });
    for (const animation of ["attack", "hit", "die"] as const) {
      expect(advanceDirectionalSpriteFrame(animation, 14)).toEqual({ frame: 0, completed: true });
    }
  });

  it("queues death as TakeDamage then Die", () => {
    expect(getDirectionalAnimationQueue([{
      unitId: "knight", eventId: "death", kind: "death", direction: 2,
    }]).map((entry) => entry.animation)).toEqual(["hit", "die"]);
  });

  it("builds the exact local URLs and renders one background window", () => {
    expect(getHdCharacterSheetUrl("1Knight", "attack")).toBe("/local-assets/hd-character/1Knight/Melee.png");
    expect(getHdCharacterSheetUrl("2Archer", "attack")).toBe("/local-assets/hd-character/2Archer/Attack1.png");
    const markup = renderToStaticMarkup(
      <HdCharacterSpriteFrame character="4Paladin" animation="attack" direction={1} frame={14} />,
    );
    expect(markup).toContain("data-asset-pack=\"hd-character\"");
    expect(markup).toContain("data-character=\"4Paladin\"");
    expect(markup).toContain("background-size:1500% 800%");
    expect(markup).toContain("background-position:100% 100%");
  });

  it("derives attack direction from attacker to target", () => {
    const before = createInitialGameState();
    const knight = unit("infantry", { id: "knight", position: { kind: "tile", x: 3, y: 3 } });
    const target = unit("cavalry", { id: "target", ownerTeamId: "team-2", position: { kind: "tile", x: 4, y: 2 } });
    before.units = [knight, target];
    const presentation = createUnitVisualPresentation(before, structuredClone(before), [{
      teamId: "team-1", attackerUnitId: knight.id, target: { kind: "unit", unitId: target.id }, pass: false,
    }]);
    expect(presentation.events).toMatchObject([{ unitId: "knight", kind: "attack", direction: 3 }]);
  });
});

describe("HD Character UnitToken routing and fallback", () => {
  it("renders the four mapped characters and preserves the heavy badge", () => {
    const fixtures = [
      [unit("infantry"), "1Knight"],
      [unit("archer"), "2Archer"],
      [unit("infantry", { formation: "heavy", hp: 2 }), "4Paladin"],
      [unit("ninja"), "7DeathKnight"],
    ] as const;
    for (const [entry, character] of fixtures) {
      const markup = renderToStaticMarkup(<UnitToken unit={entry} />);
      expect(markup).toContain(`data-character=\"${character}\"`);
      if (character === "4Paladin") expect(markup).toContain("formation-badge");
    }
  });

  it("preserves UnitToken interaction classes, labels, HP, formation, and retreat badges", () => {
    const paladin = unit("infantry", {
      formation: "heavy",
      hp: 2,
      statuses: [{ kind: "retreating", retreatTargetBaseId: "home-1" }],
    });
    const markup = renderToStaticMarkup(<UnitToken
      unit={paladin}
      team={{ id: "team-1", name: "Blue", color: "#123456", status: "active", controlledBaseIds: [] }}
      selected
      attackTarget
      attackReady
      attackComplete
    />);
    expect(markup).toContain("selected");
    expect(markup).toContain("attack-target");
    expect(markup).toContain("attack-ready");
    expect(markup).toContain("attack-complete");
    expect(markup).toContain("title=\"Blue infantry HP:2\"");
    expect(markup).toContain("aria-label=\"Blue infantry HP:2\"");
    expect(markup).toContain("unit-sprite-team-badge");
    expect(markup).toContain("hp-badge");
    expect(markup).toContain("formation-badge");
    expect(markup).toContain("retreat-badge");
  });

  it("keeps cavalry and apprentice ninja on their legacy labels", () => {
    expect(renderToStaticMarkup(<UnitToken unit={unit("cavalry")} />)).toContain("馬");
    expect(renderToStaticMarkup(<UnitToken unit={unit("apprentice_ninja")} />)).toContain("見");
  });

  it("creates a presentation-only death overlay for a Character-pack unit", () => {
    const before = createInitialGameState();
    before.units = [unit("archer", { id: "archer", position: { kind: "tile", x: 7, y: 8 } })];
    const after = structuredClone(before);
    after.units[0].position = { kind: "removed", reason: "defeated" };
    expect(createUnitVisualPresentation(before, after, []).deathOverlays).toMatchObject([{
      unit: { id: "archer", type: "archer" },
      coord: { x: 7, y: 8 },
      events: [{ kind: "death" }],
    }]);
  });

  it("offers a deduplicated missing-asset warning and text fallback", async () => {
    const warning = vi.fn();
    expect(handleHdCharacterAssetError("/missing-character.png", warning)).toBe(false);
    expect(handleHdCharacterAssetError("/missing-character.png", warning)).toBe(false);
    expect(warning).toHaveBeenCalledTimes(1);
    expect(renderToStaticMarkup(<HdCharacterAssetFallback>弓</HdCharacterAssetFallback>))
      .toContain("HD Character asset unavailable");

    const consoleWarning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(await loadHdCharacterManifest(async () => new Response(null, { status: 404 }))).toBeUndefined();
    expect(consoleWarning).toHaveBeenCalled();
    consoleWarning.mockRestore();
  });

  it("exposes all four preview characters", () => {
    expect(HD_CHARACTER_CHARACTERS).toEqual(["1Knight", "2Archer", "4Paladin", "7DeathKnight"]);
  });
});
