import type { Unit } from "../game/types";
import {
  DIRECTIONAL_ATTACK_FRAME_MS,
  DIRECTIONAL_DIE_FRAME_MS,
  DIRECTIONAL_HIT_FRAME_MS,
  DIRECTIONAL_IDLE_FRAME_MS,
  DIRECTIONAL_SPRITE_COLUMNS,
  DIRECTIONAL_SPRITE_DIRECTION_ROWS,
  DIRECTIONAL_SPRITE_FRAME_HEIGHT,
  DIRECTIONAL_SPRITE_FRAME_WIDTH,
  DIRECTIONAL_SPRITE_ROWS,
  type DirectionalAnimationState,
} from "./directionalSpriteSheet";
import { getUnitSpriteKind } from "./unitVisualEvents";

export const HD_CHARACTER_CHARACTERS = ["1Knight", "2Archer", "4Paladin", "7DeathKnight"] as const;
export type HdCharacter = typeof HD_CHARACTER_CHARACTERS[number];
export type HdCharacterAnimationState = DirectionalAnimationState;

export const HD_CHARACTER_IDLE_FRAME_MS = DIRECTIONAL_IDLE_FRAME_MS;
export const HD_CHARACTER_ATTACK_FRAME_MS = DIRECTIONAL_ATTACK_FRAME_MS;
export const HD_CHARACTER_HIT_FRAME_MS = DIRECTIONAL_HIT_FRAME_MS;
export const HD_CHARACTER_DIE_FRAME_MS = DIRECTIONAL_DIE_FRAME_MS;

export const HD_CHARACTER_STATE_SHEETS: Record<
  HdCharacter,
  Record<HdCharacterAnimationState, string>
> = {
  "1Knight": { idle: "Idle.png", attack: "Melee.png", hit: "TakeDamage.png", die: "Die.png" },
  "2Archer": { idle: "Idle.png", attack: "Attack1.png", hit: "TakeDamage.png", die: "Die.png" },
  "4Paladin": { idle: "Idle.png", attack: "Melee.png", hit: "TakeDamage.png", die: "Die.png" },
  "7DeathKnight": { idle: "Idle.png", attack: "Melee.png", hit: "TakeDamage.png", die: "Die.png" },
};

export const HD_CHARACTER_UNIT_MAPPING = {
  normalInfantry: "1Knight",
  archer: "2Archer",
  heavyInfantry: "4Paladin",
  ninja: "7DeathKnight",
} as const;

export type HdCharacterManifest = {
  version: 1;
  sourceFps: "unverified";
  frameWidth: 128;
  frameHeight: 128;
  columns: 15;
  rows: 8;
  directionRows: readonly number[];
  unitMapping: typeof HD_CHARACTER_UNIT_MAPPING;
  characters: Record<HdCharacter, { sheets: Record<HdCharacterAnimationState, string> }>;
};

export function getHdCharacterForUnit(unit: Unit): HdCharacter | undefined {
  const kind = getUnitSpriteKind(unit);
  return HD_CHARACTER_CHARACTERS.find((character) => character === kind);
}

export function getHdCharacterSheetUrl(
  character: HdCharacter,
  animation: HdCharacterAnimationState,
) {
  return `/local-assets/hd-character/${character}/${HD_CHARACTER_STATE_SHEETS[character][animation]}`;
}

function isHdCharacterManifest(value: unknown): value is HdCharacterManifest {
  if (!value || typeof value !== "object") return false;
  const manifest = value as Partial<HdCharacterManifest>;
  return manifest.version === 1
    && manifest.sourceFps === "unverified"
    && manifest.frameWidth === DIRECTIONAL_SPRITE_FRAME_WIDTH
    && manifest.frameHeight === DIRECTIONAL_SPRITE_FRAME_HEIGHT
    && manifest.columns === DIRECTIONAL_SPRITE_COLUMNS
    && manifest.rows === DIRECTIONAL_SPRITE_ROWS
    && Array.isArray(manifest.directionRows)
    && manifest.directionRows.join(",") === DIRECTIONAL_SPRITE_DIRECTION_ROWS.join(",")
    && JSON.stringify(manifest.unitMapping) === JSON.stringify(HD_CHARACTER_UNIT_MAPPING)
    && HD_CHARACTER_CHARACTERS.every((character) => (
      manifest.characters?.[character]
      && Object.entries(HD_CHARACTER_STATE_SHEETS[character]).every(([state, filename]) => (
        manifest.characters?.[character]?.sheets[state as HdCharacterAnimationState] === filename
      ))
    ));
}

export async function loadHdCharacterManifest(
  fetcher: typeof fetch = fetch,
): Promise<HdCharacterManifest | undefined> {
  try {
    const response = await fetcher("/local-assets/hd-character/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest: unknown = await response.json();
    if (!isHdCharacterManifest(manifest)) throw new Error("manifest schema mismatch");
    return manifest;
  } catch (error) {
    console.warn("HD Character asset unavailable", error);
    return undefined;
  }
}
