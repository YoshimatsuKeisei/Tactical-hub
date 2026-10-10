import type { Unit } from "../game/types";
import {
  getUnitSpriteKind,
  type UnitDirection,
  type UnitVisualEvent,
} from "./unitVisualEvents";
import {
  DIRECTIONAL_ATTACK_FRAME_MS,
  DIRECTIONAL_DIE_FRAME_MS,
  DIRECTIONAL_HIT_FRAME_MS,
  DIRECTIONAL_IDLE_FRAME_MS,
  DIRECTIONAL_SPRITE_COLUMNS,
  DIRECTIONAL_SPRITE_DIRECTION_ROWS,
  DIRECTIONAL_SPRITE_FRAME_COUNT,
  DIRECTIONAL_SPRITE_FRAME_HEIGHT,
  DIRECTIONAL_SPRITE_FRAME_WIDTH,
  DIRECTIONAL_SPRITE_ROWS,
  advanceDirectionalSpriteFrame,
  getDirectionalAnimationQueue,
  getDirectionalSpriteBackgroundPosition,
  getDirectionalSpriteFrameDuration,
  getDirectionalSpriteRow,
  type DirectionalAnimationQueueEntry,
  type DirectionalAnimationState,
} from "./directionalSpriteSheet";

export const HD_ENEMY_CHARACTERS = ["6Crusader", "10Caster"] as const;
export type HdEnemyCharacter = typeof HD_ENEMY_CHARACTERS[number];
export type HdEnemyAnimationState = DirectionalAnimationState;

export const HD_ENEMY_FRAME_WIDTH = DIRECTIONAL_SPRITE_FRAME_WIDTH;
export const HD_ENEMY_FRAME_HEIGHT = DIRECTIONAL_SPRITE_FRAME_HEIGHT;
export const HD_ENEMY_COLUMNS = DIRECTIONAL_SPRITE_COLUMNS;
export const HD_ENEMY_ROWS = DIRECTIONAL_SPRITE_ROWS;
export const HD_ENEMY_FRAME_COUNT = DIRECTIONAL_SPRITE_FRAME_COUNT;

export const HD_ENEMY_DIRECTION_ROWS = DIRECTIONAL_SPRITE_DIRECTION_ROWS;

// Provisional presentation timings. The purchased pack's source FPS is unverified.
export const HD_ENEMY_IDLE_FRAME_MS = DIRECTIONAL_IDLE_FRAME_MS;
export const HD_ENEMY_ATTACK_FRAME_MS = DIRECTIONAL_ATTACK_FRAME_MS;
export const HD_ENEMY_HIT_FRAME_MS = DIRECTIONAL_HIT_FRAME_MS;
export const HD_ENEMY_DIE_FRAME_MS = DIRECTIONAL_DIE_FRAME_MS;

export const HD_ENEMY_STATE_SHEETS: Record<HdEnemyAnimationState, string> = {
  idle: "Idle.png",
  attack: "Attack1.png",
  hit: "TakeDamage.png",
  die: "Die.png",
};

export type HdEnemyQueueEntry = DirectionalAnimationQueueEntry;

export type HdEnemyManifest = {
  version: 1;
  sourceFps: "unverified";
  frameWidth: 128;
  frameHeight: 128;
  columns: 15;
  rows: 8;
  directionRows: readonly number[];
  stateMapping: Record<HdEnemyAnimationState, string>;
  characters: Record<HdEnemyCharacter, { sheets: Record<HdEnemyAnimationState, string> }>;
};

export function getHdEnemyCharacterForUnit(unit: Unit): HdEnemyCharacter | undefined {
  const kind = getUnitSpriteKind(unit);
  return kind === "6Crusader" || kind === "10Caster" ? kind : undefined;
}

export function getHdEnemyDirectionRow(direction: UnitDirection) {
  return getDirectionalSpriteRow(direction);
}

export function getHdEnemyBackgroundPosition(
  frame: number,
  direction: UnitDirection,
) {
  return getDirectionalSpriteBackgroundPosition(frame, direction, "HD Enemy");
}

export function getHdEnemySheetUrl(
  character: HdEnemyCharacter,
  animation: HdEnemyAnimationState,
) {
  return `/local-assets/hd-enemy/${character}/${HD_ENEMY_STATE_SHEETS[animation]}`;
}

export function getHdEnemyFrameDuration(animation: HdEnemyAnimationState) {
  return getDirectionalSpriteFrameDuration(animation);
}

export function advanceHdEnemyFrame(
  animation: HdEnemyAnimationState,
  frame: number,
): { frame: number; completed: boolean } {
  return advanceDirectionalSpriteFrame(animation, frame);
}

export function getHdEnemyAnimationQueue(
  visualEvents: readonly UnitVisualEvent[],
): HdEnemyQueueEntry[] {
  return getDirectionalAnimationQueue(visualEvents);
}

function isHdEnemyManifest(value: unknown): value is HdEnemyManifest {
  if (!value || typeof value !== "object") return false;
  const manifest = value as Partial<HdEnemyManifest>;
  return manifest.version === 1
    && manifest.sourceFps === "unverified"
    && manifest.frameWidth === HD_ENEMY_FRAME_WIDTH
    && manifest.frameHeight === HD_ENEMY_FRAME_HEIGHT
    && manifest.columns === HD_ENEMY_COLUMNS
    && manifest.rows === HD_ENEMY_ROWS
    && Array.isArray(manifest.directionRows)
    && manifest.directionRows.join(",") === HD_ENEMY_DIRECTION_ROWS.join(",")
    && HD_ENEMY_CHARACTERS.every((character) => (
      manifest.characters?.[character]
      && Object.entries(HD_ENEMY_STATE_SHEETS).every(([state, filename]) => (
        manifest.characters?.[character]?.sheets[state as HdEnemyAnimationState] === filename
      ))
    ));
}

export async function loadHdEnemyManifest(
  fetcher: typeof fetch = fetch,
): Promise<HdEnemyManifest | undefined> {
  try {
    const response = await fetcher("/local-assets/hd-enemy/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest: unknown = await response.json();
    if (!isHdEnemyManifest(manifest)) throw new Error("manifest schema mismatch");
    return manifest;
  } catch (error) {
    console.warn("HD Enemy asset unavailable", error);
    return undefined;
  }
}
