import type { Unit } from "../game/types";
import {
  getUnitSpriteKind,
  type UnitDirection,
  type UnitVisualEvent,
} from "./unitVisualEvents";

export const HD_ENEMY_CHARACTERS = ["6Crusader", "10Caster"] as const;
export type HdEnemyCharacter = typeof HD_ENEMY_CHARACTERS[number];
export type HdEnemyAnimationState = "idle" | "attack" | "hit" | "die";

export const HD_ENEMY_FRAME_WIDTH = 128;
export const HD_ENEMY_FRAME_HEIGHT = 128;
export const HD_ENEMY_COLUMNS = 15;
export const HD_ENEMY_ROWS = 8;
export const HD_ENEMY_FRAME_COUNT = 15;

export const HD_ENEMY_DIRECTION_ROWS = [6, 7, 0, 1, 2, 3, 4, 5] as const;

// Provisional presentation timings. The purchased pack's source FPS is unverified.
export const HD_ENEMY_IDLE_FRAME_MS = 110;
export const HD_ENEMY_ATTACK_FRAME_MS = 70;
export const HD_ENEMY_HIT_FRAME_MS = 70;
export const HD_ENEMY_DIE_FRAME_MS = 85;

export const HD_ENEMY_STATE_SHEETS: Record<HdEnemyAnimationState, string> = {
  idle: "Idle.png",
  attack: "Attack1.png",
  hit: "TakeDamage.png",
  die: "Die.png",
};

export type HdEnemyQueueEntry = {
  eventId: string;
  animation: Exclude<HdEnemyAnimationState, "idle">;
  direction?: UnitDirection;
};

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
  return HD_ENEMY_DIRECTION_ROWS[direction];
}

export function getHdEnemyBackgroundPosition(
  frame: number,
  direction: UnitDirection,
) {
  if (!Number.isInteger(frame) || frame < 0 || frame >= HD_ENEMY_COLUMNS) {
    throw new RangeError(`HD Enemy frame must be 0-${HD_ENEMY_COLUMNS - 1}.`);
  }
  return {
    xPercent: (frame / (HD_ENEMY_COLUMNS - 1)) * 100,
    yPercent: (getHdEnemyDirectionRow(direction) / (HD_ENEMY_ROWS - 1)) * 100,
  };
}

export function getHdEnemySheetUrl(
  character: HdEnemyCharacter,
  animation: HdEnemyAnimationState,
) {
  return `/local-assets/hd-enemy/${character}/${HD_ENEMY_STATE_SHEETS[animation]}`;
}

export function getHdEnemyFrameDuration(animation: HdEnemyAnimationState) {
  if (animation === "attack") return HD_ENEMY_ATTACK_FRAME_MS;
  if (animation === "hit") return HD_ENEMY_HIT_FRAME_MS;
  if (animation === "die") return HD_ENEMY_DIE_FRAME_MS;
  return HD_ENEMY_IDLE_FRAME_MS;
}

export function advanceHdEnemyFrame(
  animation: HdEnemyAnimationState,
  frame: number,
): { frame: number; completed: boolean } {
  if (frame + 1 < HD_ENEMY_FRAME_COUNT) {
    return { frame: frame + 1, completed: false };
  }
  return { frame: 0, completed: animation !== "idle" };
}

export function getHdEnemyAnimationQueue(
  visualEvents: readonly UnitVisualEvent[],
): HdEnemyQueueEntry[] {
  const queue: HdEnemyQueueEntry[] = [];
  for (const event of visualEvents) {
    if (event.kind === "death") {
      queue.push(
        { eventId: `${event.eventId}:damage`, animation: "hit" as const, direction: event.direction },
        { eventId: `${event.eventId}:die`, animation: "die" as const, direction: event.direction },
      );
    } else {
      queue.push({
        eventId: event.eventId,
        animation: event.kind,
        direction: event.direction,
      });
    }
  }
  return queue;
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
