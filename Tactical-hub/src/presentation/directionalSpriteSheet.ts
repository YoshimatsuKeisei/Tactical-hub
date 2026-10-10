import type { UnitDirection, UnitVisualEvent } from "./unitVisualEvents";

export type DirectionalAnimationState = "idle" | "attack" | "hit" | "die";

export const DIRECTIONAL_SPRITE_FRAME_WIDTH = 128;
export const DIRECTIONAL_SPRITE_FRAME_HEIGHT = 128;
export const DIRECTIONAL_SPRITE_COLUMNS = 15;
export const DIRECTIONAL_SPRITE_ROWS = 8;
export const DIRECTIONAL_SPRITE_FRAME_COUNT = 15;
export const DIRECTIONAL_SPRITE_DIRECTION_ROWS = [6, 7, 0, 1, 2, 3, 4, 5] as const;

// Provisional presentation timings shared by the two 15x8 HD packs.
// Both purchased packs have unverified source FPS.
export const DIRECTIONAL_IDLE_FRAME_MS = 110;
export const DIRECTIONAL_ATTACK_FRAME_MS = 70;
export const DIRECTIONAL_HIT_FRAME_MS = 70;
export const DIRECTIONAL_DIE_FRAME_MS = 85;

export type DirectionalAnimationQueueEntry = {
  eventId: string;
  animation: Exclude<DirectionalAnimationState, "idle">;
  direction?: UnitDirection;
};

export function getDirectionalSpriteRow(direction: UnitDirection) {
  return DIRECTIONAL_SPRITE_DIRECTION_ROWS[direction];
}

export function getDirectionalSpriteBackgroundPosition(
  frame: number,
  direction: UnitDirection,
  label = "Directional sprite",
) {
  if (!Number.isInteger(frame) || frame < 0 || frame >= DIRECTIONAL_SPRITE_COLUMNS) {
    throw new RangeError(`${label} frame must be 0-${DIRECTIONAL_SPRITE_COLUMNS - 1}.`);
  }
  return {
    xPercent: (frame / (DIRECTIONAL_SPRITE_COLUMNS - 1)) * 100,
    yPercent: (getDirectionalSpriteRow(direction) / (DIRECTIONAL_SPRITE_ROWS - 1)) * 100,
  };
}

export function getDirectionalSpriteFrameDuration(animation: DirectionalAnimationState) {
  if (animation === "attack") return DIRECTIONAL_ATTACK_FRAME_MS;
  if (animation === "hit") return DIRECTIONAL_HIT_FRAME_MS;
  if (animation === "die") return DIRECTIONAL_DIE_FRAME_MS;
  return DIRECTIONAL_IDLE_FRAME_MS;
}

export function advanceDirectionalSpriteFrame(
  animation: DirectionalAnimationState,
  frame: number,
): { frame: number; completed: boolean } {
  if (frame + 1 < DIRECTIONAL_SPRITE_FRAME_COUNT) {
    return { frame: frame + 1, completed: false };
  }
  return { frame: 0, completed: animation !== "idle" };
}

export function getDirectionalAnimationQueue(
  visualEvents: readonly UnitVisualEvent[],
): DirectionalAnimationQueueEntry[] {
  const queue: DirectionalAnimationQueueEntry[] = [];
  for (const event of visualEvents) {
    if (event.kind === "death") {
      queue.push(
        { eventId: `${event.eventId}:damage`, animation: "hit", direction: event.direction },
        { eventId: `${event.eventId}:die`, animation: "die", direction: event.direction },
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
