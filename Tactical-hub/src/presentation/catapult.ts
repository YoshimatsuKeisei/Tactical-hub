import type { AttackIntent, GameState, Unit } from "../game/types";
import {
  UNIT_DIRECTION_LABELS,
  UNIT_VISUAL_EVENT_RETENTION_MS,
  createUnitVisualPresentation,
  getUnitAttackDirection,
  getUnitSpriteKind,
  unitDirectionFromDelta,
  type UnitDirection,
  type UnitVisualEvent,
} from "./unitVisualEvents";

export const CATAPULT_DIRECTION_LABELS = UNIT_DIRECTION_LABELS;
export type CatapultDirection = UnitDirection;
export type { UnitVisualEvent } from "./unitVisualEvents";
export type CatapultAnimationState = "idle" | "attack" | "hit" | "break";
export type CatapultFrameAnimation = "move" | "load" | "throw" | "break";

export const CATAPULT_MOVE_FRAME_COUNT = 31;
export const CATAPULT_LOAD_FRAME_COUNT = 31;
export const CATAPULT_THROW_FRAME_COUNT = 16;
export const CATAPULT_BREAK_FRAME_COUNT = 31;

// Provisional presentation timings. The source asset FPS is unverified.
export const CATAPULT_LOAD_FRAME_MS = 70;
export const CATAPULT_THROW_FRAME_MS = 70;
export const CATAPULT_BREAK_FRAME_MS = 70;
export const CATAPULT_HIT_DURATION_MS = 220;
export const CATAPULT_VISUAL_EVENT_RETENTION_MS = UNIT_VISUAL_EVENT_RETENTION_MS;

export type CatapultFrame = {
  animation: CatapultFrameAnimation;
  direction: CatapultDirection;
  frame: number;
  src: string;
  durationMs: number | null;
};

export type CatapultAssetManifest = {
  version: 1;
  asset: "Catapult - Isometric";
  attribution: string;
  sourceFps: "unverified";
  directions: { index: CatapultDirection; label: typeof CATAPULT_DIRECTION_LABELS[number] }[];
  idle: string[];
  animations: Record<CatapultFrameAnimation, { direction: CatapultDirection; frames: string[] }[]>;
};

export function isEngineerCatapultUnit(unit: Unit) {
  return getUnitSpriteKind(unit) === "catapult";
}

export function catapultDirectionFromDelta(dx: number, dy: number): CatapultDirection {
  return unitDirectionFromDelta(dx, dy);
}

function framePath(
  animation: CatapultFrameAnimation,
  direction: CatapultDirection,
  frame: number,
) {
  return `/local-assets/catapult/${animation}/dir${direction}/${String(frame).padStart(4, "0")}.png`;
}

export function getCatapultIdleFrame(direction: CatapultDirection): CatapultFrame {
  return {
    animation: "move",
    direction,
    frame: 0,
    src: `/local-assets/catapult/idle/dir${direction}.png`,
    durationMs: null,
  };
}

export function getCatapultAttackFrames(direction: CatapultDirection): CatapultFrame[] {
  return [
    ...Array.from({ length: CATAPULT_LOAD_FRAME_COUNT }, (_, frame) => ({
      animation: "load" as const,
      direction,
      frame,
      src: framePath("load", direction, frame),
      durationMs: CATAPULT_LOAD_FRAME_MS,
    })),
    ...Array.from({ length: CATAPULT_THROW_FRAME_COUNT }, (_, frame) => ({
      animation: "throw" as const,
      direction,
      frame,
      src: framePath("throw", direction, frame),
      durationMs: CATAPULT_THROW_FRAME_MS,
    })),
  ];
}

export function getCatapultBreakFrames(direction: CatapultDirection): CatapultFrame[] {
  return Array.from({ length: CATAPULT_BREAK_FRAME_COUNT }, (_, frame) => ({
    animation: "break" as const,
    direction,
    frame,
    src: framePath("break", direction, frame),
    durationMs: CATAPULT_BREAK_FRAME_MS,
  }));
}

export function getCatapultFrames(
  state: CatapultAnimationState,
  direction: CatapultDirection,
): CatapultFrame[] {
  if (state === "attack") return getCatapultAttackFrames(direction);
  if (state === "break") return getCatapultBreakFrames(direction);
  return [getCatapultIdleFrame(direction)];
}

export function getCatapultAttackDirection(
  state: GameState,
  attacker: Unit,
  intent: AttackIntent,
  fallback: CatapultDirection = 0,
): CatapultDirection {
  return getUnitAttackDirection(state, attacker, intent, fallback);
}

export function createCatapultVisualEvents(
  before: GameState,
  after: GameState,
  attackIntents: readonly AttackIntent[] = before.turnState.actionIntents.flatMap(
    (entry) => entry.attackIntents ?? [],
  ),
): UnitVisualEvent[] {
  return createUnitVisualPresentation(before, after, attackIntents).events.filter((event) => {
    const unit = before.units.find((candidate) => candidate.id === event.unitId);
    return Boolean(unit && isEngineerCatapultUnit(unit));
  });
}

function isManifest(value: unknown): value is CatapultAssetManifest {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<CatapultAssetManifest>;
  const hasFrames = (
    entries: CatapultAssetManifest["animations"][CatapultFrameAnimation] | undefined,
    frameCount: number,
  ) => entries?.length === 8
    && entries.every((entry, direction) => (
      entry.direction === direction && entry.frames.length === frameCount
    ));
  return Boolean(candidate.version === 1
    && candidate.sourceFps === "unverified"
    && candidate.directions?.length === 8
    && candidate.idle?.length === 8
    && hasFrames(candidate.animations?.move, CATAPULT_MOVE_FRAME_COUNT)
    && hasFrames(candidate.animations?.load, CATAPULT_LOAD_FRAME_COUNT)
    && hasFrames(candidate.animations?.throw, CATAPULT_THROW_FRAME_COUNT)
    && hasFrames(candidate.animations?.break, CATAPULT_BREAK_FRAME_COUNT));
}

export async function loadCatapultManifest(
  fetcher: typeof fetch = fetch,
): Promise<CatapultAssetManifest | undefined> {
  try {
    const response = await fetcher("/local-assets/catapult/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest: unknown = await response.json();
    if (!isManifest(manifest)) throw new Error("manifest schema mismatch");
    return manifest;
  } catch (error) {
    console.warn("Catapult asset unavailable", error);
    return undefined;
  }
}
