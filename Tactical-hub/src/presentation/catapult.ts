import type { AttackIntent, GameState, Unit } from "../game/types";
import { getPositionCoord } from "../game/utils/roadTopology";

export const CATAPULT_DIRECTION_LABELS = [
  "down",
  "down-right",
  "right",
  "up-right",
  "up",
  "up-left",
  "left",
  "down-left",
] as const;

export type CatapultDirection = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
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
export const CATAPULT_VISUAL_EVENT_RETENTION_MS =
  CATAPULT_LOAD_FRAME_COUNT * CATAPULT_LOAD_FRAME_MS
  + CATAPULT_THROW_FRAME_COUNT * CATAPULT_THROW_FRAME_MS
  + 250;

export type CatapultFrame = {
  animation: CatapultFrameAnimation;
  direction: CatapultDirection;
  frame: number;
  src: string;
  durationMs: number | null;
};

export type UnitVisualEvent = {
  unitId: string;
  kind: "attack" | "hit" | "break";
  direction?: CatapultDirection;
  eventId: string;
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

export function isBuilderCatapultUnit(unit: Unit) {
  return unit.type === "strategist" && unit.role === "builder";
}

export function catapultDirectionFromDelta(dx: number, dy: number): CatapultDirection {
  const horizontal = Math.sign(dx);
  const vertical = Math.sign(dy);
  if (horizontal === 0 && vertical >= 0) return 0;
  if (horizontal > 0 && vertical > 0) return 1;
  if (horizontal > 0 && vertical === 0) return 2;
  if (horizontal > 0 && vertical < 0) return 3;
  if (horizontal === 0 && vertical < 0) return 4;
  if (horizontal < 0 && vertical < 0) return 5;
  if (horizontal < 0 && vertical === 0) return 6;
  return 7;
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

function getUnitPresentationCoord(state: GameState, unit: Unit) {
  const position = unit.position;
  const direct = getPositionCoord(state, position);
  if (direct || position.kind !== "base") return direct;
  const base = state.bases.find((candidate) => candidate.id === position.baseId);
  const slot = base?.slots.find((candidate) => candidate.id === position.slotId);
  if (!base || !slot) return undefined;
  const minX = Math.min(...base.coords.map((coord) => coord.x));
  const minY = Math.min(...base.coords.map((coord) => coord.y));
  return { x: minX + slot.localCol, y: minY + slot.localRow };
}

export function getCatapultAttackDirection(
  state: GameState,
  attacker: Unit,
  intent: AttackIntent,
  fallback: CatapultDirection = 0,
): CatapultDirection {
  if (!intent.target) return fallback;
  const target = state.units.find((unit) => unit.id === intent.target?.unitId);
  const from = getUnitPresentationCoord(state, attacker);
  const to = target ? getUnitPresentationCoord(state, target) : undefined;
  if (!from || !to || (from.x === to.x && from.y === to.y)) return fallback;
  return catapultDirectionFromDelta(to.x - from.x, to.y - from.y);
}

export function createCatapultVisualEvents(
  before: GameState,
  after: GameState,
  attackIntents: readonly AttackIntent[] = before.turnState.actionIntents.flatMap(
    (entry) => entry.attackIntents ?? [],
  ),
): UnitVisualEvent[] {
  const events: UnitVisualEvent[] = [];
  const eventPrefix = `battle-${before.turnNumber}`;

  for (const intent of attackIntents) {
    if (intent.pass || !intent.target) continue;
    const attacker = before.units.find((unit) => unit.id === intent.attackerUnitId);
    if (!attacker || !isBuilderCatapultUnit(attacker)) continue;
    events.push({
      unitId: attacker.id,
      kind: "attack",
      direction: getCatapultAttackDirection(before, attacker, intent),
      eventId: `${eventPrefix}:attack:${attacker.id}`,
    });
  }

  for (const previous of before.units.filter(isBuilderCatapultUnit)) {
    if (previous.position.kind === "removed" || previous.hp <= 0) continue;
    const current = after.units.find((unit) => unit.id === previous.id);
    const removed = !current || current.hp <= 0 || current.position.kind === "removed";
    if (removed) {
      events.push({
        unitId: previous.id,
        kind: "break",
        eventId: `${eventPrefix}:break:${previous.id}`,
      });
    } else if (current.hp < previous.hp) {
      events.push({
        unitId: previous.id,
        kind: "hit",
        eventId: `${eventPrefix}:hit:${previous.id}`,
      });
    }
  }
  return events;
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
