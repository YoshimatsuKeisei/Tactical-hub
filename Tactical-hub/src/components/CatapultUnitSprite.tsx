import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  CATAPULT_HIT_DURATION_MS,
  getCatapultFrames,
  type CatapultAnimationState,
  type CatapultDirection,
  type UnitVisualEvent,
} from "../presentation/catapult";

type Props = {
  visualEvents?: readonly UnitVisualEvent[];
  fallback: ReactNode;
  initialDirection?: CatapultDirection;
  onAnimationEnd?: (state: CatapultAnimationState) => void;
};

const warnedMissingFrames = new Set<string>();

type CatapultQueueEntry = {
  eventId: string;
  animation: Exclude<CatapultAnimationState, "idle">;
  direction?: CatapultDirection;
};

export function handleCatapultImageError(
  src: string,
  warn: (message: string) => void = console.warn,
) {
  if (!warnedMissingFrames.has(src)) {
    warnedMissingFrames.add(src);
    warn(`Catapult asset unavailable: ${src}`);
  }
  return false;
}

export function CatapultAssetFallback({ children }: { children: ReactNode }) {
  return <span className="catapult-asset-fallback" title="Catapult asset unavailable">{children}</span>;
}

export function CatapultUnitSprite({
  visualEvents = [],
  fallback,
  initialDirection = 0,
  onAnimationEnd,
}: Props) {
  const [facingDirection, setFacingDirection] = useState<CatapultDirection>(initialDirection);
  const [animationQueue, setAnimationQueue] = useState<CatapultQueueEntry[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [assetAvailable, setAssetAvailable] = useState(true);
  const handledEvents = useRef(new Set<string>());

  useEffect(() => {
    const additions: CatapultQueueEntry[] = [];
    for (const event of visualEvents) {
      if (handledEvents.current.has(event.eventId)) continue;
      handledEvents.current.add(event.eventId);
      if (event.direction !== undefined) setFacingDirection(event.direction);
      additions.push({
        eventId: event.eventId,
        animation: event.kind === "death" ? "break" : event.kind,
        direction: event.direction,
      });
    }
    if (additions.length) setAnimationQueue((current) => [...current, ...additions]);
  }, [visualEvents]);

  const active = animationQueue.length > 0 ? animationQueue[0] : undefined;
  const animation: CatapultAnimationState = active?.animation ?? "idle";
  const direction = active?.direction ?? facingDirection;

  useEffect(() => setFrameIndex(0), [active?.eventId]);

  const frames = useMemo(() => getCatapultFrames(animation, direction), [animation, direction]);
  const currentFrame = frames[Math.min(frameIndex, frames.length - 1)];

  useEffect(() => {
    if (animation === "idle") return;
    const durationMs = animation === "hit"
      ? CATAPULT_HIT_DURATION_MS
      : currentFrame.durationMs;
    if (durationMs === null) return;
    const timer = window.setTimeout(() => {
      if (animation !== "hit" && frameIndex + 1 < frames.length) {
        setFrameIndex(frameIndex + 1);
        return;
      }
      const completed = animation;
      setAnimationQueue((current) => current.slice(1));
      onAnimationEnd?.(completed);
    }, durationMs);
    return () => window.clearTimeout(timer);
  }, [animation, currentFrame.durationMs, frameIndex, frames.length, onAnimationEnd]);

  if (!assetAvailable) return <CatapultAssetFallback>{fallback}</CatapultAssetFallback>;

  return (
    <span
      className="catapult-sprite-stage"
      data-animation={animation}
      data-direction={direction}
    >
      <img
        className={`catapult-sprite-image ${animation === "hit" ? "catapult-hit-feedback" : ""}`}
        src={currentFrame.src}
        alt=""
        aria-hidden="true"
        draggable={false}
        style={{ "--catapult-hit-duration": `${CATAPULT_HIT_DURATION_MS}ms` } as CSSProperties}
        onError={() => setAssetAvailable(handleCatapultImageError(currentFrame.src))}
      />
    </span>
  );
}
