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
  const [direction, setDirection] = useState<CatapultDirection>(initialDirection);
  const [animation, setAnimation] = useState<CatapultAnimationState>("idle");
  const [frameIndex, setFrameIndex] = useState(0);
  const [hitFeedback, setHitFeedback] = useState(false);
  const [assetAvailable, setAssetAvailable] = useState(true);
  const handledEvents = useRef(new Set<string>());
  const hitTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => {
    if (hitTimer.current !== undefined) window.clearTimeout(hitTimer.current);
  }, []);

  useEffect(() => {
    for (const event of visualEvents) {
      if (handledEvents.current.has(event.eventId)) continue;
      handledEvents.current.add(event.eventId);
      if (event.direction !== undefined) setDirection(event.direction);
      if (event.kind === "attack") {
        setAnimation("attack");
        setFrameIndex(0);
      } else if (event.kind === "break") {
        setAnimation("break");
        setFrameIndex(0);
      } else {
        if (hitTimer.current !== undefined) window.clearTimeout(hitTimer.current);
        setHitFeedback(true);
        hitTimer.current = window.setTimeout(() => {
          setHitFeedback(false);
          onAnimationEnd?.("hit");
        }, CATAPULT_HIT_DURATION_MS);
      }
    }
  }, [onAnimationEnd, visualEvents]);

  const frames = useMemo(() => getCatapultFrames(animation, direction), [animation, direction]);
  const currentFrame = frames[Math.min(frameIndex, frames.length - 1)];

  useEffect(() => {
    if (animation === "idle" || currentFrame.durationMs === null) return;
    const timer = window.setTimeout(() => {
      if (frameIndex + 1 < frames.length) {
        setFrameIndex(frameIndex + 1);
        return;
      }
      const completed = animation;
      setAnimation("idle");
      setFrameIndex(0);
      onAnimationEnd?.(completed);
    }, currentFrame.durationMs);
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
        className={`catapult-sprite-image ${hitFeedback ? "catapult-hit-feedback" : ""}`}
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
