import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  DIRECTIONAL_SPRITE_COLUMNS,
  DIRECTIONAL_SPRITE_ROWS,
  advanceDirectionalSpriteFrame,
  getDirectionalAnimationQueue,
  getDirectionalSpriteBackgroundPosition,
  getDirectionalSpriteFrameDuration,
  type DirectionalAnimationQueueEntry,
  type DirectionalAnimationState,
} from "../presentation/directionalSpriteSheet";
import type { UnitDirection, UnitVisualEvent } from "../presentation/unitVisualEvents";

type FrameProps = {
  assetPack: string;
  character: string;
  animation: DirectionalAnimationState;
  direction: UnitDirection;
  frame: number;
  src: string;
  onAssetError?: (src: string) => void;
};

type Props = {
  assetPack: string;
  character: string;
  getSheetUrl: (animation: DirectionalAnimationState) => string;
  checkAssets: () => Promise<boolean>;
  handleAssetError: (src: string) => boolean;
  visualEvents?: readonly UnitVisualEvent[];
  fallback: ReactNode;
  fallbackTitle: string;
  initialDirection?: UnitDirection;
  onAnimationEnd?: (state: DirectionalAnimationState) => void;
};

export function DirectionalSpriteSheetFallback({
  children,
  title,
}: {
  children: ReactNode;
  title: string;
}) {
  return <span className="directional-sprite-asset-fallback" title={title}>{children}</span>;
}

export function DirectionalSpriteSheetFrame({
  assetPack,
  character,
  animation,
  direction,
  frame,
  src,
  onAssetError,
}: FrameProps) {
  const position = getDirectionalSpriteBackgroundPosition(frame, direction);
  const style = {
    backgroundImage: `url("${src}")`,
    backgroundPosition: `${position.xPercent}% ${position.yPercent}%`,
    backgroundSize: `${DIRECTIONAL_SPRITE_COLUMNS * 100}% ${DIRECTIONAL_SPRITE_ROWS * 100}%`,
  } as CSSProperties;
  return (
    <span
      className="directional-unit-sprite-stage"
      data-asset-pack={assetPack}
      data-character={character}
      data-animation={animation}
      data-direction={direction}
      data-frame={frame}
    >
      <span className="directional-unit-sprite-image" style={style} aria-hidden="true" />
      <img
        className="directional-unit-sprite-asset-probe"
        src={src}
        alt=""
        aria-hidden="true"
        draggable={false}
        onError={() => onAssetError?.(src)}
      />
    </span>
  );
}

export function DirectionalUnitSpriteSheet({
  assetPack,
  character,
  getSheetUrl,
  checkAssets,
  handleAssetError,
  visualEvents = [],
  fallback,
  fallbackTitle,
  initialDirection = 0,
  onAnimationEnd,
}: Props) {
  const [facingDirection, setFacingDirection] = useState<UnitDirection>(initialDirection);
  const [animationQueue, setAnimationQueue] = useState<DirectionalAnimationQueueEntry[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [assetAvailable, setAssetAvailable] = useState(true);
  const handledEvents = useRef(new Set<string>());

  useEffect(() => {
    let cancelled = false;
    void checkAssets().then((available) => {
      if (!cancelled && !available) setAssetAvailable(false);
    });
    return () => { cancelled = true; };
  }, [checkAssets]);

  useEffect(() => {
    const freshEvents = visualEvents.filter((event) => {
      if (handledEvents.current.has(event.eventId)) return false;
      handledEvents.current.add(event.eventId);
      if (event.direction !== undefined) setFacingDirection(event.direction);
      return true;
    });
    const additions = getDirectionalAnimationQueue(freshEvents);
    if (additions.length) setAnimationQueue((current) => [...current, ...additions]);
  }, [visualEvents]);

  const active = animationQueue.length > 0 ? animationQueue[0] : undefined;
  const animation: DirectionalAnimationState = active?.animation ?? "idle";
  const direction = active?.direction ?? facingDirection;

  useEffect(() => setFrameIndex(0), [active?.eventId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const next = advanceDirectionalSpriteFrame(animation, frameIndex);
      if (!next.completed) {
        setFrameIndex(next.frame);
        return;
      }
      const completed = animation;
      setAnimationQueue((current) => current.slice(1));
      onAnimationEnd?.(completed);
    }, getDirectionalSpriteFrameDuration(animation));
    return () => window.clearTimeout(timer);
  }, [animation, frameIndex, onAnimationEnd]);

  if (!assetAvailable) {
    return <DirectionalSpriteSheetFallback title={fallbackTitle}>{fallback}</DirectionalSpriteSheetFallback>;
  }

  const src = getSheetUrl(animation);
  return (
    <DirectionalSpriteSheetFrame
      assetPack={assetPack}
      character={character}
      animation={animation}
      direction={direction}
      frame={frameIndex}
      src={src}
      onAssetError={(failedSrc) => setAssetAvailable(handleAssetError(failedSrc))}
    />
  );
}
