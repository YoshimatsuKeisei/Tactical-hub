import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import {
  HD_ENEMY_COLUMNS,
  HD_ENEMY_ROWS,
  advanceHdEnemyFrame,
  getHdEnemyAnimationQueue,
  getHdEnemyBackgroundPosition,
  getHdEnemyFrameDuration,
  getHdEnemySheetUrl,
  loadHdEnemyManifest,
  type HdEnemyAnimationState,
  type HdEnemyCharacter,
  type HdEnemyQueueEntry,
} from "../presentation/hdEnemy";
import type { UnitDirection, UnitVisualEvent } from "../presentation/unitVisualEvents";

type Props = {
  character: HdEnemyCharacter;
  visualEvents?: readonly UnitVisualEvent[];
  fallback: ReactNode;
  initialDirection?: UnitDirection;
  onAnimationEnd?: (state: HdEnemyAnimationState) => void;
};

type FrameProps = {
  character: HdEnemyCharacter;
  animation: HdEnemyAnimationState;
  direction: UnitDirection;
  frame: number;
  onAssetError?: (src: string) => void;
};

const warnedMissingAssets = new Set<string>();
let sharedManifestPromise: Promise<Awaited<ReturnType<typeof loadHdEnemyManifest>>> | undefined;

function getSharedManifest() {
  sharedManifestPromise ??= loadHdEnemyManifest();
  return sharedManifestPromise;
}

export function handleHdEnemyAssetError(
  src: string,
  warn: (message: string) => void = console.warn,
) {
  if (!warnedMissingAssets.has(src)) {
    warnedMissingAssets.add(src);
    warn(`HD Enemy asset unavailable: ${src}`);
  }
  return false;
}

export function HdEnemyAssetFallback({ children }: { children: ReactNode }) {
  return <span className="hd-enemy-asset-fallback" title="HD Enemy asset unavailable">{children}</span>;
}

export function HdEnemySpriteFrame({
  character,
  animation,
  direction,
  frame,
  onAssetError,
}: FrameProps) {
  const src = getHdEnemySheetUrl(character, animation);
  const position = getHdEnemyBackgroundPosition(frame, direction);
  const style = {
    backgroundImage: `url("${src}")`,
    backgroundPosition: `${position.xPercent}% ${position.yPercent}%`,
    backgroundSize: `${HD_ENEMY_COLUMNS * 100}% ${HD_ENEMY_ROWS * 100}%`,
  } as CSSProperties;
  return (
    <span
      className="hd-enemy-sprite-stage"
      data-character={character}
      data-animation={animation}
      data-direction={direction}
      data-frame={frame}
    >
      <span className="hd-enemy-sprite-image" style={style} aria-hidden="true" />
      <img
        className="hd-enemy-asset-probe"
        src={src}
        alt=""
        aria-hidden="true"
        draggable={false}
        onError={() => onAssetError?.(src)}
      />
    </span>
  );
}

export function HdEnemyUnitSprite({
  character,
  visualEvents = [],
  fallback,
  initialDirection = 0,
  onAnimationEnd,
}: Props) {
  const [facingDirection, setFacingDirection] = useState<UnitDirection>(initialDirection);
  const [animationQueue, setAnimationQueue] = useState<HdEnemyQueueEntry[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [assetAvailable, setAssetAvailable] = useState(true);
  const handledEvents = useRef(new Set<string>());

  useEffect(() => {
    let cancelled = false;
    void getSharedManifest().then((manifest) => {
      if (!cancelled && !manifest) setAssetAvailable(false);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const freshEvents = visualEvents.filter((event) => {
      if (handledEvents.current.has(event.eventId)) return false;
      handledEvents.current.add(event.eventId);
      if (event.direction !== undefined) setFacingDirection(event.direction);
      return true;
    });
    const additions = getHdEnemyAnimationQueue(freshEvents);
    if (additions.length) setAnimationQueue((current) => [...current, ...additions]);
  }, [visualEvents]);

  const active = animationQueue.length > 0 ? animationQueue[0] : undefined;
  const animation: HdEnemyAnimationState = active?.animation ?? "idle";
  const direction = active?.direction ?? facingDirection;

  useEffect(() => setFrameIndex(0), [active?.eventId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const next = advanceHdEnemyFrame(animation, frameIndex);
      if (!next.completed) {
        setFrameIndex(next.frame);
        return;
      }
      const completed = animation;
      setAnimationQueue((current) => current.slice(1));
      onAnimationEnd?.(completed);
    }, getHdEnemyFrameDuration(animation));
    return () => window.clearTimeout(timer);
  }, [animation, frameIndex, onAnimationEnd]);

  if (!assetAvailable) return <HdEnemyAssetFallback>{fallback}</HdEnemyAssetFallback>;

  return (
    <HdEnemySpriteFrame
      character={character}
      animation={animation}
      direction={direction}
      frame={frameIndex}
      onAssetError={(src) => setAssetAvailable(handleHdEnemyAssetError(src))}
    />
  );
}
