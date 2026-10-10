import type { ReactNode } from "react";
import {
  getHdCharacterSheetUrl,
  loadHdCharacterManifest,
  type HdCharacter,
  type HdCharacterAnimationState,
} from "../presentation/hdCharacter";
import type { UnitDirection, UnitVisualEvent } from "../presentation/unitVisualEvents";
import {
  DirectionalSpriteSheetFallback,
  DirectionalSpriteSheetFrame,
  DirectionalUnitSpriteSheet,
} from "./DirectionalUnitSpriteSheet";

type Props = {
  character: HdCharacter;
  visualEvents?: readonly UnitVisualEvent[];
  fallback: ReactNode;
  initialDirection?: UnitDirection;
  onAnimationEnd?: (state: HdCharacterAnimationState) => void;
};

const warnedMissingAssets = new Set<string>();
let sharedManifestPromise: Promise<Awaited<ReturnType<typeof loadHdCharacterManifest>>> | undefined;

function getSharedManifest() {
  sharedManifestPromise ??= loadHdCharacterManifest();
  return sharedManifestPromise;
}

async function checkHdCharacterAssets() {
  return Boolean(await getSharedManifest());
}

export function handleHdCharacterAssetError(
  src: string,
  warn: (message: string) => void = console.warn,
) {
  if (!warnedMissingAssets.has(src)) {
    warnedMissingAssets.add(src);
    warn(`HD Character asset unavailable: ${src}`);
  }
  return false;
}

export function HdCharacterAssetFallback({ children }: { children: ReactNode }) {
  return <DirectionalSpriteSheetFallback title="HD Character asset unavailable">{children}</DirectionalSpriteSheetFallback>;
}

export function HdCharacterSpriteFrame({
  character,
  animation,
  direction,
  frame,
  onAssetError,
}: {
  character: HdCharacter;
  animation: HdCharacterAnimationState;
  direction: UnitDirection;
  frame: number;
  onAssetError?: (src: string) => void;
}) {
  const src = getHdCharacterSheetUrl(character, animation);
  return (
    <DirectionalSpriteSheetFrame
      assetPack="hd-character"
      character={character}
      animation={animation}
      direction={direction}
      frame={frame}
      src={src}
      onAssetError={onAssetError}
    />
  );
}

export function HdCharacterUnitSprite({
  character,
  visualEvents = [],
  fallback,
  initialDirection = 0,
  onAnimationEnd,
}: Props) {
  return (
    <DirectionalUnitSpriteSheet
      assetPack="hd-character"
      character={character}
      getSheetUrl={(animation) => getHdCharacterSheetUrl(character, animation)}
      checkAssets={checkHdCharacterAssets}
      handleAssetError={handleHdCharacterAssetError}
      visualEvents={visualEvents}
      fallback={fallback}
      fallbackTitle="HD Character asset unavailable"
      initialDirection={initialDirection}
      onAnimationEnd={onAnimationEnd}
    />
  );
}
