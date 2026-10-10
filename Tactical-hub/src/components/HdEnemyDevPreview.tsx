import { useEffect, useState } from "react";
import {
  HD_ENEMY_CHARACTERS,
  type HdEnemyCharacter,
} from "../presentation/hdEnemy";
import { HD_CHARACTER_CHARACTERS, type HdCharacter } from "../presentation/hdCharacter";
import {
  DIRECTIONAL_SPRITE_COLUMNS,
  getDirectionalSpriteFrameDuration,
  type DirectionalAnimationState,
} from "../presentation/directionalSpriteSheet";
import type { UnitDirection } from "../presentation/unitVisualEvents";
import { HdCharacterSpriteFrame } from "./HdCharacterUnitSprite";
import { HdEnemySpriteFrame } from "./HdEnemyUnitSprite";

const PREVIEW_STATES: DirectionalAnimationState[] = ["idle", "attack", "hit", "die"];
const PREVIEW_STATE_LABELS: Record<DirectionalAnimationState, string> = {
  idle: "Idle",
  attack: "Attack",
  hit: "TakeDamage",
  die: "Die",
};
type PreviewCharacter =
  | { id: `hd-enemy:${HdEnemyCharacter}`; pack: "hd-enemy"; character: HdEnemyCharacter }
  | { id: `hd-character:${HdCharacter}`; pack: "hd-character"; character: HdCharacter };

const PREVIEW_CHARACTERS: PreviewCharacter[] = [
  ...HD_ENEMY_CHARACTERS.map((character) => ({
    id: `hd-enemy:${character}` as const,
    pack: "hd-enemy" as const,
    character,
  })),
  ...HD_CHARACTER_CHARACTERS.map((character) => ({
    id: `hd-character:${character}` as const,
    pack: "hd-character" as const,
    character,
  })),
];

export function HdEnemyDevPreview() {
  const [characterId, setCharacterId] = useState<PreviewCharacter["id"]>("hd-enemy:6Crusader");
  const [animation, setAnimation] = useState<DirectionalAnimationState>("idle");
  const [direction, setDirection] = useState<UnitDirection>(0);
  const [frame, setFrame] = useState(0);
  const selected = PREVIEW_CHARACTERS.find((entry) => entry.id === characterId) ?? PREVIEW_CHARACTERS[0];

  useEffect(() => setFrame(0), [animation, characterId, direction]);
  useEffect(() => {
    const timer = window.setTimeout(
      () => setFrame((current) => (current + 1) % DIRECTIONAL_SPRITE_COLUMNS),
      getDirectionalSpriteFrameDuration(animation),
    );
    return () => window.clearTimeout(timer);
  }, [animation, frame]);

  return (
    <details className="hd-enemy-dev-preview">
      <summary>HD directional sprite preview</summary>
      <div className="hd-enemy-dev-preview-controls">
        <label>
          Character
          <select value={characterId} onChange={(event) => setCharacterId(event.target.value as PreviewCharacter["id"])}>
            {PREVIEW_CHARACTERS.map((entry) => <option key={entry.id} value={entry.id}>{entry.character}</option>)}
          </select>
        </label>
        <label>
          State
          <select value={animation} onChange={(event) => setAnimation(event.target.value as DirectionalAnimationState)}>
            {PREVIEW_STATES.map((entry) => <option key={entry} value={entry}>{PREVIEW_STATE_LABELS[entry]}</option>)}
          </select>
        </label>
        <label>
          Direction
          <select value={direction} onChange={(event) => setDirection(Number(event.target.value) as UnitDirection)}>
            {Array.from({ length: 8 }, (_, value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
      </div>
      <div className="hd-enemy-dev-preview-stage">
        {selected.pack === "hd-enemy" ? (
          <HdEnemySpriteFrame
            character={selected.character}
            animation={animation}
            direction={direction}
            frame={frame}
          />
        ) : (
          <HdCharacterSpriteFrame
            character={selected.character}
            animation={animation}
            direction={direction}
            frame={frame}
          />
        )}
      </div>
      <small>frame {frame}/14 · source FPS unverified</small>
    </details>
  );
}
