import { useEffect, useState } from "react";
import {
  HD_ENEMY_CHARACTERS,
  HD_ENEMY_COLUMNS,
  getHdEnemyFrameDuration,
  type HdEnemyAnimationState,
  type HdEnemyCharacter,
} from "../presentation/hdEnemy";
import type { UnitDirection } from "../presentation/unitVisualEvents";
import { HdEnemySpriteFrame } from "./HdEnemyUnitSprite";

const PREVIEW_STATES: HdEnemyAnimationState[] = ["idle", "attack", "hit", "die"];

export function HdEnemyDevPreview() {
  const [character, setCharacter] = useState<HdEnemyCharacter>("6Crusader");
  const [animation, setAnimation] = useState<HdEnemyAnimationState>("idle");
  const [direction, setDirection] = useState<UnitDirection>(0);
  const [frame, setFrame] = useState(0);

  useEffect(() => setFrame(0), [animation, character, direction]);
  useEffect(() => {
    const timer = window.setTimeout(
      () => setFrame((current) => (current + 1) % HD_ENEMY_COLUMNS),
      getHdEnemyFrameDuration(animation),
    );
    return () => window.clearTimeout(timer);
  }, [animation, frame]);

  return (
    <details className="hd-enemy-dev-preview">
      <summary>HD Enemy sprite preview</summary>
      <div className="hd-enemy-dev-preview-controls">
        <label>
          Character
          <select value={character} onChange={(event) => setCharacter(event.target.value as HdEnemyCharacter)}>
            {HD_ENEMY_CHARACTERS.map((entry) => <option key={entry}>{entry}</option>)}
          </select>
        </label>
        <label>
          State
          <select value={animation} onChange={(event) => setAnimation(event.target.value as HdEnemyAnimationState)}>
            {PREVIEW_STATES.map((entry) => <option key={entry}>{entry}</option>)}
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
        <HdEnemySpriteFrame
          character={character}
          animation={animation}
          direction={direction}
          frame={frame}
        />
      </div>
      <small>frame {frame}/14 · source FPS unverified</small>
    </details>
  );
}
