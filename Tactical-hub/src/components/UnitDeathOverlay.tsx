import { useState } from "react";
import { UNIT_STATS } from "../game/constants";
import { getHdEnemyCharacterForUnit } from "../presentation/hdEnemy";
import { getUnitSpriteKind, type UnitDeathOverlay as Overlay } from "../presentation/unitVisualEvents";
import { CatapultUnitSprite } from "./CatapultUnitSprite";
import { HdEnemyUnitSprite } from "./HdEnemyUnitSprite";

type Props = {
  overlay: Overlay;
  teamColor?: string;
};

export function UnitDeathOverlay({ overlay, teamColor }: Props) {
  const [visible, setVisible] = useState(true);
  if (!visible) return null;

  const fallback = UNIT_STATS[overlay.unit.type].label;
  const spriteKind = getUnitSpriteKind(overlay.unit);
  const hdCharacter = getHdEnemyCharacterForUnit(overlay.unit);

  return (
    <span
      className="unit-death-overlay"
      data-unit-id={overlay.unit.id}
      data-sprite-kind={spriteKind}
      aria-hidden="true"
    >
      {spriteKind === "catapult" ? (
        <CatapultUnitSprite
          fallback={fallback}
          initialDirection={overlay.direction}
          visualEvents={overlay.events}
          onAnimationEnd={(state) => {
            if (state === "break") setVisible(false);
          }}
        />
      ) : hdCharacter ? (
        <HdEnemyUnitSprite
          character={hdCharacter}
          fallback={fallback}
          initialDirection={overlay.direction}
          visualEvents={overlay.events}
          onAnimationEnd={(state) => {
            if (state === "die") setVisible(false);
          }}
        />
      ) : null}
      <span
        className="unit-sprite-team-badge"
        style={{ background: teamColor ?? "#777" }}
      />
    </span>
  );
}
