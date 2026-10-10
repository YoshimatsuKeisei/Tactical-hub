import { UNIT_STATS } from "../game/constants";
import { isRetreating, type RetreatDirectionIndicator } from "../game/engine/retreat";
import type { Team, Unit } from "../game/types";
import { getHdEnemyCharacterForUnit } from "../presentation/hdEnemy";
import { getUnitSpriteKind, type UnitVisualEvent } from "../presentation/unitVisualEvents";
import { CatapultUnitSprite } from "./CatapultUnitSprite";
import { HdEnemyUnitSprite } from "./HdEnemyUnitSprite";

type Props = {
  unit: Unit;
  team?: Team;
  selected?: boolean;
  attackTarget?: boolean;
  attackReady?: boolean;
  attackComplete?: boolean;
  retreatIndicators?: RetreatDirectionIndicator[];
  visualEvents?: readonly UnitVisualEvent[];
  onClick?: () => void;
};

const directionArrows: Record<RetreatDirectionIndicator["directionLabel"], string> = {
  up: "↑",
  "up-right": "↗",
  right: "→",
  "down-right": "↘",
  down: "↓",
  "down-left": "↙",
  left: "←",
  "up-left": "↖",
};

export function UnitToken({ unit, team, selected, attackTarget, attackReady, attackComplete, retreatIndicators = [], visualEvents = [], onClick }: Props) {
  const spriteKind = getUnitSpriteKind(unit);
  const hdEnemyCharacter = getHdEnemyCharacterForUnit(unit);
  const tokenLabel = `${team?.name ?? unit.ownerTeamId} ${unit.type} HP:${unit.hp}`;
  return (
    <button
      className={`unit-token ${spriteKind ? "animated-unit-token" : ""} ${spriteKind === "catapult" ? "catapult-token" : ""} ${selected ? "selected" : ""} ${attackTarget ? "attack-target" : ""} ${attackReady ? "attack-ready" : ""} ${attackComplete ? "attack-complete" : ""}`}
      style={{ background: spriteKind ? "transparent" : team?.color ?? "#777" }}
      title={tokenLabel}
      aria-label={tokenLabel}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
    >
      {spriteKind ? (
        <>
          {spriteKind === "catapult" ? (
            <CatapultUnitSprite
              visualEvents={visualEvents}
              fallback={UNIT_STATS[unit.type].label}
            />
          ) : hdEnemyCharacter ? (
            <HdEnemyUnitSprite
              character={hdEnemyCharacter}
              visualEvents={visualEvents}
              fallback={UNIT_STATS[unit.type].label}
            />
          ) : UNIT_STATS[unit.type].label}
          <span
            className="unit-sprite-team-badge"
            style={{ background: team?.color ?? "#777" }}
            aria-hidden="true"
          />
        </>
      ) : UNIT_STATS[unit.type].label}
      {unit.type === "infantry" && unit.formation === "heavy" ? <span className="formation-badge">重</span> : null}
      {unit.hp > 1 ? <span className="hp-badge">{unit.hp}</span> : null}
      {isRetreating(unit) ? <span className="retreat-badge">R</span> : null}
      {retreatIndicators.map((indicator) => (
        <span
          key={indicator.key}
          className={`retreat-indicator retreat-indicator-${indicator.key} direction-${indicator.directionLabel}`}
          aria-hidden="true"
        >
          <span>{directionArrows[indicator.directionLabel]}</span>
          <span>{indicator.label}</span>
        </span>
      ))}
    </button>
  );
}
