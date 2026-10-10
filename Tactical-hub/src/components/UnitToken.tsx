import { UNIT_STATS } from "../game/constants";
import { isRetreating, type RetreatDirectionIndicator } from "../game/engine/retreat";
import type { Team, Unit } from "../game/types";
import { isBuilderCatapultUnit, type UnitVisualEvent } from "../presentation/catapult";
import { CatapultUnitSprite } from "./CatapultUnitSprite";

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
  const builderCatapult = isBuilderCatapultUnit(unit);
  const tokenLabel = `${team?.name ?? unit.ownerTeamId} ${unit.type} HP:${unit.hp}`;
  return (
    <button
      className={`unit-token ${builderCatapult ? "catapult-token" : ""} ${selected ? "selected" : ""} ${attackTarget ? "attack-target" : ""} ${attackReady ? "attack-ready" : ""} ${attackComplete ? "attack-complete" : ""}`}
      style={{ background: builderCatapult ? "transparent" : team?.color ?? "#777" }}
      title={tokenLabel}
      aria-label={tokenLabel}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
    >
      {builderCatapult ? (
        <>
          <CatapultUnitSprite
            visualEvents={visualEvents}
            fallback={UNIT_STATS[unit.type].label}
          />
          <span
            className="catapult-team-badge"
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
