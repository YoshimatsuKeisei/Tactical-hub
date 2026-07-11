import { UNIT_STATS } from "../game/constants";
import { isRetreating, type RetreatDirectionIndicator } from "../game/engine/retreat";
import type { Team, Unit } from "../game/types";

type Props = {
  unit: Unit;
  team?: Team;
  selected?: boolean;
  attackTarget?: boolean;
  retreatIndicators?: RetreatDirectionIndicator[];
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

export function UnitToken({ unit, team, selected, attackTarget, retreatIndicators = [], onClick }: Props) {
  return (
    <button
      className={`unit-token ${selected ? "selected" : ""} ${attackTarget ? "attack-target" : ""}`}
      style={{ background: team?.color ?? "#777" }}
      title={`${team?.name ?? unit.ownerTeamId} ${unit.type} HP:${unit.hp}`}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
    >
      {UNIT_STATS[unit.type].label}
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
