import { UNIT_STATS } from "../game/constants";
import type { Team, Unit } from "../game/types";

type Props = {
  unit: Unit;
  team?: Team;
  selected?: boolean;
  attackTarget?: boolean;
  onClick?: () => void;
};

export function UnitToken({ unit, team, selected, attackTarget, onClick }: Props) {
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
    </button>
  );
}
