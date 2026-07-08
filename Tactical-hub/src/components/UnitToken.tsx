import { UNIT_STATS } from "../game/constants";
import type { Team, Unit } from "../game/types";

type Props = {
  unit: Unit;
  team?: Team;
  selected?: boolean;
  onClick?: () => void;
};

export function UnitToken({ unit, team, selected, onClick }: Props) {
  return (
    <button
      className={`unit-token ${selected ? "selected" : ""}`}
      style={{ background: team?.color ?? "#777" }}
      title={`${team?.name ?? unit.ownerTeamId} ${unit.type}`}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
    >
      {UNIT_STATS[unit.type].label}
    </button>
  );
}
