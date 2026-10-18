import { UNIT_STATS } from "../game/constants";
import type { Team, Unit } from "../game/types";

type Props = {
  unit: Unit;
  team?: Team;
  selected?: boolean;
  attackTarget?: boolean;
  attackReady?: boolean;
  attackComplete?: boolean;
  ninjaTeleportUsed?: boolean;
  onClick?: () => void;
};

export function UnitToken({ unit, team, selected, attackTarget, attackReady, attackComplete, ninjaTeleportUsed = false, onClick }: Props) {
  const ninjaTeleportLabel = unit.type === "ninja"
    ? ninjaTeleportUsed ? "used" : "available"
    : undefined;
  return (
    <button
      className={`unit-token ${selected ? "selected" : ""} ${attackTarget ? "attack-target" : ""} ${attackReady ? "attack-ready" : ""} ${attackComplete ? "attack-complete" : ""}`}
      style={{ background: team?.color ?? "#777" }}
      title={`${team?.name ?? unit.ownerTeamId} ${unit.type} HP:${unit.hp}${ninjaTeleportLabel ? ` Ninja teleport: ${ninjaTeleportLabel}` : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
    >
      {UNIT_STATS[unit.type].label}
      {unit.type === "infantry" && unit.formation === "heavy" ? <span className="formation-badge">重</span> : null}
      {unit.type === "ninja" ? <span className={`ninja-teleport-badge ${ninjaTeleportUsed ? "used" : "available"}`} aria-label={`Ninja teleport ${ninjaTeleportLabel}`}>{ninjaTeleportUsed ? "済" : "瞬"}</span> : null}
      {unit.hp > 1 ? <span className="hp-badge">{unit.hp}</span> : null}
    </button>
  );
}
