import type { UnitType } from "./types";

export const UNIT_STATS: Record<UnitType, { label: string; hp: number; move: number; priority: number }> = {
  king: { label: "王", hp: 3, move: 1, priority: 1 },
  infantry: { label: "歩", hp: 1, move: 1, priority: 2 },
  cavalry: { label: "馬", hp: 1, move: 2, priority: 3 },
  archer: { label: "弓", hp: 1, move: 1, priority: 4 },
  ninja: { label: "忍", hp: 1, move: 1, priority: 5 },
  engineer: { label: "工", hp: 1, move: 1, priority: 6 },
  strategist: { label: "帥", hp: 1, move: 1, priority: 7 },
  apprentice_ninja: { label: "見", hp: 1, move: 1, priority: 8 },
};

export const PRODUCIBLE_UNIT_TYPES: UnitType[] = [
  "infantry",
  "cavalry",
  "archer",
  "engineer",
  "ninja",
  "strategist",
];
