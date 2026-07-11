import type { ReactNode } from "react";
import type { Tile } from "../game/types";

const terrainLabels: Record<Tile["terrain"], string> = {
  outside: "",
  road: "道",
  lake: "湖",
  base: "拠",
  baseGate: "門",
  reorganize: "編",
};

type Props = {
  tile: Tile;
  highlighted: boolean;
  attackHighlighted?: boolean;
  encourageHighlighted?: boolean;
  onClick: () => void;
  children?: ReactNode;
};

export function TileView({
  tile,
  highlighted,
  attackHighlighted = false,
  encourageHighlighted = false,
  onClick,
  children,
}: Props) {
  return (
    <button
      className={`tile ${tile.terrain} ${highlighted ? "highlighted" : ""} ${attackHighlighted ? "attack-highlighted" : ""} ${
        encourageHighlighted ? "encourage-highlighted" : ""
      }`}
      onClick={onClick}
    >
      <span className="terrain-label">{terrainLabels[tile.terrain]}</span>
      <div className="tile-units">{children}</div>
    </button>
  );
}
