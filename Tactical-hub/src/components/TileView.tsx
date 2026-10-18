import type { ReactNode } from "react";
import type { Tile } from "../game/types";

const terrainLabels: Partial<Record<Tile["terrain"], string>> = {
  outside: "",
  //road: "道",
  //lake: "湖",
  //base: "拠",
  //baseGate: "門",
  //reorganize: "編",
};

type Props = {
  tile: Tile;
  highlighted: boolean;
  attackHighlighted?: boolean;
  constructionPreview?: boolean;
  bridgeCandidateMarker?: boolean;
  onPointerEnter?: () => void;
  bridge?: boolean;
  obstacle?: boolean;
  onClick: () => void;
  children?: ReactNode;
};

export function TileView({
  tile,
  highlighted,
  attackHighlighted = false,
  constructionPreview = false,
  bridgeCandidateMarker = false,
  onPointerEnter,
  bridge = false,
  obstacle = false,
  onClick,
  children,
}: Props) {
  return (
    <button
      className={`tile ${tile.terrain} ${highlighted ? "highlighted" : ""} ${attackHighlighted ? "attack-highlighted" : ""} ${
        constructionPreview ? "construction-preview" : ""
      } ${bridgeCandidateMarker ? "bridge-candidate-marker" : ""} ${bridge ? "active-bridge" : ""} ${obstacle ? "active-obstacle" : ""
      }`}
      onClick={onClick}
      onPointerEnter={onPointerEnter}
    >
      <span className="terrain-label">{terrainLabels[tile.terrain] ?? ""}</span>
      <div className="tile-units">{children}</div>
    </button>
  );
}
