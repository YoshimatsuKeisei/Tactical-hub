import { getMovementCandidates } from "../game/engine/movement";
import type { Base, GameState, UnitPosition } from "../game/types";
import { getUnitAtBoardCell, positionKey } from "../game/utils/position";
import { TileView } from "./TileView";
import { UnitToken } from "./UnitToken";

type Props = {
  state: GameState;
  selectedUnitId?: string;
  onSelectUnit: (unitId: string) => void;
  onChooseDestination: (position: UnitPosition) => void;
};

export function BoardView({ state, selectedUnitId, onSelectUnit, onChooseDestination }: Props) {
  const selectedCandidates = selectedUnitId ? getMovementCandidates(state, selectedUnitId) : [];
  const candidateByKey = new Map(selectedCandidates.map((candidate) => [positionKey(candidate), candidate]));

  function getBaseUnitForTile(base: Base, x: number, y: number) {
    const minX = Math.min(...base.coords.map((coord) => coord.x));
    const minY = Math.min(...base.coords.map((coord) => coord.y));
    const slot = base.slots.find((candidate) => candidate.localCol === x - minX && candidate.localRow === y - minY);
    return slot?.unitId ? state.units.find((unit) => unit.id === slot.unitId) : undefined;
  }

  return (
    <div className="board" style={{ gridTemplateColumns: `repeat(${state.map.width}, minmax(26px, 1fr))` }}>
      {state.map.tiles.map((tile) => {
        const boardUnit = getUnitAtBoardCell(state, tile.x, tile.y);
        const base = tile.baseId ? state.bases.find((candidate) => candidate.id === tile.baseId) : undefined;
        const baseUnit = base ? getBaseUnitForTile(base, tile.x, tile.y) : undefined;
        const tileCandidate =
          candidateByKey.get(positionKey({ kind: "tile", x: tile.x, y: tile.y })) ??
          candidateByKey.get(positionKey({ kind: "water", x: tile.x, y: tile.y }));
        const baseCandidate = base ? selectedCandidates.find((candidate) => candidate.kind === "base" && candidate.baseId === base.id) : undefined;
        const destination = tileCandidate ?? baseCandidate;

        return (
          <TileView
            key={`${tile.x}-${tile.y}`}
            tile={tile}
            highlighted={Boolean(destination)}
            onClick={() => {
              if (destination) onChooseDestination(destination);
            }}
          >
            {boardUnit && (
              <UnitToken
                unit={boardUnit}
                team={state.teams.find((team) => team.id === boardUnit.ownerTeamId)}
                selected={boardUnit.id === selectedUnitId}
                onClick={() => onSelectUnit(boardUnit.id)}
              />
            )}
            {baseUnit && (
              <UnitToken
                unit={baseUnit}
                team={state.teams.find((team) => team.id === baseUnit.ownerTeamId)}
                selected={baseUnit.id === selectedUnitId}
                onClick={() => onSelectUnit(baseUnit.id)}
              />
            )}
          </TileView>
        );
      })}
    </div>
  );
}
