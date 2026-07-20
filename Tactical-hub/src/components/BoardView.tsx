import { getMovementCandidates } from "../game/engine/movement";
import { getAttackCandidates } from "../game/engine/battle";
import { getEncourageAreaTileKeys } from "../game/engine/encouragement";
import { getRetreatDirectionIndicators } from "../game/engine/retreat";
import { getBridgeCandidates, getConstructionAt, getObstacleCandidates, getOwnStrategistPreview } from "../game/engine/construction";
import type { AttackTarget, Base, GameState, UnitPosition } from "../game/types";
import { getUnitAtBoardCell, tileKey } from "../game/utils/position";
import { getPositionCoord } from "../game/utils/roadTopology";
import { TileView } from "./TileView";
import { UnitToken } from "./UnitToken";

type Props = {
  state: GameState;
  selectedUnitId?: string;
  onSelectUnit: (unitId: string) => void;
  onChooseDestination: (position: UnitPosition) => void;
  onChooseAttackTarget: (target: AttackTarget) => void;
  manualTeamId: string;
  constructionMode?: "bridge" | "obstacle";
  onChooseConstruction: (unitId: string, kind: "bridge" | "obstacle", tiles: { x: number; y: number }[]) => void;
};

export function getMovementCandidateByBoardCell(
  state: GameState,
  candidates: UnitPosition[],
) {
  const byCell = new Map<string, UnitPosition>();
  for (const candidate of candidates) {
    const coord = getPositionCoord(state, candidate);
    if (coord) byCell.set(tileKey(coord.x, coord.y), candidate);
  }
  return byCell;
}

export function BoardView({ state, selectedUnitId, onSelectUnit, onChooseDestination, onChooseAttackTarget, manualTeamId, constructionMode, onChooseConstruction }: Props) {
  const selectedCandidates = state.phase === "movement_input" && selectedUnitId ? getMovementCandidates(state, selectedUnitId) : [];
  const attackCandidates = state.phase === "attack_input" && selectedUnitId ? getAttackCandidates(state, selectedUnitId) : [];
  const selectedUnit = state.units.find((unit) => unit.id === selectedUnitId);
  const encourageAreaKeys =
    selectedUnit && selectedUnit.type === "strategist" && selectedUnit.role === "encourage"
      ? getEncourageAreaTileKeys(state, selectedUnit)
      : new Set<string>();
  const retreatIndicators = getRetreatDirectionIndicators(state, selectedUnitId);
  const previewKeys = new Set((state.phase === "strategist_action_input" ? getOwnStrategistPreview(state, manualTeamId) : []).map((entry) => `${entry.x},${entry.y}`));
  const constructionCandidates = state.phase === "strategist_action_input" && selectedUnit?.ownerTeamId === manualTeamId && selectedUnit.role === "builder"
    ? constructionMode === "bridge" ? getBridgeCandidates(state, selectedUnit.id) : constructionMode === "obstacle" ? getObstacleCandidates(state, selectedUnit.id).map((cell) => [cell]) : []
    : [];
  const constructionByTile = new Map<string, { x: number; y: number }[]>();
  for (const candidate of constructionCandidates) for (const cell of candidate) if (!constructionByTile.has(`${cell.x},${cell.y}`)) constructionByTile.set(`${cell.x},${cell.y}`, candidate);
  const candidateByCell = getMovementCandidateByBoardCell(state, selectedCandidates);
  const attackByUnitId = new Map(attackCandidates.map((candidate) => [candidate.unitId, candidate]));

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
        const attackTarget = boardUnit ? attackByUnitId.get(boardUnit.id) : baseUnit ? attackByUnitId.get(baseUnit.id) : undefined;
        const tileCandidate = candidateByCell.get(tileKey(tile.x, tile.y));
        const baseCandidate = base ? selectedCandidates.find((candidate) => candidate.kind === "base" && candidate.baseId === base.id) : undefined;
        const destination = tileCandidate ?? baseCandidate;
        const bridge = getConstructionAt(state, tile.x, tile.y, "bridge");
        const obstacle = getConstructionAt(state, tile.x, tile.y, "obstacle");

        return (
          <TileView
            key={`${tile.x}-${tile.y}`}
            tile={tile}
            highlighted={Boolean(destination)}
            attackHighlighted={Boolean(attackTarget)}
            encourageHighlighted={encourageAreaKeys.has(`${tile.x},${tile.y}`)}
            constructionPreview={previewKeys.has(`${tile.x},${tile.y}`) || constructionByTile.has(`${tile.x},${tile.y}`)}
            bridge={Boolean(bridge)}
            obstacle={Boolean(obstacle)}
            onClick={() => {
              const construction = constructionByTile.get(`${tile.x},${tile.y}`);
              if (construction && selectedUnit && constructionMode) onChooseConstruction(selectedUnit.id, constructionMode, construction);
              else if (attackTarget) onChooseAttackTarget(attackTarget);
              else if (destination) onChooseDestination(destination);
            }}
          >
            {boardUnit && (
              <UnitToken
                unit={boardUnit}
                team={state.teams.find((team) => team.id === boardUnit.ownerTeamId)}
                selected={boardUnit.id === selectedUnitId}
                attackTarget={attackByUnitId.has(boardUnit.id)}
                retreatIndicators={boardUnit.id === selectedUnitId ? retreatIndicators : []}
                onClick={() => {
                  const target = attackByUnitId.get(boardUnit.id);
                  if (target) onChooseAttackTarget(target);
                  else onSelectUnit(boardUnit.id);
                }}
              />
            )}
            {baseUnit && (
              <UnitToken
                unit={baseUnit}
                team={state.teams.find((team) => team.id === baseUnit.ownerTeamId)}
                selected={baseUnit.id === selectedUnitId}
                attackTarget={attackByUnitId.has(baseUnit.id)}
                retreatIndicators={baseUnit.id === selectedUnitId ? retreatIndicators : []}
                onClick={() => {
                  const target = attackByUnitId.get(baseUnit.id);
                  if (target) onChooseAttackTarget(target);
                  else onSelectUnit(baseUnit.id);
                }}
              />
            )}
          </TileView>
        );
      })}
    </div>
  );
}
