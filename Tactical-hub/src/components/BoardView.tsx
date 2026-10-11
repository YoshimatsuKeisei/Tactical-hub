import { getMovementCandidates } from "../game/engine/movement";
import { getAttackCandidates, getTeamAttackCandidates } from "../game/engine/battle";
import { getEncourageAreaTileKeys } from "../game/engine/encouragement";
import { getRetreatDirectionIndicators } from "../game/engine/retreat";
import { getBridgeCandidates, getConstructionAt, getObstacleCandidates, getOwnStrategistPreview } from "../game/engine/construction";
import type { AttackTarget, Base, GameState, UnitPosition } from "../game/types";
import { getUnitAtBoardCell, tileKey } from "../game/utils/position";
import { getPositionCoord } from "../game/utils/roadTopology";
import type { UnitDeathOverlay as DeathOverlay, UnitVisualEvent } from "../presentation/unitVisualEvents";
import { TileView } from "./TileView";
import { UnitDeathOverlay } from "./UnitDeathOverlay";
import { UnitToken } from "./UnitToken";
import { useMemo, useRef, useState } from "react";
import { PixiBoardTerrain } from "./PixiBoardTerrain";
import { buildPixiBoardTerrain } from "../presentation/pixiBoardTerrain";

type Props = {
  state: GameState;
  selectedUnitId?: string;
  onSelectUnit: (unitId: string) => void;
  onChooseDestination: (position: UnitPosition) => void;
  onChooseAttackTarget: (target: AttackTarget) => void;
  manualTeamId: string;
  constructionMode?: "bridge" | "obstacle";
  onChooseConstruction: (unitId: string, kind: "bridge" | "obstacle", tiles: { x: number; y: number }[]) => void;
  visualEvents?: readonly UnitVisualEvent[];
  deathOverlays?: readonly DeathOverlay[];
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

export function BoardView({ state, selectedUnitId, onSelectUnit, onChooseDestination, onChooseAttackTarget, manualTeamId, constructionMode, onChooseConstruction, visualEvents = [], deathOverlays = [] }: Props) {
  const [terrainMode, setTerrainMode] = useState("css");
  const [terrainError, setTerrainError] = useState("");
  const boardRef = useRef<HTMLDivElement>(null);
  const usePixi = (import.meta as ImportMeta & { env: { DEV: boolean } }).env.DEV && terrainMode === "pixi";
  const terrain = useMemo(() => usePixi ? buildPixiBoardTerrain(state) : [], [usePixi, state.map, state.constructions]);
  const [hoveredBridge, setHoveredBridge] = useState<{ key: string; cells: { x: number; y: number }[] }>();
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
  const bridgeMarkers = new Map<string, { key: string; cells: { x: number; y: number }[] }>();
  for (const candidate of constructionCandidates) {
    const candidateKey = candidate.map((cell) => `${cell.x},${cell.y}`).join("|");
    if (constructionMode === "obstacle") {
      for (const cell of candidate) constructionByTile.set(`${cell.x},${cell.y}`, candidate);
    } else if (constructionMode === "bridge") {
      for (const cell of [candidate[0], candidate.at(-1)!]) if (!bridgeMarkers.has(`${cell.x},${cell.y}`)) bridgeMarkers.set(`${cell.x},${cell.y}`, { key: candidateKey, cells: candidate });
      if (hoveredBridge?.key === candidateKey) for (const cell of candidate) constructionByTile.set(`${cell.x},${cell.y}`, candidate);
    }
  }
  const candidateByCell = getMovementCandidateByBoardCell(state, selectedCandidates);
  const attackByUnitId = new Map(attackCandidates.map((candidate) => [candidate.unitId, candidate]));
  const savedAttackIds = new Set(state.turnState.actionIntents.flatMap((intent) => intent.attackIntents ?? []).map((intent) => intent.attackerUnitId));
  const teamAttackers = state.phase === "attack_input" ? getTeamAttackCandidates(state, manualTeamId).filter((entry) => entry.targets.length > 0) : [];
  const attackReadyIds = new Set(teamAttackers.filter((entry) => !savedAttackIds.has(entry.attackerUnitId)).map((entry) => entry.attackerUnitId));
  const attackCompleteIds = new Set(teamAttackers.filter((entry) => savedAttackIds.has(entry.attackerUnitId)).map((entry) => entry.attackerUnitId));
  const visualEventsByUnit = new Map<string, UnitVisualEvent[]>();
  for (const event of visualEvents) {
    const current = visualEventsByUnit.get(event.unitId) ?? [];
    current.push(event);
    visualEventsByUnit.set(event.unitId, current);
  }
  const deathOverlaysByTile = new Map<string, DeathOverlay[]>();
  for (const overlay of deathOverlays) {
    const key = tileKey(overlay.coord.x, overlay.coord.y);
    const current = deathOverlaysByTile.get(key) ?? [];
    current.push(overlay);
    deathOverlaysByTile.set(key, current);
  }

  function getBaseUnitForTile(base: Base, x: number, y: number) {
    const minX = Math.min(...base.coords.map((coord) => coord.x));
    const minY = Math.min(...base.coords.map((coord) => coord.y));
    const slot = base.slots.find((candidate) => candidate.localCol === x - minX && candidate.localRow === y - minY);
    return slot?.unitId ? state.units.find((unit) => unit.id === slot.unitId) : undefined;
  }

  return (
    <>
    {(import.meta as ImportMeta & { env: { DEV: boolean } }).env.DEV && <div className="board-renderer-controls">
      <label>Terrain renderer (DEV): <select value={terrainMode} onChange={(event) => { setTerrainError(""); setTerrainMode(event.target.value); }}>
        <option value="css">CSS (default)</option><option value="pixi">PixiJS (experimental)</option>
      </select></label>
      {usePixi && <span>HTML units/highlights remain above all terrain; exact occlusion is not supported.</span>}
      {terrainError && <span role="alert">PixiJS failed; using CSS: {terrainError}</span>}
    </div>}
    <div ref={boardRef} className={`board${usePixi ? " board-pixi" : ""}`} style={{ gridTemplateColumns: `repeat(${state.map.width}, minmax(26px, 1fr))` }}>
      {usePixi && <PixiBoardTerrain cells={terrain} boardRef={boardRef} onError={(message) => { setTerrainError(message); setTerrainMode("css"); }} />}
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
        const bridgeMarker = bridgeMarkers.get(`${tile.x},${tile.y}`);

        return (
          <TileView
            key={`${tile.x}-${tile.y}`}
            tile={tile}
            highlighted={Boolean(destination)}
            attackHighlighted={Boolean(attackTarget)}
            encourageHighlighted={encourageAreaKeys.has(`${tile.x},${tile.y}`)}
            constructionPreview={previewKeys.has(`${tile.x},${tile.y}`) || constructionByTile.has(`${tile.x},${tile.y}`)}
            bridgeCandidateMarker={Boolean(bridgeMarker)}
            onPointerEnter={bridgeMarker ? () => setHoveredBridge(bridgeMarker) : undefined}
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
                attackReady={attackReadyIds.has(boardUnit.id)}
                attackComplete={attackCompleteIds.has(boardUnit.id)}
                retreatIndicators={boardUnit.id === selectedUnitId ? retreatIndicators : []}
                visualEvents={visualEventsByUnit.get(boardUnit.id)}
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
                attackReady={attackReadyIds.has(baseUnit.id)}
                attackComplete={attackCompleteIds.has(baseUnit.id)}
                retreatIndicators={baseUnit.id === selectedUnitId ? retreatIndicators : []}
                visualEvents={visualEventsByUnit.get(baseUnit.id)}
                onClick={() => {
                  const target = attackByUnitId.get(baseUnit.id);
                  if (target) onChooseAttackTarget(target);
                  else onSelectUnit(baseUnit.id);
                }}
              />
            )}
            {(deathOverlaysByTile.get(tileKey(tile.x, tile.y)) ?? []).map((overlay) => (
              <UnitDeathOverlay
                key={overlay.overlayId}
                overlay={overlay}
                teamColor={state.teams.find((team) => team.id === overlay.unit.ownerTeamId)?.color}
              />
            ))}
          </TileView>
        );
      })}
    </div>
    </>
  );
}
