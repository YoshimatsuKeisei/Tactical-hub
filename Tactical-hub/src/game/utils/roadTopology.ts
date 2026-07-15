import type { GameState, TerrainType, UnitPosition } from "../types";
import { getTile } from "./position";

const ROAD_TERRAINS = new Set<TerrainType>(["road", "baseGate", "reorganize"]);

const adjacentDirections = [-1, 0, 1]
  .flatMap((dx) =>
    [-1, 0, 1].map((dy) => ({
      dx,
      dy,
    })),
  )
  .filter(({ dx, dy }) => dx !== 0 || dy !== 0);

export function getRoadSectionIdAtTile(
  state: GameState,
  x: number,
  y: number,
): string | undefined {
  const tile = getTile(state.map.tiles, x, y);

  if (!tile) {
    return undefined;
  }

  if (!ROAD_TERRAINS.has(tile.terrain)) {
    return undefined;
  }

  return tile.roadSectionId;
}

export function getRoadSectionIdForPosition(
  state: GameState,
  position: UnitPosition,
): string | undefined {
  if (position.kind !== "tile") {
    return undefined;
  }

  return getRoadSectionIdAtTile(state, position.x, position.y);
}

function activeBridge(state: GameState, bridgeId: string) {
  return state.constructions.find((entry) => entry.active && entry.kind === "bridge" && entry.id === bridgeId);
}

export function getBridgePositionAt(state: GameState, x: number, y: number): Extract<UnitPosition, { kind: "bridge" }> | undefined {
  const bridge = state.constructions.find((entry) => entry.active && entry.kind === "bridge" && entry.tiles.some((cell) => cell.x === x && cell.y === y));
  if (!bridge) return undefined;
  return { kind: "bridge", bridgeId: bridge.id, cellIndex: bridge.tiles.findIndex((cell) => cell.x === x && cell.y === y) };
}

export function getPositionCoord(state: GameState, position: UnitPosition) {
  if (position.kind === "tile" || position.kind === "water") return { x: position.x, y: position.y };
  if (position.kind === "bridge") return activeBridge(state, position.bridgeId)?.tiles[position.cellIndex];
  return undefined;
}

function bridgeRoadSections(state: GameState, bridgeId: string) {
  const bridge = activeBridge(state, bridgeId);
  const sections = new Set<string>();
  for (const cell of bridge?.tiles ?? []) for (const { dx, dy } of adjacentDirections) {
    if (Math.abs(dx) + Math.abs(dy) !== 1) continue;
    const section = getRoadSectionIdAtTile(state, cell.x + dx, cell.y + dy);
    if (section) sections.add(section);
  }
  return [...sections];
}

function positionRoadSections(state: GameState, position: UnitPosition) {
  if (position.kind === "tile") return [getRoadSectionIdForPosition(state, position)].filter((id): id is string => Boolean(id));
  if (position.kind === "bridge") return bridgeRoadSections(state, position.bridgeId);
  if (position.kind === "base") return getBaseConnectedRoadSectionIds(state, position.baseId);
  return [];
}

export function areRoadSectionsDynamicallyConnected(state: GameState, a: string, b: string) {
  if (a === b) return true;
  const adjacency = new Map<string, Set<string>>();
  for (const bridge of state.constructions.filter((entry) => entry.active && entry.kind === "bridge")) {
    const sections = bridgeRoadSections(state, bridge.id);
    for (const left of sections) for (const right of sections) if (left !== right) {
      if (!adjacency.has(left)) adjacency.set(left, new Set()); adjacency.get(left)!.add(right);
    }
  }
  const queue = [a], seen = new Set(queue);
  while (queue.length) { const current = queue.shift()!; for (const next of adjacency.get(current) ?? []) { if (next === b) return true; if (!seen.has(next)) { seen.add(next); queue.push(next); } } }
  return false;
}

/**
 * 拠点の周囲8方向に存在する道区間IDを取得する。
 *
 * Baseへ接続IDを重複保存せず、
 * 実際のマップ配置から自動計算する。
 */
export function getBaseConnectedRoadSectionIds(
  state: GameState,
  baseId: string,
): string[] {
  const base = state.bases.find((candidate) => candidate.id === baseId);

  if (!base) {
    return [];
  }

  const roadSectionIds = new Set<string>();

  for (const coord of base.coords) {
    for (const { dx, dy } of adjacentDirections) {
      const roadSectionId = getRoadSectionIdAtTile(
        state,
        coord.x + dx,
        coord.y + dy,
      );

      if (roadSectionId) {
        roadSectionIds.add(roadSectionId);
      }
    }
  }

  return [...roadSectionIds];
}

export function isGroundPositionConnectedToBase(
  state: GameState,
  position: UnitPosition,
  baseId: string,
): boolean {
  const roadSectionId = getRoadSectionIdForPosition(state, position);

  if (!roadSectionId) {
    return false;
  }

  return getBaseConnectedRoadSectionIds(state, baseId).includes(roadSectionId);
}

/**
 * 通常の地上セル間移動が同一道区間内か判定する。
 *
 * 水上移動については既存の忍者用判定を維持する。
 */
export function canMoveBetweenGroundPositions(
  state: GameState,
  from: UnitPosition,
  to: UnitPosition,
): boolean {
  if (from.kind === "water" || to.kind === "water") {
    return true;
  }

  if (from.kind === "bridge" || to.kind === "bridge") {
    const fromCoord = getPositionCoord(state, from), toCoord = getPositionCoord(state, to);
    if (!fromCoord || !toCoord || Math.max(Math.abs(fromCoord.x - toCoord.x), Math.abs(fromCoord.y - toCoord.y)) !== 1) return false;
    if (from.kind === "bridge" && to.kind === "bridge") return from.bridgeId === to.bridgeId;
    const tilePosition = from.kind === "tile" ? from : to.kind === "tile" ? to : undefined;
    return Boolean(
      tilePosition && getRoadSectionIdForPosition(state, tilePosition),
    );
  }
  if (from.kind !== "tile" || to.kind !== "tile") {
    return false;
  }

  const fromRoadSectionId = getRoadSectionIdForPosition(state, from);

  const toRoadSectionId = getRoadSectionIdForPosition(state, to);

  // Movement is expanded one physical board cell at a time. A bridge may make
  // two road sections part of one attack network, but it must never turn every
  // adjacent pair in those sections into a one-step movement edge.
  return Boolean(fromRoadSectionId && fromRoadSectionId === toRoadSectionId);
}

/**
 * 射程や兵種相性ではなく、
 * 道区間上の位置関係だけを判定する。
 */
export function canAttackAcrossRoadTopology(
  state: GameState,
  attackerPosition: UnitPosition,
  targetPosition: UnitPosition,
): boolean {
  /*
   * 水上戦は既存の水上忍者ルールへ任せる。
   */
  if (attackerPosition.kind === "water" || targetPosition.kind === "water") {
    return true;
  }

  /*
   * 地上同士は同一道区間のみ攻撃可能。
   */
  if (["tile", "bridge"].includes(attackerPosition.kind) && ["tile", "bridge"].includes(targetPosition.kind)) {
    const attackerSections = positionRoadSections(state, attackerPosition);
    const targetSections = positionRoadSections(state, targetPosition);
    return attackerSections.some((left) => targetSections.some((right) => areRoadSectionsDynamicallyConnected(state, left, right)));
  }

  /*
   * 地上から拠点内への攻撃。
   *
   * 攻撃者がいる道区間が対象拠点へ接続していれば、
   * 従来どおり拠点内の敵を攻撃できる。
   */
  if (attackerPosition.kind === "tile" && targetPosition.kind === "base") {
    return isGroundPositionConnectedToBase(
      state,
      attackerPosition,
      targetPosition.baseId,
    );
  }

  /*
   * 拠点内から地上への攻撃。
   *
   * 弓兵などは、所属拠点に接続する道路上の敵を
   * 既存射程内なら攻撃できる。
   */
  if (attackerPosition.kind === "base" && targetPosition.kind === "tile") {
    return isGroundPositionConnectedToBase(
      state,
      targetPosition,
      attackerPosition.baseId,
    );
  }

  /*
   * 拠点内同士は同じ拠点内のみ。
   */
  if (attackerPosition.kind === "base" && targetPosition.kind === "base") {
    return attackerPosition.baseId === targetPosition.baseId;
  }

  return false;
}
