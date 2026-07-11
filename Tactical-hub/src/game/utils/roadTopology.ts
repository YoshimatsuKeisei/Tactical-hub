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

  if (from.kind !== "tile" || to.kind !== "tile") {
    return false;
  }

  const fromRoadSectionId = getRoadSectionIdForPosition(state, from);

  const toRoadSectionId = getRoadSectionIdForPosition(state, to);

  return Boolean(
    fromRoadSectionId &&
    toRoadSectionId &&
    fromRoadSectionId === toRoadSectionId,
  );
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
  if (attackerPosition.kind === "tile" && targetPosition.kind === "tile") {
    const attackerRoadSectionId = getRoadSectionIdForPosition(
      state,
      attackerPosition,
    );

    const targetRoadSectionId = getRoadSectionIdForPosition(
      state,
      targetPosition,
    );

    return Boolean(
      attackerRoadSectionId &&
      targetRoadSectionId &&
      attackerRoadSectionId === targetRoadSectionId,
    );
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
