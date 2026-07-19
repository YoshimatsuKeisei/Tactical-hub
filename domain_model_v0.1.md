# ドメインモデル図 v0.1

この文書は、戦略陣取りゲームの実装前設計として、主要な「概念」と「関係」を整理したドメインモデル図です。  
Codex などのコード生成エージェントが、ゲーム全体の構造を把握するための資料として使います。

---

## Phase 3-A: Retreat State Additions

`GameState` keeps `unitTurnFlags` as machine-readable per-turn flags derived from the previous battle.
These flags are not parsed from logs and are cleared after movement resolution.

```ts
type UnitTurnFlags = {
  unitId: string;
  battleTurnNumber: number;
  wasAliveAtBattleStart: boolean;
  survivedPreviousBattle: boolean;
  attackedInPreviousBattle: boolean;
  wasTargetedInPreviousBattle: boolean;
  retreatEligible: boolean;
};
```

`UnitStatus.kind = "retreating"` is the active retreat marker.
Retreat eligibility is separate from `UnitStatus`; eligibility means the next legal movement toward the nearest friendly controlled base can start retreat.
Retreating units cannot attack, but can be attacked normally.
Phase 3-A does not add retreat defense probability changes.

`unitTurnFlags` also records battle-time retreat diagnostics such as `positionAtBattleStart`, `enemyBaseDistanceAtBattleStart`, `enemyBaseWithin3AtBattleStart`, and `retreatEligibilityReason`.
The retreat eligibility participant set comes from valid BattleEvents before hit resolution, so both attackers and targets count even when the attack misses.
Enemy controlled bases are resolved from enemy teams' `controlledBaseIds` plus their home-owned bases.

# 1. ドメインモデル図とは何か

ドメインモデル図とは、実装対象の世界に登場する主要な概念を洗い出し、それらがどのように関係しているかを表す図です。

このゲームで言えば、以下のような「ゲーム内の概念」を整理します。

- 試合
- プレイヤー/チーム
- 駒
- 拠点
- マップ
- 道/湖/橋
- 障害物
- ターン
- 行動入力
- 戦闘イベント
- 得点

## 1.1 クラス図やER図との違い

| 図 | 目的 |
|---|---|
| ドメインモデル図 | ゲーム世界の概念と関係を整理する |
| クラス図 | 実装上のクラス/型/メソッドまで整理する |
| ER図 | DBテーブルとリレーションを整理する |

この段階では、DB設計やメソッド設計ではなく、「何をデータとして持つべきか」「どの概念がどの概念に所属するか」を明確にします。

---

# 2. 設計方針

- ゲームロジックは2Dグリッド座標で管理する。
- isometric表示は描画専用であり、ドメインモデルには直接入れない。
- 拠点内の駒は通常座標ではなく、拠点内部リストで管理する。
- 橋はマップ上の `BridgeSlot` に設置される。
- 橋・障害物は建設軍師に紐づく。
- 橋/障害物のクールダウンは別管理とする。
- ターン中の入力は `ActionIntent` として保持し、移動解決・戦闘解決で処理する。
- 中立CPU拠点は、プレイヤー代替CPUとは別扱いにする。

---

# 3. ドメインモデル図

```mermaid
classDiagram
    direction LR

    class GameSession {
      +id
      +status
      +config
      +state
      +logs
    }

    class GameConfig {
      +playerCount
      +turnLimit
      +matchMode
      +productionInterval
      +bridgeCooldownTurns
      +obstacleCooldownTurns
    }

    class GameState {
      +turnNumber
      +phase
      +teams
      +units
      +bases
      +activeBridges
      +activeObstacles
      +scores
    }

    class TurnState {
      +turnNumber
      +phase
      +submittedTeamIds
      +actionIntents
    }

    class Team {
      +id
      +name
      +color
      +status
      +homeBaseId
      +controlledBaseIds
    }

    class PlayerSlot {
      +slotId
      +kind
      +userId
      +teamId
      +readyStatus
    }

    class ScoreState {
      +teamId
      +killPoints
      +kingDamagePoints
      +baseControlPoints
      +survivalBonus
      +total
    }

    class BoardMap {
      +id
      +name
      +width
      +height
      +tiles
      +bases
      +bridgeSlots
    }

    class Tile {
      +x
      +y
      +terrainType
      +lakeId
      +roadGroupId
    }

    class Base {
      +id
      +baseType
      +ownerTeamId
      +coords
      +slots
      +protectedSlotId
      +occupationPriorityTeamId
    }

    class BaseSlot {
      +slotId
      +baseId
      +kind
      +unitId
    }

    class BridgeSlot {
      +id
      +lakeId
      +cells
      +orientation
      +startAdjacentRoad
      +endAdjacentRoad
    }

    class ActiveBridge {
      +id
      +ownerTeamId
      +createdByUnitId
      +bridgeSlotId
      +status
      +cooldownRemaining
    }

    class ActiveObstacle {
      +id
      +ownerTeamId
      +createdByUnitId
      +position
      +status
      +cooldownRemaining
    }

    class Unit {
      +id
      +ownerTeamId
      +unitType
      +hp
      +position
      +statuses
      +role
    }

    class UnitPosition {
      +kind
      +x
      +y
      +baseId
      +slotId
      +bridgeId
      +bridgeCellIndex
    }

    class UnitStatus {
      +kind
      +remainingTurns
      +sourceId
    }

    class ActionIntent {
      +teamId
      +turnNumber
      +productionChoices
      +movementIntents
      +attackIntents
      +strategistActions
    }

    class MovementIntent {
      +unitId
      +from
      +to
      +stay
    }

    class AttackIntent {
      +attackerUnitId
      +targetKind
      +targetUnitId
      +targetBaseId
    }

    class StrategistAction {
      +unitId
      +actionType
      +target
    }

    class BattleEvent {
      +id
      +attackerUnitId
      +targetUnitId
      +targetBaseId
      +baseSuccessDenominator
      +finalSuccessDenominator
      +encouraged
      +result
    }

    class BattleLog {
      +id
      +turnNumber
      +message
      +eventType
      +relatedIds
    }

    GameSession "1" --> "1" GameConfig
    GameSession "1" --> "1" GameState
    GameSession "1" --> "1" BoardMap
    GameSession "1" --> "*" PlayerSlot

    GameState "1" --> "1" TurnState
    GameState "1" --> "*" Team
    GameState "1" --> "*" Unit
    GameState "1" --> "*" Base
    GameState "1" --> "*" ActiveBridge
    GameState "1" --> "*" ActiveObstacle
    GameState "1" --> "*" ScoreState

    BoardMap "1" --> "*" Tile
    BoardMap "1" --> "*" Base
    BoardMap "1" --> "*" BridgeSlot

    Base "1" --> "*" BaseSlot
    BaseSlot "0..1" --> "1" Unit : contains

    Team "1" --> "*" Unit : owns
    Team "1" --> "*" Base : controls
    Team "1" --> "1" ScoreState

    Unit "1" --> "1" UnitPosition
    Unit "1" --> "*" UnitStatus

    ActiveBridge "1" --> "1" BridgeSlot
    ActiveBridge "1" --> "1" Unit : createdBy
    ActiveObstacle "1" --> "1" Unit : createdBy

    TurnState "1" --> "*" ActionIntent
    ActionIntent "1" --> "*" MovementIntent
    ActionIntent "1" --> "*" AttackIntent
    ActionIntent "1" --> "*" StrategistAction

    BattleEvent "0..*" --> "0..1" Unit : attacker
    BattleEvent "0..*" --> "0..1" Unit : target
    BattleEvent "0..*" --> "0..1" Base : targetBase
    BattleLog "0..*" --> "0..*" BattleEvent
```

---

# 4. 主要概念の説明

## 4.1 GameSession

1試合そのものを表す。

保持するもの:

- 試合ID
- 試合状態
- 設定
- 現在のゲーム状態
- マップ
- 参加枠
- ログ

オンライン化した場合も、1つのルーム/試合に対応する単位として扱える。

## 4.2 GameConfig

試合開始前に決まる設定。

例:

- プレイヤー人数
- 試合形式
- 制限ターン数
- 生産間隔
- 橋クールダウン
- 障害物クールダウン
- マップID

ゲーム中に基本的には変更しない。

## 4.3 GameState

試合中に変化する状態の中心。

保持するもの:

- 現在ターン数
- 現在フェーズ
- チーム一覧
- 駒一覧
- 拠点一覧
- 設置中の橋
- 設置中の障害物
- 得点
- ターン状態

Codex実装では、まず `GameState` を中心に純粋関数で処理するのが安全。

## 4.4 TurnState

現在ターンの入力状況を表す。

同時行動型なので、各チームの入力を `ActionIntent` として集めてから解決する。

## 4.5 Team

ゲーム内の陣営。

ユーザー/CPUとは別に、ゲーム内の陣営として扱う。

## 4.6 PlayerSlot

参加枠。

CPUやオンライン参加待ちを「モード」ではなく、参加枠の種類として扱う。

| kind | 意味 |
|---|---|
| human_local | ローカル人間 |
| human_online | オンライン人間 |
| cpu | CPU |
| open | 募集中 |
| empty | 空き |

## 4.7 BoardMap

マップ定義。

保持するもの:

- タイル一覧
- 拠点一覧
- 橋候補スロット一覧
- 湖ID
- 道グループID

ゲームロジック上のマップは2D座標で持つ。

## 4.8 Tile

通常のマス。

地形種別:

- road
- lake

橋や障害物は静的な `Tile` に直接書き込まず、`ActiveBridge` と `ActiveObstacle` から動的に判定する。

## 4.9 Base

拠点。

保持するもの:

- 拠点ID
- 拠点種別
  - home
  - neutral
  - normal
- 所有者チームID
- 2×2座標
- 内部スロット
- 奥座敷スロットID
- 占拠優先権チームID

拠点内の駒は、通常地上座標ではなく `BaseSlot` で管理する。

## 4.10 BaseSlot

拠点内の収容枠。

スロット例:

- front_1
- front_2
- front_3
- protected

本拠地の場合、`protected` が奥座敷枠となる。

## 4.11 Unit

駒。

保持するもの:

- 駒ID
- 所有チームID
- 兵種
- HP
- 現在位置
- 状態
- 軍師の場合は役割

`Unit` は必ず `UnitPosition` を持つ。
初期配置される各チームの軍師は `role = encourage` を持つ鼓舞型軍師とする。  
王は本拠地のprotected slotに配置し、初期鼓舞型軍師は王とは別のBaseSlotに配置する。

## 4.12 UnitPosition

駒の位置。

| kind | 意味 |
|---|---|
| tile | 通常地上マス |
| water | 水面上 |
| base | 拠点内 |
| bridge | 橋上 |
| removed | 撃破/除去済み |

リプレイやログを考えるなら、撃破済み駒を削除せず `removed` 状態として残す方が扱いやすい。

## 4.13 UnitStatus

駒に付く一時状態。

例:

| status | 意味 |
|---|---|
| retreating | 撤退中 |
| encouraged | 鼓舞中 |
| cannot_attack | 攻撃不可 |

撤退中は全兵種に発生し得る共通状態。  
ただし、歩兵のみ防御補正を得る。

## 4.14 ActiveBridge

現在設置中の橋。

保持するもの:

- 所有チームID
- 作成した建設軍師ID
- 使用している `BridgeSlot`
- 状態
  - none
  - active
  - cooldown
- クールダウン残りターン

橋そのものの形は `BridgeSlot` が持つ。  
`ActiveBridge` は「どのチームがどの橋候補を使っているか」を表す。

## 4.15 BridgeSlot

マップ上で橋を架けられる候補。

保持するもの:

- 湖ID
- 橋セル座標一覧
- 縦/横
- 始点側の隣接道
- 終点側の隣接道

橋は自由生成ではなく、マップ側に定義された `BridgeSlot` の中から選ぶ。

## 4.16 ActiveObstacle

現在設置中の障害物。

保持するもの:

- 所有チームID
- 作成した建設軍師ID
- 位置
- 状態
- クールダウン残りターン

橋上に設置されている場合、位置は `bridge` または該当座標として表す。  
橋が消滅した場合、その橋上の障害物も消滅する。

## 4.17 ActionIntent

各チームが1ターン中に入力した行動予定。

含むもの:

- 生産選択
- 移動予定
- 攻撃対象選択
- 軍師アクション

同時行動ゲームなので、入力をすぐ反映せず、フェーズ解決時にまとめて処理する。

## 4.18 MovementIntent

移動予定。

- 駒ID
- 移動前位置
- 移動先位置
- 移動しないか

解決時に合法性を再判定する。

## 4.19 AttackIntent

攻撃予定。

- 攻撃する駒
- 攻撃対象
- 対象が駒か拠点か
- 拠点攻撃の場合、拠点内のどの駒を狙うか

## 4.20 StrategistAction

軍師アクション。

想定:

- placeBridge
- resetBridge
- placeObstacle
- resetObstacle
- teleportUnit

鼓舞はパッシブ扱いのため、Action にしない可能性が高い。

## 4.21 BattleEvent

戦闘解決時に生成される攻撃イベント。

保持するもの:

- 攻撃者
- 対象
- 対象拠点
- 基本攻撃成功確率
- 鼓舞補正の有無
- 最終攻撃成功確率
- 結果

戦闘開始時点で有効な攻撃予定だけを攻撃イベントに変換し、すべての有効イベントを完全同時に判定する。  
鼓舞補正の有無は、戦闘解決開始直前の盤面から鼓舞対象をスナップショットして決定する。  
priorityTierによる階層解決は廃止案とし、正式仕様としては採用しない。

## 4.22 BattleLog

表示/検証用ログ。

例:

- 水計
- 橋消滅
- 拠点攻撃
- 王撃破
- 撤退解除
- 中立拠点制圧

---

# 5. Codex向け実装メモ

## 5.1 最初に実装すべき中心型

```ts
type GameState = {
  turnNumber: number;
  phase: TurnPhase;
  teams: Team[];
  units: Unit[];
  bases: Base[];
  activeBridges: ActiveBridge[];
  activeObstacles: ActiveObstacle[];
  scores: ScoreState[];
  turnState: TurnState;
};
```

## 5.2 駒位置型

```ts
type UnitPosition =
  | { kind: "tile"; x: number; y: number }
  | { kind: "water"; x: number; y: number }
  | { kind: "base"; baseId: string; slotId: string }
  | { kind: "bridge"; bridgeId: string; cellIndex: number }
  | { kind: "removed"; reason: "defeated" | "water_trap" | "king_defeat_reset" };
```

## 5.3 兵種型

```ts
type UnitType =
  | "king"
  | "infantry"
  | "cavalry"
  | "archer"
  | "engineer"
  | "ninja"
  | "apprentice_ninja"
  | "strategist";
```

## 5.4 軍師役割型

```ts
type StrategistRole =
  | "encourage"
  | "builder"
  | "teleporter";
```

## 5.5 状態型

```ts
type UnitStatusKind =
  | "retreating"
  | "encouraged"
  | "cannot_attack";
```

撤退中は全兵種に発生し得る。  
ただし、防御補正は歩兵のみ。

## 5.6 橋と障害物

```ts
type ActiveBridge = {
  id: string;
  ownerTeamId: string;
  createdByUnitId: string;
  bridgeSlotId: string | null;
  status: "none" | "active" | "cooldown";
  cooldownRemaining: number;
};
```

```ts
type ActiveObstacle = {
  id: string;
  ownerTeamId: string;
  createdByUnitId: string;
  position: UnitPosition | null;
  status: "none" | "active" | "cooldown";
  cooldownRemaining: number;
};
```

## 5.7 実装上の注意

- `Tile` に橋や障害物を直接書き込まず、`activeBridges` と `activeObstacles` から動的に通行可否を判定する。
- 拠点内の駒を通常座標に置かない。
- 本拠地奥座敷は `BaseSlot.kind = "protected"` で表す。
- 橋リセット後は即再設置しない。5ターンクールダウンに入る。
- 橋と障害物のクールダウンは別管理。
- `ActionIntent` は入力情報であり、解決前にゲーム状態へ反映しない。
- 移動候補や射程は2Dグリッド座標で計算する。
- isometric表示座標をロジックに使わない。

---

# 6. 未確定/TBD

| 項目 | 状態 |
|---|---|
| 得点計算の最終式 | TBD |
| 転送軍師の移動可能範囲詳細 | TBD |
| 軍師の正式な総数上限 | TBD |
| 軍師役割ごとの複数生産可否 | TBD |
| 鼓舞範囲重複時の生産・移動制限 | TBD |
| 生産上限の最終値 | 暫定 |
| CPUプレイヤーの意思決定モデル | TBD |
| DB保存形式 | オンライン化時に検討 |
| API設計 | オンライン化時に検討 |
# 攻略・褒賞ドメイン追補

`GameState` は拠点別 `SiegeState[]` と複数の `RewardPlacementRequest[]` を持つ。`SiegeState` は攻略対象所有チーム、チーム別撃破数・有効攻撃ターン数、最後の有効攻撃ターン、継続状態、守備駒損失、陥落決定候補を保持する。`RewardPlacementRequest` は要求ID、対象チーム、`capture_reward | contribution_compensation`、発生元拠点、固定・選択式配置先、兵種選択、完了・失効状態と理由を保持する。

`GameState.kingCampaignStates` は王Unitごとの `KingCampaignState` を保持する。各状態は王Unit ID、所属チームID、チーム別 `cumulativeDamage` と `effectiveAttackTurns` を持つ。拠点攻略状態と異なり無攻撃リセットを持たない。

`RewardPlacementRequest.rewardType` は既存2種に `king_conquest_reward`、`king_contribution_compensation`、`overridden_capture_compensation` を加える。王撃破褒賞は継承拠点固定、その他2種は対象チーム所有拠点から選択する。

所有権移転は `Base.ownerTeamId`、旧・新チームの `controlledBaseIds` を一括更新し、攻略状態を即時リセットする。これにより `getBaseControllerTeamId` と騎兵の自軍拠点経由判定も新所有者を参照する。
# Phase 3-B 撤退状態の不変条件（2026-07-14）

- `retreatEligible` は戦闘開始時の参加資格と、最終盤面における生存・active所属・現在所有自軍拠点への合法経路・距離が短くなる合法移動先の両方を満たす場合だけ成立する。
- `retreating` は撤退資格を持つ駒が自軍拠点への合法経路距離を短縮する移動を実行した後の正式状態であり、固定した `retreatTargetBaseId` を保持する。死亡、チーム敗北、目標拠点到着、継戦移動、待機、目標の所有権喪失または目標までの合法経路消失で除去される。
- `retreatTargetBaseId` が失効しても、別の到達可能拠点へ自動再割当てしない。失効時は撤退状態と資格を解除して通常状態へ戻す。
- 撤退経路探索は参照処理であり、GameState、拠点所有権、撤退フラグを変更しない。
- 防御対象が `infantry` かつ `retreating` の場合、既存最終命中確率へ0.5を一度だけ乗算する。
# Phase 4-A 建設状態（2026-07-15）

- `Construction` はID、`bridge | obstacle`、所有チーム、管理軍師、構成座標、設置ターン、active状態を持つ。権利はチーム共有ではなく管理軍師単位である。
- `StrategistActionIntent` はチーム、軍師、アクション種別、対象座標または既存設備IDを持つ。未解決Intentは盤面地形や移動可能性を変更しない。
- `StrategistCooldown` は管理軍師と設備種別ごとの絶対再使用ターンを持つ。橋と障害物は独立する。
- 橋による道路接続はGameState上のactive設備から動的に導出する。障害物は移動グラフだけから除外し、攻撃グラフには影響させない。
- `strategistSubmittedTeamIds` が全activeチームを含むまで設備状態を解決しない。
- デバッグUIの操作チームIDはGameStateの所有権を変更せず、Production入力対象、軍師入力対象、自チーム用プレビューの表示スコープだけを切り替える。
- 橋候補の同一性は構成タイル列の順方向・逆方向に依存しない正規化キーで判定する。
# Phase 4-B ドメインモデル追補（2026-07-18確定）

`Construction.ownerTeamId` と `Construction.managerUnitId` は任意値とする。管理軍師死亡時は `managerUnitId` だけを解除し、active状態と所有権、盤面効果を維持する。単独王撃破による征服では `ownerTeamId` を征服チームへ移し、`managerUnitId` を解除する。複数王同時撃破による中立化では両方を解除し、永続的な残留設備として扱う。

`Team.defeatedUnitCount` は水計による敵駒除去を含むチーム撃破実績であり、`SiegeTeamRecord.defenderKills` および `KingAttackContribution` とは独立する。`Team.conqueredTeamIds` は単独王撃破で正式に征服したチームIDの一意集合で、中立化は含めない。`Team.constructionCapacityBonusStrategistId` は正式征服数1のときに追加管理枠を割り当てた生存建設軍師を示す。

管理可能数は `conqueredTeamIds.length` と追加枠割当先から導出する。0件は各軍師各種1、1件は割当軍師のみ各種2、2件以上は両軍師各種2である。管理設備は配列で扱い、単一IDを前提としない。管理権割当は、active、所有者あり、管理者なしの設備と、同一チームの生存建設軍師を指定して行い、種別ごとの上限を検証する。

手動橋リセット解決は、全有効リセットIntentの橋・橋上駒をスナップショットし、橋と重複障害物をinactive化して各管理枠へT+5クールタイムを設定した後、水計対象を一括処理する。忍者は元座標の `UnitPosition.kind = "water"` へ変換する。非忍者一般兵は `reason = "water_trap"` で除去する。王は1ダメージを受け、生存時は全盤面の最寄り空き通常道路、所有拠点BaseSlot、死亡の順で重複なしに割り当てる。通常道路候補は作戦圏、所有関係、`roadSectionId` を問わず、元橋タイルとのチェビシェフ距離で比較し、駒、active障害物、active橋が存在するタイルを除外する。同距離候補は座標で正規化してから `resolveStrategistActions` へ注入したRNGで選択する。

作戦圏は所有拠点に接続する静的道路区間に加え、その作戦圏道路へ接続するactive橋の全構成タイルを含む。橋の所有チームは問わないが、橋の対岸にある別道路区間は橋の存在だけでは作戦圏へ追加しない。これは障害物候補等の作戦圏判定で用い、静的 `roadSectionId`、通常移動、攻撃トポロジー、橋設置条件は変更しない。

水計で湖へ残った忍者の通常の湖内移動、および橋へ上がれない・橋を横切れない処理は後続Phaseの実装事項とする。

水計による敵王ダメージは `recordKingDamage` を使用するが `recordKingAttackTurns` は呼ばない。王死亡時は `DefeatedKingPlan` を生成して既存の `resolveKingDefeats` へ渡し、通常戦闘と同じ征服・中立化・チーム敗北・褒賞処理へ接続する。
