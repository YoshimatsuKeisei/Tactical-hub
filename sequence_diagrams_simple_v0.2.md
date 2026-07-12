# シーケンス図 v0.2（簡略版）

この文書は、戦略陣取りゲームの主要処理を、できるだけシンプルな登場人物で表したシーケンス図です。  
v0.1 は実装サービス名を細かく分けすぎていたため、まずはこの v0.2 を読み取り用・仕様共有用の正本とします。

---

## Phase 3-A: 撤退処理シーケンス

```mermaid
sequenceDiagram
    autonumber
    participant App as Game logic
    participant State as GameState
    participant Plan as ActionIntent
    participant Log as GameLog

    App->>Plan: AttackIntent を取得する
    App->>State: 戦闘開始時点の生存駒と鼓舞対象をスナップショットする
    App->>App: 撤退中の攻撃者は BattleEvent に変換しない
    App->>App: 有効な BattleEvent を完全同時に解決する
    App->>State: HP、撃破、王撃破を反映する
    App->>State: 有効 BattleEvent の攻撃者/対象から unitTurnFlags を作成する
    App->>Log: 撤退資格を得た駒を記録する

    App->>Plan: MovementIntent を取得する
    App->>State: 合法移動か再判定する
    alt 撤退資格あり and 最寄り自軍拠点へ近づく
        App->>State: retreating 状態を付ける
    else 撤退中 and 距離が同じ/短い
        App->>State: retreating 状態を維持する
    else 撤退中 and 距離が長い
        App->>State: retreating 状態を解除する
    else 撤退中 and 自軍拠点へ入る
        App->>State: 拠点内へ入り retreating 状態を解除する
    else 撤退中 and stay
        App->>State: retreating 状態を解除する
    end
    App->>State: 移動解決後に unitTurnFlags をクリアする
```

Phase 3-A では、攻撃フェーズで攻撃を選んで撤退解除する処理は実装しない。
撤退解除は移動解決時の stay、遠ざかる合法移動、自軍拠点到達で行う。
撤退中の駒は攻撃不可だが、被攻撃確率補正はまだ発生しない。

撤退資格の参加者一覧は、有効な BattleEvent を確定した直後、命中判定より前に攻撃者と対象者の両方から作る。
このため、攻撃が失敗してダメージが0でも、対象者が戦闘後に生存していれば撤退資格の判定対象になる。
敵拠点3マス以内の判定には、BattleEvent 確定時点の位置を使う。

# 1. この版の方針

## 1.1 登場人物

この版では、登場人物を基本的に以下だけに絞ります。

| 登場人物   | 意味                                                   |
| ---------- | ------------------------------------------------------ |
| プレイヤー | 操作する人                                             |
| 画面       | 盤面UI、ボタン、候補表示など                           |
| ゲーム処理 | ルール判定、移動解決、戦闘解決などを行うアプリ内部処理 |
| 盤面データ | 現在の駒、拠点、橋、障害物、得点などの状態             |
| 入力予定   | 同時行動を解決する前に、一時保存する入力               |
| ログ       | 戦闘結果や水計などの記録                               |
| 乱数       | 攻撃成功判定に使う乱数                                 |

## 1.2 v0.1との違い

v0.1では、以下のような細かい担当を出していました。

- GameEngine
- RuleService
- MovementService
- BattleService
- BaseService
- BridgeService
- ScoreService

これは実装詳細に寄りすぎています。  
学習・仕様理解段階では、まず「ゲーム処理」にまとめて構いません。

---

# 2. 全体ターン進行

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ
    participant Log as ログ

    App->>State: 現在ターンを開始する
    App->>UI: 生産フェーズを表示する

    P->>UI: 生産する駒と拠点を選ぶ
    UI->>Plan: 生産予定を保存する

    App->>Plan: 全員の生産予定を取得する
    App->>State: 拠点内に駒を生産する
    App->>Log: 生産ログを記録する

    App->>UI: 移動入力フェーズを表示する
    P->>UI: 駒の移動先を選ぶ
    UI->>Plan: 移動予定を保存する

    App->>Plan: 全員の移動予定を取得する
    App->>State: 移動をまとめて解決する
    App->>Log: 移動ログを記録する

    App->>UI: 攻撃対象選択フェーズを表示する
    P->>UI: 攻撃対象を選ぶ
    UI->>Plan: 攻撃予定を保存する

    App->>Plan: 全員の攻撃予定を取得する
    App->>State: 戦闘をまとめて解決する
    App->>Log: 戦闘ログを記録する

    App->>Plan: 軍師アクション予定を取得する
    App->>State: 橋・障害物・転送を処理する
    App->>Log: 軍師アクションログを記録する

    App->>State: 王撃破/拠点/得点/クールダウンを後処理する

    alt 試合終了
        App->>UI: 結果画面を表示する
    else 試合継続
        App->>State: 次ターンへ進める
        App->>UI: 次ターンを表示する
    end
```

## 日本語で言うと

1ターンの流れを、上から順に示した図です。  
重要なのは、プレイヤーが入力した瞬間に盤面を変えるのではなく、まず `入力予定` に保存し、フェーズ解決時にまとめて `盤面データ` を更新することです。

---

# 3. 生産処理

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ

    P->>UI: 生産画面を開く
    UI->>App: 生産可能な拠点と兵種を確認する
    App->>State: 自軍拠点の空き枠を確認する
    State-->>App: 空き枠情報を返す
    App-->>UI: 生産候補を表示する

    P->>UI: 生産する兵種と拠点を選ぶ
    UI->>Plan: 生産予定を保存する

    App->>Plan: 生産予定を取得する
    App->>State: 拠点内の空き枠に駒を追加する

    Note over App,State: 生産された駒は通常地上ではなく、必ず拠点内に配置する
```

---

# 4. 移動入力

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ

    P->>UI: 駒を選ぶ
    UI->>App: この駒の移動候補を確認する
    App->>State: 駒の現在位置と周囲の地形を確認する
    State-->>App: 位置・道・湖・拠点・橋・障害物の情報を返す
    App-->>UI: 移動候補を表示する

    P->>UI: 移動先を選ぶ
    UI->>Plan: 移動予定を保存する

    Note over UI,Plan: この時点では、まだ盤面データの駒位置は変えない
```

---

# 5. 移動解決

```mermaid
sequenceDiagram
    autonumber
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ
    participant Log as ログ

    App->>Plan: 全プレイヤーの移動予定を取得する
    App->>App: 移動優先順で並べる

    loop 各移動予定
        App->>State: 移動先が今でも空いているか確認する
        State-->>App: 合法/非合法を返す

        alt 合法
            App->>State: 駒の位置を更新する
            App->>Log: 移動成功を記録する
        else 非合法
            App->>State: 駒を現在位置に残す
            App->>Log: 移動失敗を記録する
        end
    end

    App->>State: 移動後盤面を確定する
```

## 日本語で言うと

全員の移動予定をまとめて取り出し、優先順で1つずつ処理します。  
入力時には合法だった移動でも、他の駒の移動結果によって移動先が埋まる可能性があるため、解決時に再判定します。

---

# 6. 攻撃対象選択

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ

    App->>State: 移動後の盤面を確認する
    App->>UI: 攻撃対象候補を表示する

    P->>UI: 攻撃対象を選ぶ

    alt 撤退中の駒で攻撃しようとした場合
        UI->>P: 撤退を解除して攻撃するか確認する
        alt 防御継続
            UI->>Plan: 攻撃予定を作らない
        else 撤退解除して攻撃
            UI->>Plan: 撤退解除と攻撃予定を保存する
        end
    else 通常の攻撃
        UI->>Plan: 攻撃予定を保存する
    end
```

## 日本語で言うと

攻撃対象は、移動前ではなく移動解決後の盤面を見て選びます。  
撤退中の駒は攻撃できないため、攻撃する場合は撤退解除を選ばせます。

---

# 7. 戦闘解決

```mermaid
sequenceDiagram
    autonumber
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ
    participant Dice as 乱数
    participant Log as ログ

    App->>Plan: 全プレイヤーの攻撃予定を取得する
    App->>State: 戦闘開始直前の鼓舞対象を確定する
    App->>App: 攻撃予定を戦闘イベントに変換する
    App->>App: 各イベントの攻撃成功確率を決定する

    loop すべての有効な戦闘イベント
        App->>Dice: 攻撃成功判定を行う
        Dice-->>App: 成功/失敗を返す
        App->>App: 判定結果を一時保存する
    end

    App->>App: 対象ごとに成功数を合算する
    App->>State: HP減少・撃破・王撃破をまとめて反映する
    App->>Log: 成功/失敗/ダメージ/撃破を記録する
    App->>State: 戦闘後盤面を確定する
```

---

# 8. 拠点攻撃と奥座敷

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ

    P->>UI: 拠点を攻撃対象として選ぶ
    UI->>App: 拠点内の攻撃可能駒を確認する
    App->>State: 拠点内リストを取得する
    App->>State: 奥座敷の保護条件を確認する

    alt 奥座敷の駒が保護中
        App-->>UI: 奥座敷の駒を候補から除外して表示する
    else 奥座敷の駒も攻撃可能
        App-->>UI: 拠点内の全攻撃可能駒を表示する
    end

    P->>UI: 拠点内の攻撃対象を選ぶ
    UI->>Plan: 拠点攻撃予定を保存する
```

---

# 9. 撤退処理

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant Plan as 入力予定
    participant State as 盤面データ

    App->>State: 直前の戦闘参加情報を確認する
    App->>State: 駒が敵拠点から3マス以内で交戦したか確認する
    App->>State: 駒が自軍拠点方向へ移動したか確認する

    alt 撤退条件を満たす
        App->>State: 駒に撤退中状態を付ける
    else 撤退条件を満たさない
        App->>State: 通常状態のままにする
    end

    alt 撤退中の駒が敵と接している
        UI->>P: 防御継続か撤退解除して攻撃か確認する
        alt 防御継続
            UI->>Plan: 攻撃予定なしとして保存する
        else 撤退解除して攻撃
            UI->>Plan: 撤退解除と攻撃予定を保存する
        end
    end

    Note over App,State: 撤退中は全兵種攻撃不可。歩兵のみ被攻撃成功確率低下。
```

---

# 10. 橋リセットと水計

```mermaid
sequenceDiagram
    autonumber
    participant P as プレイヤー
    participant UI as 画面
    participant App as ゲーム処理
    participant State as 盤面データ
    participant Log as ログ

    P->>UI: 建設軍師で橋リセットを選ぶ
    UI->>App: 橋リセット可能か確認する
    App->>State: 橋上に自軍王がいないか確認する
    App->>State: 橋上に管理軍師本人がいないか確認する

    alt リセット不可
        App-->>UI: リセット不可として表示する
    else リセット可能
        App->>State: 橋上の駒を確認する

        loop 橋上の各駒
            alt 敵駒
                App->>State: 敵駒を水没除去する
                App->>State: 橋所有者に撃破点を加算する
                App->>Log: 敵駒水没を記録する
            else 味方駒
                App->>State: 味方駒を水没除去する
                App->>Log: 味方駒水没を記録する
            end
        end

        App->>State: 橋上障害物を消滅させる
        App->>State: 橋をクールダウン状態にする
        App->>Log: 橋リセットを記録する
    end
```

---

# 11. 後処理と試合終了判定

```mermaid
sequenceDiagram
    autonumber
    participant App as ゲーム処理
    participant State as 盤面データ
    participant UI as 画面
    participant Log as ログ

    App->>State: 王が撃破されたチームを確認する

    alt 王撃破チームあり
        App->>State: チームを脱落状態にする
        App->>State: そのチームの橋と障害物を消滅させる
        App->>Log: チーム脱落を記録する
    end

    App->>State: 拠点支配と占拠優先権を更新する
    App->>State: 橋/障害物のクールダウンを減らす
    App->>State: 得点を更新する

    alt 試合終了条件を満たす
        App->>UI: 結果画面を表示する
    else 試合継続
        App->>State: 次ターンへ進める
        App->>UI: 次ターンを表示する
    end
```

---

# 12. 補足

## 12.1 この簡略版で十分な理由

実装初期では、`RuleService` や `BattleService` のような細かい担当を図に出さなくてもよいです。  
まずは、以下の区別ができていれば十分です。

- 画面が受け取る
- 入力予定に保存する
- ゲーム処理がまとめて解決する
- 盤面データを更新する
- ログに残す

## 12.2 実装が進んだら分けるもの

将来的に処理が複雑になったら、以下を内部サービスとして分けてもよいです。

| 簡略版     | 詳細実装で分けるなら |
| ---------- | -------------------- |
| ゲーム処理 | GameEngine           |
| ゲーム処理 | MovementService      |
| ゲーム処理 | BattleService        |
| ゲーム処理 | BaseService          |
| ゲーム処理 | BridgeService        |
| ゲーム処理 | ScoreService         |

ただし、最初から全部を図に出すと読みにくくなるため、この簡略版ではまとめています。
# 占領シーケンス追補

守備隊全滅: `battle_resolution → capture_resolution相当処理 → 所有権移転・要求生成 → reward_placement → 全要求完了/失効 → 次通常フェーズ`。

戦闘中放棄: `movement_resolution → 放棄検出 → capture_resolution相当処理 → 即時所有権移転・要求生成 → reward_placement → 全要求完了/失効 → attack_input`。後入城待ちは行わない。

単純放棄への入城: `movement_resolution → 所有権移転 → attack_input`。褒賞配置要求は生成しない。
