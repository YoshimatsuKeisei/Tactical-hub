# UIブランチと新u54 PPOブランチのゲーム仕様差分監査

監査日: 2026-10-19 (Asia/Tokyo)

## 比較対象

- UI側: `experiment/pixi-tilemap`、HEAD `20c01777095261ad5f872c85f002499401ecd621`。比較時点で多数の追跡済み変更と未追跡ファイルが存在するため、HEADだけでなく現在の作業ツリー内容を読み取り専用で確認した。
- PPO側: `experiment/u54-newrules-ability-audit`、固定commit `096ad1daff4975db8697f728b8590d94de5b3f88`。
- UI側へのcheckout、reset、clean、add、commit、merge、cherry-pick、ファイル変更は行っていない。

分類は「一致」「実装差はあるが確認できた動作は一致」「ゲーム挙動差」「片側のみ」「意図確認が必要」「未検証」とした。

## 機能別結果

| 機能 | 分類 | UI側の現在実装 | PPO側の現在実装 | 根拠・リスク | 統合優先度 |
|---|---|---|---|---|---:|
| 撤退モード廃止 | 一致 | `retreat.ts`で旧状態を消去し、Encoder予約列を0固定 | 同じ互換方式 | `clearLegacyRetreatState()`、`buildUnitTurnFlag()`、retreat/Encoderテスト。通常の離脱移動は残る | 済 |
| 通常移動 | 実装差はあるが基本動作は一致 | 即時移動、拠点入退場、道路接続を使用 | 同系統だが拠点接続判定のキャッシュ化、RL事前検証、final-duel同期を追加 | `movement.ts`。基本移動の全面不一致は未検出だが、特殊移動と終局同期が違う | 中 |
| 攻撃成功率 | ゲーム挙動差 | archer→ninja は1/5、ninja→archer は1/7 | archer→ninja は1/7、ninja→archer は1/3 | `battle.ts`の`SUCCESS_DENOMINATORS`。学習分布へ直接影響 | 最優先 |
| 戦闘演出メタデータ | UI側のみ | `BattlePresentationMetadata`を生成しUIへ渡す | ゲーム状態を変えない能力監査イベントのみ | `battle.ts`、`cpuStep.ts`、`App.tsx`。UI統合時に演出を落とす恐れ | 高 |
| 王の鼓舞能力 | ゲーム挙動差 | 鼓舞役は`strategist`かつrole=`encourage`のみ | 王と軍師の両方が鼓舞源 | 両側の`encouragement.ts`。命中率と能力統計へ影響 | 最優先 |
| 王敗北時の強制退却・再配置 | 実装差はあるが確認できた動作は一致 | construction phaseの既存退却ロジックを保持 | 同ロジックにfinal-duel同期を追加 | `construction.ts`、`constructionPhase4B.test.ts`。撤退モードとは別機能 | 高 |
| 軍師の役割 | ゲーム挙動差 | `builder` / `encourage` / `teleporter`を別駒として生産・判定 | 全軍師が建設・鼓舞・転送可能。role列は互換用 | `production.ts`、`construction.ts`、`encouragement.ts`、`teleport.ts` | 最優先 |
| 軍師の鼓舞範囲表示 | UI側のみ | `BoardView.tsx`が`getEncourageAreaTileKeys()`でタイルを強調 | 表示を削除し能力は内部保持 | ユーザー指定とPPO側は一致。UI側は表示が残る | 高 |
| 橋・障害物 | 実装差 | UI側はbuilder roleのみ使用可能 | PPO側は統合軍師が使用可能 | 配置・競合・cooldown本体は概ね共通。能力者の資格が違う | 最優先 |
| 軍師転送 | ゲーム挙動差 | teleporter roleのみ | 全軍師 | `teleport.ts::getTeleportStrategists()` | 最優先 |
| 忍者の水上移動 | 実装差はあるが確認できた動作は一致 | 水上移動・視認処理あり | 同機能に追加能力と監査を付加 | `ninjaWaterMovement.test.ts`は両側に存在 | 中 |
| 忍者の生涯1回瞬間移動 | PPO側のみ | 実装なし | `ninjaTeleport.ts`と`ninjaTeleportUsedUnitIds` | UIでは操作不能。PPOが学習した行動をUIゲームで再現できない | 最優先 |
| 忍者の障害物越え | PPO側のみ | 実装なし | `movement.ts`に障害物1マス越え経路 | UIでは合法Actionが欠落 | 最優先 |
| 重装歩兵への統合 | 一致 | `heavyInfantry.ts`あり | ファイル内容に差分なし | UIとPPOでエンジン本体は同一。RL schema v2のみPPO側で強く検証 | 高 |
| 拠点制圧 | ゲーム挙動差 | neutral王撃破時に敗北完了・拠点中立化、neutral/敗北勢力を征服候補から除外 | 旧処理で`selectConquestTeam()`がteam状態を見ず、征服者なしでは敗北処理を抜ける | `defeat.ts`、`kingCampaign.ts`、`kingDefeat.test.ts`。ゾンビ勢力・不正褒章リスク | 最優先 |
| 通常生産 | ゲーム挙動差 | UIは空きslotを明示選択し`ProductionChoice.slotId`へ保存 | PPOは拠点内の最初の空きslotへ自動配置 | `production.ts`、`types.ts`、`ui/unitPlacement.ts`。盤面状態とAction表現が異なる | 最優先・設計判断 |
| Random CPUの生産兵種 | ゲーム挙動差 | infantry/archer/cavalryへ制限 | 全6兵種から選択 | `randomCpuPolicy.ts`。以前確認された不具合修正はUI側だけに存在 | 最優先 |
| Heuristic CPUの生産兵種 | 一致 | `PREFERRED_TYPES`で3基本兵種を選び、なければpass | 同じ優先ロジック | `heuristicCpuPolicy.ts::choosePreferredProduction()`。軍師固定移動だけ役割統合差あり | 高 |
| 褒章配置 | ゲーム挙動差 | active非neutral、所有拠点、fixed base、空slot、兵種を再検証。不正要求を失効しphase復帰。slot明示選択 | 旧処理。受取資格とfixed base防御が不足し、不正要求失効の自動復帰なし | `reward.ts`、`rewardPlacementFlow.test.ts`。実測seed 417で2442 action後に非終局・行動者なしstall | 最優先 |
| neutral勢力 | ゲーム挙動差 | 王撃破の征服者にならず褒章も得ない | neutralが候補になり得る旧処理 | 上記king/reward差。通常neutral自動攻撃自体は両側に存在 | 最優先 |
| 最終決闘・試合終了 | PPO側のみ | active team数による従来終局のみ | `finalDuel.ts`、連続優勢・timeout勝敗/引分を保持 | 学習のterminal/reward分布とUIの勝敗が不一致 | 最優先・設計判断 |
| PPO安全上限・時間切れ裁定 | PPO側のみ | UI通常試合には非該当 | safety turn/action adjudicationあり | 学習専用のためUIへそのまま移植不要。ただしUI対戦結果との比較時に区別が必要 | 維持 |
| Active Team=`all`表示 | UI側のみ・意図確認済み | 同時行動phaseで`all` | 学習UIなし | `App.tsx`、`PlayScreenLayout.test.tsx`。ゲームルール差ではない | 不要 |
| CONFIRM補足文の英語化 | UI側のみ・意図確認済み | 英語文へ修正済み | 学習UIなし | `App.tsx`とlayout test | 不要 |

## PPO学習へ影響する差分

特に影響が大きいのは、攻撃成功率、王の鼓舞、軍師統合、忍者2能力、final duel、neutral王撃破、褒章失効、Random CPU生産制限、slot指定生産である。これらは合法Action集合、状態遷移、命中分布、終局、対戦相手分布のいずれかを変える。

PPO側のObservation/Action Feature Specは今回変更していないが、UI側の`slotId`付き生産をそのままActionへ追加するとAction schemaとCheckpoint互換性を破る可能性がある。UIのslot選択をゲーム表示だけの選択とみなすか、PPO側でもAction化するかは、統合前に仕様判断が必要である。

## 学習済みモデルをUIゲームへ接続する際のリスク

1. UI側に忍者瞬間移動・障害物越え・統合軍師・final duelがないため、モデルが選ぶ合法ActionをUI側が生成または適用できない。
2. UIとPPOで攻撃成功率および鼓舞源が違い、同じ盤面・同じ乱数でも結果が分岐する。
3. UI側のslot指定生産とPPO側の自動slot選択で、同じ生産兵種でも以後の盤面が分岐する。
4. PPO側にneutral征服・不正褒章の旧バグが残り、長時間試合がreward placementで停止し得る。
5. Random CPU対戦相手の生産分布が違い、UIで想定する3基本兵種縛りと学習相手が一致しない。
6. UI側の戦闘演出メタデータはゲーム結果とは独立だが、統合時に落とすと画面上の攻撃・命中演出が欠落する。

## 推奨する最小統合方針

1. まず正とするゲーム仕様を機能ごとに決定する。特に攻撃成功率、王の鼓舞、統合軍師、slot指定生産、final duelは推測で選ばない。
2. 確定済み不具合修正として、neutral王撃破・褒章防御/失効・Random CPU 3兵種制限をPPO固定コードへ最小移植し、Feature Spec不変を再検証する。
3. UI側へは忍者2能力、統合軍師、必要なfinal-duel表示/操作をゲームエンジン単位で移植し、UI素材やレイアウト変更とは分離する。
4. 生産slotは既存PPO Actionを増やさず互換層で解決できるか先に設計する。Action type/width変更が必要なら既存Checkpoint継続とは別案件にする。
5. 統合後、同一seed・同一Action Key列で状態hash、乱数、終局を比較してから新u54学習へ進む。

## 未検証

- UI作業ツリーは未コミットのため、将来の編集後も本レポートと同じ状態である保証はない。
- UIとPPOの全合法Actionを全盤面で網羅比較してはいない。
- 視覚演出・アセット・レイアウトの画面確認は今回の禁止事項に従い実施していない。
- 差分は報告のみであり、今回ゲーム仕様の統合作業は行っていない。
