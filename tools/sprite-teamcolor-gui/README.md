# Tactical-hub Sprite Team Color GUI

SAM 2.1でクリック指定したキャラクター部位をアニメーション方向ごとに追跡し、
元の陰影を残したままTactical-hubの4チーム色へ再着色するローカルGUIツールです。

## 現在のMVP仕様

- 対象: 128x128フレームで構成されたPNGスプライトシート
- 列数/行数は画像サイズから自動判定
- 左クリック: positive（色を変えたい対象）
- 右クリック: negative（除外）
- クリックはフレームごとに保持
- SAM 2.1 Video Predictorで現在の方向行を全フレーム追跡
- 誤認したフレームへ追加クリックして再Track可能
- マスクは元PNGのalpha領域へ制限
- 元の明暗を保持したチームカラー変換
- 極端に暗い輪郭は保護可能
- 4チーム分をPNGとして一括出力

Tactical-hub本体の色定義を使用します。

- red: #d94a4a
- blue: #3e7bd8
- green: #36a166
- yellow: #c58a2b

## Windowsセットアップ

PowerShellでこのフォルダを開き、次を実行します。

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
~~~

このセットアップは以下を行います。

1. Python 3.12用の .venv を作成
2. CPU版 torch 2.7.1 / torchvision 0.22.1 を導入
3. Pillow / NumPy を導入
4. Meta公式SAM 2 commit 2b90b9f5ceec907a1c18123530e92e794ad901a4 を導入
5. 公式 sam2.1_hiera_tiny.pt をダウンロード
6. checkpoint SHA-256を検証

確認済みcheckpoint SHA-256:

7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69

## 起動

セットアップ後、run_windows.bat をダブルクリックします。

またはPowerShellから:

~~~powershell
.\.venv\Scripts\python.exe .\app.py
~~~

## 基本操作

1. PNGを開く
2. 方向行を選ぶ
3. まずframe 1で、色を変えたい部位を左クリック
4. 必要なら色を変えたくない近接部位を右クリック
5. 「この方向をTrack」
6. フレームスライダーで追跡結果を確認
7. 誤認があれば、そのフレームへ移動して左/右クリックを追加
8. 「この方向をTrack」をもう一度押す
9. 他の方向行も同様に処理
10. red / blue / green / yellowでプレビュー
11. 「4チーム分のPNGを書き出す」

未Trackの方向がある状態でも書き出しは可能ですが、その方向は元画像のまま残るため警告を表示します。

## 重要

- 元画像を生成AIで描き直す処理ではありません。
- 選択マスク外のピクセルは変更しません。
- alpha値は変更しません。
- 再着色は元の明暗を保持するluminance mappingです。
- SAM 2モデルの初回ロードとCPU追跡には時間がかかります。GUIは別スレッドで実行し、進捗を表示します。
- 現時点ではWindows実機GUI起動は未確認です。GitHub Actionsではコア処理の自動テストを行います。
