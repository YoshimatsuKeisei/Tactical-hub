import { useState } from "react";
import {
  DEFAULT_ADJUSTMENT, LAKE_FRAME_KEYS, missingLakeAnimationAssets, moveHighlightSettings,
  roadSouthShadowSettings, TILE_ASSET_KEYS, TILE_ASSET_LABELS, type TileAssetKey,
} from "../presentation/tileAssets";
import { useTileAssets } from "./TileAssetProvider";
export function TileAssetImportPanel() {
  const [open, setOpen] = useState(false);
  const { store, snapshot } = useTileAssets();
  const moveHighlight = moveHighlightSettings(snapshot);
  const missingLakeFrames = missingLakeAnimationAssets(snapshot);
  const assetField = (key: TileAssetKey) => {
    const asset = snapshot.assets[key];
    const isShadow = key === "road_south_shadow";
    const isAdditionalLakeFrame = key !== "lake" && LAKE_FRAME_KEYS.some((lakeKey) => lakeKey === key);
    const transparentOptional = key === "road_south_face" || isShadow || key.startsWith("base_front_wall") || key.startsWith("base_flag_");
    const shadowSettings = roadSouthShadowSettings(snapshot);
    return <fieldset key={key}>
      <legend>{TILE_ASSET_LABELS[key]}{transparentOptional ? " (透過PNG・任意)" : ""}</legend>
      {isShadow && <label>
        <input type="checkbox" checked={shadowSettings.enabled} onChange={(event) => store?.setRoadSouthShadowEnabled(event.target.checked)} />
        影を表示
      </label>}
      <input aria-label={`${key} PNG`} type="file" accept="image/png,.png" onChange={(event) => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (file) void store?.importFile(key, file);
      }} />
      <span role="status">{snapshot.statuses[key] ?? (isAdditionalLakeFrame ? "未設定：アニメーション停止" : "未設定：仮図形")}{asset ? ` — ${asset.name} ${asset.width}×${asset.height}` : ""}</span>
      <button type="button" onClick={() => store?.reset(key)}>Reset image</button>
      {isShadow ? <div>
        {(["scaleX", "scaleY", "offsetX", "offsetY"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`${key} ${property}`} step={property === "scaleX" || property === "scaleY" ? 0.1 : 1}
            min={property === "scaleX" || property === "scaleY" ? 0.1 : undefined} max={property === "scaleX" || property === "scaleY" ? 4 : undefined}
            value={shadowSettings[property]} onChange={(event) => { if (event.target.value !== "") store?.adjustRoadSouthShadow({ [property]: Number(event.target.value) }); }} />
        </label>)}
        <label>opacity
          <input type="number" aria-label="road_south_shadow opacity" step={0.05} min={0} max={1}
            value={shadowSettings.opacity} onChange={(event) => { if (event.target.value !== "") store?.setRoadSouthShadowOpacity(Number(event.target.value)); }} />
        </label>
        <button type="button" onClick={() => store?.resetRoadSouthShadowSettings()}>Reset shadow settings</button>
      </div> : asset && !isAdditionalLakeFrame && <div>
        {(["scale", "offsetX", "offsetY"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`${key} ${property}`} step={property === "scale" ? 0.1 : 1} min={property === "scale" ? 0.1 : undefined} max={property === "scale" ? 4 : undefined}
            value={asset[property]} onChange={(event) => { if (event.target.value !== "") store?.adjust(key, { [property]: Number(event.target.value) }); }} />
        </label>)}
        <button type="button" onClick={() => store?.adjust(key, DEFAULT_ADJUSTMENT)}>Reset adjustment</button>
      </div>}
    </fieldset>;
  };
  return <section className="tile-asset-import">
    <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>Tile Asset Preview / Import {open ? "— Close" : "(DEV)"}</button>
    {open && <div className="tile-asset-import-panel">
      <h2>Tile Asset Preview / Import</h2>
      <p>ブラウザ内のみ。再読み込み・ゲーム画面終了で消えます。ファイルのupload／保存は行いません。パネルを閉じても共有素材は保持します。</p>
      <label>Pixi素材表示 <select value={snapshot.enabled ? "images" : "shapes"} onChange={(event) => store?.setEnabled(event.target.value === "images")}>
        <option value="shapes">仮図形（Graphics）</option><option value="images">インポート素材＋未設定は仮図形</option>
      </select></label>
      <button type="button" onClick={() => store?.resetAll()}>Reset All</button>
      {snapshot.renderError && <p role="alert">画像描画に失敗したため仮図形へ戻しました：{snapshot.renderError}</p>}
      <p>実盤面はTerrain rendererをPixiJSに切り替えてください。CSSには適用されません。
        X/Yは48px基準の位置調整。通常画像は幅1セル、拠点部品は2×2実測範囲を基準に縦横比を維持します。
        Team 1～4 flagは現在の非中立チーム定義順に対応し、ID文字列は解析しません。道路南側面は同じ高さが南へ続かない境界だけに表示します。</p>
      <fieldset>
        <legend>Lake animation (DEV)</legend>
        <label><input type="checkbox" checked={snapshot.lakeAnimationEnabled === true}
          disabled={!snapshot.enabled || missingLakeFrames.length > 0}
          onChange={(event) => store?.setLakeAnimationEnabled(event.target.checked)} /> 4 frames / 800ms / cross-fade 200ms</label>
        {!snapshot.enabled && <p>「Pixi素材表示」をインポート素材へ切り替えると有効化できます。</p>}
        {missingLakeFrames.length > 0
          ? <p role="status">不足画像：{missingLakeFrames.map((key) => TILE_ASSET_LABELS[key]).join("、")}</p>
          : <p role="status">4枚読み込み済み。湖面アニメーションを有効化できます。</p>}
      </fieldset>
      {LAKE_FRAME_KEYS.map(assetField)}
      <fieldset>
        <legend>Normal unit move highlight / 通常ユニット移動強調</legend>
        <label><input type="checkbox" checked={moveHighlight.normal.enabled}
          onChange={(event) => store?.adjustMoveHighlight("normal", { enabled: event.target.checked })} /> 表示</label>
        {(["topOpacity", "southFaceOpacity", "darken", "outlineOpacity", "outlineLighten"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`normal move highlight ${property}`} min={0} max={property === "darken" ? 0.85 : 1} step={0.05}
            value={moveHighlight.normal[property]} onChange={(event) => { if (event.target.value !== "") store?.adjustMoveHighlight("normal", { [property]: Number(event.target.value) }); }} />
        </label>)}
        <label><input type="checkbox" checked={moveHighlight.normal.outlineEnabled}
          onChange={(event) => store?.adjustMoveHighlight("normal", { outlineEnabled: event.target.checked })} /> 境界線を表示</label>
        <button type="button" onClick={() => store?.resetMoveHighlightSettings("normal")}>Reset normal highlight</button>
      </fieldset>
      <fieldset>
        <legend>Ninja move highlight / 忍者移動強調</legend>
        <label><input type="checkbox" checked={moveHighlight.ninja.enabled}
          onChange={(event) => store?.adjustMoveHighlight("ninja", { enabled: event.target.checked })} /> 表示</label>
        {(["topOpacity", "southFaceOpacity"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`ninja move highlight ${property}`} min={0} max={1} step={0.05}
            value={moveHighlight.ninja[property]} onChange={(event) => { if (event.target.value !== "") store?.adjustMoveHighlight("ninja", { [property]: Number(event.target.value) }); }} />
        </label>)}
        <label>gray (0 black – 160 gray)
          <input type="number" aria-label="ninja move highlight gray" min={0} max={160} step={4}
            value={moveHighlight.ninja.gray} onChange={(event) => { if (event.target.value !== "") store?.adjustMoveHighlight("ninja", { gray: Number(event.target.value) }); }} />
        </label>
        {(["outlineOpacity", "outlineLighten"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`ninja move highlight ${property}`} min={0} max={1} step={0.05}
            value={moveHighlight.ninja[property]} onChange={(event) => { if (event.target.value !== "") store?.adjustMoveHighlight("ninja", { [property]: Number(event.target.value) }); }} />
        </label>)}
        <label><input type="checkbox" checked={moveHighlight.ninja.outlineEnabled}
          onChange={(event) => store?.adjustMoveHighlight("ninja", { outlineEnabled: event.target.checked })} /> 境界線を表示</label>
        <button type="button" onClick={() => store?.resetMoveHighlightSettings("ninja")}>Reset ninja highlight</button>
      </fieldset>
      {TILE_ASSET_KEYS.filter((key) => !LAKE_FRAME_KEYS.some((lakeKey) => lakeKey === key)).map(assetField)}
      <p>拠点前壁はclick-throughの前景Canvasでキャラ本体より手前、HP・badge・選択表示より奥へ配置します。
        その他の障害物とHTMLキャラの正確な部分遮蔽は未対応です。</p>
    </div>}
  </section>;
}
