import { useState } from "react";
import {
  DEFAULT_ADJUSTMENT, moveHighlightSettings, roadSouthShadowSettings, TILE_ASSET_KEYS, TILE_ASSET_LABELS,
} from "../presentation/tileAssets";
import { useTileAssets } from "./TileAssetProvider";
export function TileAssetImportPanel() {
  const [open, setOpen] = useState(false);
  const { store, snapshot } = useTileAssets();
  const moveHighlight = moveHighlightSettings(snapshot);
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
        X/Yは48px基準の位置調整。画像幅を1セルに合わせ、縦横比を維持します。道路南側面は同じ高さが南へ続かない境界だけに表示します。焼き込み側面は自動除去できません。</p>
      <fieldset>
        <legend>Normal unit move highlight / 通常ユニット移動強調</legend>
        <label><input type="checkbox" checked={moveHighlight.normal.enabled}
          onChange={(event) => store?.adjustMoveHighlight("normal", { enabled: event.target.checked })} /> 表示</label>
        {(["topOpacity", "southFaceOpacity", "darken"] as const).map((property) => <label key={property}>{property}
          <input type="number" aria-label={`normal move highlight ${property}`} min={0} max={property === "darken" ? 0.85 : 1} step={0.05}
            value={moveHighlight.normal[property]} onChange={(event) => { if (event.target.value !== "") store?.adjustMoveHighlight("normal", { [property]: Number(event.target.value) }); }} />
        </label>)}
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
        <button type="button" onClick={() => store?.resetMoveHighlightSettings("ninja")}>Reset ninja highlight</button>
      </fieldset>
      {TILE_ASSET_KEYS.map((key) => {
        const asset = snapshot.assets[key];
        const isShadow = key === "road_south_shadow";
        const shadowSettings = roadSouthShadowSettings(snapshot);
        return <fieldset key={key}>
          <legend>{TILE_ASSET_LABELS[key]}{key === "road_south_face" || isShadow ? " (透過PNG・任意)" : ""}</legend>
          {isShadow && <label>
            <input type="checkbox" checked={shadowSettings.enabled} onChange={(event) => store?.setRoadSouthShadowEnabled(event.target.checked)} />
            影を表示
          </label>}
          <input aria-label={`${key} PNG`} type="file" accept="image/png,.png" onChange={(event) => {
            const file = event.target.files?.[0]; event.target.value = "";
            if (file) void store?.importFile(key, file);
          }} />
          <span role="status">{snapshot.statuses[key] ?? "未設定：仮図形"}{asset ? ` — ${asset.name} ${asset.width}×${asset.height}` : ""}</span>
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
          </div> : asset && <div>
            {(["scale", "offsetX", "offsetY"] as const).map((property) => <label key={property}>{property}
              <input type="number" aria-label={`${key} ${property}`} step={property === "scale" ? 0.1 : 1} min={property === "scale" ? 0.1 : undefined} max={property === "scale" ? 4 : undefined}
                value={asset[property]} onChange={(event) => { if (event.target.value !== "") store?.adjust(key, { [property]: Number(event.target.value) }); }} />
            </label>)}
            <button type="button" onClick={() => store?.adjust(key, DEFAULT_ADJUSTMENT)}>Reset adjustment</button>
          </div>}
        </fieldset>;
      })}
      <p>HTMLキャラはPixi全体より上です。障害物との正確な部分遮蔽は未対応です。</p>
    </div>}
  </section>;
}
