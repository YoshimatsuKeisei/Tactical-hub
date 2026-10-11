import { useState } from "react";
import { DEFAULT_ADJUSTMENT, TILE_ASSET_KEYS } from "../presentation/tileAssets";
import { useTileAssets } from "./TileAssetProvider";
export function TileAssetImportPanel() {
  const [open, setOpen] = useState(false);
  const { store, snapshot } = useTileAssets();
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
        X/Yは48px基準の位置調整。画像幅を1セルに合わせ、縦横比を維持します。焼き込み側面は自動除去できません。</p>
      {TILE_ASSET_KEYS.map((key) => {
        const asset = snapshot.assets[key];
        return <fieldset key={key}>
          <legend>{key}{key === "road_south_face" ? " (optional display part)" : ""}</legend>
          <input aria-label={`${key} PNG`} type="file" accept="image/png,.png" onChange={(event) => {
            const file = event.target.files?.[0]; event.target.value = "";
            if (file) void store?.importFile(key, file);
          }} />
          <span role="status">{snapshot.statuses[key] ?? "未設定：仮図形"}{asset ? ` — ${asset.name} ${asset.width}×${asset.height}` : ""}</span>
          <button type="button" onClick={() => store?.reset(key)}>Reset image</button>
          {asset && <div>
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
