import { useEffect, useRef, useState } from "react";
import { useTileAssets } from "./TileAssetProvider";

function PixiTileCanvas() {
  const host = useRef<HTMLDivElement>(null);
  const { store, snapshot } = useTileAssets();
  const latest = useRef(snapshot); latest.current = snapshot;
  const rendererRef = useRef<Awaited<ReturnType<typeof import("../presentation/pixiTilePreviewRenderer").mountPixiTilePreview>> | undefined>(undefined);
  useEffect(() => {
    try { rendererRef.current?.update(snapshot); }
    catch (error) { if (snapshot.enabled) store?.fallbackShapes(String(error)); setStatus(`画像描画に失敗しました: ${String(error)}`); }
  }, [snapshot, store]);
  const [status, setStatus] = useState("Initializing PixiJS…");
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const controller = new AbortController();
    let dispose: (() => void) | undefined;
    void import("../presentation/pixiTilePreviewRenderer").then(async ({ mountPixiTilePreview }) => {
      if (controller.signal.aborted) return;
      const renderer = await mountPixiTilePreview(element, controller.signal, setStatus);
      dispose = renderer.dispose;
      if (controller.signal.aborted) dispose();
      else { rendererRef.current = renderer; renderer.update(latest.current); setStatus("Click either grid to inspect its logical cell (coordinates start at 0)."); }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        if (latest.current.enabled) store?.fallbackShapes(String(error));
        setStatus(`PixiJS initialization failed: ${String(error)}`);
      }
    });
    return () => { controller.abort(); dispose?.(); rendererRef.current = undefined; };
  }, [store]);
  return <>
    <p aria-live="polite">{status}</p>
    <div className="pixi-tile-preview-scroll" ref={host} />
  </>;
}

export function PixiTileDevPreview() {
  const [open, setOpen] = useState(false);
  return <section className="pixi-tile-dev-preview" aria-label="DEV square tile rendering experiment">
    <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
      {open ? "Close PixiJS tile preview" : "Open PixiJS tile preview (DEV)"}
    </button>
    {open && <div className="pixi-tile-preview-panel">
      <h2>PixiJS square tile experiment</h2>
      <p>Synthetic map only. Gray: road; gold: base; blue: lake. Bridge caps mark start/end.
        Left: flat; right: raised. Red obstacles are separate overlays.</p>
      <PixiTileCanvas />
    </div>}
  </section>;
}
