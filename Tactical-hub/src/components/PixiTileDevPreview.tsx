import { useEffect, useRef, useState } from "react";

function PixiTileCanvas() {
  const host = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("Initializing PixiJS…");
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const controller = new AbortController();
    let dispose: (() => void) | undefined;
    void import("../presentation/pixiTilePreviewRenderer").then(async ({ mountPixiTilePreview }) => {
      if (controller.signal.aborted) return;
      dispose = await mountPixiTilePreview(element, controller.signal, setStatus);
      if (controller.signal.aborted) dispose();
      else setStatus("Click either grid to inspect its logical cell (coordinates start at 0).");
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setStatus(`PixiJS initialization failed: ${String(error)}`);
    });
    return () => { controller.abort(); dispose?.(); };
  }, []);
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
