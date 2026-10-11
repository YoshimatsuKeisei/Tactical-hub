import { useEffect, useRef, type RefObject } from "react";
import type { CellBounds, PixiBoardTerrain as Terrain } from "../presentation/pixiBoardTerrain";
import { relativeCellBounds } from "../presentation/pixiBoardTerrain";
import { useTileAssets } from "./TileAssetProvider";

export function PixiBoardTerrain({ cells, boardRef, onError }: {
  cells: Terrain; boardRef: RefObject<HTMLDivElement | null>; onError: (message: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const foregroundHost = useRef<HTMLDivElement>(null);
  const { store, snapshot } = useTileAssets();
  const latest = useRef({ cells, onError, snapshot, store });
  latest.current = { cells, onError, snapshot, store };
  const refresh = useRef<() => void>(() => {});
  useEffect(() => { refresh.current(); }, [cells, snapshot]);
  useEffect(() => {
    const board = boardRef.current, element = host.current, foregroundElement = foregroundHost.current;
    if (!board || !element || !foregroundElement) return;
    const controller = new AbortController();
    let renderer: Awaited<ReturnType<typeof import("../presentation/pixiBoardRenderer").mountPixiBoard>> | undefined;
    let observer: ResizeObserver | undefined;
    let pending = 0;
    const draw = () => {
      pending = 0;
      if (controller.signal.aborted || !renderer) return;
      const origin = board.getBoundingClientRect();
      const bounds = new Map<string, CellBounds>();
      board.querySelectorAll<HTMLButtonElement>(":scope > .tile").forEach((tile) => {
        const rect = tile.getBoundingClientRect();
        bounds.set(`${tile.dataset.boardX},${tile.dataset.boardY}`, relativeCellBounds({ x: rect.x, y: rect.y, width: rect.width, height: rect.height }, origin));
      });
      try { renderer.update(latest.current.cells, bounds, origin.width, origin.height, latest.current.snapshot); }
      catch (error) {
        if (latest.current.snapshot.enabled) latest.current.store?.fallbackShapes(String(error));
        latest.current.onError(String(error));
      }
    };
    refresh.current = () => { if (!pending) pending = requestAnimationFrame(draw); };
    void import("../presentation/pixiBoardRenderer").then(async ({ mountPixiBoard }) => {
      if (controller.signal.aborted) return;
      renderer = await mountPixiBoard(element, foregroundElement, controller.signal);
      if (controller.signal.aborted) { renderer.dispose(); return; }
      observer = new ResizeObserver(refresh.current);
      observer.observe(board);
      board.querySelectorAll(":scope > .tile").forEach((tile) => observer!.observe(tile));
      refresh.current();
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) latest.current.onError(String(error));
    });
    return () => {
      controller.abort(); observer?.disconnect(); cancelAnimationFrame(pending);
      renderer?.dispose(); refresh.current = () => {};
    };
  }, [boardRef]);
  return <>
    <div ref={host} className="pixi-board-canvas" aria-hidden="true" />
    <div ref={foregroundHost} className="pixi-board-foreground" aria-hidden="true" />
  </>;
}
