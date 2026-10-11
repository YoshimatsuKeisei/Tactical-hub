import { useEffect, useRef, type RefObject } from "react";
import type { CellBounds, PixiBoardTerrain as Terrain } from "../presentation/pixiBoardTerrain";
import { relativeCellBounds } from "../presentation/pixiBoardTerrain";

export function PixiBoardTerrain({ cells, boardRef, onError }: {
  cells: Terrain; boardRef: RefObject<HTMLDivElement | null>; onError: (message: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ cells, onError });
  latest.current = { cells, onError };
  const refresh = useRef<() => void>(() => {});
  useEffect(() => { refresh.current(); }, [cells]);
  useEffect(() => {
    const board = boardRef.current, element = host.current;
    if (!board || !element) return;
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
      try { renderer.update(latest.current.cells, bounds, origin.width, origin.height); }
      catch (error) { latest.current.onError(String(error)); }
    };
    refresh.current = () => { if (!pending) pending = requestAnimationFrame(draw); };
    void import("../presentation/pixiBoardRenderer").then(async ({ mountPixiBoard }) => {
      if (controller.signal.aborted) return;
      renderer = await mountPixiBoard(element, controller.signal);
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
  return <div ref={host} className="pixi-board-canvas" aria-hidden="true" />;
}
