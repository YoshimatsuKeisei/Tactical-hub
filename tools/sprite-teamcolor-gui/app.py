from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from core import (
    FRAME_SIZE,
    TEAM_COLORS,
    PromptPoint,
    apply_all_tracked_rows,
    extract_frame,
    extract_row_frames,
    infer_grid,
    load_rgba_png,
    recolor_rgba,
    save_team_variants,
)
from sam_backend import Sam2RowTracker

APP_TITLE = "Tactical-hub Sprite Team Color Tool"
PREVIEW_SCALE = 5
CHECKER_A = (44, 44, 44, 255)
CHECKER_B = (66, 66, 66, 255)


class TeamColorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1060x760")
        self.minsize(980, 700)

        self.sheet_path: Path | None = None
        self.sheet: Image.Image | None = None
        self.grid = None
        self.checkpoint_path: Path | None = self._find_default_checkpoint()
        self.tracker: Sam2RowTracker | None = None

        # row -> frame -> prompt list
        self.prompts: dict[int, dict[int, list[PromptPoint]]] = {}
        # row -> one mask per frame
        self.row_masks: dict[int, list[np.ndarray]] = {}

        self.row_var = tk.IntVar(value=1)
        self.frame_var = tk.IntVar(value=1)
        self.team_var = tk.StringVar(value="original")
        self.strength_var = tk.DoubleVar(value=100.0)
        self.preserve_outline_var = tk.BooleanVar(value=True)
        self.device_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="PNGを開いてください")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._preview_photo: ImageTk.PhotoImage | None = None
        self._busy = False

        self._build_ui()
        self._refresh_checkpoint_label()
        self._render_preview()

    def _find_default_checkpoint(self) -> Path | None:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "models" / "sam2.1_hiera_tiny.pt",
            here.parent / "sprite-teamcolor-sam2-smoke" / "models" / "sam2.1_hiera_tiny.pt",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=330)
        controls.pack(side="left", fill="y", padx=(0, 12))
        controls.pack_propagate(False)

        preview_panel = ttk.Frame(outer)
        preview_panel.pack(side="right", fill="both", expand=True)

        ttk.Label(controls, text="入力", font=("", 11, "bold")).pack(anchor="w")
        ttk.Button(controls, text="PNGを開く", command=self._open_png).pack(fill="x", pady=(4, 4))
        self.file_label = ttk.Label(controls, text="未選択", wraplength=300)
        self.file_label.pack(anchor="w", pady=(0, 8))

        ttk.Button(controls, text="SAM 2 checkpointを選択", command=self._choose_checkpoint).pack(fill="x")
        self.checkpoint_label = ttk.Label(controls, text="", wraplength=300)
        self.checkpoint_label.pack(anchor="w", pady=(4, 10))

        nav = ttk.LabelFrame(controls, text="方向 / フレーム", padding=8)
        nav.pack(fill="x", pady=(0, 10))

        ttk.Label(nav, text="方向行").grid(row=0, column=0, sticky="w")
        self.row_spin = ttk.Spinbox(
            nav,
            from_=1,
            to=1,
            width=6,
            textvariable=self.row_var,
            command=self._on_nav_change,
        )
        self.row_spin.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(nav, text="フレーム").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.frame_scale = ttk.Scale(
            nav,
            from_=1,
            to=1,
            variable=self.frame_var,
            orient="horizontal",
            command=lambda _value: self._on_nav_change(),
        )
        self.frame_scale.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.frame_label = ttk.Label(nav, text="1 / 1")
        self.frame_label.grid(row=2, column=1, sticky="e")
        nav.columnconfigure(1, weight=1)

        prompt_box = ttk.LabelFrame(controls, text="クリック指定", padding=8)
        prompt_box.pack(fill="x", pady=(0, 10))
        ttk.Label(
            prompt_box,
            text="左クリック = 対象（positive）\n右クリック = 除外（negative）",
        ).pack(anchor="w")
        button_row = ttk.Frame(prompt_box)
        button_row.pack(fill="x", pady=(8, 0))
        ttk.Button(button_row, text="1つ戻す", command=self._undo_prompt).pack(side="left", expand=True, fill="x")
        ttk.Button(button_row, text="現在フレームをクリア", command=self._clear_current_prompts).pack(
            side="left", expand=True, fill="x", padx=(6, 0)
        )
        ttk.Button(prompt_box, text="この方向の全クリックをクリア", command=self._clear_row_prompts).pack(
            fill="x", pady=(6, 0)
        )

        track_box = ttk.LabelFrame(controls, text="SAM 2追跡", padding=8)
        track_box.pack(fill="x", pady=(0, 10))
        device_row = ttk.Frame(track_box)
        device_row.pack(fill="x")
        ttk.Label(device_row, text="Device").pack(side="left")
        ttk.Combobox(
            device_row,
            textvariable=self.device_var,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=8,
        ).pack(side="right")
        self.track_button = ttk.Button(track_box, text="この方向をTrack", command=self._track_current_row)
        self.track_button.pack(fill="x", pady=(8, 0))
        ttk.Button(track_box, text="この方向のマスクを削除", command=self._clear_row_mask).pack(
            fill="x", pady=(6, 0)
        )

        color_box = ttk.LabelFrame(controls, text="チーム色プレビュー", padding=8)
        color_box.pack(fill="x", pady=(0, 10))
        team_combo = ttk.Combobox(
            color_box,
            textvariable=self.team_var,
            values=("original", "red", "blue", "green", "yellow"),
            state="readonly",
        )
        team_combo.pack(fill="x")
        team_combo.bind("<<ComboboxSelected>>", lambda _event: self._render_preview())

        ttk.Label(color_box, text="色の強さ").pack(anchor="w", pady=(8, 0))
        strength = ttk.Scale(
            color_box,
            from_=0,
            to=100,
            variable=self.strength_var,
            orient="horizontal",
            command=lambda _value: self._render_preview(),
        )
        strength.pack(fill="x")
        ttk.Checkbutton(
            color_box,
            text="極暗色の輪郭を保護",
            variable=self.preserve_outline_var,
            command=self._render_preview,
        ).pack(anchor="w", pady=(6, 0))

        ttk.Button(controls, text="4チーム分のPNGを書き出す", command=self._export_variants).pack(
            fill="x", pady=(0, 10)
        )

        self.progress = ttk.Progressbar(
            controls,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress.pack(fill="x")
        ttk.Label(controls, textvariable=self.status_var, wraplength=300).pack(
            anchor="w", pady=(6, 0)
        )

        ttk.Label(
            preview_panel,
            text="128×128 frame preview",
            font=("", 11, "bold"),
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            preview_panel,
            width=FRAME_SIZE * PREVIEW_SCALE,
            height=FRAME_SIZE * PREVIEW_SCALE,
            highlightthickness=1,
            highlightbackground="#777",
            background="#222",
        )
        self.canvas.pack(anchor="center", expand=True)
        self.canvas.bind("<Button-1>", self._positive_click)
        self.canvas.bind("<Button-3>", self._negative_click)

        self.preview_info = ttk.Label(preview_panel, text="")
        self.preview_info.pack(anchor="center", pady=(8, 0))

    def _refresh_checkpoint_label(self) -> None:
        if self.checkpoint_path:
            self.checkpoint_label.config(text=str(self.checkpoint_path))
        else:
            self.checkpoint_label.config(text="未選択")

    def _set_status(self, text: str) -> None:
        self.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, current: int, total: int) -> None:
        value = 0 if total <= 0 else (current / total) * 100
        self.after(0, lambda: self.progress_var.set(value))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.track_button.config(state=("disabled" if busy else "normal"))

    def _open_png(self) -> None:
        if self._busy:
            return
        path_text = filedialog.askopenfilename(
            title="Sprite sheet PNGを選択",
            filetypes=[("PNG image", "*.png")],
        )
        if not path_text:
            return
        try:
            image = load_rgba_png(path_text)
            grid = infer_grid(image)
        except Exception as error:
            messagebox.showerror("読み込み失敗", str(error))
            return

        self.sheet_path = Path(path_text)
        self.sheet = image
        self.grid = grid
        self.prompts.clear()
        self.row_masks.clear()
        self.row_var.set(1)
        self.frame_var.set(1)
        self.row_spin.config(to=grid.rows)
        self.frame_scale.config(to=grid.columns)
        self.file_label.config(
            text=f"{self.sheet_path.name}\n{image.width}×{image.height} / "
            f"{grid.columns}列×{grid.rows}行"
        )
        self.status_var.set("読み込み完了。frame 1をクリックしてください")
        self.progress_var.set(0)
        self._render_preview()

    def _choose_checkpoint(self) -> None:
        if self._busy:
            return
        path_text = filedialog.askopenfilename(
            title="sam2.1_hiera_tiny.pt を選択",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if not path_text:
            return
        self.checkpoint_path = Path(path_text)
        self.tracker = None
        self._refresh_checkpoint_label()
        self.status_var.set("checkpointを選択しました")

    def _row_index(self) -> int:
        if not self.grid:
            return 0
        return max(0, min(int(round(self.row_var.get())) - 1, self.grid.rows - 1))

    def _frame_index(self) -> int:
        if not self.grid:
            return 0
        return max(0, min(int(round(self.frame_var.get())) - 1, self.grid.columns - 1))

    def _on_nav_change(self) -> None:
        if self.grid:
            self.row_var.set(self._row_index() + 1)
            self.frame_var.set(self._frame_index() + 1)
        self._render_preview()

    def _checkerboard(self, image: Image.Image) -> Image.Image:
        out = Image.new("RGBA", image.size, CHECKER_A)
        draw = ImageDraw.Draw(out)
        cell = 8
        for y in range(0, image.height, cell):
            for x in range(0, image.width, cell):
                if (x // cell + y // cell) % 2:
                    draw.rectangle(
                        (x, y, min(x + cell - 1, image.width - 1), min(y + cell - 1, image.height - 1)),
                        fill=CHECKER_B,
                    )
        out.alpha_composite(image.convert("RGBA"))
        return out

    def _preview_frame(self) -> Image.Image:
        assert self.sheet is not None
        row = self._row_index()
        frame_index = self._frame_index()
        frame = extract_frame(self.sheet, row, frame_index)

        masks = self.row_masks.get(row)
        team = self.team_var.get()
        if masks is not None and frame_index < len(masks):
            mask = masks[frame_index]
            if team in TEAM_COLORS:
                frame = recolor_rgba(
                    frame,
                    mask,
                    TEAM_COLORS[team],
                    strength=self.strength_var.get() / 100.0,
                    preserve_outline=self.preserve_outline_var.get(),
                )
            else:
                rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
                selected = mask & (rgba[:, :, 3] > 0)
                if selected.any():
                    tint = np.asarray([255, 45, 45], dtype=np.float32)
                    original = rgba[selected, :3].astype(np.float32)
                    rgba[selected, :3] = np.rint(original * 0.45 + tint * 0.55).astype(np.uint8)
                    frame = Image.fromarray(rgba, mode="RGBA")
        return frame

    def _render_preview(self) -> None:
        self.canvas.delete("all")
        if self.sheet is None or self.grid is None:
            self.canvas.create_text(
                FRAME_SIZE * PREVIEW_SCALE // 2,
                FRAME_SIZE * PREVIEW_SCALE // 2,
                text="PNGを開いてください",
                fill="white",
            )
            self.preview_info.config(text="")
            return

        row = self._row_index()
        frame_index = self._frame_index()
        frame = self._checkerboard(self._preview_frame())
        enlarged = frame.resize(
            (FRAME_SIZE * PREVIEW_SCALE, FRAME_SIZE * PREVIEW_SCALE),
            Image.Resampling.NEAREST,
        )
        self._preview_photo = ImageTk.PhotoImage(enlarged)
        self.canvas.create_image(0, 0, image=self._preview_photo, anchor="nw")

        for point in self.prompts.get(row, {}).get(frame_index, []):
            cx = point.x * PREVIEW_SCALE + PREVIEW_SCALE / 2
            cy = point.y * PREVIEW_SCALE + PREVIEW_SCALE / 2
            radius = 5
            color = "#3cff6f" if point.label == 1 else "#ff4141"
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=color,
                outline="white",
                width=1,
            )

        prompt_count = len(self.prompts.get(row, {}).get(frame_index, []))
        tracked = row in self.row_masks
        self.frame_label.config(text=f"{frame_index + 1} / {self.grid.columns}")
        self.preview_info.config(
            text=f"方向 {row + 1}/{self.grid.rows}  |  "
            f"frame {frame_index + 1}/{self.grid.columns}  |  "
            f"clicks {prompt_count}  |  "
            f"{'MASKあり' if tracked else '未Track'}"
        )

    def _canvas_to_frame(self, event: tk.Event) -> tuple[int, int]:
        x = max(0, min(FRAME_SIZE - 1, int(event.x // PREVIEW_SCALE)))
        y = max(0, min(FRAME_SIZE - 1, int(event.y // PREVIEW_SCALE)))
        return x, y

    def _add_prompt(self, event: tk.Event, label: int) -> None:
        if self.sheet is None or self._busy:
            return
        row = self._row_index()
        frame_index = self._frame_index()
        x, y = self._canvas_to_frame(event)
        frame = extract_frame(self.sheet, row, frame_index)
        if frame.getpixel((x, y))[3] == 0:
            self.status_var.set("透明部分です。キャラクター上をクリックしてください")
            return
        self.prompts.setdefault(row, {}).setdefault(frame_index, []).append(
            PromptPoint(x, y, label)
        )
        self.status_var.set(
            f"{'positive' if label else 'negative'} click: ({x}, {y})"
        )
        self._render_preview()

    def _positive_click(self, event: tk.Event) -> None:
        self._add_prompt(event, 1)

    def _negative_click(self, event: tk.Event) -> None:
        self._add_prompt(event, 0)

    def _undo_prompt(self) -> None:
        row = self._row_index()
        frame_index = self._frame_index()
        points = self.prompts.get(row, {}).get(frame_index, [])
        if points:
            points.pop()
            self.status_var.set("最後のクリックを戻しました")
        self._render_preview()

    def _clear_current_prompts(self) -> None:
        row = self._row_index()
        frame_index = self._frame_index()
        self.prompts.setdefault(row, {}).pop(frame_index, None)
        self.status_var.set("現在フレームのクリックをクリアしました")
        self._render_preview()

    def _clear_row_prompts(self) -> None:
        row = self._row_index()
        self.prompts.pop(row, None)
        self.status_var.set("この方向のクリックを全てクリアしました")
        self._render_preview()

    def _clear_row_mask(self) -> None:
        row = self._row_index()
        self.row_masks.pop(row, None)
        self.status_var.set("この方向のマスクを削除しました")
        self._render_preview()

    def _get_tracker(self) -> Sam2RowTracker:
        if self.checkpoint_path is None:
            raise FileNotFoundError("SAM 2 checkpointを選択してください")
        requested = self.device_var.get()
        if (
            self.tracker is None
            or self.tracker.checkpoint != self.checkpoint_path
            or self.tracker.requested_device != requested
        ):
            self.tracker = Sam2RowTracker(
                self.checkpoint_path,
                device=requested,
            )
        return self.tracker

    def _track_current_row(self) -> None:
        if self.sheet is None or self.grid is None or self._busy:
            return
        row = self._row_index()
        prompts = self.prompts.get(row, {})
        if not any(
            point.label == 1 for points in prompts.values() for point in points
        ):
            messagebox.showwarning(
                "positive clickが必要",
                "この方向のどこかのフレームで、対象部位を左クリックしてください。",
            )
            return

        frames = extract_row_frames(self.sheet, row)
        prompt_copy = {
            frame_index: list(points)
            for frame_index, points in prompts.items()
            if points
        }

        self._set_busy(True)
        self.progress_var.set(0)
        self.status_var.set("Trackを開始します…")

        def worker() -> None:
            try:
                tracker = self._get_tracker()
                result = tracker.track(
                    frames,
                    prompt_copy,
                    status=self._set_status,
                    progress=self._set_progress,
                )
            except Exception as error:
                details = traceback.format_exc()
                self.after(0, lambda: self._track_failed(error, details))
                return
            self.after(0, lambda: self._track_done(row, result))

        threading.Thread(target=worker, daemon=True).start()

    def _track_failed(self, error: Exception, details: str) -> None:
        self._set_busy(False)
        self.status_var.set(f"Track失敗: {error}")
        messagebox.showerror("SAM 2 Track失敗", f"{error}\n\n{details}")

    def _track_done(self, row: int, result) -> None:
        self.row_masks[row] = result.masks
        self._set_busy(False)
        self.progress_var.set(100)
        self.status_var.set(
            f"方向 {row + 1} Track完了: {result.elapsed_seconds:.1f}秒 / {result.device}"
        )
        self._render_preview()

    def _export_variants(self) -> None:
        if self.sheet is None or self.sheet_path is None or self.grid is None:
            return
        if not self.row_masks:
            messagebox.showwarning("マスクなし", "まず少なくとも1方向をTrackしてください。")
            return

        missing = [row + 1 for row in range(self.grid.rows) if row not in self.row_masks]
        if missing:
            proceed = messagebox.askyesno(
                "未Track方向があります",
                "未Trackの方向は元画像のまま書き出します。\n"
                f"未Track: {missing}\n\n続行しますか？",
            )
            if not proceed:
                return

        output_text = filedialog.askdirectory(title="出力フォルダを選択")
        if not output_text:
            return

        self.status_var.set("4チーム分を書き出し中…")
        self.update_idletasks()
        try:
            written = save_team_variants(
                self.sheet,
                self.row_masks,
                output_text,
                self.sheet_path.stem,
                strength=self.strength_var.get() / 100.0,
                preserve_outline=self.preserve_outline_var.get(),
            )
        except Exception as error:
            messagebox.showerror("書き出し失敗", str(error))
            self.status_var.set(f"書き出し失敗: {error}")
            return

        self.status_var.set("書き出し完了")
        messagebox.showinfo(
            "書き出し完了",
            "\n".join(f"{team}: {path.name}" for team, path in written.items()),
        )


def main() -> None:
    app = TeamColorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
