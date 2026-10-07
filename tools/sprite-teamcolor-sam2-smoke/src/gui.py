#!/usr/bin/env python3
"""Tkinter GUI MVP for one-row SAM 2 sprite team-color processing."""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from gui_controller import (
    LassoSelectionState,
    TEAM_COLOR_NAMES,
    PromptState,
    Sam2GuiSession,
    SheetLayout,
    choose_display_scale,
    clip_mask_to_frame,
    display_to_frame,
    extract_row_frames,
    frame_output_names,
    load_sprite_png,
    make_mask_overlay,
    make_selection_overlay,
    parse_team_color,
    recolor_frame_sequence,
    save_results,
)
from recolor import make_contact_sheet


ROOT = Path(__file__).resolve().parents[1]


class TeamColorApp:
    PREFERRED_DISPLAY_SCALE = 4

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sprite Team Color - SAM 2 MVP")
        self.screen_size = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        window_width = min(1080, max(1, self.screen_size[0] - 40))
        window_height = min(900, max(1, self.screen_size[1] - 80))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min(640, window_width), min(480, window_height))

        self.sprite_path = tk.StringVar()
        self.checkpoint_path = tk.StringVar(value=str(ROOT / "models" / "sam2.1_hiera_tiny.pt"))
        self.frame_width = tk.StringVar(value="128")
        self.frame_height = tk.StringVar(value="128")
        self.columns = tk.StringVar(value="15")
        self.rows = tk.StringVar(value="8")
        self.target_row = tk.StringVar(value="0")
        self.frame_count = tk.StringVar(value="15")
        self.device = tk.StringVar(value="auto")
        self.selection_mode = tk.StringVar(value="area")
        self.lasso_operation = tk.StringVar(value="add")
        self.prompt_mode = tk.IntVar(value=1)
        self.color_preset = tk.StringVar(value="Red")
        self.custom_color = tk.StringVar(value="#D23030")
        self.image_info = tk.StringVar(value="No spritesheet loaded")
        self.selection_info = tk.StringVar(value="Selected: 0 pixels")
        self.prompt_info = tk.StringVar(value="Positive: 0 / Negative: 0")
        self.status = tk.StringVar(value="Ready")

        self.layout: SheetLayout | None = None
        self.sprite: Image.Image | None = None
        self.frames: list[Image.Image] = []
        self.lasso_selection: LassoSelectionState | None = None
        self.prompts: PromptState | None = None
        self.quick_frame0_mask = None
        self.frame0_mask = None
        self.masks = None
        self.recolored_frames: list[Image.Image] = []
        self.recolored_contact: Image.Image | None = None
        self.sam_session: Sam2GuiSession | None = None
        self.session_checkpoint: Path | None = None
        self.session_device_name: str | None = None
        self._prompt_display_size = (384, 384)
        self._lasso_points: list[tuple[int, int]] = []
        self._lasso_display_points: list[tuple[int, int]] = []
        self._prompt_photo: ImageTk.PhotoImage | None = None
        self._contact_photo: ImageTk.PhotoImage | None = None
        self._busy = False
        self._events: queue.Queue[tuple] = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        viewport = ttk.Frame(self.root)
        viewport.pack(fill="both", expand=True)
        content_canvas = tk.Canvas(viewport, highlightthickness=0)
        vertical_scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=content_canvas.yview)
        horizontal_scrollbar = ttk.Scrollbar(viewport, orient="horizontal", command=content_canvas.xview)
        content_canvas.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        content_canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)

        container = ttk.Frame(content_canvas, padding=10)
        content_canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind(
            "<Configure>",
            lambda _event: content_canvas.configure(scrollregion=content_canvas.bbox("all")),
        )

        input_box = ttk.LabelFrame(container, text="A. Input", padding=8)
        input_box.pack(fill="x")
        ttk.Label(input_box, text="Spritesheet:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_box, textvariable=self.sprite_path, width=72).grid(row=0, column=1, columnspan=7, sticky="ew")
        ttk.Button(input_box, text="Browse...", command=self._browse_sprite).grid(row=0, column=8, padx=(6, 0))

        fields = (
            ("Frame width", self.frame_width),
            ("Frame height", self.frame_height),
            ("Columns", self.columns),
            ("Rows", self.rows),
            ("Target row", self.target_row),
            ("Frame count", self.frame_count),
        )
        for index, (label, variable) in enumerate(fields):
            column = (index % 3) * 3
            row = 1 + index // 3
            ttk.Label(input_box, text=f"{label}:").grid(row=row, column=column, sticky="e", padx=(4, 2), pady=3)
            ttk.Entry(input_box, textvariable=variable, width=8).grid(row=row, column=column + 1, sticky="w")
        ttk.Button(input_box, text="Load / Apply", command=self._load_sheet).grid(row=1, column=8, rowspan=2, sticky="ns")
        ttk.Label(input_box, textvariable=self.image_info).grid(row=3, column=0, columnspan=9, sticky="w", pady=(5, 0))
        input_box.columnconfigure(1, weight=1)

        sam_box = ttk.LabelFrame(container, text="B. SAM 2", padding=8)
        sam_box.pack(fill="x", pady=(8, 0))
        ttk.Label(sam_box, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        ttk.Entry(sam_box, textvariable=self.checkpoint_path, width=76).grid(row=0, column=1, sticky="ew")
        ttk.Button(sam_box, text="Browse...", command=self._browse_checkpoint).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(sam_box, text="Device:").grid(row=0, column=3, padx=(12, 2))
        ttk.Combobox(sam_box, textvariable=self.device, values=("auto", "cpu", "cuda"), width=7, state="readonly").grid(row=0, column=4)
        sam_box.columnconfigure(1, weight=1)

        middle = ttk.Frame(container)
        middle.pack(fill="x", pady=(8, 0))
        prompt_box = ttk.LabelFrame(middle, text="C/D. Selection editor and mask preview", padding=8)
        prompt_box.pack(side="left", fill="y", expand=False)

        selection_mode_controls = ttk.Frame(prompt_box)
        selection_mode_controls.pack(fill="x")
        ttk.Label(selection_mode_controls, text="Selection mode:").pack(side="left")
        self.selection_mode_buttons = (
            ttk.Radiobutton(
                selection_mode_controls,
                text="Select Area",
                variable=self.selection_mode,
                value="area",
                command=self._on_selection_mode_change,
            ),
            ttk.Radiobutton(
                selection_mode_controls,
                text="Quick Select",
                variable=self.selection_mode,
                value="quick",
                command=self._on_selection_mode_change,
            ),
        )
        self.selection_mode_buttons[0].pack(side="left", padx=(8, 0))
        self.selection_mode_buttons[1].pack(side="left", padx=(8, 0))

        editor_controls = ttk.Frame(prompt_box)
        editor_controls.pack(fill="x", pady=(5, 0))
        self.lasso_controls = ttk.Frame(editor_controls)
        ttk.Radiobutton(self.lasso_controls, text="Add", variable=self.lasso_operation, value="add").pack(side="left")
        ttk.Radiobutton(self.lasso_controls, text="Subtract", variable=self.lasso_operation, value="subtract").pack(
            side="left", padx=(8, 0)
        )
        self.lasso_undo_button = ttk.Button(self.lasso_controls, text="Undo", command=self._undo_lasso)
        self.lasso_undo_button.pack(side="left", padx=(16, 0))
        self.lasso_clear_button = ttk.Button(
            self.lasso_controls,
            text="Clear Selection",
            command=self._clear_lasso,
        )
        self.lasso_clear_button.pack(side="left", padx=(6, 0))
        self.lasso_controls.grid(row=0, column=0, sticky="w")

        self.quick_controls = ttk.Frame(editor_controls)
        ttk.Radiobutton(self.quick_controls, text="Positive", variable=self.prompt_mode, value=1).pack(side="left")
        ttk.Radiobutton(self.quick_controls, text="Negative", variable=self.prompt_mode, value=0).pack(
            side="left", padx=(8, 0)
        )
        self.quick_undo_button = ttk.Button(self.quick_controls, text="Undo", command=self._undo_prompt)
        self.quick_undo_button.pack(side="left", padx=(16, 0))
        self.quick_clear_button = ttk.Button(self.quick_controls, text="Clear clicks", command=self._clear_prompts)
        self.quick_clear_button.pack(side="left", padx=(6, 0))
        self.quick_controls.grid(row=0, column=0, sticky="w")

        self.selection_info_label = ttk.Label(prompt_box, textvariable=self.selection_info)
        self.selection_info_label.pack(anchor="w", pady=(4, 0))
        self.prompt_info_label = ttk.Label(prompt_box, textvariable=self.prompt_info)
        self.prompt_info_label.pack(anchor="w", pady=(4, 0))
        self.prompt_canvas = tk.Canvas(
            prompt_box,
            width=self._prompt_display_size[0],
            height=self._prompt_display_size[1],
            background="#202020",
            highlightthickness=0,
        )
        self.prompt_canvas.pack(pady=(6, 0))
        self.prompt_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.prompt_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.prompt_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        action_box = ttk.LabelFrame(middle, text="E/F/H. Process and save", padding=10)
        action_box.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.generate_button = ttk.Button(action_box, text="Generate Mask (Quick Select)", command=self._generate_mask)
        self.generate_button.pack(fill="x", pady=3)
        self.track_button = ttk.Button(action_box, text="Track Across Frames", command=self._track_frames)
        self.track_button.pack(fill="x", pady=3)
        ttk.Separator(action_box).pack(fill="x", pady=10)
        ttk.Label(action_box, text="Team color preset:").pack(anchor="w")
        ttk.Combobox(
            action_box,
            textvariable=self.color_preset,
            values=TEAM_COLOR_NAMES,
            state="readonly",
        ).pack(fill="x", pady=(2, 6))
        ttk.Label(action_box, text="Custom (#RRGGBB or R,G,B):").pack(anchor="w")
        ttk.Entry(action_box, textvariable=self.custom_color).pack(fill="x", pady=(2, 6))
        self.recolor_button = ttk.Button(action_box, text="Recolor", command=self._recolor)
        self.recolor_button.pack(fill="x", pady=3)
        self.save_button = ttk.Button(action_box, text="Save Results...", command=self._save)
        self.save_button.pack(fill="x", pady=3)
        ttk.Separator(action_box).pack(fill="x", pady=10)
        ttk.Label(action_box, text="Status:").pack(anchor="w")
        ttk.Label(action_box, textvariable=self.status, wraplength=340).pack(anchor="w", fill="x", pady=(3, 0))

        preview_box = ttk.LabelFrame(container, text="G. Recolored contact sheet preview", padding=8)
        preview_box.pack(fill="both", expand=True, pady=(8, 0))
        self.contact_label = ttk.Label(preview_box, text="Run Track Across Frames, then Recolor")
        self.contact_label.pack(expand=True)
        self._show_editor_controls()
        self._update_buttons()

    def _browse_sprite(self) -> None:
        path = filedialog.askopenfilename(title="Select spritesheet PNG", filetypes=(("PNG images", "*.png"),))
        if path:
            self.sprite_path.set(path)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title="Select SAM 2.1 checkpoint", filetypes=(("PyTorch checkpoint", "*.pt"), ("All files", "*.*")))
        if path:
            self.checkpoint_path.set(path)

    def _read_layout(self) -> SheetLayout:
        try:
            return SheetLayout(
                frame_width=int(self.frame_width.get()),
                frame_height=int(self.frame_height.get()),
                columns=int(self.columns.get()),
                rows=int(self.rows.get()),
                target_row=int(self.target_row.get()),
                frame_count=int(self.frame_count.get()),
            )
        except ValueError as error:
            raise ValueError("Frame settings must use integers") from error

    def _load_sheet(self) -> None:
        if self._busy:
            return
        try:
            path = Path(self.sprite_path.get()).expanduser()
            sprite = load_sprite_png(path)
            layout = self._read_layout()
            frames = extract_row_frames(sprite, layout)
        except Exception as error:
            messagebox.showerror("Could not load spritesheet", str(error))
            return
        self._close_session()
        self.sprite = sprite
        self.layout = layout
        self.frames = frames
        self.lasso_selection = LassoSelectionState(layout.frame_size)
        self.prompts = PromptState(layout.frame_size)
        available_size = (
            max(1, self.screen_size[0] - 480),
            max(1, self.screen_size[1] - 360),
        )
        display_scale = choose_display_scale(
            layout.frame_size,
            available_size,
            preferred_scale=self.PREFERRED_DISPLAY_SCALE,
        )
        self._prompt_display_size = (
            layout.frame_width * display_scale,
            layout.frame_height * display_scale,
        )
        self._lasso_points = []
        self._lasso_display_points = []
        self.quick_frame0_mask = None
        self.frame0_mask = None
        self.masks = None
        self.recolored_frames = []
        self.recolored_contact = None
        self.image_info.set(
            f"{path.name}: {sprite.width}x{sprite.height}, mode {sprite.mode}; "
            f"loaded row {layout.target_row}, {len(frames)} frames"
        )
        if self.selection_mode.get() == "area":
            self.status.set("Spritesheet loaded. Drag around the target area on frame 0.")
        else:
            self.status.set("Spritesheet loaded. Add at least one positive Quick Select click.")
        self._clear_contact_preview()
        self._redraw_prompt()
        self._refresh_selection_info()
        self._refresh_prompt_info()
        self._update_buttons()

    def _event_frame_point(self, event: tk.Event) -> tuple[int, int] | None:
        if self.layout is None:
            return None
        try:
            return display_to_frame(event.x, event.y, self._prompt_display_size, self.layout.frame_size)
        except ValueError:
            return None

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self._busy or not self.frames or self.prompts is None or self.layout is None:
            return
        point = self._event_frame_point(event)
        if point is None:
            return
        if self.selection_mode.get() == "quick":
            self.prompts.add(*point, self.prompt_mode.get())
            self._invalidate_after_prompt_change()
            return
        self._lasso_points = [point]
        self._lasso_display_points = [(event.x, event.y)]
        self.prompt_canvas.delete("lasso_draft")

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._busy or self.selection_mode.get() != "area" or not self._lasso_points:
            return
        point = self._event_frame_point(event)
        if point is None or point == self._lasso_points[-1]:
            return
        previous_x, previous_y = self._lasso_display_points[-1]
        self._lasso_points.append(point)
        self._lasso_display_points.append((event.x, event.y))
        self.prompt_canvas.create_line(
            previous_x,
            previous_y,
            event.x,
            event.y,
            fill="#ffdc00",
            width=2,
            tags="lasso_draft",
        )

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._busy or self.selection_mode.get() != "area" or not self._lasso_points:
            return
        point = self._event_frame_point(event)
        if point is not None and point != self._lasso_points[-1]:
            self._lasso_points.append(point)
            self._lasso_display_points.append((event.x, event.y))
        points = tuple(self._lasso_points)
        self._lasso_points = []
        self._lasso_display_points = []
        self.prompt_canvas.delete("lasso_draft")
        if len(points) < 3 or self.lasso_selection is None:
            self.status.set("Lasso needs at least three distinct frame pixels.")
            return
        try:
            self.lasso_selection.apply(points, self.lasso_operation.get())
        except ValueError as error:
            self.status.set(str(error))
            return
        self._invalidate_after_lasso_change()

    def _on_selection_mode_change(self) -> None:
        if self._busy:
            return
        self._lasso_points = []
        self._lasso_display_points = []
        self.prompt_canvas.delete("lasso_draft")
        self.masks = None
        self.recolored_frames = []
        self.recolored_contact = None
        self._clear_contact_preview()
        if self.selection_mode.get() == "area":
            self._sync_lasso_mask()
            self._refresh_selection_info()
            if self.frame0_mask is None:
                self.status.set("Select Area: drag around the target, then Track Across Frames.")
            else:
                self.status.set("Initial Mask Ready. Review it, then Track Across Frames.")
        else:
            self.frame0_mask = self.quick_frame0_mask.copy() if self.quick_frame0_mask is not None else None
            self.status.set("Quick Select: add clicks, then Generate Mask.")
        self._show_editor_controls()
        self._redraw_prompt()
        self._update_buttons()

    def _show_editor_controls(self) -> None:
        if self.selection_mode.get() == "area":
            self.quick_controls.grid_remove()
            self.lasso_controls.grid()
            self.prompt_info_label.pack_forget()
            self.selection_info_label.pack(anchor="w", pady=(4, 0), before=self.prompt_canvas)
        else:
            self.lasso_controls.grid_remove()
            self.quick_controls.grid()
            self.selection_info_label.pack_forget()
            self.prompt_info_label.pack(anchor="w", pady=(4, 0), before=self.prompt_canvas)

    def _sync_lasso_mask(self) -> None:
        if self.lasso_selection is None or not self.frames or self.lasso_selection.is_empty:
            self.frame0_mask = None
            return
        self.frame0_mask = clip_mask_to_frame(self.lasso_selection.mask, self.frames[0])
        if not self.frame0_mask.any():
            self.frame0_mask = None

    def _invalidate_after_lasso_change(self) -> None:
        self._sync_lasso_mask()
        self.masks = None
        self.recolored_frames = []
        self.recolored_contact = None
        self._clear_contact_preview()
        self._refresh_selection_info()
        self._redraw_prompt()
        if self.frame0_mask is None:
            self.status.set("Selection is empty. Use Add and draw around the target area.")
        else:
            self.status.set("Initial Mask Ready. Review it, then Track Across Frames.")
        self._update_buttons()

    def _undo_lasso(self) -> None:
        if self._busy or self.lasso_selection is None:
            return
        if self.lasso_selection.undo() is not None:
            self._invalidate_after_lasso_change()

    def _clear_lasso(self) -> None:
        if self._busy or self.lasso_selection is None:
            return
        self.lasso_selection.clear()
        self._invalidate_after_lasso_change()

    def _undo_prompt(self) -> None:
        if self._busy:
            return
        if self.prompts and self.prompts.undo() is not None:
            self._invalidate_after_prompt_change()

    def _clear_prompts(self) -> None:
        if self._busy:
            return
        if self.prompts:
            self.prompts.clear()
            self._invalidate_after_prompt_change()

    def _invalidate_after_prompt_change(self) -> None:
        self.quick_frame0_mask = None
        self.frame0_mask = None
        self.masks = None
        self.recolored_frames = []
        self.recolored_contact = None
        self._clear_contact_preview()
        self._refresh_prompt_info()
        self._redraw_prompt()
        self.status.set("Prompts changed. Generate the frame 0 mask again.")
        self._update_buttons()

    def _refresh_prompt_info(self) -> None:
        clicks = self.prompts.clicks if self.prompts else ()
        positives = sum(click.label == 1 for click in clicks)
        negatives = sum(click.label == 0 for click in clicks)
        self.prompt_info.set(f"Positive: {positives} / Negative: {negatives}")

    def _refresh_selection_info(self) -> None:
        selected = (
            int(self.frame0_mask.sum())
            if self.selection_mode.get() == "area" and self.frame0_mask is not None
            else 0
        )
        self.selection_info.set(f"Selected: {selected} pixels")

    def _redraw_prompt(self) -> None:
        self.prompt_canvas.delete("all")
        if not self.frames or self.layout is None:
            return
        source = self.frames[0]
        if self.frame0_mask is not None:
            if self.selection_mode.get() == "area":
                source = make_selection_overlay(source, self.frame0_mask)
            else:
                source = make_mask_overlay(source, self.frame0_mask)
        display = source.resize(self._prompt_display_size, Image.Resampling.NEAREST)
        self._prompt_photo = ImageTk.PhotoImage(display)
        self.prompt_canvas.configure(width=display.width, height=display.height)
        self.prompt_canvas.create_image(0, 0, image=self._prompt_photo, anchor="nw")
        if self.selection_mode.get() == "quick":
            for click in self.prompts.clicks if self.prompts else ():
                cx = (click.x + 0.5) * display.width / self.layout.frame_width
                cy = (click.y + 0.5) * display.height / self.layout.frame_height
                if click.label == 1:
                    self.prompt_canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, outline="#00ff66", width=3)
                else:
                    self.prompt_canvas.create_line(cx - 6, cy - 6, cx + 6, cy + 6, fill="#ff3344", width=3)
                    self.prompt_canvas.create_line(cx + 6, cy - 6, cx - 6, cy + 6, fill="#ff3344", width=3)

    def _generate_mask(self) -> None:
        if self.selection_mode.get() != "quick":
            messagebox.showinfo("Select Area is ready", "Use Track Across Frames after completing the lasso selection.")
            return
        if not self.prompts or not self.prompts.has_positive:
            messagebox.showerror("Missing positive prompt", "Add at least one positive click.")
            return
        checkpoint = Path(self.checkpoint_path.get()).expanduser().resolve()
        clicks = self.prompts.clicks
        device_name = self.device.get()

        def worker() -> object:
            self._ensure_sam_session(checkpoint, device_name)
            return self.sam_session.generate_frame0_mask(clicks, self._queue_status)

        self._start_task("generate", "Starting frame 0 mask generation...", worker)

    def _track_frames(self) -> None:
        if self.frame0_mask is None:
            messagebox.showerror("Mask required", "Create and review the frame 0 selection first.")
            return
        if self.selection_mode.get() == "area":
            checkpoint = Path(self.checkpoint_path.get()).expanduser().resolve()
            device_name = self.device.get()
            initial_mask = self.frame0_mask.copy()

            def worker() -> object:
                self._ensure_sam_session(checkpoint, device_name)
                self.sam_session.set_frame0_mask(initial_mask, self._queue_status)
                return self.sam_session.track_across_frames(self._queue_status)

            self._start_task("track", "Preparing lasso mask for propagation...", worker)
            return
        if self.sam_session is None:
            messagebox.showerror("Mask required", "Generate and review the Quick Select mask first.")
            return
        self._start_task(
            "track",
            "Starting mask propagation...",
            lambda: self.sam_session.track_across_frames(self._queue_status),
        )

    def _ensure_sam_session(self, checkpoint: Path, device_name: str) -> None:
        if (
            self.sam_session is None
            or self.session_checkpoint != checkpoint
            or self.session_device_name != device_name
        ):
            self._close_session()
            self.sam_session = Sam2GuiSession(self.frames, checkpoint, device_name=device_name)
            self.session_checkpoint = checkpoint
            self.session_device_name = device_name

    def _recolor(self) -> None:
        if self.masks is None or self.layout is None:
            messagebox.showerror("Tracking required", "Track the mask across frames first.")
            return
        try:
            target = parse_team_color(self.color_preset.get(), self.custom_color.get())
        except Exception as error:
            messagebox.showerror("Invalid team color", str(error))
            return

        def worker() -> object:
            recolored = recolor_frame_sequence(self.frames, self.masks, target)
            labels = [Path(name).stem for name in frame_output_names(len(recolored))]
            contact = make_contact_sheet(recolored, labels, columns=min(5, len(recolored)))
            return recolored, contact

        self._start_task("recolor", "Applying team color...", worker)

    def _save(self) -> None:
        if self.masks is None or not self.recolored_frames or self.layout is None:
            messagebox.showerror("Nothing to save", "Complete tracking and recoloring first.")
            return
        directory = filedialog.askdirectory(title="Select output directory")
        if not directory:
            return
        try:
            outputs = save_results(
                Path(directory),
                self.frames,
                self.masks,
                self.recolored_frames,
                columns=min(5, len(self.frames)),
            )
        except Exception as error:
            messagebox.showerror("Save failed", str(error))
            return
        self.status.set(f"Saved results to {Path(directory).resolve()}")
        messagebox.showinfo("Results saved", "Saved masks, recolored frames, and contact sheets.\n" + "\n".join(str(path) for path in outputs.values()))

    def _start_task(self, name: str, status: str, worker: Callable[[], object]) -> None:
        if self._busy:
            return
        self._busy = True
        self.status.set(status)
        self._update_buttons()

        def run_worker() -> None:
            try:
                result = worker()
            except Exception as error:
                self._events.put(("error", name, str(error), traceback.format_exc()))
            else:
                self._events.put(("success", name, result))

        threading.Thread(target=run_worker, daemon=True).start()

    def _queue_status(self, message: str) -> None:
        self._events.put(("status", message))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "status":
                    self.status.set(event[1])
                elif event[0] == "error":
                    _kind, name, message, details = event
                    self._busy = False
                    print(details, file=sys.stderr)
                    self.status.set(f"{name} failed: {message}")
                    messagebox.showerror(f"{name.title()} failed", message)
                    self._update_buttons()
                elif event[0] == "success":
                    _kind, name, result = event
                    self._busy = False
                    self._handle_success(name, result)
                    self._update_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_success(self, name: str, result: object) -> None:
        if name == "generate":
            self.frame0_mask = result
            self.quick_frame0_mask = result.copy()
            self.status.set("Frame 0 mask ready. Review the overlay, then track across frames.")
            self._redraw_prompt()
        elif name == "track":
            self.masks = result
            self.status.set(f"Tracking complete for {len(self.masks)} frames. Choose a color and Recolor.")
        elif name == "recolor":
            self.recolored_frames, self.recolored_contact = result
            self.status.set("Recolor complete. Review the contact sheet, then save results.")
            self._show_contact_preview()

    def _show_contact_preview(self) -> None:
        if self.recolored_contact is None:
            return
        display = self.recolored_contact.copy()
        display.thumbnail((960, 240), Image.Resampling.NEAREST)
        self._contact_photo = ImageTk.PhotoImage(display)
        self.contact_label.configure(image=self._contact_photo, text="")

    def _clear_contact_preview(self) -> None:
        self._contact_photo = None
        self.contact_label.configure(image="", text="Run Track Across Frames, then Recolor")

    def _update_buttons(self) -> None:
        disabled = self._busy
        quick_ready = self.prompts is not None and self.prompts.has_positive
        generate_disabled = disabled or not self.frames or self.selection_mode.get() != "quick" or not quick_ready
        self.generate_button.configure(state="disabled" if generate_disabled else "normal")
        self.track_button.configure(state="disabled" if disabled or self.frame0_mask is None else "normal")
        self.recolor_button.configure(state="disabled" if disabled or self.masks is None else "normal")
        self.save_button.configure(state="disabled" if disabled or not self.recolored_frames else "normal")
        mode_state = "disabled" if disabled else "normal"
        for button in self.selection_mode_buttons:
            button.configure(state=mode_state)
        self.lasso_undo_button.configure(
            state="disabled"
            if disabled or self.lasso_selection is None or not self.lasso_selection.can_undo
            else "normal"
        )
        self.lasso_clear_button.configure(
            state="disabled"
            if disabled or self.lasso_selection is None or self.lasso_selection.is_empty
            else "normal"
        )
        self.quick_undo_button.configure(
            state="disabled" if disabled or self.prompts is None or not self.prompts.clicks else "normal"
        )
        self.quick_clear_button.configure(
            state="disabled" if disabled or self.prompts is None or not self.prompts.clicks else "normal"
        )

    def _close_session(self) -> None:
        if self.sam_session is not None:
            self.sam_session.close()
        self.sam_session = None
        self.session_checkpoint = None
        self.session_device_name = None

    def _on_close(self) -> None:
        self._close_session()
        self.root.destroy()


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Could not start Tkinter GUI: {error}", file=sys.stderr)
        return 1
    TeamColorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
