#!/usr/bin/env python3
"""Tkinter GUI MVP for one-row SAM 2 sprite team-color processing."""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from gui_controller import (
    LassoSelectionState,
    DEFAULT_MASK_THRESHOLD,
    MAX_ZOOM,
    MAX_MASK_THRESHOLD,
    MIN_ZOOM,
    MIN_MASK_THRESHOLD,
    PRE_GATE_PREVIEW_DEFAULT_THRESHOLD,
    PRE_GATE_PREVIEW_MAX_THRESHOLD,
    PRE_GATE_PREVIEW_MIN_THRESHOLD,
    MaskSourceState,
    RecolorPreviewState,
    SamPreGateFrameDiagnostics,
    TemporalPreGateFilterResult,
    TEAM_COLOR_NAMES,
    PromptState,
    Sam2GuiSession,
    SheetLayout,
    apply_mask_cleanup,
    choose_display_scale,
    clamp_zoom,
    clip_mask_to_frame,
    cursor_centered_zoom_offset,
    diagnose_sam_logits,
    derive_masks_from_sam_logits,
    derive_pre_gate_masks,
    derive_reverse_masks_from_sam_logits,
    ellipse_to_mask,
    extract_row_frames,
    frame_output_names,
    format_sam_logit_diagnostics,
    format_pre_gate_diagnostics,
    format_temporal_filter_diagnostics,
    load_sprite_png,
    make_mask_overlay,
    make_pre_gate_contact_sheet,
    make_temporal_filter_contact_sheet,
    make_selection_overlay,
    pan_scroll_offset,
    parse_team_color,
    percentage_to_unit,
    rectangle_to_mask,
    recolor_frame_sequence,
    save_results,
    select_all_mask,
    shape_bounds,
    temporal_filter_pre_gate_masks,
    viewport_to_frame,
    validate_mask_threshold,
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
        self.selection_tool = tk.StringVar(value="freehand")
        self.lasso_operation = tk.StringVar(value="add")
        self.shape_constraint = tk.BooleanVar(value=False)
        self.fill_enclosed_holes = tk.BooleanVar(value=True)
        self.mask_threshold = tk.DoubleVar(value=DEFAULT_MASK_THRESHOLD)
        self.mask_threshold_text = tk.StringVar(value=f"{DEFAULT_MASK_THRESHOLD:+.2f}")
        self.tracked_mask_info = tk.StringVar(value="Total mask pixels: 0")
        self.mask_sources_info = tk.StringVar(value="Sources: Regular")
        self.mask_adoption_info = tk.StringVar(value="")
        self.add_pre_gate_source = tk.BooleanVar(value=False)
        self.add_reverse_source = tk.BooleanVar(value=False)
        self.auto_filter_pre_gate = tk.BooleanVar(value=False)
        self.pre_gate_threshold = tk.DoubleVar(value=PRE_GATE_PREVIEW_DEFAULT_THRESHOLD)
        self.pre_gate_threshold_text = tk.StringVar(
            value=f"{PRE_GATE_PREVIEW_DEFAULT_THRESHOLD:+.1f}"
        )
        self.pre_gate_fill_holes = tk.BooleanVar(value=False)
        self.prompt_mode = tk.IntVar(value=1)
        self.color_preset = tk.StringVar(value="Red")
        self.custom_color = tk.StringVar(value="#D23030")
        self.recolor_strength_percent = tk.DoubleVar(value=85.0)
        self.outline_protection_percent = tk.DoubleVar(value=100.0)
        self.image_info = tk.StringVar(value="No spritesheet loaded")
        self.selection_info = tk.StringVar(value="Selected: 0 pixels")
        self.prompt_info = tk.StringVar(value="Positive: 0 / Negative: 0")
        self.prompt_zoom_text = tk.StringVar(value=f"Zoom: {self.PREFERRED_DISPLAY_SCALE}x")
        self.mask_cleanup_info = tk.StringVar(value="Mask pixels: 0 → 0")
        self.status = tk.StringVar(value="Ready")

        self.layout: SheetLayout | None = None
        self.sprite: Image.Image | None = None
        self.frames: list[Image.Image] = []
        self.lasso_selection: LassoSelectionState | None = None
        self.prompts: PromptState | None = None
        self.quick_frame0_mask: np.ndarray | None = None
        self.frame0_mask: np.ndarray | None = None
        self.masks: list[np.ndarray] | None = None
        self.raw_masks: list[np.ndarray] | None = None
        self.sam_raw_logits: list[np.ndarray] | None = None
        self.pre_gate_diagnostics: list[SamPreGateFrameDiagnostics] | None = None
        self.regular_raw_masks: list[np.ndarray] | None = None
        self.regular_masks: list[np.ndarray] | None = None
        self.pre_gate_masks: list[np.ndarray] | None = None
        self.filtered_pre_gate_masks: list[np.ndarray] | None = None
        self.temporal_filter_result: TemporalPreGateFilterResult | None = None
        self.reverse_raw_logits: list[np.ndarray] | None = None
        self.reverse_raw_masks: list[np.ndarray] | None = None
        self.reverse_masks: list[np.ndarray] | None = None
        self.combined_preview_masks: list[np.ndarray] | None = None
        self.mask_source_state: MaskSourceState | None = None
        self.quick_frame0_logits: np.ndarray | None = None
        self.recolored_frames: list[Image.Image] = []
        self.recolored_contact: Image.Image | None = None
        self.recolor_target: tuple[int, int, int] | None = None
        self.sam_session: Sam2GuiSession | None = None
        self.session_checkpoint: Path | None = None
        self.session_device_name: str | None = None
        self._prompt_display_size = (384, 384)
        self._prompt_viewport_size = (384, 384)
        self._prompt_zoom = self.PREFERRED_DISPLAY_SCALE
        self._lasso_points: list[tuple[int, int]] = []
        self._lasso_display_points: list[tuple[int, int]] = []
        self._shape_start: tuple[int, int] | None = None
        self._prompt_photo: ImageTk.PhotoImage | None = None
        self._contact_photo: ImageTk.PhotoImage | None = None
        self.review_state = RecolorPreviewState()
        self.mask_editor: MaskEditorWindow | None = None
        self.review_window: RecolorReviewWindow | None = None
        self.tracked_mask_window: TrackedMaskPreviewWindow | None = None
        self.logit_diagnostics_window: LogitDiagnosticsWindow | None = None
        self.pre_gate_diagnostics_window: PreGateDiagnosticsWindow | None = None
        self.temporal_filter_diagnostics_window: TemporalFilterDiagnosticsWindow | None = None
        self._pending_threshold_update: str | None = None
        self._pending_pre_gate_update: str | None = None
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
            ttk.Radiobutton(
                selection_mode_controls,
                text="Select All",
                variable=self.selection_mode,
                value="all",
                command=self._on_selection_mode_change,
            ),
        )
        self.selection_mode_buttons[0].pack(side="left", padx=(8, 0))
        self.selection_mode_buttons[1].pack(side="left", padx=(8, 0))
        self.selection_mode_buttons[2].pack(side="left", padx=(8, 0))
        self.open_mask_editor_button = ttk.Button(
            selection_mode_controls,
            text="Open Mask Editor...",
            command=self._open_mask_editor,
        )
        self.open_mask_editor_button.pack(side="right", padx=(12, 0))
        ttk.Button(selection_mode_controls, text="+", width=3, command=lambda: self._change_prompt_zoom(1)).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(selection_mode_controls, text="-", width=3, command=lambda: self._change_prompt_zoom(-1)).pack(
            side="right", padx=(8, 0)
        )
        ttk.Label(selection_mode_controls, textvariable=self.prompt_zoom_text).pack(side="right", padx=(8, 0))

        editor_controls = ttk.Frame(prompt_box)
        editor_controls.pack(fill="x", pady=(5, 0))
        self.lasso_controls = ttk.Frame(editor_controls)
        tool_controls = ttk.Frame(self.lasso_controls)
        tool_controls.pack(fill="x")
        ttk.Label(tool_controls, text="Selection tool:").pack(side="left")
        tools = (("Freehand", "freehand"), ("Rectangle", "rectangle"), ("Ellipse", "ellipse"))
        for label, value in tools:
            ttk.Radiobutton(
                tool_controls,
                text=label,
                variable=self.selection_tool,
                value=value,
                command=self._on_selection_tool_change,
            ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            tool_controls,
            text="Square / circle lock",
            variable=self.shape_constraint,
        ).pack(side="left", padx=(12, 0))

        operation_controls = ttk.Frame(self.lasso_controls)
        operation_controls.pack(fill="x", pady=(4, 0))
        ttk.Radiobutton(operation_controls, text="Add", variable=self.lasso_operation, value="add").pack(side="left")
        ttk.Radiobutton(operation_controls, text="Subtract", variable=self.lasso_operation, value="subtract").pack(
            side="left", padx=(8, 0)
        )
        self.lasso_undo_button = ttk.Button(operation_controls, text="Undo", command=self._undo_lasso)
        self.lasso_undo_button.pack(side="left", padx=(16, 0))
        self.lasso_clear_button = ttk.Button(
            operation_controls,
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
        self.prompt_viewport = ttk.Frame(prompt_box)
        self.prompt_viewport.pack(pady=(6, 0))
        self.prompt_canvas = tk.Canvas(
            self.prompt_viewport,
            width=self._prompt_display_size[0],
            height=self._prompt_display_size[1],
            background="#202020",
            highlightthickness=0,
        )
        prompt_vertical = ttk.Scrollbar(self.prompt_viewport, orient="vertical", command=self.prompt_canvas.yview)
        prompt_horizontal = ttk.Scrollbar(self.prompt_viewport, orient="horizontal", command=self.prompt_canvas.xview)
        self.prompt_canvas.configure(
            yscrollcommand=prompt_vertical.set,
            xscrollcommand=prompt_horizontal.set,
        )
        self.prompt_canvas.grid(row=0, column=0, sticky="nsew")
        prompt_vertical.grid(row=0, column=1, sticky="ns")
        prompt_horizontal.grid(row=1, column=0, sticky="ew")
        self.prompt_viewport.rowconfigure(0, weight=1)
        self.prompt_viewport.columnconfigure(0, weight=1)
        self.prompt_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.prompt_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.prompt_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.prompt_canvas.bind("<Control-MouseWheel>", self._on_prompt_zoom_wheel)
        self.prompt_canvas.bind("<MouseWheel>", self._on_prompt_vertical_wheel)
        self.prompt_canvas.bind("<Shift-MouseWheel>", self._on_prompt_horizontal_wheel)
        self.prompt_canvas.bind("<KeyPress-plus>", self._on_prompt_zoom_key)
        self.prompt_canvas.bind("<KeyPress-equal>", self._on_prompt_zoom_key)
        self.prompt_canvas.bind("<KeyPress-minus>", self._on_prompt_zoom_key)
        self.prompt_canvas.bind("<KeyPress-KP_Add>", self._on_prompt_zoom_key)
        self.prompt_canvas.bind("<KeyPress-KP_Subtract>", self._on_prompt_zoom_key)

        action_box = ttk.LabelFrame(middle, text="E/F/H. Process and save", padding=10)
        action_box.pack(side="left", fill="both", expand=True, padx=(8, 0))
        cleanup_controls = ttk.Frame(action_box)
        cleanup_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(cleanup_controls, text="Mask cleanup:").pack(side="left")
        self.cleanup_checkbox = ttk.Checkbutton(
            cleanup_controls,
            text="Fill enclosed holes",
            variable=self.fill_enclosed_holes,
            command=self._on_cleanup_toggle,
        )
        self.cleanup_checkbox.pack(side="left", padx=(8, 0))
        threshold_controls = ttk.Frame(action_box)
        threshold_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(threshold_controls, text="SAM Mask Threshold:").grid(row=0, column=0, sticky="w")
        self.threshold_scale = ttk.Scale(
            threshold_controls,
            from_=MIN_MASK_THRESHOLD,
            to=MAX_MASK_THRESHOLD,
            variable=self.mask_threshold,
            command=self._schedule_threshold_update,
        )
        self.threshold_scale.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Label(threshold_controls, textvariable=self.mask_threshold_text, width=7).grid(
            row=0,
            column=2,
            sticky="e",
        )
        ttk.Label(threshold_controls, text="Lower = more inclusive").grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="w",
        )
        threshold_controls.columnconfigure(1, weight=1)
        ttk.Label(action_box, textvariable=self.mask_cleanup_info, wraplength=340).pack(
            anchor="w",
            fill="x",
            pady=(0, 5),
        )
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
        ttk.Label(action_box, text="Recolor opens a review window before export.", wraplength=340).pack(
            anchor="w", pady=(4, 0)
        )
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
        self._close_mask_editor()
        self._discard_recolor_preview()
        self._close_tracked_mask_preview()
        self._close_logit_diagnostics()
        self._clear_pre_gate_diagnostics()
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
        self._prompt_zoom = display_scale
        self.prompt_zoom_text.set(f"Zoom: {display_scale}x")
        self._prompt_viewport_size = (
            layout.frame_width * display_scale,
            layout.frame_height * display_scale,
        )
        self._prompt_display_size = self._prompt_viewport_size
        self._lasso_points = []
        self._lasso_display_points = []
        self._shape_start = None
        self.quick_frame0_mask = None
        self.frame0_mask = None
        self.sam_raw_logits = None
        self.pre_gate_diagnostics = None
        self._reset_mask_sources(clear_active=True)
        self.quick_frame0_logits = None
        self.recolored_frames = []
        self.recolored_contact = None
        self.tracked_mask_info.set("Total mask pixels: 0")
        self.image_info.set(
            f"{path.name}: {sprite.width}x{sprite.height}, mode {sprite.mode}; "
            f"loaded row {layout.target_row}, {len(frames)} frames"
        )
        if self.selection_mode.get() == "area":
            self.status.set("Spritesheet loaded. Drag around the target area on frame 0.")
        elif self.selection_mode.get() == "all":
            self._sync_select_all_mask()
            self.status.set("Select All: visible frame 0 pixels selected. Review, then track.")
        else:
            self.status.set("Spritesheet loaded. Add at least one positive Quick Select click.")
        self._clear_contact_preview()
        self._redraw_prompt()
        self.prompt_canvas.xview_moveto(0)
        self.prompt_canvas.yview_moveto(0)
        self._refresh_selection_info()
        self._refresh_prompt_info()
        self._refresh_cleanup_info()
        self._update_buttons()

    def _event_frame_point(self, event: tk.Event) -> tuple[int, int] | None:
        if self.layout is None:
            return None
        try:
            return viewport_to_frame(
                event.x,
                event.y,
                self._prompt_zoom,
                (self.prompt_canvas.canvasx(0), self.prompt_canvas.canvasy(0)),
                self.layout.frame_size,
            )
        except ValueError:
            return None

    def _prompt_view_size(self) -> tuple[int, int]:
        return max(1, self.prompt_canvas.winfo_width()), max(1, self.prompt_canvas.winfo_height())

    def _set_prompt_scroll(self, offset: tuple[float, float]) -> None:
        content_width, content_height = self._prompt_display_size
        self.prompt_canvas.xview_moveto(offset[0] / content_width if content_width else 0)
        self.prompt_canvas.yview_moveto(offset[1] / content_height if content_height else 0)

    def _change_prompt_zoom(self, step: int, cursor: tuple[float, float] | None = None) -> None:
        if not self.frames or self.layout is None:
            return
        old_zoom = self._prompt_zoom
        new_zoom = clamp_zoom(old_zoom + step)
        if new_zoom == old_zoom:
            return
        viewport_size = self._prompt_view_size()
        if cursor is None:
            cursor = (viewport_size[0] / 2, viewport_size[1] / 2)
        new_scroll = cursor_centered_zoom_offset(
            cursor,
            old_zoom,
            new_zoom,
            (self.prompt_canvas.canvasx(0), self.prompt_canvas.canvasy(0)),
            viewport_size,
            self.layout.frame_size,
        )
        self._prompt_zoom = new_zoom
        self.prompt_zoom_text.set(f"Zoom: {new_zoom}x")
        self._prompt_display_size = (
            self.layout.frame_width * new_zoom,
            self.layout.frame_height * new_zoom,
        )
        self._redraw_prompt()
        self._set_prompt_scroll(new_scroll)
        self.status.set(f"Mask preview zoom: {new_zoom}x")

    def _on_prompt_zoom_wheel(self, event: tk.Event) -> str:
        self._change_prompt_zoom(1 if event.delta > 0 else -1, (event.x, event.y))
        return "break"

    def _on_prompt_zoom_key(self, event: tk.Event) -> str:
        step = -1 if event.keysym in ("minus", "KP_Subtract") else 1
        self._change_prompt_zoom(step)
        return "break"

    def _on_prompt_vertical_wheel(self, event: tk.Event) -> str:
        units = -max(1, abs(event.delta) // 120) if event.delta > 0 else max(1, abs(event.delta) // 120)
        self.prompt_canvas.yview_scroll(units, "units")
        return "break"

    def _on_prompt_horizontal_wheel(self, event: tk.Event) -> str:
        units = -max(1, abs(event.delta) // 120) if event.delta > 0 else max(1, abs(event.delta) // 120)
        self.prompt_canvas.xview_scroll(units, "units")
        return "break"

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self._busy or not self.frames or self.prompts is None or self.layout is None:
            return
        self.prompt_canvas.focus_set()
        point = self._event_frame_point(event)
        if point is None:
            return
        if self.selection_mode.get() == "quick":
            self.prompts.add(*point, self.prompt_mode.get())
            self._invalidate_after_prompt_change()
            return
        if self.selection_mode.get() != "area":
            return
        self._clear_main_draft()
        if self.selection_tool.get() == "freehand":
            self._lasso_points = [point]
            self._lasso_display_points = [
                (self.prompt_canvas.canvasx(event.x), self.prompt_canvas.canvasy(event.y))
            ]
        else:
            self._shape_start = point
            self._draw_main_shape_draft(point, event)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._busy or self.selection_mode.get() != "area":
            return
        if self.selection_tool.get() != "freehand":
            if self._shape_start is not None:
                self._draw_main_shape_draft(self._event_frame_point(event), event)
            return
        if not self._lasso_points:
            return
        point = self._event_frame_point(event)
        if point is None or point == self._lasso_points[-1]:
            return
        canvas_x = self.prompt_canvas.canvasx(event.x)
        canvas_y = self.prompt_canvas.canvasy(event.y)
        previous_x, previous_y = self._lasso_display_points[-1]
        self._lasso_points.append(point)
        self._lasso_display_points.append((canvas_x, canvas_y))
        self.prompt_canvas.create_line(
            previous_x,
            previous_y,
            canvas_x,
            canvas_y,
            fill="#ffdc00",
            width=2,
            tags="selection_draft",
        )

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._busy or self.selection_mode.get() != "area":
            return
        if self.selection_tool.get() != "freehand":
            self._commit_main_shape(event)
            return
        if not self._lasso_points:
            return
        point = self._event_frame_point(event)
        if point is not None and point != self._lasso_points[-1]:
            self._lasso_points.append(point)
            self._lasso_display_points.append(
                (self.prompt_canvas.canvasx(event.x), self.prompt_canvas.canvasy(event.y))
            )
        points = tuple(self._lasso_points)
        self._lasso_points = []
        self._lasso_display_points = []
        self.prompt_canvas.delete("selection_draft")
        if len(points) < 3 or self.lasso_selection is None:
            self.status.set("Lasso needs at least three distinct frame pixels.")
            return
        try:
            self.lasso_selection.apply(points, self.lasso_operation.get())
        except ValueError as error:
            self.status.set(str(error))
            return
        self._invalidate_after_lasso_change()

    def _shape_is_constrained(self, event: tk.Event) -> bool:
        return self.shape_constraint.get() or bool(event.state & 0x0001)

    def _clear_main_draft(self) -> None:
        self._lasso_points = []
        self._lasso_display_points = []
        self._shape_start = None
        self.prompt_canvas.delete("selection_draft")

    def _draw_main_shape_draft(self, end: tuple[int, int] | None, event: tk.Event) -> None:
        if self._shape_start is None or end is None or self.layout is None:
            return
        bounds = shape_bounds(
            self._shape_start,
            end,
            self.layout.frame_size,
            self._shape_is_constrained(event),
        )
        x0, y0, x1, y1 = bounds
        coordinates = (
            x0 * self._prompt_zoom,
            y0 * self._prompt_zoom,
            (x1 + 1) * self._prompt_zoom,
            (y1 + 1) * self._prompt_zoom,
        )
        self.prompt_canvas.delete("selection_draft")
        draw = (
            self.prompt_canvas.create_rectangle
            if self.selection_tool.get() == "rectangle"
            else self.prompt_canvas.create_oval
        )
        draw(*coordinates, outline="#ffdc00", width=2, dash=(5, 3), tags="selection_draft")

    def _commit_main_shape(self, event: tk.Event) -> None:
        start = self._shape_start
        end = self._event_frame_point(event)
        self._clear_main_draft()
        if start is None or end is None or self.layout is None or self.lasso_selection is None:
            return
        constrained = self._shape_is_constrained(event)
        if self.selection_tool.get() == "rectangle":
            mask = rectangle_to_mask(start, end, self.layout.frame_size, constrained)
        else:
            mask = ellipse_to_mask(start, end, self.layout.frame_size, constrained)
        self.lasso_selection.apply_mask(mask, self.lasso_operation.get())
        self._invalidate_after_lasso_change()

    def _on_selection_tool_change(self) -> None:
        self._clear_main_draft()
        if self.mask_editor is not None and self.mask_editor.exists():
            self.mask_editor.cancel_draft()
        self.status.set(f"Selection tool: {self.selection_tool.get().title()}")

    def _on_selection_mode_change(self) -> None:
        if self._busy:
            return
        self._clear_main_draft()
        self._reset_sam_tracking_target()
        self.quick_frame0_mask = None
        self.quick_frame0_logits = None
        self.sam_raw_logits = None
        self._clear_pre_gate_diagnostics()
        self._reset_mask_sources(clear_active=True)
        self._close_tracked_mask_preview()
        self._close_logit_diagnostics()
        self._discard_recolor_preview()
        if self.selection_mode.get() == "area":
            self._sync_lasso_mask()
            self._refresh_selection_info()
            if self.frame0_mask is None:
                self.status.set("Select Area: drag around the target, then Track Across Frames.")
            else:
                self.status.set("Initial Mask Ready. Review it, then Track Across Frames.")
        elif self.selection_mode.get() == "quick":
            self._close_mask_editor()
            self._sync_quick_mask()
            self.status.set(
                "Selection mode changed. Previous tracking state cleared; "
                "review prompts, then Generate Mask."
            )
        else:
            self._close_mask_editor()
            self._sync_select_all_mask()
            self._refresh_selection_info()
            self.status.set("Select All: visible frame 0 pixels selected. Review, then track.")
        self._show_editor_controls()
        self._refresh_cleanup_info()
        self._redraw_prompt()
        self._update_buttons()

    def _show_editor_controls(self) -> None:
        if self.selection_mode.get() == "area":
            self.quick_controls.grid_remove()
            self.lasso_controls.grid()
            self.prompt_info_label.pack_forget()
            self.selection_info_label.pack(anchor="w", pady=(4, 0), before=self.prompt_viewport)
        elif self.selection_mode.get() == "quick":
            self.lasso_controls.grid_remove()
            self.quick_controls.grid()
            self.selection_info_label.pack_forget()
            self.prompt_info_label.pack(anchor="w", pady=(4, 0), before=self.prompt_viewport)
        else:
            self.lasso_controls.grid_remove()
            self.quick_controls.grid_remove()
            self.prompt_info_label.pack_forget()
            self.selection_info_label.pack(anchor="w", pady=(4, 0), before=self.prompt_viewport)

    def _sync_lasso_mask(self) -> None:
        if self.lasso_selection is None or not self.frames or self.lasso_selection.is_empty:
            self.frame0_mask = None
            return
        raw = clip_mask_to_frame(self.lasso_selection.mask, self.frames[0])
        self._set_active_frame0_mask(raw)

    def _sync_select_all_mask(self) -> None:
        raw = select_all_mask(self.frames[0]) if self.frames else None
        self._set_active_frame0_mask(raw)

    def _sync_quick_mask(self) -> None:
        self._set_active_frame0_mask(self.quick_frame0_mask)

    def _set_active_frame0_mask(self, raw_mask: np.ndarray | None) -> None:
        if raw_mask is None or not self.frames:
            self.frame0_mask = None
            return
        alpha_clipped = clip_mask_to_frame(raw_mask, self.frames[0])
        active = apply_mask_cleanup(alpha_clipped, self.fill_enclosed_holes.get())
        self.frame0_mask = active if active.any() else None

    def _raw_frame0_mask(self) -> np.ndarray | None:
        if not self.frames:
            return None
        if self.selection_mode.get() == "area":
            if self.lasso_selection is None or self.lasso_selection.is_empty:
                return None
            return clip_mask_to_frame(self.lasso_selection.mask, self.frames[0])
        if self.selection_mode.get() == "quick":
            return self.quick_frame0_mask.copy() if self.quick_frame0_mask is not None else None
        return select_all_mask(self.frames[0])

    def _authoritative_frame0_mask(self) -> np.ndarray | None:
        if self.selection_mode.get() not in ("area", "all"):
            return None
        return self._raw_frame0_mask()

    def _reset_mask_sources(self, clear_active: bool = True) -> None:
        self.add_pre_gate_source.set(False)
        self.add_reverse_source.set(False)
        self.auto_filter_pre_gate.set(False)
        self.regular_raw_masks = None
        self.regular_masks = None
        self.pre_gate_masks = None
        self.filtered_pre_gate_masks = None
        self.temporal_filter_result = None
        self._close_temporal_filter_diagnostics()
        self.reverse_raw_logits = None
        self.reverse_raw_masks = None
        self.reverse_masks = None
        self.combined_preview_masks = None
        self.mask_source_state = None
        self.mask_sources_info.set("Sources: Regular")
        self.mask_adoption_info.set("")
        self.tracked_mask_info.set("Total mask pixels: 0")
        if self.sam_session is not None:
            self.sam_session.reverse_raw_logits = None
        if clear_active:
            self.raw_masks = None
            self.masks = None

    def _sync_mask_source_preview(self) -> None:
        if self.mask_source_state is None:
            self.combined_preview_masks = None
            return
        self.combined_preview_masks = self.mask_source_state.rebuild_preview()
        self.mask_sources_info.set(f"Sources: {self.mask_source_state.source_label}")
        self.mask_adoption_info.set(
            "Preview changed. Click Use These Masks to apply."
            if self.mask_source_state.preview_dirty
            else "Current preview is applied for Recolor / Export."
        )

    def _rebuild_masks_from_logits(self) -> None:
        if self.sam_raw_logits is None:
            return
        regular_raw, regular = derive_masks_from_sam_logits(
            self.sam_raw_logits,
            self.frames,
            threshold=validate_mask_threshold(self.mask_threshold.get()),
            fill_holes=self.fill_enclosed_holes.get(),
            authoritative_frame0=self._authoritative_frame0_mask(),
        )
        self.regular_raw_masks = regular_raw
        self.regular_masks = regular
        self.raw_masks = [mask.copy() for mask in regular_raw]
        if self.mask_source_state is None:
            self.mask_source_state = MaskSourceState(regular)
            self.masks = self.mask_source_state.adopt_preview()
        else:
            self.mask_source_state.set_regular_masks(regular)

        if self.pre_gate_diagnostics is not None:
            self.pre_gate_masks = derive_pre_gate_masks(
                self.pre_gate_diagnostics,
                self.frames,
                threshold=self.pre_gate_threshold.get(),
                fill_holes=self.pre_gate_fill_holes.get(),
            )
            self.mask_source_state.set_pre_gate_masks(self.pre_gate_masks)
        if self.reverse_raw_logits is not None:
            reverse_raw, reverse = derive_reverse_masks_from_sam_logits(
                self.reverse_raw_logits,
                self.frames,
                regular,
                threshold=validate_mask_threshold(self.mask_threshold.get()),
                fill_holes=self.fill_enclosed_holes.get(),
            )
            self.reverse_raw_masks = reverse_raw
            self.reverse_masks = reverse
            self.mask_source_state.set_reverse_masks(reverse)
        self._recompute_temporal_filter()
        self.mask_source_state.set_enabled(
            pre_gate=self.add_pre_gate_source.get(),
            reverse=self.add_reverse_source.get(),
        )
        self._sync_mask_source_preview()
        if regular:
            self.frame0_mask = regular[0].copy()

    def _recompute_temporal_filter(self) -> None:
        if self.mask_source_state is None:
            return
        if not self.auto_filter_pre_gate.get():
            self.mask_source_state.set_pre_gate_filter_enabled(False)
            return
        if (
            self.regular_masks is None
            or self.pre_gate_masks is None
            or self.reverse_masks is None
        ):
            return
        result = temporal_filter_pre_gate_masks(
            self.regular_masks,
            self.pre_gate_masks,
            self.reverse_masks,
        )
        self.temporal_filter_result = result
        self.filtered_pre_gate_masks = [mask.copy() for mask in result.filtered_masks]
        self.mask_source_state.set_filtered_pre_gate_masks(self.filtered_pre_gate_masks)
        self.mask_source_state.set_pre_gate_filter_enabled(True)
        if (
            self.temporal_filter_diagnostics_window is not None
            and self.temporal_filter_diagnostics_window.exists()
        ):
            self.temporal_filter_diagnostics_window.update_result(result)

    def _rebuild_quick_frame0_from_logits(self) -> None:
        if self.quick_frame0_logits is None or not self.frames:
            return
        raw, active = derive_masks_from_sam_logits(
            [self.quick_frame0_logits],
            [self.frames[0]],
            threshold=validate_mask_threshold(self.mask_threshold.get()),
            fill_holes=self.fill_enclosed_holes.get(),
        )
        self.quick_frame0_mask = raw[0]
        self.frame0_mask = active[0] if active[0].any() else None

    def _schedule_threshold_update(self, value: str) -> None:
        rounded = round(float(value) / 0.05) * 0.05
        rounded = validate_mask_threshold(rounded)
        self.mask_threshold.set(rounded)
        self.mask_threshold_text.set(f"{rounded:+.2f}")
        if self._pending_threshold_update is not None:
            self.root.after_cancel(self._pending_threshold_update)
        self._pending_threshold_update = self.root.after(75, self._apply_threshold_update)

    def _apply_threshold_update(self) -> None:
        self._pending_threshold_update = None
        threshold = validate_mask_threshold(self.mask_threshold.get())
        self.mask_threshold_text.set(f"{threshold:+.2f}")
        updated_from_logits = False
        if self.sam_raw_logits is not None:
            self._rebuild_masks_from_logits()
            updated_from_logits = True
        elif self.selection_mode.get() == "quick" and self.quick_frame0_logits is not None:
            self._rebuild_quick_frame0_from_logits()
            updated_from_logits = True
        self._discard_recolor_preview()
        self._refresh_cleanup_info()
        self._refresh_selection_info()
        self._redraw_prompt()
        self._refresh_mask_editor()
        self._refresh_tracked_mask_preview()
        if updated_from_logits:
            suffix = (
                " Preview changed; click Use These Masks to apply."
                if self.mask_source_state is not None and self.mask_source_state.preview_dirty
                else ""
            )
            self.status.set(
                f"Mask threshold {threshold:+.2f} applied from retained logits; "
                f"SAM 2 was not rerun.{suffix}"
            )
        else:
            self.status.set(f"Mask threshold set to {threshold:+.2f} for the next SAM mask.")
        self._update_buttons()

    def _refresh_cleanup_info(self) -> None:
        if self.regular_raw_masks is not None and self.regular_masks is not None:
            raw_masks = self.regular_raw_masks
            active_masks = self.regular_masks
        else:
            raw = self._raw_frame0_mask()
            if raw is None or self.frame0_mask is None:
                self.mask_cleanup_info.set("Mask pixels: 0 → 0")
                return
            raw_masks = [raw]
            active_masks = [self.frame0_mask]
        before = sum(int(mask.sum()) for mask in raw_masks)
        after = sum(int(mask.sum()) for mask in active_masks)
        if self.fill_enclosed_holes.get():
            self.mask_cleanup_info.set(
                f"Mask pixels: raw {before} → active {after} (alpha clip + hole fill)"
            )
        else:
            self.mask_cleanup_info.set(f"Mask pixels: raw {before} → active {after} (alpha clip; cleanup off)")

    def _on_cleanup_toggle(self) -> None:
        if self._busy:
            return
        if self.sam_raw_logits is not None:
            self._rebuild_masks_from_logits()
        elif self.selection_mode.get() == "quick" and self.quick_frame0_logits is not None:
            self._rebuild_quick_frame0_from_logits()
        else:
            raw_frame0 = self._raw_frame0_mask()
            self._set_active_frame0_mask(raw_frame0)
        self._discard_recolor_preview()
        self._refresh_selection_info()
        self._refresh_cleanup_info()
        self._redraw_prompt()
        self._refresh_mask_editor()
        self._refresh_tracked_mask_preview()
        state = "enabled" if self.fill_enclosed_holes.get() else "disabled"
        suffix = (
            " Preview changed; click Use These Masks to apply."
            if self.mask_source_state is not None and self.mask_source_state.preview_dirty
            else ""
        )
        self.status.set(f"Fill enclosed holes {state}; tracking state retained.{suffix}")
        self._update_buttons()

    def _invalidate_after_lasso_change(self) -> None:
        self._sync_lasso_mask()
        self._reset_sam_tracking_target()
        self.sam_raw_logits = None
        self._clear_pre_gate_diagnostics()
        self._reset_mask_sources(clear_active=True)
        self._close_tracked_mask_preview()
        self._close_logit_diagnostics()
        self._discard_recolor_preview()
        self._refresh_selection_info()
        self._refresh_cleanup_info()
        self._redraw_prompt()
        self._refresh_mask_editor()
        if self.frame0_mask is None:
            self.status.set(
                "Selection changed. Previous tracking state cleared. "
                "Use Add and draw around the target area."
            )
        else:
            self.status.set(
                "Selection changed. Previous tracking state cleared. "
                "Review it, then Track Across Frames."
            )
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

    def _open_mask_editor(self) -> None:
        if not self.frames or self.lasso_selection is None:
            messagebox.showerror("Spritesheet required", "Load a spritesheet before opening the Mask Editor.")
            return
        if self.selection_mode.get() != "area":
            self.selection_mode.set("area")
            self._on_selection_mode_change()
        if self.mask_editor is not None and self.mask_editor.exists():
            self.mask_editor.focus()
            return
        self.mask_editor = MaskEditorWindow(self)

    def _refresh_mask_editor(self) -> None:
        if self.mask_editor is not None and self.mask_editor.exists():
            self.mask_editor.redraw()
            self.mask_editor.update_controls()

    def _close_mask_editor(self) -> None:
        if self.mask_editor is not None:
            editor = self.mask_editor
            self.mask_editor = None
            editor.destroy()

    def _mask_editor_closed(self, editor: "MaskEditorWindow") -> None:
        if self.mask_editor is editor:
            self.mask_editor = None

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
        self._reset_sam_tracking_target()
        self.quick_frame0_mask = None
        self.quick_frame0_logits = None
        self.tracked_mask_info.set("Total mask pixels: 0")
        self.frame0_mask = None
        self.sam_raw_logits = None
        self._clear_pre_gate_diagnostics()
        self._reset_mask_sources(clear_active=True)
        self._close_tracked_mask_preview()
        self._close_logit_diagnostics()
        self._discard_recolor_preview()
        self._refresh_prompt_info()
        self._refresh_cleanup_info()
        self._redraw_prompt()
        self.status.set(
            "Prompts changed. Previous tracking state cleared. "
            "Generate the frame 0 mask again."
        )
        self._update_buttons()

    def _refresh_prompt_info(self) -> None:
        clicks = self.prompts.clicks if self.prompts else ()
        positives = sum(click.label == 1 for click in clicks)
        negatives = sum(click.label == 0 for click in clicks)
        self.prompt_info.set(f"Positive: {positives} / Negative: {negatives}")

    def _refresh_selection_info(self) -> None:
        selected = (
            int(self.frame0_mask.sum())
            if self.selection_mode.get() in ("area", "all") and self.frame0_mask is not None
            else 0
        )
        self.selection_info.set(f"Selected: {selected} pixels")

    def _redraw_prompt(self) -> None:
        self.prompt_canvas.delete("all")
        if not self.frames or self.layout is None:
            return
        source = self.frames[0]
        if self.frame0_mask is not None:
            if self.selection_mode.get() in ("area", "all"):
                source = make_selection_overlay(source, self.frame0_mask)
            else:
                source = make_mask_overlay(source, self.frame0_mask)
        display = source.resize(self._prompt_display_size, Image.Resampling.NEAREST)
        self._prompt_photo = ImageTk.PhotoImage(display)
        self.prompt_canvas.configure(
            width=self._prompt_viewport_size[0],
            height=self._prompt_viewport_size[1],
            scrollregion=(0, 0, display.width, display.height),
        )
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
            messagebox.showinfo("Selection is ready", "Use Track Across Frames after reviewing the selection.")
            return
        if not self.prompts or not self.prompts.has_positive:
            messagebox.showerror("Missing positive prompt", "Add at least one positive click.")
            return
        checkpoint = Path(self.checkpoint_path.get()).expanduser().resolve()
        clicks = self.prompts.clicks
        device_name = self.device.get()
        threshold = validate_mask_threshold(self.mask_threshold.get())

        def worker() -> object:
            self._ensure_sam_session(checkpoint, device_name)
            mask = self.sam_session.generate_frame0_mask(
                clicks,
                self._queue_status,
                threshold=threshold,
            )
            return mask, self.sam_session.frame0_logits

        self._start_task("generate", "Starting frame 0 mask generation...", worker)

    def _track_frames(self) -> None:
        if self.frame0_mask is None:
            messagebox.showerror("Mask required", "Create and review the frame 0 selection first.")
            return
        self._discard_recolor_preview()
        self._close_logit_diagnostics()
        self._clear_pre_gate_diagnostics()
        self.sam_raw_logits = None
        self._reset_mask_sources(clear_active=True)
        if self.selection_mode.get() in ("area", "all"):
            checkpoint = Path(self.checkpoint_path.get()).expanduser().resolve()
            device_name = self.device.get()
            initial_mask = self.frame0_mask.copy()

            def worker() -> object:
                self._ensure_sam_session(checkpoint, device_name)
                self.sam_session.set_frame0_mask(initial_mask, self._queue_status)
                return self.sam_session.track_across_frames_logits(self._queue_status)

            self._start_task("track", "Preparing selection mask for propagation...", worker)
            return
        if self.sam_session is None:
            messagebox.showerror("Mask required", "Generate and review the Quick Select mask first.")
            return
        self._start_task(
            "track",
            "Starting mask propagation...",
            lambda: self.sam_session.track_across_frames_logits(self._queue_status),
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

    def _reset_sam_tracking_target(self) -> None:
        if self.sam_session is not None:
            self.sam_session.reset_tracking_state()

    def _recolor(self) -> None:
        if self.masks is None or self.layout is None:
            messagebox.showerror("Tracking required", "Track the mask across frames first.")
            return
        if self.mask_source_state is not None and self.mask_source_state.preview_dirty:
            messagebox.showinfo(
                "Apply preview masks",
                "Preview changed. Click Use These Masks before Recolor.",
            )
            return
        try:
            target = parse_team_color(self.color_preset.get(), self.custom_color.get())
        except Exception as error:
            messagebox.showerror("Invalid team color", str(error))
            return
        strength = percentage_to_unit(round(self.recolor_strength_percent.get()))
        protection = percentage_to_unit(round(self.outline_protection_percent.get()))
        self._discard_recolor_preview()

        def worker() -> object:
            recolored = recolor_frame_sequence(
                self.frames,
                self.masks,
                target,
                strength=strength,
                shadow_protect_amount=protection,
            )
            labels = [Path(name).stem for name in frame_output_names(len(recolored))]
            contact = make_contact_sheet(recolored, labels, columns=min(5, len(recolored)))
            return recolored, contact, target

        self._start_task("recolor", "Applying team color...", worker)

    def _update_recolor_preview(self, strength: float, protection: float) -> None:
        """Recompute only the in-memory recolor candidate from existing masks."""
        if self.masks is None or self.recolor_target is None:
            raise RuntimeError("No tracked recolor preview is available")
        recolored = recolor_frame_sequence(
            self.frames,
            self.masks,
            self.recolor_target,
            strength=strength,
            shadow_protect_amount=protection,
        )
        labels = [Path(name).stem for name in frame_output_names(len(recolored))]
        contact = make_contact_sheet(recolored, labels, columns=min(5, len(recolored)))
        self.review_state.begin(recolored, contact)
        self.recolored_frames = list(self.review_state.frames)
        self.recolored_contact = self.review_state.contact_sheet
        self._show_contact_preview()
        if self.review_window is not None and self.review_window.exists():
            self.review_window.set_contact(contact)
        self.status.set(
            f"Preview updated: strength {round(strength * 100)}%, "
            f"protection {round(protection * 100)}%."
        )

    def _open_recolor_review(self) -> None:
        if not self.review_state.has_preview:
            return
        self._close_review_window()
        self.review_window = RecolorReviewWindow(self)

    def _accept_recolor_preview(self) -> None:
        if not self.review_state.has_preview or self.masks is None or self.layout is None:
            messagebox.showerror("Nothing to export", "Create and review a recolor preview first.")
            return
        directory = filedialog.askdirectory(title="Select export directory")
        if not directory:
            return
        try:
            outputs = self.review_state.accept(
                lambda pending_frames: save_results(
                    Path(directory),
                    self.frames,
                    self.masks,
                    pending_frames,
                    columns=min(5, len(self.frames)),
                )
            )
        except Exception as error:
            messagebox.showerror("Export failed", str(error))
            return
        self.recolored_frames = []
        self.recolored_contact = None
        self.recolor_target = None
        self._clear_contact_preview()
        self._close_review_window()
        self.status.set(f"Accepted and exported results to {Path(directory).resolve()}")
        messagebox.showinfo(
            "Results exported",
            "Saved masks, recolored frames, and contact sheets.\n"
            + "\n".join(str(path) for path in outputs.values()),
        )

    def _reject_recolor_preview(self) -> None:
        self.review_state.reject()
        self.recolored_frames = []
        self.recolored_contact = None
        self.recolor_target = None
        self._clear_contact_preview()
        self._close_review_window()
        if self.masks is not None:
            self.status.set("Recolor preview rejected. Tracking masks retained; choose another color and Recolor.")
        self._update_buttons()

    def _discard_recolor_preview(self) -> None:
        self.review_state.reject()
        self.recolored_frames = []
        self.recolored_contact = None
        self.recolor_target = None
        self._clear_contact_preview()
        self._close_review_window()

    def _close_review_window(self) -> None:
        if self.review_window is not None:
            review = self.review_window
            self.review_window = None
            review.destroy()

    def _tracked_mask_contact(self) -> Image.Image:
        if self.combined_preview_masks is None:
            raise RuntimeError("No tracked masks are available")
        overlays = [
            make_mask_overlay(frame, mask)
            for frame, mask in zip(self.frames, self.combined_preview_masks, strict=True)
        ]
        labels = [Path(name).stem for name in frame_output_names(len(overlays))]
        return make_contact_sheet(overlays, labels, columns=min(5, len(overlays)))

    def _open_tracked_mask_preview(self) -> None:
        if self.combined_preview_masks is None:
            return
        self._close_tracked_mask_preview()
        self.tracked_mask_window = TrackedMaskPreviewWindow(self)

    def _refresh_tracked_mask_preview(self) -> None:
        if self.combined_preview_masks is None:
            self.tracked_mask_info.set("Total mask pixels: 0")
            return
        total = sum(int(mask.sum()) for mask in self.combined_preview_masks)
        self.tracked_mask_info.set(f"Total mask pixels: {total}")
        if self.tracked_mask_window is not None and self.tracked_mask_window.exists():
            self.tracked_mask_window.set_contact(self._tracked_mask_contact())
            self.tracked_mask_window.update_controls()

    def _mark_mask_source_preview_changed(self) -> None:
        self._discard_recolor_preview()
        self._sync_mask_source_preview()
        self._refresh_tracked_mask_preview()
        self.status.set("Preview changed. Click Use These Masks to apply.")
        self._update_buttons()

    def _on_pre_gate_source_toggle(self) -> None:
        if self._busy or self.mask_source_state is None:
            return
        try:
            self.mask_source_state.set_enabled(pre_gate=self.add_pre_gate_source.get())
        except ValueError as error:
            self.add_pre_gate_source.set(False)
            messagebox.showerror("Pre-Gate unavailable", str(error))
            return
        self._mark_mask_source_preview_changed()

    def _on_reverse_source_toggle(self) -> None:
        if self._busy or self.mask_source_state is None:
            return
        if not self.add_reverse_source.get():
            self.mask_source_state.set_enabled(reverse=False)
            self._mark_mask_source_preview_changed()
            return
        if self.reverse_masks is not None:
            self.mask_source_state.set_reverse_masks(self.reverse_masks)
            self.mask_source_state.set_enabled(reverse=True)
            self._mark_mask_source_preview_changed()
            return
        if not self._start_reverse_generation():
            self.add_reverse_source.set(False)

    def _start_reverse_generation(self) -> bool:
        if self.sam_session is None or self.regular_masks is None:
            messagebox.showerror("Reverse unavailable", "Complete Regular tracking first.")
            return False
        if len(self.regular_masks) <= 14:
            messagebox.showerror(
                "Reverse unavailable",
                "Reverse tracking requires frames 000 through 014.",
            )
            return False
        if not self.regular_masks[14].any():
            messagebox.showerror(
                "Reverse unavailable",
                "Reverse requires a non-empty Regular mask on frame 014.",
            )
            return False
        regular_snapshot = [mask.copy() for mask in self.regular_masks]
        self._start_task(
            "reverse",
            "Generating reverse tracking from frame 014...",
            lambda: self.sam_session.track_reverse_logits(regular_snapshot, self._queue_status),
        )
        return True

    def _on_auto_filter_pre_gate_toggle(self) -> None:
        if self._busy or self.mask_source_state is None:
            return
        if not self.auto_filter_pre_gate.get():
            self.mask_source_state.set_pre_gate_filter_enabled(False)
            self._mark_mask_source_preview_changed()
            return
        if not self.add_pre_gate_source.get():
            self.auto_filter_pre_gate.set(False)
            messagebox.showerror("Pre-Gate required", "Enable Add Pre-Gate before Auto-filter.")
            return
        if self.reverse_masks is None:
            if not self._start_reverse_generation():
                self.auto_filter_pre_gate.set(False)
            return
        self._recompute_temporal_filter()
        self._mark_mask_source_preview_changed()

    def _schedule_pre_gate_update(self, value: str) -> None:
        rounded = max(
            PRE_GATE_PREVIEW_MIN_THRESHOLD,
            min(PRE_GATE_PREVIEW_MAX_THRESHOLD, round(float(value) * 2.0) / 2.0),
        )
        self.pre_gate_threshold.set(rounded)
        self.pre_gate_threshold_text.set(f"{rounded:+.1f}")
        if self._pending_pre_gate_update is not None:
            self.root.after_cancel(self._pending_pre_gate_update)
        self._pending_pre_gate_update = self.root.after(75, self._apply_pre_gate_settings)

    def _apply_pre_gate_settings(self) -> None:
        self._pending_pre_gate_update = None
        if self.pre_gate_diagnostics is None or not self.frames:
            return
        self.pre_gate_masks = derive_pre_gate_masks(
            self.pre_gate_diagnostics,
            self.frames,
            threshold=self.pre_gate_threshold.get(),
            fill_holes=self.pre_gate_fill_holes.get(),
        )
        if self.mask_source_state is not None:
            self.mask_source_state.set_pre_gate_masks(self.pre_gate_masks)
            self._recompute_temporal_filter()
            self._sync_mask_source_preview()
            self._refresh_tracked_mask_preview()
            if self.add_pre_gate_source.get():
                self._discard_recolor_preview()
        if self.pre_gate_diagnostics_window is not None and self.pre_gate_diagnostics_window.exists():
            self.pre_gate_diagnostics_window.refresh_contact()
        if self.add_pre_gate_source.get():
            self.status.set("Pre-Gate source preview changed. Click Use These Masks to apply.")
        else:
            self.status.set("Pre-Gate diagnostic preview updated; Combined mask is unchanged.")
        self._update_buttons()

    def _close_tracked_mask_preview(self) -> None:
        if self.tracked_mask_window is not None:
            preview = self.tracked_mask_window
            self.tracked_mask_window = None
            preview.destroy()

    def _open_logit_diagnostics(self) -> None:
        if self.sam_raw_logits is None or not self.frames:
            messagebox.showerror("Diagnostics unavailable", "Track masks before opening logit diagnostics.")
            return
        try:
            diagnostics = diagnose_sam_logits(self.sam_raw_logits, self.frames)
            diagnostic_text = format_sam_logit_diagnostics(diagnostics)
        except Exception as error:
            messagebox.showerror("Could not calculate diagnostics", str(error))
            return
        self._close_logit_diagnostics()
        self.logit_diagnostics_window = LogitDiagnosticsWindow(self, diagnostic_text)
        checks = (
            diagnostics.all_float32_2d,
            diagnostics.has_continuous_values,
            diagnostics.threshold_zero_compatible,
            diagnostics.monotonic_nonincreasing,
        )
        if all(checks):
            self.status.set("Raw logit diagnostics ready; all integrity checks passed.")
        else:
            self.status.set("Raw logit diagnostics found a failed integrity check. Copy and review the report.")

    def _close_logit_diagnostics(self) -> None:
        if self.logit_diagnostics_window is not None:
            diagnostics = self.logit_diagnostics_window
            self.logit_diagnostics_window = None
            diagnostics.destroy()

    def _logit_diagnostics_window_closed(self, diagnostics: "LogitDiagnosticsWindow") -> None:
        if self.logit_diagnostics_window is diagnostics:
            self.logit_diagnostics_window = None

    def _open_pre_gate_diagnostics(self) -> None:
        if self.pre_gate_diagnostics is None or not self.frames:
            messagebox.showerror(
                "Diagnostics unavailable",
                "Track masks before opening pre-gate diagnostics.",
            )
            return
        self._close_pre_gate_diagnostics()
        self.pre_gate_diagnostics_window = PreGateDiagnosticsWindow(self)
        self.status.set("Pre-gate diagnostics ready. Standard tracked masks are unchanged.")

    def _close_pre_gate_diagnostics(self) -> None:
        if self.pre_gate_diagnostics_window is not None:
            diagnostics = self.pre_gate_diagnostics_window
            self.pre_gate_diagnostics_window = None
            diagnostics.destroy()

    def _clear_pre_gate_diagnostics(self) -> None:
        self._close_pre_gate_diagnostics()
        self.pre_gate_diagnostics = None

    def _pre_gate_diagnostics_window_closed(self, diagnostics: "PreGateDiagnosticsWindow") -> None:
        if self.pre_gate_diagnostics_window is diagnostics:
            self.pre_gate_diagnostics_window = None

    def _open_temporal_filter_diagnostics(self) -> None:
        if self.temporal_filter_result is None:
            messagebox.showerror(
                "Diagnostics unavailable",
                "Enable Add Pre-Gate and Auto-filter Pre-Gate after Reverse cache is ready.",
            )
            return
        self._close_temporal_filter_diagnostics()
        self.temporal_filter_diagnostics_window = TemporalFilterDiagnosticsWindow(self)
        self.status.set("Temporal Pre-Gate Filter diagnostics ready.")

    def _close_temporal_filter_diagnostics(self) -> None:
        if self.temporal_filter_diagnostics_window is not None:
            diagnostics = self.temporal_filter_diagnostics_window
            self.temporal_filter_diagnostics_window = None
            diagnostics.destroy()

    def _temporal_filter_diagnostics_window_closed(
        self,
        diagnostics: "TemporalFilterDiagnosticsWindow",
    ) -> None:
        if self.temporal_filter_diagnostics_window is diagnostics:
            self.temporal_filter_diagnostics_window = None

    def _tracked_mask_window_closed(self, preview: "TrackedMaskPreviewWindow") -> None:
        if self.tracked_mask_window is preview:
            self.tracked_mask_window = None

    def _use_tracked_masks(self) -> None:
        if self._pending_threshold_update is not None:
            self.root.after_cancel(self._pending_threshold_update)
            self._pending_threshold_update = None
            self._apply_threshold_update()
        if self._pending_pre_gate_update is not None:
            self.root.after_cancel(self._pending_pre_gate_update)
            self._pending_pre_gate_update = None
            self._apply_pre_gate_settings()
        if self.mask_source_state is None:
            messagebox.showerror("Tracking required", "No Combined masks are available.")
            return
        self.masks = self.mask_source_state.adopt_preview()
        self._sync_mask_source_preview()
        threshold = validate_mask_threshold(self.mask_threshold.get())
        self.status.set(
            f"Using {self.mask_source_state.source_label} masks at threshold {threshold:+.2f}; "
            "choose a color and Recolor."
        )
        self._close_tracked_mask_preview()
        self._update_buttons()

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
                    if name == "reverse":
                        self.add_reverse_source.set(False)
                        self.auto_filter_pre_gate.set(False)
                        self.reverse_raw_logits = None
                        self.reverse_raw_masks = None
                        self.reverse_masks = None
                        self.filtered_pre_gate_masks = None
                        self.temporal_filter_result = None
                        self._close_temporal_filter_diagnostics()
                        if self.sam_session is not None:
                            self.sam_session.reverse_raw_logits = None
                        if self.mask_source_state is not None:
                            self.mask_source_state.set_enabled(reverse=False)
                            self.mask_source_state.set_pre_gate_filter_enabled(False)
                            self._sync_mask_source_preview()
                            self._refresh_tracked_mask_preview()
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
            generated_mask, raw_logits = result
            self.quick_frame0_mask = generated_mask.copy()
            self.quick_frame0_logits = raw_logits
            self.sam_raw_logits = None
            self.pre_gate_diagnostics = None
            self._reset_mask_sources(clear_active=True)
            self._rebuild_quick_frame0_from_logits()
            self.status.set("Frame 0 mask ready. Review the overlay, then track across frames.")
            self._refresh_cleanup_info()
            self._redraw_prompt()
        elif name == "track":
            self.sam_raw_logits = list(result)
            self.pre_gate_diagnostics = list(self.sam_session.pre_gate_diagnostics or ())
            self._rebuild_masks_from_logits()
            self._refresh_cleanup_info()
            self._redraw_prompt()
            self._refresh_mask_editor()
            self._open_tracked_mask_preview()
            self.status.set(
                f"Tracking complete for {len(self.masks)} frames. Review tracked masks before Recolor."
            )
        elif name == "reverse":
            self.reverse_raw_logits = [logits.copy() for logits in result]
            self._rebuild_masks_from_logits()
            if self.mask_source_state is None or self.reverse_masks is None:
                raise RuntimeError("Reverse masks were not retained after propagation")
            self.mask_source_state.set_reverse_masks(self.reverse_masks)
            self.mask_source_state.set_enabled(reverse=self.add_reverse_source.get())
            self._recompute_temporal_filter()
            self._sync_mask_source_preview()
            self._refresh_tracked_mask_preview()
            if self.auto_filter_pre_gate.get():
                self.status.set(
                    "Reverse cache ready; Temporal Pre-Gate Filter updated. "
                    "Click Use These Masks to apply."
                )
            elif self.add_reverse_source.get():
                self.status.set(
                    "Reverse tracking ready and added to the Combined preview. "
                    "Click Use These Masks to apply."
                )
            else:
                self.status.set("Reverse cache ready.")
        elif name == "recolor":
            self.recolored_frames, self.recolored_contact, self.recolor_target = result
            self.review_state.begin(self.recolored_frames, self.recolored_contact)
            self.recolored_frames = list(self.review_state.frames)
            self.recolored_contact = self.review_state.contact_sheet
            self.status.set("Recolor preview ready. Accept & Export or Reject it.")
            self._show_contact_preview()
            self._open_recolor_review()

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
        preview_dirty = self.mask_source_state is not None and self.mask_source_state.preview_dirty
        self.recolor_button.configure(
            state="disabled" if disabled or self.masks is None or preview_dirty else "normal"
        )
        self.cleanup_checkbox.configure(state="disabled" if disabled or not self.frames else "normal")
        self.threshold_scale.configure(state="disabled" if disabled or not self.frames else "normal")
        self.open_mask_editor_button.configure(
            state="disabled"
            if disabled or not self.frames or self.selection_mode.get() != "area"
            else "normal"
        )
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
        if self.mask_editor is not None and self.mask_editor.exists():
            self.mask_editor.update_controls()
        if self.tracked_mask_window is not None and self.tracked_mask_window.exists():
            self.tracked_mask_window.update_controls()

    def _close_session(self) -> None:
        if self.sam_session is not None:
            self.sam_session.close()
        self.sam_session = None
        self.session_checkpoint = None
        self.session_device_name = None
        self.sam_raw_logits = None
        self.pre_gate_diagnostics = None
        self.regular_raw_masks = None
        self.regular_masks = None
        self.pre_gate_masks = None
        self.filtered_pre_gate_masks = None
        self.temporal_filter_result = None
        self.reverse_raw_logits = None
        self.reverse_raw_masks = None
        self.reverse_masks = None
        self.combined_preview_masks = None
        self.mask_source_state = None
        self.raw_masks = None
        self.masks = None
        self.quick_frame0_logits = None

    def _on_close(self) -> None:
        self._discard_recolor_preview()
        self._close_tracked_mask_preview()
        self._close_logit_diagnostics()
        self._close_pre_gate_diagnostics()
        self._close_temporal_filter_diagnostics()
        self._close_mask_editor()
        self._close_session()
        self.root.destroy()


class MaskEditorWindow:
    """Large zoomable editor sharing the main app's LassoSelectionState."""

    def __init__(self, app: TeamColorApp) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Mask Editor - Select Area")
        width = min(max(1, app.screen_size[0] - 40), max(480, int(app.screen_size[0] * 0.85)))
        height = min(max(1, app.screen_size[1] - 80), max(400, int(app.screen_size[1] * 0.85)))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(640, width), min(480, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.zoom = choose_display_scale(
            app.frames[0].size,
            (max(1, width - 100), max(1, height - 170)),
            preferred_scale=8,
        )
        self.zoom_text = tk.StringVar(value=f"Zoom: {self.zoom}x")
        self._photo: ImageTk.PhotoImage | None = None
        self._lasso_points: list[tuple[int, int]] = []
        self._lasso_canvas_points: list[tuple[float, float]] = []
        self._shape_start: tuple[int, int] | None = None

        tool_controls = ttk.Frame(self.window, padding=(8, 8, 8, 2))
        tool_controls.pack(fill="x")
        ttk.Label(tool_controls, text="Selection tool:").pack(side="left")
        tools = (("Freehand", "freehand"), ("Rectangle", "rectangle"), ("Ellipse", "ellipse"))
        for label, value in tools:
            ttk.Radiobutton(
                tool_controls,
                text=label,
                variable=app.selection_tool,
                value=value,
                command=app._on_selection_tool_change,
            ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            tool_controls,
            text="Square / circle lock",
            variable=app.shape_constraint,
        ).pack(side="left", padx=(12, 0))

        controls = ttk.Frame(self.window, padding=(8, 2, 8, 4))
        controls.pack(fill="x")
        ttk.Label(controls, text="Operation:").pack(side="left")
        ttk.Radiobutton(controls, text="Add", variable=app.lasso_operation, value="add").pack(
            side="left", padx=(8, 0)
        )
        ttk.Radiobutton(controls, text="Subtract", variable=app.lasso_operation, value="subtract").pack(
            side="left", padx=(8, 0)
        )
        self.undo_button = ttk.Button(controls, text="Undo", command=app._undo_lasso)
        self.undo_button.pack(side="left", padx=(16, 0))
        self.clear_button = ttk.Button(controls, text="Clear Selection", command=app._clear_lasso)
        self.clear_button.pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Zoom -", command=lambda: self.change_zoom(-1)).pack(side="right")
        ttk.Button(controls, text="Zoom +", command=lambda: self.change_zoom(1)).pack(side="right", padx=(6, 0))
        ttk.Label(controls, textvariable=self.zoom_text).pack(side="right", padx=(12, 0))
        ttk.Label(self.window, textvariable=app.selection_info, padding=(8, 0)).pack(anchor="w")

        viewport = ttk.Frame(self.window, padding=(8, 4, 8, 8))
        viewport.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewport, background="#202020", highlightthickness=0)
        vertical = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(viewport, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom_wheel)
        self.canvas.bind("<MouseWheel>", self._on_vertical_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_horizontal_wheel)
        for sequence in ("<KeyPress-plus>", "<KeyPress-equal>", "<KeyPress-KP_Add>"):
            self.window.bind(sequence, lambda _event: self._zoom_key(1))
        for sequence in ("<KeyPress-minus>", "<KeyPress-KP_Subtract>"):
            self.window.bind(sequence, lambda _event: self._zoom_key(-1))
        self.window.bind("<Left>", lambda _event: self._arrow_pan(-1, 0))
        self.window.bind("<Right>", lambda _event: self._arrow_pan(1, 0))
        self.window.bind("<Up>", lambda _event: self._arrow_pan(0, -1))
        self.window.bind("<Down>", lambda _event: self._arrow_pan(0, 1))

        self.redraw()
        self.update_controls()
        self.window.after(50, self.canvas.focus_set)

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def focus(self) -> None:
        self.window.deiconify()
        self.window.lift()
        self.canvas.focus_set()

    def destroy(self) -> None:
        if self.exists():
            self.window.destroy()

    def close(self) -> None:
        self.app._mask_editor_closed(self)
        self.destroy()

    def _frame_point(self, event: tk.Event) -> tuple[int, int] | None:
        try:
            return viewport_to_frame(
                event.x,
                event.y,
                self.zoom,
                (self.canvas.canvasx(0), self.canvas.canvasy(0)),
                self.app.frames[0].size,
            )
        except ValueError:
            return None

    def _on_press(self, event: tk.Event) -> None:
        if self.app._busy:
            return
        self.canvas.focus_set()
        point = self._frame_point(event)
        if point is None:
            return
        self.cancel_draft()
        if self.app.selection_tool.get() == "freehand":
            self._lasso_points = [point]
            self._lasso_canvas_points = [(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))]
        else:
            self._shape_start = point
            self._draw_shape_draft(point, event)

    def _on_drag(self, event: tk.Event) -> None:
        if self.app._busy:
            return
        if self.app.selection_tool.get() != "freehand":
            if self._shape_start is not None:
                self._draw_shape_draft(self._frame_point(event), event)
            return
        if not self._lasso_points:
            return
        point = self._frame_point(event)
        if point is None or point == self._lasso_points[-1]:
            return
        canvas_point = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        previous = self._lasso_canvas_points[-1]
        self._lasso_points.append(point)
        self._lasso_canvas_points.append(canvas_point)
        self.canvas.create_line(*previous, *canvas_point, fill="#ffdc00", width=2, tags="selection_draft")

    def _on_release(self, event: tk.Event) -> None:
        if self.app._busy:
            return
        if self.app.selection_tool.get() != "freehand":
            self._commit_shape(event)
            return
        if not self._lasso_points:
            return
        point = self._frame_point(event)
        if point is not None and point != self._lasso_points[-1]:
            self._lasso_points.append(point)
        points = tuple(self._lasso_points)
        self._lasso_points = []
        self._lasso_canvas_points = []
        self.canvas.delete("selection_draft")
        if len(points) < 3 or self.app.lasso_selection is None:
            self.app.status.set("Lasso needs at least three distinct frame pixels.")
            return
        try:
            self.app.lasso_selection.apply(points, self.app.lasso_operation.get())
        except ValueError as error:
            self.app.status.set(str(error))
            return
        self.app._invalidate_after_lasso_change()

    def cancel_draft(self) -> None:
        self._lasso_points = []
        self._lasso_canvas_points = []
        self._shape_start = None
        self.canvas.delete("selection_draft")

    def _draw_shape_draft(self, end: tuple[int, int] | None, event: tk.Event) -> None:
        if self._shape_start is None or end is None:
            return
        constrained = self.app._shape_is_constrained(event)
        x0, y0, x1, y1 = shape_bounds(
            self._shape_start,
            end,
            self.app.frames[0].size,
            constrained,
        )
        coordinates = (
            x0 * self.zoom,
            y0 * self.zoom,
            (x1 + 1) * self.zoom,
            (y1 + 1) * self.zoom,
        )
        self.canvas.delete("selection_draft")
        draw = (
            self.canvas.create_rectangle
            if self.app.selection_tool.get() == "rectangle"
            else self.canvas.create_oval
        )
        draw(*coordinates, outline="#ffdc00", width=2, dash=(5, 3), tags="selection_draft")

    def _commit_shape(self, event: tk.Event) -> None:
        start = self._shape_start
        end = self._frame_point(event)
        self.cancel_draft()
        if start is None or end is None or self.app.lasso_selection is None:
            return
        constrained = self.app._shape_is_constrained(event)
        if self.app.selection_tool.get() == "rectangle":
            mask = rectangle_to_mask(start, end, self.app.frames[0].size, constrained)
        else:
            mask = ellipse_to_mask(start, end, self.app.frames[0].size, constrained)
        self.app.lasso_selection.apply_mask(mask, self.app.lasso_operation.get())
        self.app._invalidate_after_lasso_change()

    def _view_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _content_size(self) -> tuple[int, int]:
        frame = self.app.frames[0]
        return frame.width * self.zoom, frame.height * self.zoom

    def _set_scroll(self, offset: tuple[float, float]) -> None:
        content = self._content_size()
        self.canvas.xview_moveto(offset[0] / content[0] if content[0] else 0)
        self.canvas.yview_moveto(offset[1] / content[1] if content[1] else 0)

    def change_zoom(self, step: int, cursor: tuple[float, float] | None = None) -> None:
        old_zoom = self.zoom
        new_zoom = clamp_zoom(old_zoom + step, MIN_ZOOM, MAX_ZOOM)
        if new_zoom == old_zoom:
            return
        viewport = self._view_size()
        if cursor is None:
            cursor = (viewport[0] / 2, viewport[1] / 2)
        new_scroll = cursor_centered_zoom_offset(
            cursor,
            old_zoom,
            new_zoom,
            (self.canvas.canvasx(0), self.canvas.canvasy(0)),
            viewport,
            self.app.frames[0].size,
        )
        self.zoom = new_zoom
        self.zoom_text.set(f"Zoom: {new_zoom}x")
        self.redraw()
        self._set_scroll(new_scroll)

    def _on_zoom_wheel(self, event: tk.Event) -> str:
        self.change_zoom(1 if event.delta > 0 else -1, (event.x, event.y))
        return "break"

    def _zoom_key(self, step: int) -> str:
        self.change_zoom(step)
        return "break"

    def _wheel_units(self, delta: int) -> int:
        amount = max(1, abs(delta) // 120)
        return -amount if delta > 0 else amount

    def _on_vertical_wheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(self._wheel_units(event.delta), "units")
        return "break"

    def _on_horizontal_wheel(self, event: tk.Event) -> str:
        self.canvas.xview_scroll(self._wheel_units(event.delta), "units")
        return "break"

    def _arrow_pan(self, horizontal: int, vertical: int) -> str:
        step = max(16, self.zoom * 4)
        offset = pan_scroll_offset(
            (self.canvas.canvasx(0), self.canvas.canvasy(0)),
            (horizontal * step, vertical * step),
            self._content_size(),
            self._view_size(),
        )
        self._set_scroll(offset)
        return "break"

    def redraw(self) -> None:
        if not self.app.frames:
            return
        source = self.app.frames[0]
        if self.app.frame0_mask is not None:
            source = make_selection_overlay(source, self.app.frame0_mask)
        display_size = (source.width * self.zoom, source.height * self.zoom)
        display = source.resize(display_size, Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, display.width, display.height))

    def update_controls(self) -> None:
        selection = self.app.lasso_selection
        disabled = self.app._busy
        self.undo_button.configure(
            state="disabled" if disabled or selection is None or not selection.can_undo else "normal"
        )
        self.clear_button.configure(
            state="disabled" if disabled or selection is None or selection.is_empty else "normal"
        )


class ZoomPanImageView:
    """Shared nearest-neighbor zoom/pan canvas for review windows."""

    def __init__(self, parent: tk.Misc, window: tk.Toplevel, image: Image.Image, zoom: int = 1) -> None:
        self.window = window
        self.image = image.copy()
        self.zoom = clamp_zoom(zoom)
        self.zoom_text = tk.StringVar(value=f"Zoom: {self.zoom}x")
        self._photo: ImageTk.PhotoImage | None = None

        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, background="#181818", highlightthickness=0)
        vertical = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Control-MouseWheel>", self._on_zoom_wheel)
        self.canvas.bind("<MouseWheel>", self._on_vertical_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_horizontal_wheel)
        for sequence in ("<KeyPress-plus>", "<KeyPress-equal>", "<KeyPress-KP_Add>"):
            self.window.bind(sequence, lambda _event: self._zoom_key(1))
        for sequence in ("<KeyPress-minus>", "<KeyPress-KP_Subtract>"):
            self.window.bind(sequence, lambda _event: self._zoom_key(-1))
        self.window.bind("<Left>", lambda _event: self._arrow_pan(-1, 0))
        self.window.bind("<Right>", lambda _event: self._arrow_pan(1, 0))
        self.window.bind("<Up>", lambda _event: self._arrow_pan(0, -1))
        self.window.bind("<Down>", lambda _event: self._arrow_pan(0, 1))
        self.redraw()

    def _view_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _content_size(self) -> tuple[int, int]:
        return self.image.width * self.zoom, self.image.height * self.zoom

    def _set_scroll(self, offset: tuple[float, float]) -> None:
        content = self._content_size()
        self.canvas.xview_moveto(offset[0] / content[0] if content[0] else 0)
        self.canvas.yview_moveto(offset[1] / content[1] if content[1] else 0)

    def set_image(self, image: Image.Image) -> None:
        scroll = (self.canvas.canvasx(0), self.canvas.canvasy(0))
        self.image = image.copy()
        self.redraw()
        self._set_scroll(scroll)

    def change_zoom(self, step: int, cursor: tuple[float, float] | None = None) -> None:
        old_zoom = self.zoom
        new_zoom = clamp_zoom(old_zoom + step, MIN_ZOOM, MAX_ZOOM)
        if new_zoom == old_zoom:
            return
        viewport = self._view_size()
        if cursor is None:
            cursor = (viewport[0] / 2, viewport[1] / 2)
        new_scroll = cursor_centered_zoom_offset(
            cursor,
            old_zoom,
            new_zoom,
            (self.canvas.canvasx(0), self.canvas.canvasy(0)),
            viewport,
            self.image.size,
        )
        self.zoom = new_zoom
        self.zoom_text.set(f"Zoom: {new_zoom}x")
        self.redraw()
        self._set_scroll(new_scroll)

    def _on_zoom_wheel(self, event: tk.Event) -> str:
        self.change_zoom(1 if event.delta > 0 else -1, (event.x, event.y))
        return "break"

    def _zoom_key(self, step: int) -> str:
        self.change_zoom(step)
        return "break"

    @staticmethod
    def _wheel_units(delta: int) -> int:
        amount = max(1, abs(delta) // 120)
        return -amount if delta > 0 else amount

    def _on_vertical_wheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(self._wheel_units(event.delta), "units")
        return "break"

    def _on_horizontal_wheel(self, event: tk.Event) -> str:
        self.canvas.xview_scroll(self._wheel_units(event.delta), "units")
        return "break"

    def _arrow_pan(self, horizontal: int, vertical: int) -> str:
        step = max(16, self.zoom * 4)
        offset = pan_scroll_offset(
            (self.canvas.canvasx(0), self.canvas.canvasy(0)),
            (horizontal * step, vertical * step),
            self._content_size(),
            self._view_size(),
        )
        self._set_scroll(offset)
        return "break"

    def redraw(self) -> None:
        display = self.image.resize(self._content_size(), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, display.width, display.height))


class TrackedMaskPreviewWindow:
    """Review active tracked masks while rethresholding retained SAM logits."""

    def __init__(self, app: TeamColorApp) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Tracked Mask Preview")
        width = min(max(1, app.screen_size[0] - 40), max(480, int(app.screen_size[0] * 0.8)))
        height = min(max(1, app.screen_size[1] - 80), max(360, int(app.screen_size[1] * 0.8)))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(560, width), min(360, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        contact = app._tracked_mask_contact()
        initial_zoom = max(1, min(2, (width - 50) // contact.width, (height - 210) // contact.height))

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Tracked Mask Preview").pack(side="left")
        ttk.Button(
            header,
            text="Logit Diagnostics...",
            command=app._open_logit_diagnostics,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(
            header,
            text="Pre-Gate Diagnostics...",
            command=app._open_pre_gate_diagnostics,
        ).pack(side="left", padx=(8, 0))
        self.temporal_diagnostics_button = ttk.Button(
            header,
            text="Temporal Filter Diagnostics...",
            command=app._open_temporal_filter_diagnostics,
        )
        self.temporal_diagnostics_button.pack(side="left", padx=(8, 0))

        controls = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Mask Threshold:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            controls,
            from_=MIN_MASK_THRESHOLD,
            to=MAX_MASK_THRESHOLD,
            variable=app.mask_threshold,
            command=app._schedule_threshold_update,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Label(controls, textvariable=app.mask_threshold_text, width=7).grid(row=0, column=2)
        ttk.Label(controls, text="Lower = more inclusive").grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Fill enclosed holes",
            variable=app.fill_enclosed_holes,
            command=app._on_cleanup_toggle,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Label(controls, textvariable=app.tracked_mask_info).grid(row=1, column=2, sticky="e")
        controls.columnconfigure(1, weight=1)

        sources = ttk.LabelFrame(self.window, text="Mask Sources", padding=(8, 4))
        sources.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(sources, text="Regular SAM: ON — Base").grid(row=0, column=0, sticky="w")
        self.pre_gate_checkbox = ttk.Checkbutton(
            sources,
            text="Add Pre-Gate",
            variable=app.add_pre_gate_source,
            command=app._on_pre_gate_source_toggle,
        )
        self.pre_gate_checkbox.grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.auto_filter_checkbox = ttk.Checkbutton(
            sources,
            text="Auto-filter Pre-Gate",
            variable=app.auto_filter_pre_gate,
            command=app._on_auto_filter_pre_gate_toggle,
        )
        self.auto_filter_checkbox.grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.reverse_checkbox = ttk.Checkbutton(
            sources,
            text="Add Reverse",
            variable=app.add_reverse_source,
            command=app._on_reverse_source_toggle,
        )
        self.reverse_checkbox.grid(row=0, column=3, sticky="w", padx=(16, 0))
        ttk.Label(sources, textvariable=app.mask_sources_info).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        ttk.Label(sources, textvariable=app.mask_adoption_info).grid(
            row=1, column=3, sticky="e", pady=(4, 0)
        )
        sources.columnconfigure(3, weight=1)

        viewport = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        viewport.pack(fill="both", expand=True)
        self.image_view = ZoomPanImageView(viewport, self.window, contact, initial_zoom)
        self.image_view.frame.pack(fill="both", expand=True)
        ttk.Button(header, text="Zoom +", command=lambda: self.image_view.change_zoom(1)).pack(side="right")
        ttk.Button(header, text="Zoom -", command=lambda: self.image_view.change_zoom(-1)).pack(
            side="right", padx=(6, 0)
        )
        ttk.Label(header, textvariable=self.image_view.zoom_text).pack(side="right", padx=(12, 0))

        actions = ttk.Frame(self.window, padding=8)
        actions.pack(fill="x")
        self.use_button = ttk.Button(
            actions,
            text="Use These Masks",
            command=app._use_tracked_masks,
        )
        self.use_button.pack(side="right")
        ttk.Button(actions, text="Close", command=self.close).pack(side="right", padx=(0, 8))

        app._refresh_tracked_mask_preview()
        self.window.after(50, self.image_view.canvas.focus_set)

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def set_contact(self, contact: Image.Image) -> None:
        self.image_view.set_image(contact)

    def update_controls(self) -> None:
        disabled = self.app._busy
        self.pre_gate_checkbox.configure(
            state="disabled"
            if disabled or self.app.pre_gate_diagnostics is None
            else "normal"
        )
        self.auto_filter_checkbox.configure(
            state="disabled"
            if (
                disabled
                or self.app.pre_gate_diagnostics is None
                or not self.app.add_pre_gate_source.get()
            )
            else "normal"
        )
        reverse_ready = (
            self.app.regular_masks is not None
            and len(self.app.regular_masks) > 14
            and bool(self.app.regular_masks[14].any())
        )
        self.reverse_checkbox.configure(
            state="disabled" if disabled or not reverse_ready else "normal"
        )
        self.use_button.configure(
            state="disabled" if disabled or self.app.combined_preview_masks is None else "normal"
        )
        self.temporal_diagnostics_button.configure(
            state="disabled" if self.app.temporal_filter_result is None else "normal"
        )

    def destroy(self) -> None:
        if self.exists():
            self.window.destroy()

    def close(self) -> None:
        self.app._tracked_mask_window_closed(self)
        self.destroy()


class LogitDiagnosticsWindow:
    """Scrollable, copyable plain-text report for retained SAM raw logits."""

    def __init__(self, app: TeamColorApp, report: str) -> None:
        self.app = app
        self.report = report
        self.window = tk.Toplevel(app.root)
        self.window.title("SAM Raw Logit Diagnostics")
        width = min(max(1, app.screen_size[0] - 80), 900)
        height = min(max(1, app.screen_size[1] - 120), 720)
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(520, width), min(320, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="SAM Raw Logit Diagnostics").pack(side="left")
        ttk.Button(header, text="Copy All", command=self.copy_all).pack(side="right")
        ttk.Button(header, text="Close", command=self.close).pack(side="right", padx=(0, 8))

        viewport = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        viewport.pack(fill="both", expand=True)
        self.text = tk.Text(viewport, wrap="none", font="TkFixedFont")
        vertical = ttk.Scrollbar(viewport, orient="vertical", command=self.text.yview)
        horizontal = ttk.Scrollbar(viewport, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.text.insert("1.0", report)
        self.text.configure(state="disabled")
        self.window.after(50, self.text.focus_set)

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def copy_all(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self.report)
        self.app.status.set("Raw logit diagnostics copied to the clipboard.")

    def destroy(self) -> None:
        if self.exists():
            self.window.destroy()

    def close(self) -> None:
        self.app._logit_diagnostics_window_closed(self)
        self.destroy()


class PreGateDiagnosticsWindow:
    """Diagnostic-only pre-object-gate candidate overlay and text report."""

    def __init__(self, app: TeamColorApp) -> None:
        if app.pre_gate_diagnostics is None:
            raise RuntimeError("No pre-gate diagnostics are available")
        self.app = app
        self.diagnostics = tuple(app.pre_gate_diagnostics)
        self.report = format_pre_gate_diagnostics(app.frames, self.diagnostics)
        contact = make_pre_gate_contact_sheet(
            app.frames,
            self.diagnostics,
            threshold=app.pre_gate_threshold.get(),
            fill_holes=app.pre_gate_fill_holes.get(),
            columns=5,
        )

        self.window = tk.Toplevel(app.root)
        self.window.title("SAM Pre-Gate Diagnostics")
        width = min(max(1, app.screen_size[0] - 60), max(640, int(app.screen_size[0] * 0.85)))
        height = min(max(1, app.screen_size[1] - 100), max(480, int(app.screen_size[1] * 0.85)))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(640, width), min(480, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Pre-Gate Diagnostics").pack(side="left")
        ttk.Label(
            header,
            text="Observation only — standard masks and Recolor are unchanged.",
        ).pack(side="left", padx=(12, 0))
        ttk.Button(header, text="Copy All", command=self.copy_all).pack(side="right")
        ttk.Button(header, text="Close", command=self.close).pack(side="right", padx=(0, 8))

        controls = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Pre-Gate Preview Threshold:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            controls,
            from_=PRE_GATE_PREVIEW_MIN_THRESHOLD,
            to=PRE_GATE_PREVIEW_MAX_THRESHOLD,
            variable=app.pre_gate_threshold,
            command=app._schedule_pre_gate_update,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Label(controls, textvariable=app.pre_gate_threshold_text, width=7).grid(row=0, column=2)
        ttk.Checkbutton(
            controls,
            text="Apply hole fill for diagnostic view",
            variable=app.pre_gate_fill_holes,
            command=app._apply_pre_gate_settings,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        controls.columnconfigure(1, weight=1)

        image_viewport = ttk.Frame(self.window, padding=(8, 0, 8, 4))
        image_viewport.pack(fill="both", expand=True)
        self.image_view = ZoomPanImageView(image_viewport, self.window, contact, zoom=1)
        self.image_view.frame.pack(fill="both", expand=True)

        text_viewport = ttk.LabelFrame(self.window, text="Text diagnostics", padding=(8, 4))
        text_viewport.pack(fill="both", padx=8, pady=(0, 8))
        self.text = tk.Text(text_viewport, height=12, wrap="none", font="TkFixedFont")
        text_vertical = ttk.Scrollbar(text_viewport, orient="vertical", command=self.text.yview)
        text_horizontal = ttk.Scrollbar(text_viewport, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=text_vertical.set, xscrollcommand=text_horizontal.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        text_vertical.grid(row=0, column=1, sticky="ns")
        text_horizontal.grid(row=1, column=0, sticky="ew")
        text_viewport.rowconfigure(0, weight=1)
        text_viewport.columnconfigure(0, weight=1)
        self.text.insert("1.0", self.report)
        self.text.configure(state="disabled")
        self.window.after(50, self.image_view.canvas.focus_set)

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def refresh_contact(self) -> None:
        contact = make_pre_gate_contact_sheet(
            self.app.frames,
            self.diagnostics,
            threshold=self.app.pre_gate_threshold.get(),
            fill_holes=self.app.pre_gate_fill_holes.get(),
            columns=5,
        )
        self.image_view.set_image(contact)

    def copy_all(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self.report)
        self.app.status.set("Pre-gate diagnostics copied to the clipboard.")

    def destroy(self) -> None:
        if self.exists():
            self.window.destroy()

    def close(self) -> None:
        self.app._pre_gate_diagnostics_window_closed(self)
        self.destroy()


class TemporalFilterDiagnosticsWindow:
    """Filtered Pre-Gate overlay plus copyable per-frame geometry diagnostics."""

    def __init__(self, app: TeamColorApp) -> None:
        if app.temporal_filter_result is None:
            raise RuntimeError("No Temporal Pre-Gate Filter result is available")
        self.app = app
        self.result = app.temporal_filter_result
        self.report = format_temporal_filter_diagnostics(self.result)
        contact = make_temporal_filter_contact_sheet(app.frames, self.result, columns=5)

        self.window = tk.Toplevel(app.root)
        self.window.title("Temporal Pre-Gate Filter Diagnostics")
        width = min(max(1, app.screen_size[0] - 60), max(640, int(app.screen_size[0] * 0.85)))
        height = min(max(1, app.screen_size[1] - 100), max(480, int(app.screen_size[1] * 0.85)))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(640, width), min(480, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Temporal Pre-Gate Filter Diagnostics").pack(side="left")
        ttk.Label(
            header,
            text="Green = accepted filtered Pre-Gate pixels; trusted frames stay clear.",
        ).pack(side="left", padx=(12, 0))
        ttk.Button(header, text="Copy All", command=self.copy_all).pack(side="right")
        ttk.Button(header, text="Close", command=self.close).pack(side="right", padx=(0, 8))

        image_viewport = ttk.Frame(self.window, padding=(8, 0, 8, 4))
        image_viewport.pack(fill="both", expand=True)
        self.image_view = ZoomPanImageView(image_viewport, self.window, contact, zoom=1)
        self.image_view.frame.pack(fill="both", expand=True)

        text_viewport = ttk.LabelFrame(self.window, text="Text diagnostics", padding=(8, 4))
        text_viewport.pack(fill="both", padx=8, pady=(0, 8))
        self.text = tk.Text(text_viewport, height=14, wrap="none", font="TkFixedFont")
        vertical = ttk.Scrollbar(text_viewport, orient="vertical", command=self.text.yview)
        horizontal = ttk.Scrollbar(text_viewport, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        text_viewport.rowconfigure(0, weight=1)
        text_viewport.columnconfigure(0, weight=1)
        self._replace_report()
        self.window.after(50, self.image_view.canvas.focus_set)

    def _replace_report(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.report)
        self.text.configure(state="disabled")

    def update_result(self, result: TemporalPreGateFilterResult) -> None:
        self.result = result
        self.report = format_temporal_filter_diagnostics(result)
        self.image_view.set_image(
            make_temporal_filter_contact_sheet(self.app.frames, result, columns=5)
        )
        self._replace_report()

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def copy_all(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self.report)
        self.app.status.set("Temporal Pre-Gate Filter diagnostics copied to the clipboard.")

    def destroy(self) -> None:
        if self.exists():
            self.window.destroy()

    def close(self) -> None:
        self.app._temporal_filter_diagnostics_window_closed(self)
        self.destroy()


class RecolorReviewWindow:
    """Zoomable in-memory preview that exports only after explicit acceptance."""

    def __init__(self, app: TeamColorApp) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Recolor Preview - Accept or Reject")
        width = min(max(1, app.screen_size[0] - 40), max(480, int(app.screen_size[0] * 0.8)))
        height = min(max(1, app.screen_size[1] - 80), max(360, int(app.screen_size[1] * 0.8)))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(560, width), min(360, height))
        self.window.protocol("WM_DELETE_WINDOW", app._reject_recolor_preview)

        self.contact = app.review_state.contact_sheet
        if self.contact is None:
            raise RuntimeError("No recolor contact sheet is available")
        self._pending_recolor_update: str | None = None
        self.strength_text = tk.StringVar()
        self.protection_text = tk.StringVar()
        self._update_setting_labels()
        initial_zoom = max(
            1,
            min(2, (width - 50) // self.contact.width, (height - 230) // self.contact.height),
        )

        header = ttk.Frame(self.window, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Recolor Preview").pack(side="left")
        ttk.Label(header, text="No files are written until Accept & Export.").pack(side="left", padx=(12, 0))

        settings = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        settings.pack(fill="x")
        ttk.Label(settings, text="Recolor Strength:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            settings,
            from_=0,
            to=100,
            variable=app.recolor_strength_percent,
            command=self._schedule_recolor_update,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Label(settings, textvariable=self.strength_text, width=5).grid(row=0, column=2, sticky="e")
        ttk.Label(settings, text="Dark / Outline Protection:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Scale(
            settings,
            from_=0,
            to=100,
            variable=app.outline_protection_percent,
            command=self._schedule_recolor_update,
        ).grid(row=1, column=1, sticky="ew", padx=(8, 6), pady=(5, 0))
        ttk.Label(settings, textvariable=self.protection_text, width=5).grid(
            row=1,
            column=2,
            sticky="e",
            pady=(5, 0),
        )
        mask_pixels = sum(int(mask.sum()) for mask in app.masks) if app.masks is not None else 0
        ttk.Label(settings, text=f"Mask pixels: {mask_pixels}").grid(
            row=0,
            column=3,
            rowspan=2,
            sticky="e",
            padx=(18, 0),
        )
        settings.columnconfigure(1, weight=1)

        viewport = ttk.Frame(self.window, padding=(8, 0, 8, 8))
        viewport.pack(fill="both", expand=True)
        self.image_view = ZoomPanImageView(viewport, self.window, self.contact, initial_zoom)
        self.image_view.frame.pack(fill="both", expand=True)
        ttk.Button(header, text="Zoom +", command=lambda: self.image_view.change_zoom(1)).pack(side="right")
        ttk.Button(header, text="Zoom -", command=lambda: self.image_view.change_zoom(-1)).pack(
            side="right", padx=(6, 0)
        )
        ttk.Label(header, textvariable=self.image_view.zoom_text).pack(side="right", padx=(12, 0))

        actions = ttk.Frame(self.window, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Accept & Export...", command=self._accept_current_preview).pack(side="right")
        ttk.Button(actions, text="Reject", command=app._reject_recolor_preview).pack(side="right", padx=(0, 8))

        self.window.after(50, self.image_view.canvas.focus_set)

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def destroy(self) -> None:
        if self._pending_recolor_update is not None and self.exists():
            self.window.after_cancel(self._pending_recolor_update)
            self._pending_recolor_update = None
        if self.exists():
            self.window.destroy()

    def _update_setting_labels(self) -> None:
        self.strength_text.set(f"{round(self.app.recolor_strength_percent.get())}%")
        self.protection_text.set(f"{round(self.app.outline_protection_percent.get())}%")

    def _schedule_recolor_update(self, _value: str) -> None:
        self._update_setting_labels()
        if self._pending_recolor_update is not None:
            self.window.after_cancel(self._pending_recolor_update)
        self._pending_recolor_update = self.window.after(75, self._apply_recolor_settings)

    def _apply_recolor_settings(self) -> bool:
        self._pending_recolor_update = None
        try:
            self.app._update_recolor_preview(
                percentage_to_unit(round(self.app.recolor_strength_percent.get())),
                percentage_to_unit(round(self.app.outline_protection_percent.get())),
            )
        except Exception as error:
            self.app.status.set(f"Could not update recolor preview: {error}")
            return False
        return True

    def _accept_current_preview(self) -> None:
        if self._pending_recolor_update is not None:
            self.window.after_cancel(self._pending_recolor_update)
            self._pending_recolor_update = None
            if not self._apply_recolor_settings():
                return
        self.app._accept_recolor_preview()

    def set_contact(self, contact: Image.Image) -> None:
        self.contact = contact.copy()
        self.image_view.set_image(contact)


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
