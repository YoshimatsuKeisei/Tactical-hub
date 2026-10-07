# SAM 2.1 sprite part tracking smoke test

This directory is isolated from the Tactical-hub application. It tests whether the
official Meta SAM 2.1 video predictor can propagate point prompts from frame 0
through the 15 frames in row 0 of `input/Idle.png`.

The SAM 2 smoke inference does not recolor sprites. A separate post-processing
utility in this directory can recolor existing mask results without rerunning SAM
2. Neither utility modifies Tactical-hub game, UI, CPU, RL, PPO, or package
configuration.

The Tkinter GUI MVP described below connects the same frame extraction, official
SAM 2 predictor, mask propagation, and shading-preserving recolor operations for
one selected spritesheet row.

## Scope

- Input: `input/Idle.png` (1920 x 1024 RGBA)
- Frames: row 0, 15 frames, 128 x 128 each
- SAM input: nearest-neighbor 4x scale to 512 x 512, composited on a solid RGB background
- Model: official `sam2.1_hiera_tiny`
- Output masks are clipped to each frame's original alpha support
- Both a reproducible CLI prompt and an optional OpenCV point-picker are provided

## Environment

The checked setup is CPU-only:

- Python 3.12.14
- torch 2.7.1+cpu
- torchvision 0.22.1+cpu
- official Meta SAM 2 source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM 2 CUDA extension intentionally disabled (`SAM2_BUILD_CUDA=0`)

See `environment_report.txt` and `requirements-lock.txt` for exact details.

## Checkpoint

Only this official checkpoint is accepted:

`models/sam2.1_hiera_tiny.pt`

Official Meta URL:

`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`

No unofficial mirror or alternate implementation is used. The checkpoint is
ignored by Git.

## Run with reproducible CLI points

Coordinates use the original 128 x 128 frame, even though SAM receives a
nearest-neighbour 1024 x 1024 work image to match its image encoder exactly.

```bash
.venv/bin/python src/smoke.py \
  --positive 67,51 \
  --negative 70,54 \
  --negative 58,61 \
  --negative 66,69 \
  --device cpu
```

Repeat `--positive` or `--negative` for multiple points.

## Legacy smoke OpenCV point picker

`smoke.py --gui` is a legacy point picker for smoke testing, not the supported
team-color workflow. On a machine with a graphical display:

```bash
.venv/bin/python src/smoke.py --gui --device auto
```

- Left click: positive point
- Right click: negative point
- Enter: run
- R: reset
- Esc: cancel

The cloud execution environment has no desktop display, so its run uses CLI
coordinates. Use the Tkinter GUI MVP below for actual interactive selection,
tracking, recoloring, and saving.

## Prepare-only validation

This validates and extracts the 15 frames without loading a model:

```bash
.venv/bin/python src/smoke.py --prepare-only
```

## Outputs

Successful inference writes the requested masks, RGBA overlays, contact sheet,
`metrics.json`, and `smoke_report.md` under `output/`. The report always requires
visual review; automated heuristics never declare a PASS.

## Recolor existing masks (Melee visual review)

`src/recolor.py` is a post-processing utility for visually reviewing team-color
conversion on the 15 frames in the top row of `Melee.png`. It uses the original
PNG pixels and existing `frame_000.png` through `frame_014.png` masks; it does not
run SAM 2 inference or generate replacement artwork.

The masked RGB pixels are shifted toward the requested hue while retaining each
pixel's HSV value. The blend is reduced for near-black pixels to protect outlines.
Pixels outside the mask and the complete alpha channel remain unchanged. Built-in
`red`, `blue`, `green`, and `yellow` presets are available; `#RRGGBB` and `R,G,B`
targets are also accepted for later experiments.

Required inputs:

- `--input`: the original sprite sheet PNG, or a directory of matching
  `frame_*.png` RGBA frames
- `--masks`: an existing smoke output directory containing grayscale
  `frame_*.png` masks
- `--output`: a new output directory

From this directory, apply the existing Melee row-0 masks without rerunning SAM 2:

```bash
python3 src/recolor.py \
  --input /path/to/Melee.png \
  --masks /path/to/melee-smoke/output/masks \
  --output /path/to/melee-red-review \
  --target red \
  --row 0 \
  --frame-count 15
```

For pre-extracted frames, pass their directory to `--input`; frame names are
matched to the mask names. Sprite-sheet frame dimensions default to 128 x 128 and
can be changed with `--frame-width` and `--frame-height`.

Outputs:

- `melee-red-review/recolored_frames/frame_000.png` through `frame_014.png`
- `melee-red-review/recolored_contact_sheet.png` (5 columns x 3 rows by default)

These outputs are for visual review. They do not change the existing smoke verdict;
Melee remains `REQUIRES_VISUAL_REVIEW` until a person checks mask coverage and the
recolored result.

## Sprite team-color GUI MVP

`src/gui.py` provides a desktop workflow for processing one row of a PNG
spritesheet without entering inference or recolor commands manually. It uses
Tkinter and Pillow, and reuses the existing SAM 2 and recolor implementation.
It does not alter the source spritesheet.

### Requirements and launch

Use the environment described above, including PyTorch, Pillow, NumPy, the
official Meta SAM 2 package, and Tkinter. Download the official
`sam2.1_hiera_tiny.pt` checkpoint to `models/`, or select it from another local
path in the GUI. No user-specific absolute path is embedded in the tool.

From the repository root on Windows, with the tool virtual environment already
created, run:

```bat
tools\sprite-teamcolor-sam2-smoke\.venv\Scripts\python.exe tools\sprite-teamcolor-sam2-smoke\src\gui.py
```

If the active Python environment already contains the required packages, this is
also valid:

```bat
python tools\sprite-teamcolor-sam2-smoke\src\gui.py
```

The GUI requires a desktop display. The Codex Cloud environment is headless, so
controller and image-processing tests can run there, but the window itself must
be visually checked on a local desktop.

### Operation

1. Select the spritesheet PNG with **Browse...**.
2. Enter the frame width/height and spritesheet columns/rows, then click
   **Load / Apply**.
3. Select the single target row and frame count to process.
4. Keep the default **Select Area** mode and choose **Add**.
5. Hold the left mouse button and draw a freehand loop around the target on
   frame 0; releasing the button closes and fills the loop.
6. Refine the overlay with additional **Add** or **Subtract** loops. Use
   **Undo** for the most recent operation or **Clear Selection** to start over.
7. Review the translucent selection and yellow boundary. There is no Generate
   Mask step for Select Area.
8. Select the SAM 2.1 checkpoint, click **Track Across Frames**, and wait for
   propagation to finish.
9. Choose Red, Blue, Green, Yellow, or enter a custom color.
10. Click **Recolor** and review the contact sheet in the resizable preview
    window.
11. Click **Accept & Export...** to choose an output directory, or **Reject** to
    discard only the recolor candidate.

**Select Area** is the recommended mode. Its binary lasso mask is registered on
frame 0 through the official SAM 2 Video Predictor `add_new_mask()` API, then SAM
2 tracks it through the remaining frames. The saved frame 0 mask remains the
user-confirmed lasso selection after alpha clipping; SAM 2 does not replace it
with a different shape.

**Quick Select** is an auxiliary mode for clearly separated parts that can be
identified with a few clicks. Add Positive and optional Negative clicks, use
**Undo** or **Clear clicks**, click **Generate Mask (Quick Select)**, review the
overlay, and then click **Track Across Frames**. This retains the existing point
prompt workflow.

### Zoom and Mask Editor

The main frame-0 canvas supports integer zoom from 1x through 16x while retaining
nearest-neighbor rendering. Place the pointer over the canvas and use:

- a Windows Precision Touchpad pinch that emits `Control-MouseWheel`
- **Ctrl + mouse wheel**
- the **+** and **-** keys (or the visible +/- buttons)

Zoom is centered on the cursor when a wheel event supplies its position. Canvas
scroll offsets are included when converting every click and lasso point back to
the original frame coordinates.

Click **Open Mask Editor...** for a large, resizable Select Area window. It uses
the exact same `LassoSelectionState` as the main window, so Add, Subtract, Undo,
and Clear Selection update both views immediately. Closing the editor does not
clear the mask. The editor provides horizontal and vertical scrollbars, and these
pan controls:

- mouse wheel or touchpad scroll: vertical
- **Shift + mouse wheel**: horizontal
- arrow keys: horizontal or vertical
- touchpad pinch / **Ctrl + wheel** / **+** / **-**: zoom

### Recolor review and export

**Recolor** creates recolored frames and the contact sheet only as in-memory PIL
images. It does not write preview PNGs or create an output directory. The review
window provides:

- **Accept & Export...**: choose a folder and write the existing result set
- **Reject**: close the preview and discard only the in-memory recolored frames

Rejecting or closing the review window writes no files and retains the source
spritesheet, Lasso selection, propagated masks, and SAM 2 tracking result. Choose
another team color and run Recolor again without retracking. If the mask itself
is edited, the tracked masks are invalidated and tracking must be run again.

Long-running SAM 2 setup and propagation run on a worker thread. Status updates
are passed back to the Tkinter main thread, so the interface can continue to
paint while inference runs.

Only an accepted export directory contains:

- `masks/frame_000.png`, and subsequent masks through the selected frame count
- `recolored_frames/frame_000.png`, and subsequent RGBA/RGB recolored frames
- `recolored_contact_sheet.png`
- `overlay_contact_sheet.png`

Pixels outside each mask are unchanged, and the source alpha channel is retained.
The existing low-luminance protection and shading-preserving recolor method are
used for every preset and custom color.

This MVP intentionally processes only one selected row. Processing all eight
directions at once, batching multiple animations or characters, writing a full
spritesheet back in place, and integrating output into Tactical-hub are deferred.
