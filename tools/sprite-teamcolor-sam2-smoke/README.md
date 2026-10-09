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
4. Choose **Select Area**, **Quick Select**, or **Select All**. Select Area is
   the default.
5. For Select Area, choose **Freehand**, **Rectangle**, or **Ellipse**, then
   choose **Add** or **Subtract**. Freehand is the default tool.
6. Drag on frame 0. Freehand closes and fills the dragged loop; Rectangle and
   Ellipse show a yellow dashed outline while dragging and commit on release.
7. Refine the overlay with more Add or Subtract operations. **Undo** reverses
   the latest shape operation from the shared history; **Clear Selection**
   starts over.
8. Review the translucent selection and yellow boundary. Leave **Fill enclosed
   holes** enabled unless comparing the raw mask. There is no Generate Mask step
   for Select Area.
9. Select the SAM 2.1 checkpoint, click **Track Across Frames**, and wait for
   propagation to finish. The **Tracked Mask Preview** opens automatically.
10. Inspect all tracked overlays. If needed, adjust **SAM Mask Threshold** and
    compare with **Fill enclosed holes** on or off, then choose **Use These
    Masks**.
11. Choose Red, Blue, Green, Yellow, or enter a custom color.
12. Click **Recolor** and review the contact sheet in the resizable preview
    window.
13. Click **Accept & Export...** to choose an output directory, or **Reject** to
    discard only the recolor candidate.

**Select Area** is the recommended mode. Its binary selection mask is registered on
frame 0 through the official SAM 2 Video Predictor `add_new_mask()` API, then SAM
2 tracks it through the remaining frames. The saved frame 0 mask remains the
user-confirmed selection after alpha clipping; SAM 2 does not replace it
with a different shape.

Select Area provides these tools:

- **Freehand**: draw the existing closed lasso selection.
- **Rectangle**: drag between opposite corners of a rectangle.
- **Ellipse**: drag its bounding rectangle.

Hold **Shift** while dragging Rectangle or Ellipse to constrain it to a square
or circle. The shared **Square / circle lock** checkbox provides the same
constraint when Shift handling is inconvenient on Windows. Shape tools work in
both the main editor and **Open Mask Editor...**. Both windows share the selected
tool, Add/Subtract operation, constraint setting, selection mask, and Undo history.

**Quick Select** is an auxiliary mode for clearly separated parts that can be
identified with a few clicks. Add Positive and optional Negative clicks, use
**Undo** or **Clear clicks**, click **Generate Mask (Quick Select)**, review the
overlay, and then click **Track Across Frames**. This retains the existing point
prompt workflow.

**Select All** derives the frame 0 selection from the source image immediately.
For RGBA input it selects only pixels whose alpha is greater than zero, leaving
transparent background unselected. For RGB input it selects the complete frame.
It uses the same SAM 2 `add_new_mask()` and propagation path as Select Area.
Switching modes preserves the editable Select Area mask and Quick Select clicks;
Select All is regenerated from frame 0.

### Mask cleanup

**Mask cleanup: Fill enclosed holes** is enabled by default. It finds False
pixel regions in each binary mask and fills only regions that cannot reach an
image edge. A completely enclosed hole inside a shield-like selected part is
filled, while the outside background and any notch or gap connected to an image
edge remain unchanged. It does not dilate, erode, blur, or thicken the mask's
outer boundary.

Cleanup is applied to the visible frame 0 result and to every tracked mask used
by Recolor and Accept & Export. The raw Quick Select or tracked masks remain in
memory, so clearing the checkbox immediately derives and displays their unfilled
versions without retracking. Re-enabling it reapplies the idempotent hole fill.
The main window reports **Mask pixels: raw → active** and identifies whether
alpha clipping plus hole fill or alpha clipping alone produced the active mask.
This is intended for cases where SAM 2 captures a part's outline but leaves
internal holes; turn cleanup off to compare the unfilled, alpha-clipped mask.

### Tracked Mask Preview and diagnostic threshold

After **Track Across Frames** completes, **Tracked Mask Preview** automatically
shows the source frames with the current Combined preview masks as translucent overlays,
labelled `frame_000` through the selected frame count. It reports the total mask
pixel count and uses the same nearest-neighbor 1x–16x navigation as Recolor
Preview: Ctrl+wheel or touchpad pinch and +/- for zoom, wheel/Shift+wheel and
arrow keys for pan, plus horizontal and vertical scrollbars. **Use These Masks**
adopts the visible Combined choice for Recolor and Export; **Close** only closes
the window and does not discard tracking data or change the previously adopted
masks.

**SAM Mask Threshold** is a diagnostic control from -2.0 through +2.0, in 0.05
steps. Its default is **0.0**, which is pixel-for-pixel compatible with the
previous fixed `mask_logits > 0` behavior. Lower values are more inclusive;
higher values are more restrictive. The main window and Tracked Mask Preview
share one value. After tracking, moving the slider thresholds the retained CPU
raw logits, resizes with nearest-neighbor, clips to frame alpha, and reapplies
optional enclosed-hole filling. It does **not** rerun SAM 2 inference or video
propagation.

For **Quick Select**, threshold changes also update the frame 0 point-prompt
overlay. For **Select Area**, the explicit Freehand/Rectangle/Ellipse frame 0
mask remains authoritative. For **Select All**, the frame 0 alpha-support mask
remains authoritative. In both latter modes, the threshold applies to tracked
frames 1 onward.

For the current Melee shield diagnosis, start at 0.0 and try -0.1, -0.2, or
-0.3 gradually while watching partially visible frames such as 2–4 and 12.
Those are comparison candidates, not universal recommended values: a useful
setting must recover target pixels without unacceptable body, weapon, or
background spill. This manual control is for diagnosis; automatic threshold
selection remains future work.

If the visible mask does not change across the current -2.0 through +2.0 slider,
click **Logit Diagnostics...** in Tracked Mask Preview before changing that
range. The copyable plain-text report shows, for every tracked frame:

- retained array shape and dtype
- min, max, mean, and p01/p05/p25/p50/p75/p95/p99
- unique-value count
- alpha-clipped mask pixel counts at thresholds -20, -10, -5, -2, -1, -0.5,
  0, +0.5, +1, +2, +5, +10, and +20

The report also marks whether every retained array is 2D float32, whether at
least one frame has more than 16 unique values, whether threshold 0.0 matches
the direct legacy `logits > 0` path, and whether pixel counts are nonincreasing
as the threshold rises. Diagnostic pixel counts use only threshold, nearest-
neighbor resize, and alpha clipping; **Fill enclosed holes is deliberately
excluded**. Use **Copy All** to place the complete report on the clipboard for
comparison. The GUI slider remains -2.0 through +2.0 until real-data diagnostics
justify a different fixed or adaptive range.

### Pre-Gate Diagnostics

**Pre-Gate Diagnostics...** is a development-only view for frames where SAM 2.1
may replace its mask with `NO_OBJ_SCORE` after deciding that the object is
absent. During the normal `propagate_in_video` call, the tool temporarily adds an
observation-only PyTorch forward hook to `predictor.sam_mask_decoder`. The hook
copies the decoder's low-resolution mask candidates, IoU estimates, and object
score logit before the object-presence gate, and is always removed when
propagation finishes or raises an exception. The official SAM 2 source, config,
checkpoint, decoder output, and standard post-gate tracking result are not
modified.

After tracking, open **Pre-Gate Diagnostics...** separately from **Logit
Diagnostics...**. It shows a 5-column contact sheet with orange overlays for
unambiguous pre-gate candidates and a copyable text report. Conditioning frames
without a decoder call are labelled `NO CAPTURE`; multiple decoder calls between
predictor yields are labelled `AMBIGUOUS` and are not guessed into one overlay.
The report includes decoder call count, object score logit, object-absent gate
status, best IoU candidate, candidate statistics, post-gate `NO_OBJ_SCORE`
status, and pixel counts at 0, -5, -10, and +5.

**Pre-Gate Preview Threshold** ranges from -20 through +20 in 0.5 steps and
defaults to 0. It only re-renders retained CPU diagnostic data using threshold,
nearest-neighbor resize, and frame-alpha clipping. The optional diagnostic hole
fill starts off. Neither control reruns SAM 2 or changes the standard masks.
The same threshold and diagnostic hole-fill state are shared with the optional
Pre-Gate mask source described below. In Legacy mode they reach Recolor or
Export only after **Add Pre-Gate** is selected and the visible Combined mask is
explicitly adopted with **Use These Masks**. Identity Global can evaluate them
as candidates, but its selected path is likewise preview-only until adoption.

### Mask Sources and Combined preview

Tracked Mask Preview keeps three independent sources and always rebuilds the
visible result from them:

- **Regular SAM** is the normal Forward tracking result and is always the base.
  It cannot be turned off.
- **Pre-Gate** is the Forward candidate captured before SAM 2 replaces an
  object-absent result with `NO_OBJ_SCORE`. **Add Pre-Gate** uses the exact
  threshold and hole-fill settings currently shown by Pre-Gate Diagnostics.
- **Reverse** starts from the non-empty Regular mask on `frame_014`, creates a
  fresh SAM inference state with the already-loaded predictor/model, and calls
  official `propagate_in_video(start_frame_idx=14, reverse=True)` to track down
  through `frame_000`. It uses standard post-gate SAM output; Reverse Pre-Gate
  is not captured.

The Combined preview is deliberately only this union:

```text
Regular OR optional Reverse OR (Raw Pre-Gate or Auto-filtered Pre-Gate)
```

Use **Add Pre-Gate** and **Add Reverse** to compare Regular only, Regular +
Pre-Gate, Regular + Reverse, and all three sources. The source label and total
Combined pixel count update with the 15-frame contact sheet. Reverse inference
runs in the background only the first time it is enabled. Turning Reverse off
removes it from the union but retains its CPU raw logits; enabling it again is
immediate. SAM Mask Threshold changes rebuild both Regular and cached Reverse
from retained logits without rerunning SAM.

Source toggles change only the preview. The previous Recolor/Export masks remain
separate, and Recolor is disabled while the preview is unapplied. Click **Use
These Masks** to copy the currently visible Combined masks into the active
Recolor/Export state. Every source change rebuilds from the independent Regular,
Pre-Gate, and Reverse arrays, so removed source pixels are never baked into
Regular. Loading a sheet or row, changing the frame-0 selection or Quick Select
prompts, rerunning Forward tracking, recreating/closing the SAM session, or
exiting clears the Reverse cache.

This remains a comparison facility. It does not implement automatic source or
frame selection, confidence weighting, intersection, voting, or
object-score/IoU fallback.

### Temporal Pre-Gate Filter (experimental)

Enable **Add Pre-Gate**, then toggle **Auto-filter Pre-Gate** to replace the Raw
Pre-Gate contribution with a geometry-only filtered contribution. Raw and
filtered Pre-Gate are independent arrays and are never unioned together. Turning
Auto-filter off restores the existing Raw Pre-Gate preview; turning Add Pre-Gate
off excludes both variants. Source changes remain preview-only until **Use These
Masks** is clicked.

The filter builds a trusted mask at every frame from non-empty Regular or cached
Reverse post-gate masks. It finds consecutive gaps where that trusted mask is
empty, splits each Raw Pre-Gate candidate into 8-neighbor connected components,
and follows one spatially continuous component into each gap independently from
the left and right trusted anchors. Continuity uses centroid displacement, a
constant-velocity predicted center when two accepted masks are available, area
ratio, bounding-box width/height/diagonal ratios, and adaptive spatial
proximity. If a candidate fails, that directional chain stops instead of
reacquiring a later unrelated component.

For each frame, a candidate accepted by only one pass is retained. Candidates
accepted from both directions are unioned only when their IoU, centroid distance,
or bounding boxes show that they refer to the same spatial target; conflicting
results are marked `AMBIGUOUS` and excluded. Filtered Pre-Gate is intentionally
empty on frames already covered by Regular or Reverse. There are no frame-number
rules and no RGB, texture, model, or appearance comparison.

Auto-filter needs the Reverse cache for its right-side anchors. If the cache does
not exist when Auto-filter is first enabled, the existing background Reverse
tracking path creates it once from the Regular `frame_014` mask using a fresh SAM
state. The **Add Reverse** source does not need to remain enabled: its cached mask
can provide filter anchors without being added directly to the Combined preview.
Further filter toggles, Pre-Gate threshold/hole-fill changes, Regular threshold
changes, and cached Reverse updates recalculate only CPU binary-mask geometry;
they do not rerun SAM.

Open **Temporal Filter Diagnostics...** from Tracked Mask Preview to inspect a
green 15-frame Filtered Pre-Gate overlay and a scrollable, copyable report. For
each frame the report lists trusted status, Raw Pre-Gate pixels, component
geometry, left/right movement and size metrics, continuity score, pass result,
final result, and explicit rejection/stop reasons. **Copy All** places the report
on the clipboard for Windows test feedback.

This filter is experimental. Its purpose is to preserve temporally plausible
visible target fragments through occlusion while excluding Pre-Gate drift to a
different body part, weapon, or effect. Final quality—especially whether the
middle false positives disappear without losing the partially visible target—
must be checked on Windows with the actual Melee selection.

### Identity Global v1 (SAM-only)

Tracked Mask Preview has three independent composition modes. **Legacy Source
Union** preserves the existing Regular/Pre-Gate/Temporal Filter/Reverse union
pixel-for-pixel. **Identity Global v1 (SAM-only)** retains the original Identity
Global scoring and treats Regular,
cached Reverse, and every captured Forward Pre-Gate multimask as candidates. It
does not OR those sources together.

On first use, the tool reuses the loaded SAM 2.1 tiny predictor and calls its
official read-only `forward_image()` path in a fresh feature-only inference
state. The inspected FPN shapes determine which spatial level is cached: the
highest-resolution level whose 15 CPU `float16` maps fit the named 96 MiB cap is
chosen. With the fixed tiny model this is currently level 0, `32 x 256 x 256`,
about 60 MiB for 15 frames. Feature extraction runs in the background and does
not modify Forward or Reverse tracking state.

The authoritative user mask on frame 0 is coverage-resampled to the selected
feature resolution and used for weighted feature pooling. The resulting
L2-normalized SAM appearance descriptor is compared by cosine similarity with
descriptors pooled from each candidate. Small selections retain at least one
feature-cell support. Candidate masks come from:

- Regular post-gate masks;
- Reverse post-gate masks, generated once in the background from frame 14 when
  not already cached (Identity continues without them if Reverse fails);
- every captured Pre-Gate `low_res_multimasks` candidate, using the shared
  Pre-Gate threshold and hole-fill setting.

Each non-empty binary source is alpha-clipped and split into 8-neighbor connected
components. A full-sequence Dynamic Programming/Viterbi pass then selects one
candidate or an explicit **OCCLUDED** state per frame. Its primary node signal is
similarity to the frame-0 target; weak source priors and soft motion, area,
bounding-box, and temporal-appearance terms provide continuity without rejecting
a visually matching large movement. OCCLUDED can transition back to a distant,
strong identity match. Frame 0 is always the exact authoritative user mask.

Open **Identity Global v1 Diagnostics...** for a green selected-path contact sheet
and copyable text. The report includes the chosen feature level/shape/dtype/cache
size, target descriptor norm, all candidates and components, target/temporal
cosines, geometry penalties, node and best-DP scores, OCCLUDED score, and the
backtracked selection for each frame.

Changing Preview Mode never changes active Recolor/Export masks. Click **Use
These Masks** to adopt the currently visible Legacy or Identity result; switching
back to Legacy and adopting restores the Legacy masks without Identity pixels
being baked into any source. Pre-Gate threshold or hole-fill changes rebuild only
the candidate bank, descriptors, and DP path from cached features—SAM propagation
is not rerun.

Image features are target-independent and remain cached when the frame-0 target
changes within the same loaded row/model. Target descriptors, candidate
descriptors, and the path are cleared. Loading another sheet or row, changing
layout/frame count/checkpoint/device/model, closing the session, or exiting clears
the feature cache. This is a comparison experiment, not a quality PASS: sword,
shield, occlusion, and reappearance behavior still require Windows visual review.

### Identity Global v2 (Fused Experimental)

**Identity Global v2 (Fused Experimental)** uses the exact same authoritative
frame-0 mask and candidate masks as v1. It changes only identity evidence and the
node score used by a separate Global DP/Viterbi pass. Legacy and v1 remain
available for pixel-level regression comparison.

The first v2 selection prepares a target-independent, read-only multi-scale SAM
feature cache in the background using the already loaded predictor/model. Levels
are selected dynamically from the actual FPN shapes and `float16` byte counts:
the highest spatial-resolution level is prioritized, followed by the deepest
semantic level, and additional levels are retained only when the total stays at
or below 96 MiB. For the observed tiny-model shapes `(32,256,256)`,
`(64,128,128)`, and `(256,64,64)`, this selects levels 0 and 2 for 90 MiB across
15 frames. The choice is not hard-coded to those indices or shapes. A level-0
array already cached by v1 is shared rather than duplicated.

For each selected level, v2 coverage-pools and L2-normalizes a positive target
descriptor from the authoritative frame-0 mask. It also derives an automatic
negative context ring from visible (`alpha > 0`) unselected sprite pixels around
the target. The ring radius is based on target-bbox diagonal, bounded by named
minimum/maximum constants, expands when support is sparse, and can fall back to
visible unselected pixels in the local bbox neighborhood. Transparent background
is never used to manufacture negative evidence; if visible support remains too
small, that negative modality is disabled.

Candidate evidence uses the contrastive margin
`cos(candidate, positive) - cos(candidate, negative)`, normalized by the
frame-0 reference separation `1 - cos(positive, negative)` and clamped. Levels
whose reference separation is too small are disabled, while valid levels are
fused independently using separation-based weights. A weak local-context signal
compares candidate foreground-minus-ring contrast to frame-0 target contrast.
Lightweight native-pixel evidence uses RGB mean/std, luminance mean/std, and
luminance/saturation histograms; it is also positive-versus-negative and is not
allowed to dominate SAM features. A soft shape score compares log area ratio,
aspect ratio, bbox fill, and compactness. Extreme expansion is penalized but
never hard-rejected.

The fused identity node combines SAM contrastive, local-context, native-pixel,
and shape scores through named weights. A soft top-1/top-2 ambiguity penalty lets
the existing **OCCLUDED** state win when candidates are similarly weak, while
the constant frame-0 reference allows reacquisition after occlusion. Existing
soft temporal appearance, motion, area, and bbox transitions remain; position is
not a hard gate.

Open **Identity Global v2 Diagnostics...** for a magenta selected-path contact
sheet and **Copy All** report. It lists selected levels, shapes, dtype, per-level
and total cache MiB, negative-ring radius/support/fallback, reference separation,
active modalities, top three identity candidates and margin per frame, every
candidate's per-level positive/negative cosine and normalized margin, fused SAM,
local-context, pixel and shape scores, ambiguity/transition penalties, node/DP
scores, OCCLUDED score, and final selection.

Changing among Legacy, v1, and v2 updates only the preview. **Use These Masks**
is still required before Recolor/Export, and adopting one mode never writes its
pixels into another source. A frame-0 target change clears positive/negative,
pixel, shape, candidate-score, and v2-path state while retaining the same-row
multi-scale feature cache. Pre-Gate threshold or hole-fill changes rebuild masks,
descriptors, and both identity paths without SAM propagation. Sheet, row,
layout/frame-count, checkpoint, device/model, session-close, and application-exit
changes discard the multi-scale cache. v2 remains experimental and requires
Windows visual review; Cloud technical tests are not a quality PASS.

### Zoom and Mask Editor

The main frame-0 canvas supports integer zoom from 1x through 16x while retaining
nearest-neighbor rendering. Place the pointer over the canvas and use:

- a Windows Precision Touchpad pinch that emits `Control-MouseWheel`
- **Ctrl + mouse wheel**
- the **+** and **-** keys (or the visible +/- buttons)

Zoom is centered on the cursor when a wheel event supplies its position. Canvas
scroll offsets are included when converting every click, lasso point, and shape
corner back to the original frame coordinates.

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
window uses nearest-neighbor rendering and integer zoom from 1x through 16x.
Accept and Reject remain fixed outside the scrollable image.

The review window also provides two live recolor controls:

- **Recolor Strength** controls how strongly masked RGB pixels move toward the
  selected target hue. Its range is 0% through 100%, and its default is 85%.
- **Dark / Outline Protection** controls how strongly dark pixels and near-black
  outlines stay near their original color. Its range is 0% through 100%, and its
  default is 100%.

The existing recolor result is **Strength 85% / Protection 100%**. To check the
maximum target-hue coverage inside the mask, use **Strength 100% / Protection
0%**. This still preserves each source pixel's HSV value: dark pixels become dark
target-colored pixels rather than a flat bright fill. Slider changes recalculate
only recoloring from the existing tracked masks; they do not rerun SAM 2 tracking
and do not write files. **Mask pixels** reports the total selected pixels across
the tracked frames.

Preview navigation controls are:

- touchpad pinch or **Ctrl + mouse wheel**: cursor-centered zoom
- **+** / **-** keys or **Zoom +** / **Zoom -** buttons: zoom
- mouse wheel or touchpad scroll: vertical pan
- **Shift + mouse wheel**: horizontal pan
- arrow keys: horizontal or vertical pan
- horizontal and vertical scrollbars

Review actions are:

- **Accept & Export...**: choose a folder and write the currently previewed
  target color, strength, and protection result
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
