# SAM 2.1 sprite part tracking smoke test

This directory is isolated from the Tactical-hub application. It tests whether the
official Meta SAM 2.1 video predictor can propagate point prompts from frame 0
through the 15 frames in row 0 of `input/Idle.png`.

The SAM 2 smoke inference does not recolor sprites. A separate post-processing
utility in this directory can recolor existing mask results without rerunning SAM
2. Neither utility modifies Tactical-hub game, UI, CPU, RL, PPO, or package
configuration.

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

## Optional OpenCV point picker

On a machine with a graphical display:

```bash
.venv/bin/python src/smoke.py --gui --device auto
```

- Left click: positive point
- Right click: negative point
- Enter: run
- R: reset
- Esc: cancel

The cloud execution environment has no desktop display, so its run uses CLI
coordinates.

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
Pixels outside the mask and the complete alpha channel remain unchanged. The
current review target is the built-in `red` preset; `#RRGGBB` and `R,G,B` targets
are also accepted for later experiments.

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
