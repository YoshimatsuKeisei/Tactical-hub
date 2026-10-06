# SAM 2.1 sprite part tracking smoke test

This directory is isolated from the Tactical-hub application. It tests whether the
official Meta SAM 2.1 video predictor can propagate point prompts from frame 0
through the 15 frames in row 0 of `input/Idle.png`.

The smoke test does not recolor sprites and does not modify Tactical-hub game, UI,
CPU, RL, PPO, or package configuration.

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
