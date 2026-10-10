# Tactical Hub

## Local Catapult asset setup

Builder strategists (`unit.type === "strategist"` and `unit.role === "builder"`)
are rendered with the separately purchased **Catapult - Isometric** sprite pack
after local asset preparation. The raw pack and prepared PNG files are
local-only and must never be committed or pushed.

1. Place the purchased `Catapult - Isometric` directory at the repository root.
   Its animation root must be
   `Catapult - Isometric/isometric/`.
2. From this directory, run:

   ```bash
   npm run assets:catapult
   ```

   The default source is the repository-root directory above. For a different
   location, set `CATAPULT_ASSET_ROOT` or pass the source path as the first
   command argument.
3. Prepared byte-copied frames and `manifest.json` are written to
   `public/local-assets/catapult/`.
4. Run `npm run dev`. The existing builder strategist text token is then
   replaced by the prepared Catapult sprite; other strategist roles keep their
   existing rendering.

Both the raw directory and prepared output are covered by repository `.gitignore`
rules. CI and Cloud builds do not require the purchased pack. Their explicit
text fallback and `Catapult asset unavailable` console warning are only the
missing-local-asset path, not the intended final UI.

The idle pose is not a source animation. It uses frame `0000` from each direction
of `move`. Attack playback uses all 31 `load` frames followed by all 16 `throw`
frames. `break` exposes all 31 destruction frames. Hit feedback is a CSS-only
flash and small shake over the current frame; no source `hit`/`damage` animation
is claimed. The break animation state and sequence are available to the
presentation layer, but a retained death overlay is not wired yet because the
game removes defeated tokens from the board immediately. The frame durations
are provisional presentation constants because the source FPS is unverified.

Direction indices are centralized as: 0 down/front, 1 down-right, 2 right,
3 up-right, 4 up/back, 5 up-left, 6 left, and 7 down-left. Main sprite pixels
are displayed unchanged: no tint, blend, hue rotation, recolor, or generated
replacement is applied. The optional source shadow and projectile layers are not
used in this implementation.

Asset attribution required by the pack license:

> Bleed - http://remusprites.carbonmade.com/
