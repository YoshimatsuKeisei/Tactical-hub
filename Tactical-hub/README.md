# Tactical Hub

## Local Catapult asset setup

Engineers (`unit.type === "engineer"`) are rendered with the separately
purchased **Catapult - Isometric** sprite pack after local asset preparation.
King and strategist sprites are configured separately below; engineers remain
the only units using the Catapult pack.
The raw pack and prepared PNG files are local-only and must never be committed
or pushed.

1. Place the purchased `Catapult - Isometric` directory in the repository root
   containing `package.json`. Its animation root must be
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
4. Run `npm run dev`. The existing engineer `工` text token is then replaced by
   the prepared Catapult sprite.

Both the raw directory and prepared output are covered by repository `.gitignore`
rules. CI and Cloud builds do not require the purchased pack. Their explicit
`工` text fallback and `Catapult asset unavailable` console warning are only the
missing-local-asset path, not the intended final UI.

The idle pose is not a source animation. It uses frame `0000` from each direction
of `move`. Attack playback uses all 31 `load` frames followed by all 16 `throw`
frames. `break` exposes all 31 destruction frames. Hit feedback is a CSS-only
flash and small shake over the current frame; no source `hit`/`damage` animation
is claimed. When an engineer is defeated, a presentation-only overlay retains
its old board coordinate long enough to play `break`; game state removal is not
delayed. The frame durations are provisional presentation constants because the
source FPS is unverified.

Direction indices are centralized as: 0 down/front, 1 down-right, 2 right,
3 up-right, 4 up/back, 5 up-left, 6 left, and 7 down-left. Main sprite pixels
are displayed unchanged: no tint, blend, hue rotation, recolor, or generated
replacement is applied. The optional source shadow and projectile layers are not
used in this implementation.

Asset attribution required by the pack license:

> Bleed - http://remusprites.carbonmade.com/

## Local HD Enemy asset setup

The king (`unit.type === "king"`) uses `6Crusader`, and all strategist roles
(`builder`, `encourage`, and `teleporter`) use `10Caster` from the separately
purchased **2D HD Enemy pack 1**. Engineers continue to use Catapult. Infantry,
archer, and ninja sprites are configured separately below; cavalry and
apprentice ninja keep their current text rendering.

The source ZIP and generated files are local-only and must not be committed:

1. Put `2D HD Enemy pack 1.zip` in this repository directory, beside
   `package.json`.
2. From this directory, run:

   ```bash
   npm run assets:hd-enemy
   ```

   An explicit ZIP path and output path may optionally be passed as the first
   and second command arguments. The default command is sufficient for the
   repository-root placement above.
3. The script reads exactly these purchased entries from
   `Spritesheets/With shadow/` and byte-copies no others:

   - `6Crusader/{Idle.png,Attack1.png,TakeDamage.png,Die.png}`
   - `10Caster/{Idle.png,Attack1.png,TakeDamage.png,Die.png}`

   Each sheet must be exactly `1920x1024`: 15 columns by 8 rows of `128x128`
   frames. Prepared files and `manifest.json` are written to
   `public/local-assets/hd-enemy/`.
4. Run `npm run dev`. King and strategist tokens now display their prepared
   sprites. If the local files are absent or invalid, the original `王` / `帥`
   labels are retained as explicit fallbacks and a console warning is emitted.

The renderer uses the original spritesheets directly with CSS background
positioning; it does not create extracted frame files. Asset row order is
E, NE, N, NW, W, SW, S, SE, mapped from the game's eight directions with rows
`[6, 7, 0, 1, 2, 3, 4, 5]`. Idle loops frames 0–14. Attack and hit play
`Attack1` and `TakeDamage` once, then return to idle. A defeated HD unit uses a
presentation-only overlay to play `TakeDamage` followed by `Die`; it does not
delay or restore the removed game unit. Ordered presentation queues preserve an
attack before damage/death when both occur in one battle resolution.

No faction recolor, tint, blend mode, hue rotation, SAM processing, or generated
replacement is applied to `6Crusader` or `10Caster`; their original colors are
shown unchanged. Frame timings are deliberately named provisional constants
because the purchased pack's source FPS has not been verified.

Development builds include a collapsible directional sprite preview for both
HD Enemy characters, all four states, and all eight directions. It is also used
for the HD Character pack described below. This preview is excluded from
production builds. Cloud and CI do not need the purchased ZIP and test the
prepare pipeline with a synthetic ZIP; final appearance and animation timing
must be checked on Windows with the purchased local asset.

## Local HD Character asset setup

The separately purchased **2D HD Character pack 1 V1.2** supplies these
presentation-only mappings:

- normal infantry → `1Knight`
- heavy infantry (`type === "infantry"` and `formation === "heavy"`) → `4Paladin`
- archer → `2Archer`
- ninja → `7DeathKnight`

Cavalry and apprentice ninja remain text tokens. King, strategist, and engineer
continue to use `6Crusader`, `10Caster`, and Catapult respectively. No unit
stats, merge rules, attack behavior, visibility, CPU, or RL logic is changed.

1. Put `2D HD Character pack 1 V1.2.zip` beside `package.json`.
2. Run:

   ```bash
   npm run assets:hd-character
   ```

3. The script validates and byte-copies exactly 16 files from
   `Spritesheets/With shadow/`:

   - `1Knight/{Idle.png,Melee.png,TakeDamage.png,Die.png}`
   - `2Archer/{Idle.png,Attack1.png,TakeDamage.png,Die.png}`
   - `4Paladin/{Idle.png,Melee.png,TakeDamage.png,Die.png}`
   - `7DeathKnight/{Idle.png,Melee.png,TakeDamage.png,Die.png}`

   Output and `manifest.json` are written to
   `public/local-assets/hd-character/`. The ZIP and generated output are both
   ignored by Git.

Every sheet must be `1920x1024`, containing 15×8 frames of `128x128` pixels.
The shared directional renderer uses rows `[6, 7, 0, 1, 2, 3, 4, 5]` for game
directions 0–7. Idle loops frames 0–14. Attack and TakeDamage play once and
return to Idle. Death uses TakeDamage followed by Die on the existing
presentation-only overlay. Heavy infantry changes from Knight to Paladin as
soon as the existing merge operation sets `formation === "heavy"`; the heavy
badge remains visible.

The original Spritesheets are displayed without frame extraction, recolor,
tint, hue rotation, blend modes, or SAM processing. Frame durations are shared
named provisional constants because source FPS is unverified. Missing local
assets fall back to the existing unit labels without crashing the app.

The intended Windows local sequence is:

```powershell
cd "C:\Users\jingc\Documents\Tactical hub\Tactical-hub"
npm install
npm run assets:catapult
npm run assets:hd-enemy
npm run assets:hd-character
npm run dev
```
