# Tactical Hub

## DEV Tile Asset Preview / Import

Imported PNG terrain uses a persistent Pixi display tree. Each Canvas creates its
ground/depth layers and per-cell Graphics/Sprites once. State, asset, adjustment,
and ResizeObserver updates mutate texture, position, dimensions, and visibility;
they do not clear the stage or recreate every Sprite. Texture synchronization is
skipped while enabled state and decoded image identities are unchanged. Replacing,
resetting, or disabling an asset releases only the affected Canvas-owned texture
and switches the existing per-cell Sprite/Graphics pair in place. This avoids the
repeated GPU upload and empty display-tree interval that could produce PNG-only
flashing. The exact browser/GPU cause still requires Windows visual confirmation.

Run `npm run dev`, enter the game, and open **Tile Asset Preview / Import (DEV)**
at the top left. Select separate transparent PNG files for **Road top / 道路上面**
and optional **Road south face / 道路南側面** and **Road south shadow /
道路南側の落ち影**, plus base, lake, obstacle, and the
six bridge direction/segment slots. Each slot reports its
filename, decoded dimensions, loading state, or failure reason. Invalid formats,
signatures, and undecodable PNGs are rejected; the previous good asset is retained.

Choose **インポート素材＋未設定は仮図形** to enable images in both the independent
Flat/Raised preview and the real board's Pixi renderer. Loading a file alone
does not enable images. The real board's **Terrain renderer (DEV)** remains CSS
by default; choose PixiJS to inspect it. **仮図形（Graphics）** switches both Pixi
views back to test shapes. CSS never uses imported images.

Images are temporary browser-only data: no upload, asset-directory copy, or
persistence. Refreshing or leaving the game clears them; closing the import
panel or one preview keeps shared settings. **Reset image** removes one image;
**Reset All** clears all images/settings and restores shapes mode. **Reset
adjustment** keeps the image and restores scale 1, X/Y 0. Scale is 0.1–4; image
width defaults to one cell with aspect ratio preserved. X/Y offsets are pixels
at a 48px reference cell, proportionally scaled on the actual board. Adjustments
change visuals only, not tile positions/hitboxes. Replacing an image initializes
its placement settings.

The two renderers share decoded CPU images through a DEV React provider outside
GameState. Each Canvas owns and frees its own GPU ImageSource/Texture objects.
Object URLs are revoked after decode, failure, or cancellation; replacement/reset
cancels outdated requests. Unset or unavailable textures use Graphics. Neither
renderer uses the former Tilemap/atlas sampling path for imported images.
If GPU drawing raises an error, imported-image mode is disabled with a panel
message; the real-board view also restores CSS. The decoded images remain available
for reset/retry.

`road` is also used by baseGate/reorganize. Road, base, baseGate, and reorganize
share one raised elevation even though base keeps its gold/image top. Their south
face is hidden when the south neighbor is any of those same-height terrains, and
shown at the bottom edge or when the south neighbor is lake/outside. The separate
`road_south_face` image is reused for every such exposed raised edge; when unset,
the existing dark Graphics face is used. It is placed only below the top tile,
after terrain tops and before bridge/obstacle overlays. The optional south shadow
uses exactly the same visibility result and starts at the bottom of the face that
is actually displayed. It is drawn above low terrain and behind the unchanged
road/face/bridge/obstacle layers. With no shadow PNG, three unblurred Graphics bands
fade toward the south; this avoids one BlurFilter per tile. The shadow settings work
for both PNG and fallback: enable, opacity (default 0.24), scaleX, scaleY, and X/Y offset are
shared by both Pixi views. Shadow sizing uses independent `scaleX` and `scaleY`
controls; `scaleY` changes only its downward extent and leaves its horizontal span
unchanged, which also provides a separate vertical value for future animation work.
Turning it off restores the pre-shadow presentation without
moving the road top or face. **Reset image** returns only the shadow to Graphics;
**Reset shadow settings** restores its default controls, and **Reset All** restores
all slots and shadow/highlight settings. Lakes remain beneath transparent
bridge overlays. Bridge direction/start/middle/end and active obstacles follow
the existing presentation adapter. South-face sprites are also hidden in Flat
preview mode. A sidewall baked into a road PNG
cannot be removed automatically: use separate top/face images when needed.
HTML characters still render above the complete Pixi scene, so exact character
occlusion is unsupported. Large/offset art may overlap adjacent cells or clip at
Canvas edges; logical cell interactions are unchanged.
The real-board Canvas keeps its existing board-sized bounds, so a shadow adjusted
beyond the final Canvas row is clipped. The existing opaque HTML `outside` tile can
also cover Pixi overhang in the hybrid renderer; neither constraint changes hitboxes
or the shadow visibility decision and should be checked with the target map locally.

Manual checks: import recognizable PNGs and enable images; compare both views;
adjust/reload/reset every slot, test bridge transparency and segment selection,
toggle the south shadow, compare its PNG and Graphics versions, adjust its opacity
and placement without moving the road face,
scroll/resize/click overhangs, toggle CSS/Pixi and shapes/images, close/reopen either
view while loading, and verify no renewed flicker. GPU rendering/resource behavior
and browser visuals remain manual checks. No production images are supplied.

The same DEV panel also controls movement overlays for the real Pixi board.
Normal units use a darkened version of their team color; ninja use a team-independent
black/gray value. Each style has an enable switch plus separate top and south-face
opacity, with `darken` for normal units and `gray` for ninja. The overlay appears
only for legal destinations while a unit is selected in `movement_input`, and it
is removed when selection or phase changes. A south-face overlay is drawn only
when that terrain cell already has an exposed south face. These are presentation
settings outside `GameState`; movement legality still comes from the existing
movement engine. The selected token receives a small ring/glow in the same resolved
color. **Reset normal highlight**, **Reset ninja highlight**, and **Reset All**
restore their respective defaults.

Legal destination tops also receive a one-CSS-pixel outline so adjacent highlighted
cells remain individually readable while the normal terrain grid stays hidden.
The outline color is calculated from the resolved fill color by blending toward
white; it is therefore a lighter team color for normal units and a lighter gray
for ninja. `apprentice_ninja` continues to use the normal team-color path. Normal
and ninja settings independently expose outline enable, opacity (default `0.6`),
and lighten factor (default `0.35`). Each cell owns its top and left edges; only
cells on the east/south exterior own their right/bottom edges. This leaves every
internal boundary visible without painting a shared edge twice. The outline is
top-only: it does not extend onto the south face or south shadow.

## DEV real-board PixiJS terrain (hybrid experiment)

On the game board, **Terrain renderer (DEV)** starts at **CSS (default)**.
Choose **PixiJS (experimental)** to render the current `GameState.map` and active
constructions behind the existing HTML buttons and unit sprites. Switch back to
CSS at any time; WebGL initialization errors also restore CSS. Production has no
selector and always uses CSS.

Gray road surfaces are shared by `road`, `baseGate`, and `reorganize`; bases are
gold, lakes blue, and `outside` remains outside. These road-like surfaces and bases
share the same raised elevation for south-face suppression. Bridges retain the lake below
them and use horizontal/vertical start/middle/end segments. Active obstacles are
separate overlays. Same-height road neighbors suppress internal south faces.
The real board keeps the Graphics baseline that the user reported resolved
flicker. Both views now use the common Graphics/Sprite scene. The direct cause
of the former Tilemap/atlas issue remains unconfirmed.

Canvas positioning uses measured HTML tile rectangles, including fractional
sizes. Pixi mode removes the visual grid gap and Graphics terrain outlines so
road/lake artwork meets without permanent white cell lines; CSS mode keeps its
existing layout. A ResizeObserver updates the Canvas when layout changes; canvas and
buttons share the same scrolling parent. Canvas has no pointer events. Existing
HTML unit/tile callbacks, keyboard focus, attack selection, and construction
preview remain the interaction layer. Pixi movement destinations use persistent
top/south-face Graphics overlays while their HTML buttons retain the original
click handlers and logical coordinates.

**Hybrid limitation:** every HTML character and highlight is above the complete
Pixi terrain scene. Bridges/obstacles cannot correctly occlude character parts,
even when the terrain's own row depth is correct. This is an experiment, not a
fully depth-sorted game renderer. Real-board terrain uses measured DOM rectangles;
batching/performance at scale remains
to be assessed before production adoption.

Manual checks: compare CSS/Pixi map locations and colors; build/reset both bridge
orientations and obstacles; verify movement, attack, construction hover/click and
highlights; scroll both axes and resize the window; repeatedly toggle modes.
Confirm sprites and death animations remain unchanged, and inspect the stated
occlusion limitation. Confirm production stays CSS. Browser alignment and GPU
color rendering require local manual verification before production artwork.

## DEV PixiJS square tile experiment

Run `npm run dev`, open the game screen, and select **Open PixiJS tile preview
(DEV)** at the bottom left. Close it with the same button, which stays outside
the scrollable panel. The panel is not shown in
production; its Pixi renderer is loaded only when opened and destroyed when closed.

The independent synthetic 7×6 map compares flat and raised square tiles. Gray
roads, gold bases, and blue lakes are disposable generated shapes, not production
art. Graphics supplies unset terrain; imported PNGs use Sprites. Row-sorted Containers hold road south
faces, horizontal/vertical bridge segments (gold start/end caps), and tall red
obstacles. Same-height road/base terrain to the south suppresses internal sidewalls.
Lakes stay under bridges; bridge and obstacle overlays do not alter terrain.

Manual checks: compare both panels; inspect road clusters and exposed south edges;
check both bridge directions and obstacle occlusion. Click road sidewall overhangs
and tall obstacle tops: selection follows the logical square under the pointer,
not the graphic's owner. The text reports zero-based `(x, y)` coordinates. Scroll
the panel on narrow screens. Repeatedly open/close it, including during initial
loading, to check WebGL/remount cleanup. The renderer uses demand rendering, not
a continuous animation loop. Performance at production map scale, browser visuals,
and interaction remain manual checks; no production assets or game rules change.

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
