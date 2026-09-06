# web/ -- /manual + main app frontend

One file for this frontend instead of one per source file. See `scripts/context/scripts.md`'s Hard
negatives section for the backend/storage side (current design: a hard negative is a freely drawn
polygon, cropped at generation time either as a fixed 80m/z18 window centered on its centroid, or as
its own drawn bbox, depending on the class).

## style.css

**`.mapboxgl-ctrl-group, .mapboxgl-ctrl` pointer-events fix**: mapbox-gl-draw's controls use
`mapboxgl-ctrl*` class names (built for Mapbox GL JS), but MapLibre's own CSS only re-enables
click-through (`pointer-events`) on its own `maplibregl-ctrl*` names, leaving the draw control's
ancestor container at `pointer-events:none` -- without this, clicks silently fall through to the map
canvas and the draw buttons do nothing.

**`.maplibregl-ctrl-top-right .mapboxgl-ctrl` float/clear/margin fix**: same root cause, different
symptom -- MapLibre's top-right container only applies its stacking layout (`float`/`clear`/
`margin`) to elements carrying its own `maplibregl-ctrl` class, so the Mapbox-prefixed draw control
renders as a plain in-flow block overlapping the zoom control instead of stacking below it. Adding
the same float/clear/margin under the Mapbox-prefixed class makes it join the same top-right stack
in DOM order (zoom control, then draw, then the Upload Data Layer control from `manual.js`).

## manual.js -- Hard Negatives tab

**Free-draw reuses the same `draw` instance as Samples**: "Add Hard Negative"
(`startAddingHardNegative`) calls `draw.changeMode("draw_polygon")` -- the identical MapboxDraw
instance and mode Samples already use, not a separate tool. `draw.create` branches on
`addingHardNegative` to route the finished shape to `handleNewHardNegativeShape` (POSTs
`{class_name, polygon}`) instead of `handleNewShape` (creates a sample). An earlier version snapped
to a fixed-zoom tile around a clicked point (first z17, then z18) with a live preview box -- replaced
because even at z18 (~96m) a single marked tile on a dense refinery site routinely contained several
distinct objects, so a mark could never isolate "this one tank" from its neighbors.

**Draw/trash controls hidden outside of drawing contexts**: `drawControlsEl` (captured via
`document.querySelector(".mapboxgl-ctrl-group")` right after `map.addControl(draw)` -- specific to
mapbox-gl-draw's own container, distinct from MapLibre's `maplibregl-ctrl-group` used by the nav
control and Upload Data Layer button, see the pointer-events fix above) is shown only on the Samples
tab, or on the Hard Negatives tab while `addingHardNegative` is true
(`updateDrawControlsVisibility()`, called from `switchTab` and the add/stop handlers) -- otherwise
the draw/trash buttons sat visible over the map on every tab where clicking them did nothing useful.

**Tile footprints, not points**: assigned hard negatives render as their own drawn polygon, not a
center-point dot. Rows from before free-draw existed (tile-grid marks) come back from the API with
a synthesized rectangle as their `polygon` (see `_hard_negative_rows` in `scripts/obb.py`), so old
and new marks render identically with no frontend special-casing. Clicking a list row
`map.fitBounds`s the polygon's bbox (`polygonBbox`, shared with Samples) instead of flying to a
fixed zoom on a point, so the view frames the exact marked extent.
