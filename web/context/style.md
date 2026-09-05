# style.css

## `.mapboxgl-ctrl-group, .mapboxgl-ctrl` pointer-events fix

mapbox-gl-draw's controls use "mapboxgl-ctrl*" class names (built for Mapbox GL JS), but
MapLibre's own CSS only re-enables click-through (pointer-events) on its own "maplibregl-ctrl*"
names, leaving the draw control's ancestor container at pointer-events:none — so without this,
clicks silently fall through to the map canvas underneath and the draw buttons do nothing.

## `.maplibregl-ctrl-top-right .mapboxgl-ctrl` float/clear/margin fix

Same root cause as the pointer-events fix above, different symptom: MapLibre's own top-right
control container only applies its stacking layout (`float: right; clear: both; margin: 10px 10px
0 0`) to elements carrying its own "maplibregl-ctrl" class. The draw control's outer wrapper is
"mapboxgl-ctrl" (Mapbox-prefixed), so without this rule it isn't floated or cleared at all — it
renders as a plain in-flow block that ignores the floated siblings around it, landing at the top
of the container and visually overlapping the zoom control instead of stacking below it. Adding
the same float/clear/margin declarations under the matching Mapbox-prefixed class name makes the
draw control participate in the same top-right stack as every other control, in DOM order (zoom
control, then draw, then the "Upload Data Layer" control added after it in `manual.js`).
