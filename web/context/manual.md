# manual.html / manual.js -- Hard Negatives tab

See `scripts/context/hard_negatives.md` for the backend/storage side. This covers the map
interaction added 2026-09-05.

## Client-side tile math instead of a server round trip

`lonLatToTileXY`/`tileXYToLonLat`/`tileBoundsLonLat` mirror `common.py`'s
`lonlat_to_tile`/`tile_to_lonlat`/`tile_bounds` (same Web Mercator XYZ formulas) so the
picking-preview box can redraw on every `mousemove` without a request per pixel of cursor movement.
`HARD_NEGATIVE_ZOOM = 17` here mirrors the same constant in `scripts/api.py` -- the box always
previews the actual z17 tile that will be captured regardless of the map's own current zoom, which
only changes how big that fixed real-world extent renders on screen.

`hardNegativePreviewTileKey` caches the last tile's `"x_y"` key so the preview source's `setData`
only runs when the cursor actually crosses into a different z17 tile, not on every mousemove event
within the same one.

## Tile footprints, not points

Assigned hard negatives render as their real z17 tile rectangle (`hard-negatives-fill`/
`-line` layers, same shape as the picking preview) rather than a point marker -- a dot at the tile
center didn't convey how much ground the tile actually covers. Clicking a list row calls
`map.fitBounds` on that tile's real bounds instead of `flyTo` a fixed zoom on its center point, so
the view frames the exact extent that will be cropped.
