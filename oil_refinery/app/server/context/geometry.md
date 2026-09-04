# geometry.py

Global-pixel-space distance math for detections, used by the classifier (see
`oil_refinery/semantic_graph.md`'s "Pipeline: model router, fuser, classifier"). Distance math
stays pixel-only on purpose: at refinery-site scale a single reference latitude's
meters-per-pixel is accurate enough (same locally-constant-scale assumption `common.py` already
makes in `resample_to_target_gsd`/`bbox_crop_px`), so no detection point is converted to lon/lat
just to measure between two of them. Would need full lon/lat + haversine instead for points far
enough apart that Mercator's latitude-dependent scale distortion starts to matter — out of scope
here.

`global_pixel_to_lonlat()` is the one exception, and it's an output-shaping step, not part of the
distance math above: once the classifier has decided a cluster's boundary in pixel space (see
`classifier.polygon_for()`), that boundary has to become real lon/lat coordinates before it can go
out as GeoJSON to the frontend — pixel coordinates mean nothing to a map. It's the exact inverse of
`common.lonlat_to_tile_float`, via `common.tile_to_lonlat`'s own continuous (non-floored) math,
just taking global pixel coordinates instead of a lon/lat in the first place.
