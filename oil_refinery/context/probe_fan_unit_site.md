# probe_fan_unit_site.py

One-off: scan an arbitrary site (never used as a fan-unit training sample) with a trained fan-unit
OBB model and report any detections — used to check whether the model generalizes beyond its own
74 training samples, at a site with no ground truth of its own.

Fetches at `DETECT_ZOOM=17`, same as `oil_refinery/app/server/tile_server.py`, then GSD-resamples
each tile to `common.TARGET_GSD_M` before inference — same as that app and same as every training
crop goes through in `obb.py`. Zoom itself doesn't need to match between fetch and training; the
resample is what makes the model see a consistent real-world meters-per-pixel scale regardless of
what zoom a tile was fetched at (fetching at z17 instead of matching the samples' own z19-21
keeps the scan grid small — z17 tiles cover ~4x the ground per tile). Skips
`predict_area_obb.py`'s full-mosaic step (a 1.5km-wide grid would be a lot of pixels to hold in
memory) — only detections get saved as overlay images.

`batch_imgsz` is computed the same way as `tile_server.py`'s `_run_detection_batch`: rounded up to
the batch's largest resampled dimension, so ultralytics never silently letterboxes back down to
its default 640 and undoes the whole point of the GSD resample.

Usage:

```
python oil_refinery/probe_fan_unit_site.py --slug berendrecht_antwerp \
    --lat 51.309203 --lon 4.300917 --model models/fan-unit_obb_v2.pt
```
