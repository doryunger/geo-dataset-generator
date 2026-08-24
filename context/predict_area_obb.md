# predict_area_obb.py

OBB counterpart to `predict_area.py` — separate output dir (`predictions_obb/`, not
`predictions/`) and separate detection logic (`result.obb`, not `result.masks`), matching the
rest of the OBB track's designated-files separation (`obb.py`, `train_obb.py`, `dataset_obb/`).

## conf/iou defaults

`conf=0.15`, `iou=0.4` came from testing on a known-fence tile: the model's default NMS
(`iou=0.7`) let through ~10 heavily-overlapping low-precision boxes in the same rough area
(pairwise IoU maxed out around 0.4, so the default threshold never merged them). Tightening `iou`
to 0.4 cut that down to 3 boxes without losing the ones that were actually well-positioned.

## Chunking

`CHUNK = 8`: `model.predict()` collates the whole `source` list into one batch regardless of
`batch=`, which OOMs an 8GB GPU at higher radii. Chunked by hand instead — same reasoning as
`predict_area.py`.
