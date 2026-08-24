# train_obb.py

Trains an OBB model on a class's `dataset_obb/`, deliberately separate from `train.py` (YOLO-seg
on `dataset/`) since OBB is a different task/label format built by `obb.py`, not a variant of the
seg pipeline.

Defaults to `yolo11n-obb.pt`, Ultralytics' own DOTAv1 (aerial imagery)-pretrained checkpoint —
unlike `yolo11n-seg.pt` (COCO-pretrained, confirmed via direct testing to have zero prior exposure
to nadir/aerial views for any object, not just fence), this base model has already seen this
general viewing angle, so fine-tuning only has to learn "what is a fence" rather than also "what
does an aerial photo even look like."

`imgsz` defaults to 640, not 1280: checked the actual `dataset_obb/images/train` once `obb.py`
started emitting real-world-length pieces (median 257px, 98.5% <= 640px on their longer side), and
1280 meant every image spent most of its area as YOLO letterbox padding around a small upscaled
crop. Bump this back up only if a future dataset's own pieces actually run bigger.

`patience=30`: Ultralytics itself defaults this to 100, which combined with `epochs=100` meant
early stopping could never actually trigger (it needs 100 epochs with no improvement, but the run
itself was only ever 100 epochs long).

Run naming (`{class}_obb_{version}_run`) is namespaced separately from `train.py`'s
`{class}_{version}(.pt|_run)` so the two task types never collide or get confused for one another
in `models/`.

S3 pull happens in `main()`, not inside `train_obb_class` — see context/train_obb_kfold.md.
