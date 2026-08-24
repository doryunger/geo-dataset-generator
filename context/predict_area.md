# predict_area.py

The trained-model equivalent of `/manual`'s DINOv2 Validation tab: same hardcoded-position-via-
seed-yaml convention as `find_candidates.py` (see `seeds/seed_example.yaml`), but scores the
actual downstream detector instead of embedding similarity, so you can eyeball how well a freshly
trained model generalizes to a new area — ideally one none of your samples came from, for a
genuine held-out check.

## Design notes

- `run_name` is `<tile_id>_r<radius>`: the same area+radius always overwrites its own subfolder;
  a different one gets its own, instead of both mixing flat in one directory.
- Chunked by hand (`CHUNK = 8`): `model.predict()` given a whole list as `source` collates it into
  one batch regardless of the `batch=` kwarg (that only bounds the internal dataloader, not this
  path), which OOMs an 8GB GPU well before radius 8 (289 tiles). Looping keeps peak memory to one
  chunk's worth, independent of how many tiles the search radius covers.
- Every scanned tile gets a raw copy saved, hit or not — a 0-detection run should still leave
  something to look at, so "the model found nothing" is distinguishable from "there was nothing
  here to find" by just eyeballing the imagery.
- `--imgsz` default matches `train.py`'s default (1280) deliberately: a thin object like a fence
  can get squeezed down to a few px wide at 640 once letterboxed, well below what the network can
  represent, so inference resolution should match training resolution.
