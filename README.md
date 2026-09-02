# Setup / deploy to a new machine

Only two things are machine-specific and can't be reconstructed: the code, and `.env`
(`MAPBOX_ACCESS_TOKEN=...`). Everything else — `.venv/`, the pretrained `*.pt` checkpoints,
and the generated data dirs (`tiles/`, `embeddings/`, `models/`, `classes/`, `.scratch/`) —
is either reinstalled by `install.sh` or created lazily by the app on first use. None of it
needs to be copied.

```
./install.sh                                          # creates .venv, installs requirements.txt
set -a && source .env && set +a
.venv/bin/uvicorn api:app --app-dir scripts            # serves the UI + API on :8000
```

`ultralytics` auto-downloads its base checkpoint (`yolo11n-seg.pt`) into `weights/` the first time
`train.py` runs a fresh class — no manual step needed.

# Directory layout

## Global (shared across every class)

- `tiles/images/` — raw fetched Mapbox tiles, cached by `{z}_{x}_{y}.<ext>`. Purely a function of
  tile coordinates, not of which class searched them.
- `tiles/manifest.jsonl` — geo bounds for every cached tile.
- `embeddings/index.npy`, `embeddings/index_ids.json` — DINOv2 CLS-vector cache, parallel to the
  tile cache above (same reuse-across-classes reasoning).
- `weights/` — pretrained/base checkpoints everything fine-tunes from or benchmarks against
  (`yolo11n-obb.pt`, `yolo11n-seg.pt`, `yolo11x.pt`, `DIOR_yolov8s_backbone.pt`, ...). The single
  place all of these live — every script resolves its base-model path here rather than a bare
  filename, so nothing re-downloads a stray duplicate into whatever directory it happened to be
  run from (this used to happen: a duplicate `yolo11n-obb.pt` accumulated under `scripts/`).
  Distinct from `models/` below, which holds *this project's own* trained output, not bases.
- `models/<class>_<version>.pt` — a trained model. `models/<class>_<version>_metrics.json` next to
  it holds the training config + final metrics. `models/<class>_<version>_run/` holds the full
  Ultralytics training run for that version (loss/PR curves, confusion matrix, `weights/last.pt`,
  `args.yaml`) — everything you'd want to inspect training itself, separate from the two files
  above that are the actual product of that run.

## Per-class (`classes/<name>/`)

- `registry.jsonl` — every tile this class has ever seen, one status each: `seed`, `pending_review`,
  `confirmed`, `rejected`, or `below_threshold`. Source of truth for what round a tile belongs to.
- `labels.jsonl` — `tile_id -> auto-guessed label polygon` (DINOv2 patch-similarity, see
  `scripts/auto_labeler.py`), for every accepted candidate regardless of review outcome.
- `review/round_NNN/` — every candidate accepted in that round, as real files:
  - `<tile_id>.<ext>` — the raw tile, symlinked from the shared cache (no duplicate bytes).
  - `<tile_id>.txt` — the auto-guessed label in YOLO-seg format, if one was found.
  - `<tile_id>_labeled.jpg` — the same tile with that polygon burned onto the pixels, for quick
    visual review. Never used for training — only `dataset/` is.
- `dataset/` — the actual training set, populated only once a candidate is confirmed via
  `/api/reconcile` (or the CLI's `reconcile_review.py`):
  - `images/{train,val}/`, `labels/{train,val}/` — Ultralytics-standard layout.
  - `data.yaml` — points Ultralytics at the above.

## App

- `scripts/` — backend: `api.py` (FastAPI + background jobs), `search.py` (ring-search core),
  `common.py` (shared paths/tile-math/IO), `auto_labeler.py`, `embedder.py`, `train.py`,
  `reconcile.py`, plus thin CLI wrappers (`find_candidates.py`, `reconcile_review.py`).
- `web/` — the map UI (`index.html`, `app.js`, `style.css`), served by `api.py`.
