# api.py

Local-only web API + static frontend for the map-based collection UI.

## Job

Minimal in-memory background job: a search, pack(train), or validate run, polled via
`/api/jobs/{job_id}`. `kind`: "search" | "pack" | "validate". `status`: running | done | aborted |
error.

## lifespan

Embedder loaded once at app startup, reused across every request (`_state["embedder"]`) — DINOv2
load is expensive enough to not want to repeat per-request.

## Request models

- `CollectRequest.zoom`: the operating zoom for this search — determines the tile grid, the seed's
  containing tile, the seed crop's resolution (via `fetch_and_crop_bbox`), and every ring-search
  candidate's fetch/render resolution. Independent of whatever zoom the shape was drawn at in the
  map UI (e.g. draw precisely at 20, search/save at 17 for a wider, less-blurry reference).
  `west/south/east/north`: drawn shape's bbox, for a precise reference-image crop instead of the
  whole grid tile containing (lat, lon) — optional so direct lat/lon-only collection still works.
  `max_fetches` defaults modest (300): a search that never finds a match can otherwise run for
  many minutes on this CPU-only machine before giving up (confirmed directly — 1000+ tiles, 6+
  minutes, still nothing).
- `PackRequest.epochs` defaults low (20) — no GPU on this machine, full training is slow.
- `ManualPromoteRequest.label_polygon`: passed straight from the validation result the candidate
  came from, rather than looked up server-side — `run_validation` deliberately never persists to
  `labels.jsonl` (it's read-only, repeatable any time with no side effects), so there's nothing to
  look up by tile_id alone.

## tile_image vs dataset_image

`tile_image`: serves from the shared tile cache — used while a search is live/under review, before
a candidate has necessarily been copied into the class's own dataset folder. Content-type is
decided from the file's actual suffix (Mapbox's real extensions like jpg70/80/90, png32/64/128/256
aren't standard MIME extensions).

`dataset_image`: serves from the class's actual `dataset/images/` — used for browsing in the
Manage tab, so it reflects what's really there (including seed crops, which never lived in the
shared tile cache to begin with).

## list_rounds

`pending_review` entries carry the same auto-guessed polygon computed during the search itself
(see `auto_labeler.py`), surfaced here so the Manage tab can draw the same overlay the live Search
tab does.

## _has_dataset / Pack Data

YOLO's trainer requires non-empty train AND val — a class with too few confirmed examples (e.g.
just one seed, which always lands in train/) can't be trained yet, so Pack Data skips it rather
than crashing on it.

## /manual endpoints (see web/manual.html/.js)

`update_manual_sample`: after an edit-mode change (vertices dragged/added/removed), bbox is
recomputed from the edited ring's own extent, the crop is regenerated against that new bbox, and
the label is re-normalized — keeps the crop and its label consistent with whatever shape now
exists, regardless of how much the edit changed it.

`manual_sample_image`: unlike the shared tile cache, a sample's crop can be regenerated in place
after an edit (same URL, new bytes) — `Cache-Control: no-store` so the browser never shows a stale
thumbnail post-edit.

`manual_promote`: turns a validation candidate into a real sample, using its already-computed
auto-guessed label instead of requiring a manual redraw — an explicit opt-in action, not
automatic, since the whole point of `/manual` is that examples are normally hand-drawn. A
candidate can have more than one labeled region; a sample is one polygon, so the largest region by
area is the one promoted. The source tile is marked `confirmed` in the registry purely for dedup,
so future searches/validation runs don't re-suggest it.

`generate_package`: rebuilds both `dataset/` (segmentation) and `dataset_obb/` (OBB) from current
samples, then uploads the whole class directory — both packages together — to S3 as one fresh
timestamped snapshot. Deliberately does not train anything (see `train.py`/`train_obb.py`/
`train_obb_kfold.py` for that) — this endpoint is "process and publish what I've labeled," and
training stays a separate, explicit script-only step.

## NoCacheStaticFiles

This UI is under active iteration — never let the browser cache `app.js`/`style.css`, since a
stale copy silently breaking against a newer `index.html` is hard to diagnose.
