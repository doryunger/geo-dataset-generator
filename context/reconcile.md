# reconcile.py

Core reconcile logic, shared by the CLI (`reconcile_review.py`) and the web API (`api.py`). Given
kept tile_ids for a round, marks the rest of that round's `pending_review` candidates rejected,
and copies confirmed tile images (+ any guessed label) into `dataset/images|labels/{train,val}/`.
Also owns browsing/deleting what's already in a class's dataset (Manage Examples tab).

## Split logic

`split_for`: deterministic geographic split, bucketing by tile x-column (1 in `VAL_FRACTION`
columns go to val) so adjacent tiles — which can be near-duplicates — land on the same side of the
split, avoiding leakage. `seed_*` tile_ids always go to train (matches `search.py`'s
`_save_seed_to_dataset`).

`_dataset_ext`: Ultralytics' dataset scanner only recognizes standard image extensions. Mapbox's
raw cache filenames (`.jpg90`, `.png32`) read fine via PIL's content-sniffing (used everywhere
else) but aren't recognized by YOLO's loader, so anything copied into `dataset/` needs a normal
extension regardless of what the tile cache named it.

## generate_package

Rebuilds `dataset/images|labels/{train,val}` from scratch out of `samples.jsonl` — deterministic
and safe to re-run any time samples are added/edited/removed. Split here is round-robin by sample
order (every `VAL_FRACTION`-th -> val), not `split_for`'s tile-x-parity, since manual sample ids
aren't grid-aligned.

Always does a full wipe-and-rebuild, so a deleted sample is already correctly excluded with no
special handling. `changes_since_last_generation` in the return value is purely informational
(why this run's output might differ from the last one) — added after `dataset/`/`dataset_obb/`
were found silently holding copies of samples deleted from the UI, only noticed by manually
diffing folder contents against `samples.jsonl`.

## delete_round

Discards every not-yet-reviewed candidate from a round (`pending_review` -> rejected, its
`review/` files removed) — confirmed examples are left untouched, since a blanket "delete round"
shouldn't silently remove already-committed dataset entries. If nothing confirmed is left
afterwards, the round is purged from the registry entirely instead of lingering in the Manage tab
as a dead entry.
