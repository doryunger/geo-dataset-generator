# s3_sync.py

Backs up `classes/<class>/` (samples, crops, bend_review/error_review, dataset_obb) to S3 as
timestamped snapshots, not continuous per-write mirroring. Labeling/slicing work happens purely
locally; only the deliberate "package" step (`obb.py`'s CLI, and the `/manual` "Generate Package"
button) uploads a compressed snapshot of the whole class directory, tagged with the epoch it was
created. Training scripts pull the latest snapshot down before reading local data, so a training
run anywhere sees whatever was last explicitly packaged, not just whatever happens to be sitting
on that machine's disk.

`models/` and `tiles/` are untouched by this — trained weights stay local (retrainable), and the
Mapbox tile cache stays local (re-fetchable).

Every function no-ops (returns `None`/`False`) if `S3_BUCKET_NAME` isn't set, so local-only
development/testing works without AWS configured at all.

## latest_package_key

Keys are `<prefix><epoch>.tar.gz`. Sorted numerically on the epoch, not lexicographically — a
lexicographic sort would put `"999..."` ahead of `"1000..."`.

## download_latest_package

Extracts with tarfile's `"tar"` filter, not the stricter `"data"` default. `classes/<name>/
review|predictions/` legitimately symlink into the shared `tiles/images/` cache with absolute
targets (see `common.stage_review_candidate`), which `"data"` (meant for untrusted archives)
rejects outright. Safe to use the lighter `"tar"` filter here specifically because this archive is
self-produced by `upload_package` in this same file and never comes from anyone else — not a
general weakening for untrusted input.
