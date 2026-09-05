# Hard negatives: per-class storage, S3 sync, and /manual's Hard Negatives tab

Spans `common.py` (storage helpers), `obb.py` (`_hard_negative_tile_ids`, the training-time crop
loop), `s3_sync.py` (the `hard_negatives/<class>/` prefix), and `api.py`
(`/api/manual/hard_negatives*`) -- one feature, so one context file rather than four near-empty
ones.

## Per-class file replaces the hardcoded dict

`obb.py`'s `HARD_NEGATIVE_TILES` dict used to be the only place a hard-negative tile was recorded,
edited by hand and shared across every class in one block. It's been replaced by
`classes/<class>/hard_negatives.jsonl` (`common.load_hard_negatives`/`add_hard_negative`/
`remove_hard_negative`, same jsonl-of-rows shape as `samples.jsonl`/`registry.jsonl`), managed
through `/manual`'s Hard Negatives tab instead of a code edit. `fan-unit`'s and
`distillation-column`'s existing entries were migrated into their own files (2026-09-05); only
`fence-face`'s six entries stay in the dict, inert, since that class is discontinued and no longer
has a `classes/fence-face/` directory to hold a file. `obb.py`'s `_hard_negative_tile_ids(class_name)`
merges both sources (legacy dict entries for `class_name`, plus the per-class file) so nothing
already committed needed to change behavior.

## Why the training-time loop fetches instead of skipping

The crop-sampling loop in `_generate_pieces_for_class` used to look up each hard-negative tile's
image with `next(TILE_IMAGES_DIR.glob(...), None)` and silently skip it if not already cached
locally. That was fine when every entry was hand-added on whichever machine ran `obb.py`, but a
tile added on a different machine (see the S3 sync below) has no reason to already be cached here
-- it would otherwise silently contribute zero negative crops with no error or log line. The loop
now calls `common.fetch_tile` unconditionally, which fetches-and-caches on first use exactly like
every positive sample's crop already does.

## Why hard negatives get their own S3 prefix instead of riding the package snapshot

`classes/<class>/` (and everything in it, `hard_negatives.jsonl` included) already backs up to S3
as a timestamped tarball snapshot on "Generate Package" -- see the S3 backup section of the root
`CLAUDE.md`. That alone would work, but only at publish time, and merging two machines' *both-new*
hard-negative lists during that publish step would need the same kind of read-modify-write merge
`merge_latest_package` does for samples. Hard negatives instead get a second, independent prefix,
`hard_negatives/<class>/<tile_id>.json` -- one small object per tile rather than one combined list
file, so two machines adding *different* tiles at the same time never race on the same object; each
add/delete is an independent `PutObject`/`DeleteObject` (`s3_sync.upload_hard_negative` /
`delete_remote_hard_negative`), and `/api/manual/hard_negatives` calls `sync_hard_negatives` (pulls
remote, adds anything missing locally, additive only -- local never loses a tile) on every list
load rather than only at publish time. The point of this class of tile (something another labeler
spotted the model confusing with the real class) is exactly the kind of thing that should show up
for everyone immediately, not after someone remembers to package.

## `HARD_NEGATIVE_ZOOM = 17` in `api.py`

Fixed, not a per-request parameter -- matches the z17 grid `fetch_hard_negative_tile.py`,
`oil_refinery/probe_fan_unit_site.py`, and `tile_server.py` already fetch at for the
currently-active compact classes. `obb.py`'s `HARD_NEGATIVE_TILES` legacy dict keys are the only
place a different zoom (z19, `fence-face`) ever appeared, and that class is discontinued.
