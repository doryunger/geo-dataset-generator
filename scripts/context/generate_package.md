# generate_package.py

Usage:

    python scripts/generate_package.py --class <class>

CLI equivalent of the `/manual` UI's "Generate Package" button with "Include latest available
entry" unchecked -- rebuilds both `dataset/` (segmentation) and `dataset_obb/` (OBB) purely from
the class's current local `classes/<class>/samples.jsonl` and crop images, no S3 merge, then
uploads a fresh timestamped package to S3 (same "publish what I've labeled" step `obb.py`'s own
CLI path already does) if configured. Added because no CLI path previously covered the
segmentation half -- `obb.py --class <class>` already did the OBB half alone, but the two datasets
had to be regenerated together (through `/manual`, or by calling `reconcile.generate_package` and
`obb.generate_obb_package` separately) to keep both in sync with the same sample set.

Added 2026-09-05 after a report that a package built with the merge checkbox off seemed to contain
only newly-added samples, not ones already in the class's folder. The actual cause turned out to
be a pre-existing silent-skip bug, not anything specific to this script: `reconcile.generate_package`
and `obb.generate_obb_package` both skip a `samples.jsonl` row entirely, with no log output at all,
whenever its crop image file isn't found under `classes/<class>/samples/` -- e.g. from a sample
whose metadata is present locally but whose image never got copied over (a partial sync, or a
`merge_latest_package` run that itself failed partway through copying crops). Both now log a
`logger.warning` for each skipped sample so a shortfall like that is visible in the run's output
(and in `logs/app.log`) instead of just showing up as a smaller-than-expected train/val count.
