# Working conventions for this repo

For what the code does and where things live, see `README.md` — this file is about *how to work
on it*, distilled from actual sessions rather than aspirational.

## The one thing that matters most right now: dataset size

Every class here (fence especially) is trained on a tiny number of hand-labeled samples (fence
was at 20, then 29, then 43 pieces as of 2026-08-16 — see `classes/<class>/samples.jsonl` for the
current count). At this scale:

- Validation metrics (precision/recall/mAP50/mAP50-95) are **noisy, not signal**. A val split of
  4-9 images means one flipped detection swings a metric by 10-25 percentage points. Two training
  runs on the same data can land far apart purely from random init/augmentation — this has
  actually happened here (`fence_obb_v2` vs `fence_obb_v3`, same 20/29 samples, precision 0.32 vs
  0.045).
- Don't declare a technique "worked" or "failed" off one run's metrics at this sample count.
  Look for a consistent direction across 2+ runs, or just say plainly that it's inconclusive.
- The highest-leverage thing to improve detection quality is still **more real labeled samples**,
  not architecture choice, hyperparameters, or loss-function tweaks. Steer conversations there
  before spending a training run chasing a metric that's mostly noise.
- When reporting results, always show the current sample/image count next to the metrics table so
  it's obvious how much to trust the numbers.

## OBB training workflow (`scripts/obb.py`, `scripts/train_obb.py`)

1. New/edited samples go into `classes/<class>/samples.jsonl` via the `/manual` UI (`scripts/api.py`).
   Every create/edit/promote automatically renders a polygon-overlay image into
   `classes/<class>/bend_review/<sample_id>.jpg`.
2. **Before regenerating `dataset_obb/`**, look at any *new* files in `bend_review/` and decide by
   eye whether the ribbon is a genuine corner (needs splitting) or a straight/gently-curved line
   (doesn't). Two automatic heuristics for this were already tried and both failed (see
   `BEND_PIECES`'s docstring in `obb.py`) — this has to stay a manual/by-eye judgment call.
   Bent ones get an entry in `BEND_PIECES = {sample_id: n_pieces}` at the top of `obb.py`.
   While reviewing, also check whether any part of the ribbon runs under something that visually
   hides it (tree canopy is the case seen so far) — **trim the polygon to end where the fence
   stops being actually visible, don't label the covered stretch just because you know a fence is
   probably still there.** This isn't an image-processing problem more data can fix: the model
   can only learn from pixels it can see, and forcing it to guess at invisible content trains it
   to fire on plausible-looking textures in general (confirmed via `error_analysis_obb.py` on
   `5d3cee45db36`/`ae9e2e66bf34`/`8c2d44de40d9` — predictions under canopy came back scattered and
   misaligned, not just low-confidence). Trimming shrinks that sample's crop/bbox to match (see
   `update_manual_sample` in `api.py`) rather than discarding the sample outright — the visible
   remainder is still valid training data.
3. Run `python scripts/obb.py --class <class>` to rebuild `dataset_obb/`. Multi-piece samples are
   written as separate cropped images (one per piece), not one image with several boxes — this is
   real extra training-image count from data you already have, so check the printed train/val
   counts look right (should be ≥ the sample count, not equal to it, if any `BEND_PIECES` entries
   exist).
4. Train with `python scripts/train_obb.py --class <class> --version vN` — pick the next `vN` by
   checking `models/<class>_obb_v*.pt`. This machine has no GPU (`torch.cuda.is_available()` is
   `False`) despite earlier notes here assuming otherwise — a 100-epoch run is CPU-bound and took
   ~90 min at the 448-piece dataset size (2026-08-16), not the "~1-3 min" once assumed. Still worth
   just running rather than predicting the result, just budget real wall-clock time for it, and
   prefer `patience` (early stopping, on by default now) over guessing an epoch count.
5. Compare `models/<class>_obb_vN_metrics.json` against prior versions in a table, with the sample
   size caveat above front and center.
6. **Optional**: `python scripts/train_obb_kfold.py --class <class> --version vN --folds 5` trains
   N models, each with a different 1/N of *original samples* (not pieces) held out as val, and
   reports mean±std across folds instead of one run's number — a single split's number can be
   meaningfully optimistic (`fence_obb_v6`: single-run mAP50-95 0.455, true k-fold mean 0.33±0.08,
   confirmed 2026-08-16). Off by default (a 5-fold run is ~5x the cost of one training run,
   multi-hour on this CPU-only machine) — reach for it when a result needs to be trustworthy
   enough to act on, not for every routine iteration. Folds split by original sample id
   specifically because `dataset_obb/` pieces of the same sample aren't independent — see
   `generate_obb_package`'s `val_ids` param and the module docstring on why the *pieces* count
   (hundreds) isn't the number that matters for generalization confidence, the *original sample*
   count is (currently 29, ~27 of which are geographically distinct locations).

Hard negatives (`--hard-negatives` flag / `HARD_NEGATIVE_TILES` in `obb.py`) are off by default —
tried once at 13 positives + 6 negatives and it destabilized training (cls_loss spiked, real
confidence collapsed). Revisit only once positives comfortably outnumber any negatives added.

## S3 backup (`scripts/s3_sync.py`)

`classes/<class>/` (samples.jsonl, crops, bend_review/error_review, dataset_obb — the hand-labeled
ground truth and everything derived from it) backs up to S3 as timestamped snapshots, not
continuous per-write sync. `tiles/`, `embeddings/`, and `models/` stay local-only (all
reconstructible: tiles re-fetch from Mapbox, embeddings rebuild from samples, models retrain).

- The `/manual` editor (`api.py`) never touches S3 — every edit stays purely local while you're
  actively labeling.
- `python scripts/obb.py --class <class>` uploads a fresh timestamped package (`packages/<class>/
  <epoch>.tar.gz`) as its last step, once it's rebuilt `dataset_obb/` — this is the one deliberate
  "publish what I've labeled" action.
- `python scripts/train_obb.py` and `train_obb_kfold.py` pull the latest S3 package (if S3 is
  configured and one exists) before training, overwriting local `classes/<class>/` to match —
  so a training run always uses whatever was last explicitly packaged, not just whatever happens
  to be on that machine's disk. Both no-op back to local-only if `S3_BUCKET_NAME` isn't set.
- The pull/push only happens at these CLI entry points, not inside `generate_obb_package`/
  `train_obb_class` themselves — `train_obb_kfold.py` calls both of those once per fold against a
  fold-specific local split, and pulling/pushing mid-fold would defeat the fold split entirely.
- Extraction uses tarfile's `"tar"` filter, not the stricter `"data"` default — `review/` and
  `predictions/` legitimately contain absolute symlinks into the shared `tiles/images/` cache
  (see `stage_review_candidate`), which `"data"` rejects. Safe here specifically because this
  archive is self-produced by `upload_package` and never comes from an untrusted source.
- Needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME` in `.env`
  (same gitignored-file pattern as `MAPBOX_ACCESS_TOKEN`) — the IAM user needs `s3:GetObject`,
  `s3:PutObject`, `s3:DeleteObject` on `arn:aws:s3:::<bucket>/*` and `s3:ListBucket` on
  `arn:aws:s3:::<bucket>` itself (a separate ARN, easy to miss).

## Git commits

Dor Yunger (git user.name/user.email, configured locally) is the sole author of every commit in
this repo. Do not add a `Co-Authored-By: Claude` trailer (or any other AI-attribution line) to
commit messages — this overrides the default Claude Code commit template.

## General working style observed this session

- Prefer actually running the experiment over predicting its outcome — training here is cheap
  (GPU, small dataset), so "let's check" beats "I'd expect."
- When a metric comparison could be misleading due to sample size, say so up front rather than
  presenting the table and letting it speak for itself.
- Visually verify pipeline changes (burn labels onto images, eyeball crops) before trusting them,
  especially anything touching polygon/coordinate math — caught real issues this way already.
