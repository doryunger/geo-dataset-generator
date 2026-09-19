# scripts/loop -- site-by-site active-learning loop

The method for growing a new detection class without hand-drawing every sample: pick a site from
an OSM layer, run the current model, have a person judge what it found (triage) or mark everything
that is really there (sweep), turn the verdicts into samples, retrain, measure, repeat. Built out
on `distillation-column` in the `experiments` workspace on 2026-09-14/15 and moved here from a
session scratchpad on 2026-09-19 so it can be run without Claude in the loop. Every command takes
`--class` and honours `WORKSPACE` (see `scripts/context/scripts.md`), and all state lives under
`<workspace>/loop/<class>/` -- `sites.json`, `scans/<site>/w*.jpg`, `candidates/`, `sweeps/`,
`pages/`.

## One round

1. `sites.py --geojson <osm export>` once per class; `sites.py --list` to pick a site. Prefer
   unsampled sites with `sharpness_vs_train` near 1 (see below); scan first, then `--score`.
2. `scan.py --site X --model vN --conf 0.25` -- 120 m windows over the OSM polygon (+60 m pad),
   z18 imagery resampled to the class GSD, detections not overlapping an existing label, deduped
   within 8 m. About a minute per 500 windows once tiles are cached.
3. `pages.py triage|sweep ...` -- writes a self-contained HTML page; publish it as an Artifact.
   Triage = one candidate per screen, yes/no. Sweep = the N densest windows (Laplacian variance),
   model proposals drawn in green at a *low* threshold, the reviewer polygons every real object.
4. Reviewer uses the page's **Download JSON** button; `apply.py --review <json>` turns yeses and
   polygons into samples (same fields `/manual` writes, plus an `origin` block) and noes into
   *disabled* hard negatives. Regenerate the package, train the next version.
5. `coverage.py --review <sweep json> --models vN,vN+1` -- the number the loop is judged on.

## Why it is shaped this way

**Coverage, not precision, is the metric.** Two triage rounds on unseen sites returned 0/105 and
3/110 confirmed columns and Claude recommended abandoning the class; the user pointed out that
triage only measures "of what the model found, how much was real" and says nothing about what it
missed. The sweep exists to establish ground truth on a fixed set of windows so that
found/total can be tracked round over round. First measured round on La Rábida (26 polygons + 6
triage-confirmed): v10 22% -> v13 75% at conf 0.25, false positives 139 -> 30, after 35 samples
were added. Note that once a site's polygons are in training, rescanning it measures learning,
not generalisation -- use a fresh site for the latter.

**Polygons, not clicks or two-point axes, on the sweep page.** The first sweep marked objects
with a single click (fan-unit misses), the second with two clicks along the axis (columns). The
user rejected both for columns: at near-nadir angles a column is a small top with no axis to
click, and the honest shape is whatever the reviewer would draw in `/manual`. Polygon marks
convert to samples with no fitting step. Older single-click / two-click marks are migrated to
small polygons on load so nothing is lost.

**Low proposal threshold on sweeps.** Reviewers found real columns among 0.25-0.27 proposals that
a 0.4 cutoff would have hidden; the human is the filter, so show more.

**Rejections become disabled hard negatives.** They are the best negatives available (the
model's own confusions, tight polygons) but at 42 positive images + 49 negative crops they
collapsed a model to 0.04-0.09 confidence on its own training data (`v11`, see
`scripts/context/scripts.md`). Store them, enable them later.

**Site gate: `sharpness_vs_train`.** Mapbox coverage varies a lot by region: Burgas and Naftan
scored ~0.4x the training sites' sharpness and every proposal there was a false positive, while
Płock is a near-nadir aerial orthophoto (1.3x sharp, but every column reduced to a circle). The
score is median Laplacian variance over ~60 windows, normalised so that a training site scanned
the same way scores 1.0 (the 1.53 factor bridges 640 px crops to 960 px windows). It is a
coarse gate, not a guarantee: La Rábida scored 0.92 and was still near-nadir.

**Slugs are ASCII.** Site names with accents produced a page that could not save to the
artifact store (document ids reject non-ASCII); `loop_common.slug` strips them.

**Same 120 m / 60 m / z18 geometry everywhere.** `site_windows` is the one place the grid is
computed, so a sweep's window ids, a scan's candidates and a later coverage run line up exactly
across model versions. Changing `WINDOW_M` or `PAD_M` invalidates comparability with earlier
sweeps of the same site.

## groups.py -- control over what trains (2026-09-19)

Added after the user set the requirement that the process must let them "discard samples or hard
negatives that are reducing the performance of the model." Every sample and negative written by
the loop carries an `origin` block; `obb.group_key` collapses it to `<source>:<site>:<model>`
(`hand` for the original `/manual` samples). `groups.py` lists those groups with enabled counts,
flips a whole group (or its first N rows, `--limit`) on or off, and `--versions` shows the
enabled count per group for each trained version -- read next to a `coverage.py` table across the
same versions, that is the ablation view.

Samples now honour an `enabled` flag exactly as hard negatives do; `generate_obb_package` drops
disabled samples before anything else happens, so a disabled sample gets no crop *and* is not
drawn as a neighbour label in other crops. That second part is deliberate: a sample is disabled
either because it is wrong (then labelling it anywhere is wrong too) or because its whole group
is under test (then it must be absent, not half-present). The side effect -- a real object that
was disabled for the second reason trains as background wherever it is visible -- is accepted as
the price of a clean ablation; re-enable the group afterwards.

`generate_obb_package` writes `dataset_obb/groups.json` and `train_obb.py` copies it into
`<class>_obb_vN_metrics.json` under `groups`, so every version records what it was trained on.
Versions before `v17` have no record.

First use: `v16` = 25 negatives from the sharp training sites (`triage-rejected:-:v9`, top
confidence); `v17` = those plus the top 25 of BP Rotterdam's 77 in-place rejections
(`loop-triage-rejected:bp_raffinaderij_rotterdam:v16`), 111 positive images to 48 negative crops.
