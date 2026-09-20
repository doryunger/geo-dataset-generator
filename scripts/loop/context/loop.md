# scripts/loop -- site-by-site active-learning loop

The method for growing a new detection class without hand-drawing every sample: pick a site from
an OSM layer, run the current model, have a person judge what it found (triage) or mark everything
that is really there (sweep), turn the verdicts into samples, retrain, measure, repeat. Built out
on `distillation-column` in the `experiments` workspace on 2026-09-14/15 and moved here from a
session scratchpad on 2026-09-19 so it can be run without Claude in the loop. Every command takes
`--class` and honours `WORKSPACE` (see `scripts/context/scripts.md`), and all state lives under
`<workspace>/loop/<class>/` -- `sites.json`, `scans/<site>/w*.jpg`, `candidates/`, `sweeps/`,
`pages/`, `reviews/` (the reviewer's sweep/triage verdict JSONs, one per site and version).

## Runbook

The process as a whole -- roles, rules, the round table, the gate, adding a class from zero --
is written up for readers in `docs/training-a-new-class.md`; keep the two in step. This section
is the operator's command-level version.

This is the method for adding a new detection class, settled on 2026-09-14/19 after fan-unit
and distillation-column. The rationale and measurements behind each rule are in the sections
below this one; this section is the runbook. Everything below runs with `WORKSPACE=experiments` (see
`scripts/common.py`) and with `.env` sourced into the shell (`set -a && source .env && set +a`
-- `apply.py` fetches crops from Mapbox and dies without the token) so production `classes/`, `/manual` on port 8000 and S3 `packages/` are
never touched; `run-experiments.ps1` serves `/manual` against the experiment workspace on 8001.

**The idea.** Nobody draws a whole class from scratch. A person seeds a few dozen samples,
a weak model is trained, and from then on the model proposes and the person judges. Each round
uses one refinery site from the OSM layer and produces samples, hard negatives and a coverage
number. The class is finished when coverage on a *fresh* site stops improving.

**One round, in order:**

1. Pick a site: `python scripts/loop/sites.py --class <cls> --list` (on Windows prefix
   `PYTHONIOENCODING=utf-8` -- site names carry accents the cp1252 console cannot print and the
   listing dies mid-way otherwise). Prefer sites with
   `sampled=0` and `sharpness_vs_train` near 1.0; scan a few unscored candidates first, then
   `--score`. Sharpness is a coarse gate (Mapbox coverage is soft across much of south-east
   Europe and the model does not transfer to it); obliqueness has to be judged by eye from the
   scan windows -- tank sides visible means oblique, clean circles means nadir. **Selection is a
   queue, not a search** (rule set 2026-09-20 after a round was spent scanning for a site
   sharper than 0.85 that does not exist): score a batch of ~5 unsampled sites, work them
   top-down by score, and defer anything under the floor of **0.6** until every site above it is
   done. Don't scan for a better site than the head of the queue. Current queue: Wesseling 0.75,
   Heide 0.72, Normandie 0.72, Lingen 0.65, Gelsenkirchen Horst 0.64, Mitteldeutschland 0.61.
2. Scan it: `python scripts/loop/scan.py --class <cls> --site <name substring> --model vN --conf 0.25`.
   The substring must match exactly one site (`esso` hits Esso Belgium too, `Rotterdam` hits
   three -- `"Esso Raf"` works). Use a low threshold; the human is the filter and reviewers found
   real objects at 0.26.
3. Sweep: `python scripts/loop/pages.py sweep --class <cls> --site ... --model vN --windows 60`,
   publish the HTML under `<workspace>/loop/<cls>/pages/` as an Artifact **with
   `capabilities: {db: {}, downloads: true}`** -- the page saves every mark to the artifact's
   database as it goes, and without the declaration nothing is saved and the download button is
   dead. The reviewer polygons every real object that has **no** green proposal on it -- the
   misses -- and leaves the proposals alone. Collecting the result: the reviewer says "done"
   and the verdicts are read from the page's database (Artifact `read_db`, collection
   `reviews`, document `<cls>-sweep-<slug[:20]>-vN`; `list` the collection if the truncated
   slug is unclear) and saved as `reviews/<cls>-sweep-<site>-vN.json`. The stored document has
   the same shape as the download, so `apply.py`, `coverage.py` and `--swept-by` take it as is.
   Downloading the JSON by hand is the fallback when Claude is not in the loop.
4. Triage the proposals: `python scripts/loop/pages.py triage ... --model vN --min-conf 0.25 --swept-by <sweep json>`
   builds a yes/no page of the proposals inside the swept windows that aren't on a polygon.
   Publish and collect it the same way (document `<cls>-triage-<slug[:20]>-vN-swept`).
   Coverage of vN on the site = yeses / (yeses + polygons); precision = yeses /
   (yeses + noes). This split (misses by drawing, hits by judging) is what reviewers naturally
   do and is far cheaper than polygoning everything.
   On a site much bigger than the swept 60 windows (Normandie: 308 windows, 112 proposals of
   which 29 inside the sweep), also build `--swept-by <sweep json> --outside-sweep`: the
   proposals in the unswept windows, as a separate page and JSON (`-outside`). `apply.py` ingests
   it like any triage; it is never passed to `coverage.py`/`benchmark.py`, since those windows
   have no drawn misses to measure against. Added 2026-09-20 after the reviewer asked why the
   triage showed 28 boxes when the site had over a hundred.
5. `python scripts/loop/coverage.py --class <cls> --site ... --review <sweep json> --extra-truth <triage json> --models vN,vN+1`
   re-runs any model against that ground truth -- the generalisation number for versions that
   haven't trained on the site, memorisation for those that have. `--thresholds` matters when
   comparing versions whose confidence scale differs.
6. Ingest both JSONs: `python scripts/loop/apply.py --class <cls> --review <json>` (idempotent).
   Polygons and yeses become samples; noes become *disabled* hard negatives.
7. Regenerate and train: `python scripts/obb.py --class <cls> --hard-negatives` (the flag is what
   puts the *enabled* negatives into the package -- without it the 50 parked negatives are
   silently left out, whatever `groups.py` shows; in the experiment workspace this also uploads
   to `experiments/packages/`), then
   `python scripts/train_obb.py --class <cls> --version vN+1 --data-dir <workspace>/classes/<cls>/dataset_obb`.
8. **Gate the new version** (2026-09-20): `python scripts/loop/benchmark.py --class <cls>
   --models vN,vN+1` prints hits / false positives at >=0.5 and the >=0.7 count on every swept
   site, plus counts and max confidence on the negative (factory) sites listed in
   `<workspace>/loop/<cls>/benchmark.json`. The incumbent is replaced only on a clear win
   (more hits at no more false positives, no factory crossing 0.7); otherwise it keeps proposing
   and the new data waits for the next round. Prefer training the candidate as a fine-tune of
   the incumbent (`train_obb.py --base-model models/<cls>_obb_vN.pt --epochs 20 --lr0 0.0002`)
   over a fresh run from `yolo11n-obb.pt`; see the log below for why.
9. Next site. Re-run `coverage.py --models vN,vN+1` on every earlier sweep to see the trend.
10. Control: `python scripts/loop/groups.py --class <cls>` lists every group of samples and
   negatives by provenance with enabled counts; `--enable/--disable <group> [--limit N]`
   (add `--negatives` for the negative store) flips a group for the next package; `--versions
   vA,vB` shows what each trained version contained. To test whether a group hurts: disable it,
   regenerate, train, re-run `coverage.py` on the same sweeps, decide, re-enable or discard.

**Rules learned the hard way:**

- **Coverage is the metric, not precision.** Precision on proposals says nothing about what was
  never proposed; two "0 of 105" triage rounds looked like failure until a sweep showed the model
  simply wasn't proposing. Track found/drawn per site per model version.
- **No hard negatives below ~150 positive images.** 42 positives + 49 tight, correct negatives
  collapsed a model to 0.04-0.09 confidence on its own training data while val metrics barely
  moved (`v11`). `apply.py` stores rejections disabled; enable deliberately, in small numbers.
- **Val metrics don't flag collapse.** Ultralytics reports precision/recall at the F1-optimal
  threshold, which can be near zero. Check max confidence on known positives, and coverage.
- **Domain matters more than count at this size.** The class spans sharp/soft imagery and
  oblique/nadir views; a model trained on one corner transfers only to that corner, and adding
  14 samples from a soft site *reduced* transfer on a sharp one (`v13` 4/10 -> `v14` 1/10 at
  Godorf). Grow one domain to a stable model before mixing in the next, and always test on a
  fresh site in the domain you trained on.
- **Polygons, not points.** Reviewers rejected click and two-click marking; a polygon is what
  they would draw in `/manual` and needs no fitting. Body only, never the ground shadow (an
  elongated dark label is the shaft's shaded side seen obliquely).
- **One round at a time.** Finish and read a round before starting the next; parallel
  variants confound each other on a shared `dataset_obb/`.
- **Training is deterministic but not stable -- one-run ablations don't attribute cause.**
  `train_obb.py` runs ultralytics with `seed=0, deterministic=True`; `v20` retrained `v19`'s
  exact package and came out bit-identical, and `v21` (Esso groups disabled, 151 samples)
  reproduced `v18` exactly. But `v22` (only Esso's 12 hand-drawn polygons added back, 163
  samples) was worse than *both* `v18` and `v19` on every site including Esso itself. A
  12-sample change reshuffles the whole model at this size, so a flood like `v17`/`v19` cannot
  be pinned on the batch that preceded it from one run per variant -- the same k-fold caveat
  in root `CLAUDE.md` applies to `groups.py` ablations. Use `groups.py` to *remove data you know
  is wrong*; to test whether correct data *hurts*, average over several folds or accept that the
  answer is a trend over rounds. (`v20` duplicates `v19`; `v21` duplicates `v18`.)
- **Data is never dropped on suspicion (user's rule, 2026-09-20).** Columns vary by site, so
  the class needs many rounds of site-varied samples before versions stop swinging; until then
  every sample from every round stays enabled, and a group is disabled only with evidence
  strong enough to be certain -- which, given the determinism/instability rule above, a single
  ablation run does not provide. Candidate *models* are rejected freely by the benchmark;
  candidate *data* is not.
- **Promotion is explicit.** A class leaves `experiments/` only by a deliberate move of its
  data and a config change in `oil_refinery/app/server/`; nothing graduates as a side effect
  of training. Columns, when promoted, go in as a *booster* edge, not a `requires` edge --
  nadir-orthophoto sites like Płock will never show one.

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

## Round log and current state

**Round 10, TotalEnergies Normandie (2026-09-20), fresh, soft (0.72), 3.2 km2 / 308 windows,
v18 proposing:** 112 proposals (7 at >=0.5), 29 inside the 60 swept windows. Reviewer drew 11
misses, judged 28 inside (6 yes, 21 no, 1 unsure) and, on the new `--outside-sweep` page, 83
outside (5 yes, 76 no, 2 unsure); 17 truth. v18 fresh: 7/17 = 41% at 0.25 -- its weakest site,
and 97 of 111 proposals were rejected. Samples 227 -> 249 across 44 sites, negatives 50/473.
Candidate `v28` (v18 fine-tuned, AdamW 0.0002, 20 epochs, 249): hits 53 -> 86 on the seven
benchmark sites but FP 53 -> 142 and three factories above 0.7 (Wolfsburg 0.83); matched point
29 hits / 18 FP. **Rejected; v18 remains.** Fresh-site coverage of v18 so far: Esso 48%,
Wesseling 62%, Heide 59%, Normandie 41%. Next in queue: Lingen (0.65), Gelsenkirchen Horst
(0.64), Mitteldeutschland (0.61).

**Round 9, Raffinerie Heide (2026-09-20), fresh (0.72), v18 proposing:** 73 proposals (16 at
>=0.5); reviewer drew 12 misses, judged 52 -- 17 yes, 30 no, 5 unsure; 29 truth. v18 fresh:
17/29 = 59% at 0.25, 8 hits / 7 FP at 0.5 (v26 on the same: 12/29). Samples 198 -> 227 across
40 sites. Candidate `v27` = v18 fine-tuned 20 epochs at AdamW 0.0002 on 227: hits 51 -> 73 on
the six benchmark sites but FP 51 -> 136 and Wolfsburg 8 -> 30 detections with five above 0.7
(max 0.82); at the matched point (v27 at >=0.7 vs v18 at >=0.5) 34 hits / 29 FP against 51 /
51 with a factory now outranking a refinery. **Rejected; v18 remains.** Running tally: 151
samples (v18) has beaten every challenger trained on 172, 198 and 227 -- from scratch and as
fine-tunes at two rates. v18's fresh-site coverage: Esso 48%, Wesseling 62%, Heide 59%. Next in
queue: Normandie (0.72).

**Round 8, Shell Wesseling (2026-09-20), fresh site (0.75, head of the queue), v18 proposing:**
77 proposals at >=0.25 (18 at >=0.5). Reviewer drew 12 misses and judged 42 proposals -- 14 yes,
17 no, 11 unsure; 26 ground-truth objects. v18 fresh-site: 6 hits / 8 FP at >=0.5, 16/26 = 62%
at 0.25 with 28 FP (v16 46%/82 FP; v19 27%/54 FP). Samples 172 -> 198 across 38 sites,
negatives 50/346; `v23` trained on that and its confidence *collapsed* -- at >=0.5 it puts 0-2
boxes on BP, Godorf and Wesseling and 7 on Scholven where v18 puts 21. Between `v17` (up),
`v18` (calibrated), `v19` (up) and `v23` (down), every ~20-sample batch has swung the confidence
scale wholesale; `v18` at 151 samples is the well-calibrated one and stays the working model.

**Site-level reading (the user's frame, 2026-09-20).** The class exists to feed the refinery
classifier as a booster, so what matters is that confident (>=0.5) boxes land on refineries with
few false positives, not per-object coverage. By that measure `v18` is already usable: 3-21
confident hits per refinery seen so far at roughly 50% precision, and its unsures are
column-like objects at refineries. The remaining check before wiring it in is negative sites:
scan a few non-refinery industrial areas (chemical plants, power stations, ports) with `v18` and
confirm >=0.5 detections stay sparse. Coverage rounds continue only if that check passes and
there is appetite for more labelling; the queue (Heide, Normandie, Lingen, Horst,
Mitteldeutschland) is recorded in step 1.

**Challengers to v18 on the 198-sample package (2026-09-20), all rejected by `benchmark.py`:**
`v23` from scratch (confidence collapsed, 0-2 confident hits per refinery); `v24` fine-tuned from
v18 for 40 epochs and `v25` for 20, both at ultralytics' auto rate (identical behaviour to a
fresh run: Scholven 21 -> 11/16 hits, Godorf FP 5 -> 17/24, a factory crossing 0.7 -- and
`--lr0` was silently ignored until it also forced `optimizer=AdamW`, see `scripts.md`); `v26`
fine-tuned 20 epochs at AdamW 0.0002 -- the first that behaves as a nudge: +2 hits at Wesseling,
+5 at Esso, four of six factories quieter, but Godorf 6 -> 3 hits at 24 FP and Scholven 21 ->
16, 42 hits / 59 FP total against v18's 43 / 44. Net: 198 samples has not yet beaten 151; the
fine-tune + gate mechanism is the way each further round is tested.

**Negative-site check (2026-09-20), v18, same scan geometry everywhere.** A second OSM layer
(`industrial=factory` polygons, `~/Downloads/factories.geojson`) was merged into `sites.json`
with `sites.py --geojson ... --layer factories` (added that day: `--layer` merges by osm id
instead of overwriting, and `--list --layer` filters; the refinery rows have no `layer` key and
count as `refineries`). Six refineries not all in training vs six factories chosen to be as
refinery-like as the layer offers (steelworks, glass, sugar, wood panels, car plant with its own
power station, Continental):

| site | ≥0.5 | ≥0.7 | max |
|---|---|---|---|
| TotalEnergies Antwerpen (1.96 km²) | 24 | 4 | 0.78 |
| Shell Wesseling (1.29) | 18 | 4 | 0.84 |
| Raffinerie Heide (1.00) | 16 | 2 | 0.73 |
| Esso Rotterdam (1.78) | 12 | 1 | 0.71 |
| BP Lingen (1.26, sharp 0.65) | 8 | 0 | 0.61 |
| BP Gelsenkirchen Horst (1.44) | 8 | 1 | 0.70 |
| VW Wolfsburg (5.31) | 9 | 0 | 0.65 |
| Swiss Krono (0.49) | 4 | 0 | 0.64 |
| Ardagh Glass (0.28) | 3 | 0 | 0.56 |
| Stahlwerk Georgsmarienhütte (0.66) | 3 | 0 | 0.55 |
| Continental AG (0.51) | 2 | 0 | 0.60 |
| Suikerfabriek Vierverlaten (0.13) | 1 | 0 | 0.52 |

No factory reached 0.7; five of six refineries did. A site rule of "any detection ≥0.7, or ≥10
at ≥0.5" separates this table with BP Lingen as the one refinery miss (soft imagery) -- which is
what the booster-edge role tolerates. This is the POC-level evidence that `v18` is usable as a
refinery signal; the factory detections are unlabelled, so they are "not confident", not
"confirmed false".

**Round 7, Esso Rotterdam (2026-09-19), fresh sharp site (0.78), v18 proposing:** 66 proposals at
>=0.25 (12 at >=0.5 -- v18's scale is sane). Reviewer drew 12 misses and judged 42 proposals --
9 yes, 27 no, 6 unsure; 21 ground-truth objects. Fresh-site coverage at 0.25: v16 7/21 (88 FP),
v17 17/21 (330 FP), **v18 10/21 = 48% (34 FP)**, precision 0.50 at >=0.5. So Scholven's 91% was
the outlier; the honest home-domain number is still around half. Samples 151 -> 172 across 33
sites, negatives held at 50/329; `v19` trained on that and came out v17-shaped -- Esso 17/21 but
130 FP, Scholven 28/32 at 150 FP (v18: 59), and *lost* memorisation at Godorf (7 -> 5/10, FP 37
-> 101) and BP (7 -> 6/12, FP 12 -> 86). The full v18-vs-v19 table across the four sites is
below. Every review JSON is now under `reviews/` (earlier rounds' were pulled back from the
artifact stores).
`v20` (same package, retrained to check for run-to-run variance) reproduced `v19` exactly -- see
the determinism rule above -- so the Esso batch (12 sweep polygons + 9 triage yeses) is the
suspect for the flood. **Ablation:** `v21` (both Esso groups off, 151) reproduced `v18`
exactly; `v22` (12 sweep polygons on, 9 triage yeses off, 163) was worse than both on every
site, Esso included (Esso 6/21 at 48 FP; Scholven 22/32; Godorf 5/10; BP 4/12). That does not
support "Esso hurts" -- it shows single runs at this size move with any change (see the
determinism rule). All 21 Esso samples are **re-enabled** (they are correct labels), the package
is back to 172, and `v18` stays the proposer for round 8. Puertollano's 14 soft-site samples
(the one batch with a recorded transfer drop, `v13` -> `v14`) are still enabled and remain the
other candidate, subject to the same caveat.

| site (truth) | v18 @0.25 | v18 @0.5 | v19 @0.25 | v19 @0.5 |
|---|---|---|---|---|
| Esso (21, fresh for both) | 10, 34 FP | 3, 3 FP | 17, 130 FP | 12, 25 FP |
| Scholven (32) | 28, 59 FP | 21, 23 FP | 28, 150 FP | 17, 14 FP |
| Godorf (10) | 7, 37 FP | 6, 5 FP | 5, 101 FP | 3, 7 FP |
| BP (12) | 7, 12 FP | 7, 5 FP | 6, 86 FP | 6, 16 FP |

**Round 6, Scholven (2026-09-19), fresh sharp-oblique site, v16 proposing:** reviewer drew 3
misses and judged 94 proposals -- 29 yes, 44 no, 21 unsure. Coverage **29/32 = 91%**, precision
40% at >=0.25, 62% at >=0.4, **79% at >=0.5**. Three rounds earlier the same test at Godorf gave
4/10. Samples 119 -> 151; negatives held at 50 enabled (302 stored); `v18` trained on that.

**State as of 2026-09-19 (earlier that day):** `distillation-column` has 119 samples across 23 sites
(22 hand-drawn, the rest from four sweeps and three triages), 258 hard negatives of which 50 are
enabled (25 from sharp training sites, 25 in-place from BP Rotterdam), `v17` training on that,
and sweeps with ground truth at La Rábida (32, nadir), Puertollano (12, soft oblique), Godorf
(10, sharp oblique) and BP Rotterdam (5, sharp oblique). Fresh-site coverage in the home domain
has been ~40% at conf 0.25 (Godorf v13 4/10, BP v15 2/5). Two negative batches (v16: 25 from
training sites, v17: +25 in-place from BP) shifted confidence upward without improving
separation at matched false-positive counts, so negatives are parked at 50 and the lever is
positives again until ~150-200. **Next round: use `v16` as the proposer, not `v17`** -- v17's
inflated scores flood the candidate list (1,194 at >=0.25 on Puertollano); if v17 must be used,
show proposals at >=0.5. Expect trends over many rounds, not jumps.
