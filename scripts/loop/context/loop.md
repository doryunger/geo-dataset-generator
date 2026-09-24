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
below this one; this section is the runbook. A *new* class runs everything below with `WORKSPACE=experiments` (see
`scripts/common.py`) and with `.env` sourced into the shell (`set -a && source .env && set +a`
-- `apply.py` fetches crops from Mapbox and dies without the token) so production `classes/`, `/manual` on port 8000 and S3 `packages/` are
never touched; `run-experiments.ps1` serves `/manual` against the experiment workspace on 8001.
A promoted class (`fan-unit`; `distillation-column` since 2026-09-23) runs the same commands
with `WORKSPACE` unset -- its loop state is in `loop/<class>/` at the repo root.

**The idea.** Nobody draws a whole class from scratch. A person seeds a few dozen samples,
a weak model is trained, and from then on the model proposes and the person judges. Each round
uses one refinery site from the OSM layer and produces samples, hard negatives and a coverage
number. The class is finished when coverage on a *fresh* site stops improving.

**One round, in order:**

1. Pick a site: `python scripts/loop/sites.py --class <cls> --list` (on Windows prefix
   `PYTHONIOENCODING=utf-8` -- site names carry accents the cp1252 console cannot print and the
   listing dies mid-way otherwise). Sites marked `HELD-OUT` are off limits -- `scan.py` refuses
   them; they are `benchmark.json`'s `held_out` list and exist only to be measured. Prefer sites with
   `sampled=0` and `sharpness_vs_train` near 1.0; scan a few unscored candidates first, then
   `--score`. Sharpness is a coarse gate (Mapbox coverage is soft across much of south-east
   Europe and the model does not transfer to it); obliqueness has to be judged by eye from the
   scan windows -- tank sides visible means oblique, clean circles means nadir. **Selection is a
   queue, not a search** (rule set 2026-09-20 after a round was spent scanning for a site
   sharper than 0.85 that does not exist): score a batch of ~5 unsampled sites, work them
   top-down by score, and defer anything under the floor of **0.65** (raised from 0.6 on
   2026-09-21 after the reviewer found Mitteldeutschland at 0.61 too soft and oblique to label
   reliably -- the floor is where the *reviewer* stops being able to see columns, not where the
   model does) until every site above it is done. Don't scan for a better site than the head of
   the queue. Current queue (2026-09-21, after round 20): Gdansk 1.32, Schwechat 1.16, Port-Jerome 0.87,
   Litvinov 0.81, Grangemouth 0.80. Skipped: A Coruna 0.78 (Spain). Deferred (below 0.65):
   Sarlux 0.64, Mitteldeutschland 0.61, Sines 0.44. The score and the reviewer's eye disagree sometimes -- Fos (0.81) and Tarragona (1.08)
   both looked blurry to the reviewer; the Laplacian score rewards contrast, not clarity, so the
   floor stays a reviewer's call and the score only orders the queue. Castello (0.89) was worse
   still; the reviewer's verdict after rounds 19-20: Spanish Mapbox coverage is blurry whatever
   the score says, so A Coruna is skipped and the next batch comes from northern Europe (UK,
   Denmark, Sweden, Poland's non-orthophoto sites).
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
   **The site-level number that counts is held-out recall**: `INFERENCE_DEVICE=cuda python
   scripts/eval_sites.py --class <cls> --held-out --cache <file>` runs the app's own site path over
   refineries that have never been labelled. `positives` in `benchmark.json` are all training sites
   (every one holds 10-32 column samples), so their verdict measures memory, not generalisation.
9. Next site. Re-run `coverage.py --models vN,vN+1` on every earlier sweep to see the trend.
10. Control: `python scripts/loop/groups.py --class <cls>` lists every group of samples and
   negatives by provenance with enabled counts; `--enable/--disable <group> [--limit N]`
   (add `--negatives` for the negative store) flips a group for the next package; `--versions
   vA,vB` shows what each trained version contained. To test whether a group hurts: disable it,
   regenerate, train, re-run `coverage.py` on the same sweeps, decide, re-enable or discard.

**Rules learned the hard way:**

- **The end game is precision at a chosen threshold (user, 2026-09-21).** "Even if we might miss
  some detections, the detections themselves should be reliable." `benchmark.py` ends with a
  *reliable detections* table per version -- hits at the lowest threshold that silences every
  look-alike and keeps refinery FP within 0 / 2 / 5 -- and the FP<=0 column is the adoption
  criterion. First use, round 14: v18 4, v30 3, v32 4, **v33 18** (at 0.84); v32 had looked best
  on the 0.5 tables and v33 worst, so the old gate would have got this one backwards. Coverage (next
  rule) is still how a round's progress is measured; it is not what decides a version.
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
- **Scope is sharp imagery (user's decision, 2026-09-21).** "We cannot expect to have a class
  that will work in any condition." Soft-site coverage (Lingen 27%, Normandie 41%) is not a
  caveat on reliability and gets no rounds; the queue floor is the scope boundary.
- **Data is never dropped on suspicion (user's rule, 2026-09-20).** Columns vary by site, so
  the class needs many rounds of site-varied samples before versions stop swinging; until then
  every sample from every round stays enabled, and a group is disabled only with evidence
  strong enough to be certain -- which, given the determinism/instability rule above, a single
  ablation run does not provide. Candidate *models* are rejected freely by the benchmark;
  candidate *data* is not.
- **Promotion is explicit.** A class leaves `experiments/` only by a deliberate move of its
  data and a config change in `app/server/`; nothing graduates as a side effect
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

## Sites flagged blurry by the reviewer (candidates for a later ablation)

The reviewer said these looked blurry despite their scores, and asked (2026-09-21) that their
samples be droppable later if they turn out to confuse the model. Their groups, for
`groups.py --disable` (and `--negatives` for the rejected ones):

- Fos-sur-Mer (0.81, "quite low" but the model found most columns): `loop-sweep:rhone_energies_fos_sur_mer_refin:v33`, `loop-triage:rhone_energies_fos_sur_mer_refin:v33`, `loop-triage-rejected:rhone_energies_fos_sur_mer_refin:v33`
- Tarragona (1.08): `loop-sweep:repsol_tarragona_refinery:v37`, `loop-triage:repsol_tarragona_refinery:v37`, `loop-triage-rejected:repsol_tarragona_refinery:v37`
- Castello (0.89, "could be even worse"): `loop-sweep:refineria_de_castello:v37`, `loop-triage:refineria_de_castello:v37`, `loop-triage-rejected:refineria_de_castello:v37`

The test is the usual one: disable, repackage, train both candidates, `benchmark.py` against
the incumbent. Under the no-drop rule this is the one case where disabling is justified -- the
reviewer's own judgement that the labels are unreliable.

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

**Held-out refineries (2026-09-24).** Every `positives` site in `benchmark.json` turned out to be
a training site, so the site test's 18/18 was memory. 24 refineries with no sample or hard
negative within ~1 km were frozen as `held_out`. v46 through the app's own site path: 6/24 at the
production graph (column >= 0.65, count 3), 0/39 look-alikes. Sweep from a cache detected at a
0.4 column floor (`--floor` now lowers the detection-time floor too): distance has no effect;
count 2 -> 9/24; **floor 0.5 + count 2 -> 13/24 at 0/39**, Dow Portugal and Exxonmobil still
rejected; floor 0.4 + count 2 -> 14/24 but Dow and Chane terminal go red. 0.5 was chosen looking
at the held-out set, so 13/24 is slightly optimistic. Of the 10 still missed at 0.4 + count 2, three have no column at all (Sisak, Vega, Harwich),
St1 (0.85) and Nynas have columns but no fan-unit, the rest have at most one column.

**Promoted to production (2026-09-23).** `distillation-column` data moved out of `experiments/`
into `classes/` and `loop/`; further rounds run without `WORKSPACE`.
Paths quoted below as `experiments/...` from before this date are now at the repo root.

**Integration and the look-alike round (2026-09-21/22).** v37 was wired into `oil_refinery`
(`config.json` models + `model_gsd_m` 0.125 + gated; `semantic_graph.json` node + requires
edge) and `scripts/eval_sites.py` was written to run the server's own detection batch and
classifier over whole sites offline, with a per-tile detection cache so graph changes re-classify
in seconds. Findings, in order:

- The graph as found (2-of-5 types, `harbor` required, 600 m) called almost every look-alike a
  refinery: chimney at 0.3 and fan-unit at 0.5 fire on any industrial site. User's rule: a
  refinery is *all* its components. Now 4-of-4 (storage tank, chimney, fan-unit, column),
  `harbor` removed (inland refineries have none), radius 300 m (user: "600 is too big").
- With v37 at column >= 0.78: 12/18 refineries, 0/16 look-alikes. Lowering the column floor
  alone leaked (0.7: 16/18 but 3 look-alikes; 0.6: 18/18 but 8) because two sites --
  Wolfsburg (roof structures, 0.83) and Niederaussem (hopper tops, 12 boxes >= 0.6) -- are
  column confusers. Fan counts don't fix a column false positive; dropping chimney changes
  nothing (it is present everywhere at 0.3).
- `min_count` added to `requires` edges (optional, default 1) after the user pointed out that
  refineries have *many* fans; factory fans are real (rooftop ventilation) but few.
- Look-alike round: 32 non-refinery sites (the 16 plus 17 from a second Overpass pull --
  cement, steel, sugar, paper, incinerators, biomass/gas/coal plants, sewage works; chemical
  plants excluded because their columns are real), one multi-site triage page of v37's 216
  column proposals >= 0.4 (`pages.py triage --also ... --page-slug lookalikes`): 191 no, 4 yes
  (three at Swiss Krono, one at ThyssenKrupp -- both dropped from the negatives list), 21
  unsure. All 191 enabled (negatives 441/1031 against 476 positives). `v46` (v37 fine-tuned)
  pushed Wolfsburg 0.73 -> 0.61 (18 -> 3 boxes) and Niederaussem 12 -> 1 boxes; on the object
  gate it scored 37 clean vs v37's 52 (a few refinery-side boxes on unlabelled objects raise its
  clean threshold) but 106 vs 76 at FP<=5.
- **Site test with v46, column >= 0.6, fans >= 3: 18/18 refineries, 0/31 look-alikes**; on
  eight look-alikes v46 had never seen (Melnik, Tusimice, Clauscentrale, Ketton, Westfalenhutte,
  Stahlwerk Thuringen, Mogden, Amercentrale): 0/8, column max 0.43. **v46 adopted**, graph
  defaults set to this. The site test outranks the object gate when they disagree; the object
  table stays as the per-round proxy.

Run the site test: `INFERENCE_DEVICE=cuda python scripts/eval_sites.py --class
distillation-column --cache loop/distillation-column/site_detections_v46.json`
(`--floor`, `--count`, `--min-types`, `--max-distance-m` override the graph from the cache in
seconds; a new column model needs a new cache, ~45 min on the GPU, and OOMs if anything else
holds the card -- `--batch 4`).

**Round 21, Gdansk (2026-09-21), fresh, sharp (1.32), v37 proposing:** 35 proposals, 29
inside; reviewer drew 10 misses, judged 29 (18 yes, 4 no, 7 unsure) and 6 outside (5 no);
28 truth. v37 fresh: 18/28 = 64% at 0.25, 6 hits / 1 FP at 0.5. Samples 444 -> 472 across 62
sites, negatives 250/840. Gate (18 refineries, 16 look-alikes): `v44` 3, `v45` 0 clean vs v37
52. Rejected; v37 remains -- five candidates in a row since round 17. `benchmark.json` was
rebuilt with ASCII site keys after a cp1252 crash on "Gdansk" corrupted it; use ASCII
substrings there.

**State at the end of the agreed three-site queue (Tarragona, Castello, Gdansk):** v37, 390
samples at training time, 52 zero-FP detections with every look-alike silent at 0.78. Next
step is integration, not another site: wire v37 into `oil_refinery` as the booster edge and
test refinery vs look-alike end to end; revisit the graph radius there.

**Round 20, Castello (2026-09-21), fresh (0.89 by score, "could be even worse" than Tarragona
by eye), v37 proposing:** 93 proposals, 64 inside; reviewer drew 4 misses, judged 63 (7 yes,
33 no, 23 unsure -- Unsure used liberally on purpose); 11 truth. v37 fresh: 8/11 = 73% at 0.25
but 59 FP (precision 0.13): blur shows up as false positives, not misses. Samples 433 -> 444
across 59 sites, negatives 241/831. Gate: `v42` (fine-tune) 47, `v43` (scratch) 13 vs v37 52.
Rejected; v37 remains. Spanish sites are done with (see step 1); northern batch scored: Gdansk
1.32, Schwechat 1.16, Port-Jerome 0.87, Litvinov 0.81, Grangemouth 0.80; Coryton 1.45 is a
demolished refinery (10 proposals on 2.6 km2) and is held back as a possible negative site.

**Round 19, Repsol Tarragona (2026-09-21), fresh (1.08 by score, blurry by eye), v37
proposing:** 84 proposals, 54 inside; reviewer drew 7 misses, judged 54 inside (20 yes, 29 no,
5 unsure) and 30 outside (2 yes, 26 no); 27 truth. v37 fresh: 20/27 = 74% at 0.25, 8 hits / 3
FP at 0.5. Samples 404 -> 433 across 58 sites, negatives 226/798. Gate: `v40` (v37 fine-tuned)
47 clean at 0.85 -- 89 at FP<=2 and 114 at FP<=5, the strongest challenger yet on the looser
budgets but 5 short on the one that counts; `v41` (scratch) 24. v37 remains. Next: Castello.

**Round 18, Grandpuits (2026-09-21), fresh (0.72), v37 proposing:** 10 proposals, all inside;
reviewer drew 10 misses, judged 10 (4 yes, 5 no, 1 unsure); 14 truth. v37 fresh: 4/14 = 29% at
0.25 (v33 6/14). Samples 390 -> 404 across 57 sites, negatives 206/743. Both candidates
rejected: `v38` (v37 fine-tuned) 23 clean at 0.87, `v39` (from scratch on 404) 14 at 0.77,
against v37's 52 at 0.78 (75 at FP<=2). v39 vs v37 -- same recipe, +14 samples, a quarter of
the reliable detections -- is the instability rule again, now for from-scratch runs too. v37
remains. Queue exhausted; next batch to score: Tarragona, Castello, Sines, Sarlux, A Coruna.

v37's fresh-site record so far: Grandpuits 29% (0.72). Incumbent fresh-site coverage by site
and proposer, sharp sites only (>=0.8): Horst 79% (v30, 0.64 -- soft but the reviewer could
label it), Esso Belgium 56% (v30, 1.16), Antwerpen 74% (v33, 0.99), Gunvor 27% (v33, 0.84),
Fos 77% (v33, 0.81).

**Round 17, Fos-sur-Mer (2026-09-21), fresh (0.81, reviewer called the imagery low), v33
proposing:** 21 proposals, 19 inside; reviewer drew only 4 misses, judged 18 (9 yes, 6 no, 3
unsure); 13 truth. v33 fresh: 10/13 = 77% at 0.25, **3 hits / 0 FP at 0.5**. Samples 377 ->
390 across 56 sites, negatives 201/738. `v36` = v33 fine-tuned: 7 clean at 0.87 vs v33's 18.
Rejected. Three consecutive fine-tunes of v33 (v34 15, v35 13, v36 7 clean) have each kept
fewer reliable detections -- the fine-tune lineage looks to be drifting rather than the data
failing, so a from-scratch candidate on the same package was tried: **`v37` (yolo11n-obb.pt,
390 + 201, early-stopped) keeps 52 clean detections at 0.78 (74 at FP<=2), look-alike ceiling
0.75 -- against v33's 18 at 0.84. Adopted 2026-09-21, operating threshold 0.78.** Round 18
proposes with v37 at Grandpuits (0.72), the last site in the queue.

Lineage so far: v18 (scratch, 151) -> v30 (fine-tune, +negatives) -> v32 (fine-tune) -> v33
(fine-tune) -> v37 (scratch, 390). Fine-tunes won while the base was young; three in a row
lost once the base was three fine-tunes deep. From here every round trains both a fine-tune of
the incumbent and a from-scratch run, and the reliable-detections table picks.

**Round 16, Gunvor Rotterdam (2026-09-21), fresh, sharp (0.84), v33 proposing:** 15 proposals,
10 inside the sweep; reviewer drew 8 misses, judged 10 (3 yes, 5 no, 2 unsure); 11 truth. v33
fresh: 3/11 = 27% at 0.25 -- poor for a sharp site. Samples 366 -> 377 across 55 sites,
negatives 195/732. `v35` = v33 fine-tuned: 13 clean detections at 0.86 vs v33's 18 at 0.84
(25 vs 22 at FP<=2). Rejected on the FP<=0 column; v33 remains. Review pages were rebuilt this
round to the reviewer's layout: a plain 380 px panel on the left, image filling the rest of
one viewport, nothing below the image (`templates/sweep.html`, `templates/triage.html`).

**Round 15, TotalEnergies Antwerpen (2026-09-21), fresh, sharp (0.99), v33 proposing:** 86
proposals (42 at >=0.5), 62 inside the sweep. Reviewer drew only 8 misses, judged 61 inside (19
yes, 29 no, 13 unsure) and 24 outside (4 yes, 16 no); 27 truth. v33 fresh: 20/27 = 74% at 0.25,
14/27 at 0.5 with 19 FP -- and 0 hits at its 0.84 operating threshold: the clean point is
conservative enough that a new site can contribute nothing to it. Samples 335 -> 366 across 50
sites; negatives 190/727 (20 of Antwerpen's). `v34` = v33 fine-tuned: 15 clean detections at
0.85 against v33's 18 at 0.84. Rejected; v33 remains. The issue is now separation, not
ranking: real columns and the best look-alike confusables (hopper tops at 0.81-0.83) overlap on
the confidence scale, and only more confirmed negatives of that kind plus more sharp positives
widen the gap.

**Round 14, Zeeland revisit (2026-09-21), sharp (1.06), v32 proposing:** 14 proposals, all
inside the sweep; reviewer drew 14 misses (2 already samples), judged 14 (4 yes, 7 no, 3
unsure); 18 truth. v32 10/18 = 56% at 0.25, 4 hits / 1 FP at 0.5. Samples 319 -> 335 across 48
sites, negatives 170/682. `v33` = v32 fine-tuned: on the 0.5 tables it looked like another
inflation (hits 95 -> 141, FP 49 -> 110, three look-alikes above 0.7), but on the new
reliable-detections table it is the best version yet -- **18 true detections with zero false
positives and every look-alike silent, at threshold 0.84** (v32: 4 at 0.77; v30: 3; v18: 4).
**v33 adopted, operating threshold 0.84.** Round 15 proposes with v33 at TotalEnergies
Antwerpen (0.99).

**Round 13, Esso Belgium (2026-09-21), fresh, sharp (1.16), v30 proposing:** 37 proposals (6 at
>=0.5), 34 inside the sweep. Reviewer drew 11 misses, judged 34 (14 yes, 16 no, 4 unsure); 25
truth. v30 fresh: 14/25 = 56% at 0.25 (v18 the same), and at 0.5 **6 hits / 0 FP**. Samples
294 -> 319 across 49 sites; negatives 163/675 (Esso Belgium's 16 added). `v32` = v30
fine-tuned (AdamW 0.0002, 20 epochs): 91 hits / 48 FP on the ten refineries against v30's 79 /
51, **no look-alike above 0.7 (max 0.61)**, Niederaussem 0.67 -> 0.55, Wolfsburg 0.70 -> 0.58;
Scholven 15 -> 21, Horst 5 -> 8, Lingen 8 -> 6, Heide 14 -> 11. First clean pass of the gate.
**v32 adopted (2026-09-21)**; v30 and v18 stay on disk. Round 14 proposes with v32 at Zeeland.

Fresh-site coverage by proposer so far: v18 -- Esso 48, Wesseling 62, Heide 59, Normandie 41,
Lingen 27; v30 -- Horst 79, Esso Belgium 56 (v18 on the same two: 42, 56).

**Round 13 attempt, Mitteldeutschland/Leuna (2026-09-21):** v30 made 3 proposals on 2.8 km2;
the reviewer found the imagery too soft and oblique to label and the site was deferred (floor
raised to 0.65). Round 13 moves to Esso Belgium (1.16).

**Round 12, BP Gelsenkirchen Horst (2026-09-20), fresh, soft (0.64), v30 proposing:** 26
proposals (9 at >=0.5), 23 inside the sweep. Reviewer drew only 5 misses (previous rounds
11-19), judged 22 inside (14 yes, 4 no, 4 unsure) and 3 outside (all no); 19 truth. **v30
fresh: 15/19 = 79% at 0.25, 8 FP, precision 0.67; v18 on the same ground 8/19 = 42%** -- the
first fresh-site head-to-head between two incumbents, and the adopted one nearly doubled
coverage on a soft site. Samples 275 -> 294 across 48 sites; negatives 147/659 enabled
(Horst's 7 added to keep ~half). `v31` = v30 fine-tuned (AdamW 0.0002, 20 epochs): hits 73 ->
108 but FP 51 -> 119, Bremerhaven 0.82 (four above 0.7), Niederaussem 0.81, Chane 0.76.
Rejected -- inflation returned at the same 2:1 ratio, so the ratio was necessary for v30 but is
not sufficient; note v31 is a fine-tune of a fine-tune. v30 remains. Next: Mitteldeutschland
(0.61), the last site above the floor; then score a new batch.

**Round 11, BP Lingen (2026-09-20), fresh, soft (0.65), v18 proposing:** 188 proposals (8 at
>=0.5), 19 inside the 60 swept windows. Reviewer drew 19 misses, judged 19 inside (7 yes, 10 no,
2 unsure) and 169 outside (all no); 26 truth. v18 fresh: 7/26 = 27% -- its lowest, on its
softest site. Samples 249 -> 275 across 46 sites, negatives stored 652. `v29` (fine-tune, 50
negatives): the flood pattern at full volume -- Wolfsburg 79 detections with 12 above 0.7,
every port and power station lit. Rejected. **`v30`: same 275 samples + 140 enabled negatives
(the 50 plus the 15 highest-confidence rejections from each of Scholven, Esso, Wesseling,
Heide, Normandie, Lingen), fine-tuned from v18 at AdamW 0.0002 / 20 epochs.** First candidate
that moves the right way: 68 hits / 47 FP on the eight refineries against v18's 54 / 56, every
fresh site up (Lingen 1 -> 8, Normandie 2 -> 4, Heide 8 -> 14, Wesseling 6 -> 10, Esso 3 -> 7),
the three oldest training sites down (Scholven 21 -> 15, Godorf 6 -> 4, BP 7 -> 6), 12 of 16
negatives quieter, Niederaussem 0.72 -> 0.67 while Wolfsburg has one detection at exactly 0.70
and Chane max 0.70. The lesson: with positives at ~5x the enabled negatives, every fine-tune
inflated; at ~2x it stopped. Negatives are back on as a lever, in this proportion. **v30 adopted as incumbent (2026-09-20)**;
v18 stays on disk as the fallback. Round 12 proposes with v30.

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

**Look-alike layer (2026-09-20).** Ten sites pulled from Overpass (power=plant coal/gas,
landuse=harbour, industrial=port/oil/oil_storage within DE/NL/BE) and merged as
`sites.py --layer lookalikes` from `loop/distillation-column/lookalikes.geojson`:
power stations Weisweiler, Niederaussem, Gersteinwerk, Datteln 4; ports Bremerhaven, Dortmund,
Tollerort; tank-only terminals Nord-West Oelleitung, Grosstanklager, Chane Nieuwe Maas. Shell
Pernis / Moerdijk and Dow Schkopau came back tagged `industrial=oil` but are refineries or
crackers and were left out; Kraftwerk Scholven sits inside the Scholven refinery fence, also
out. All ten are in `benchmark.json`. v18: tank farms and three power stations max 0.38-0.44,
ports 0.49-0.66, **Niederaussem 0.72** -- four ~10 m round hopper/silo tops in near-nadir
imagery, the known nadir-circle confusion, not a refinery-vs-plant one. v28 is worse on 14 of
the 16 negatives (Niederaussem four above 0.7, Chane 0.76), confirming its rejection.

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
