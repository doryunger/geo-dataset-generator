# Training a new detection class

This is the process this repo exists to support: taking an object class that no pretrained
model knows (a distillation column, a fan unit), and growing a detector for it from a few dozen
hand-drawn samples to a model that is useful in the site classifier — without anyone drawing
the whole class by hand. It was worked out on `distillation-column` over ten rounds in
September 2026; the measurements and dead ends behind each rule are in
`scripts/loop/context/loop.md`, which is also where the per-round log lives. This document is
the process itself.

Two people-roles appear below. The **reviewer** looks at imagery and says what is there; ten
minutes per round. The **operator** runs the commands; may be a person or Claude. They are
usually the same person.

## 1. The idea

A model proposes, a person judges, the judgements become training data, and the model is
re-trained — one refinery site per round. Two numbers are tracked:

- **Coverage** — of the objects really present on a site the model has never seen, what
  fraction does it find. This is the loop's own progress metric.
- **Site-level separation** — does the model put confident detections on the kind of site the
  class belongs to (refineries) and not on look-alikes (factories, ports, power stations). This
  is what makes the model useful, and it is the metric a candidate has to win on.

Precision of proposals alone is not a metric: it says nothing about what was never proposed.

The class is **stable** when a new version trained on more data beats the incumbent on the
benchmark more often than not, and **done** when fresh-site coverage stops improving. Until
then, rounds continue, and no data collected in earlier rounds is dropped.

## 2. Ground rules

These were each learned by getting it wrong once; the evidence is in `loop.md`.

1. **Everything runs in the experiments workspace.** `WORKSPACE=experiments` for every command,
   `.env` sourced (`set -a && source .env && set +a`). Production `classes/`, the `/manual`
   editor on port 8000 and the S3 `packages/` prefix are never touched until a class is
   promoted deliberately. Once promoted (`fan-unit`, `distillation-column`), the class's rounds
   run with `WORKSPACE` unset, against the root `classes/` and `loop/`.
2. **Data is never dropped on suspicion.** Every sample and every triage "yes" from every round
   stays enabled. Candidate *models* are rejected freely; candidate *data* is disabled only with
   evidence strong enough to be certain, and a single ablation run is not that (see rule 6).
   `groups.py --disable` exists for data that is known to be wrong.
3. **Hard negatives are used in proportion, not parked.** The reviewer's "no" verdicts are
   saved as disabled negatives; enable the highest-confidence ones per site so that enabled
   negatives stay at roughly half the positive count. Below ~150 positives, none at all (42
   positives + 49 negatives collapsed a model); at 275 positives, 50 negatives let every
   fine-tune inflate its confidence and 140 stopped it (`v29` vs `v30`).
4. **The incumbent model keeps its job until a challenger beats it.** Every round trains a
   candidate; the benchmark (section 4) decides. A rejected candidate costs one GPU run and
   nothing else — the new data stays and is in the next candidate.
5. **Polygons, not points.** The reviewer draws the object body as they would in `/manual`.
   For columns: the shaft only, never the ground shadow.
6. **Training is deterministic but not stable.** The trainer is seeded, so re-training the same
   package gives the same weights; but a change of ~20 samples reshuffles the whole model at
   this size. Don't re-train to "check for variance", and don't attribute a regression to the
   last batch from one run per variant.
7. **Scope is sharp imagery.** A class is expected to work where a person can see the objects;
   soft or strongly oblique coverage (much of south-east Europe; individual sites like Leuna,
   Lingen, Puertollano) is outside the target, not a gap to close. Low coverage there is not a
   finding against the model, and no round is spent to improve it. Samples a soft site already
   contributed stay (rule 2); new effort goes to sharp sites.
8. **Site selection is a queue, not a search.** Score a batch of candidate sites, work
   top-down, defer those below a floor (0.65) until everything above is done. Imagery quality is
   what it is — but the floor is set by the reviewer: a site where a person can't reliably see
   the objects produces labels worse than none, and is deferred whatever its score.
9. **One round at a time.** Finish and read a round before starting the next.
10. **Promotion is explicit.** A class leaves `experiments/` by a deliberate move of its data and
   a config change in `app/server/`; nothing graduates as a side effect.

## 3. One round

All paths are under `<workspace>/loop/<class>/` (`experiments/loop/<class>/` for a new class, `loop/<class>/` once promoted). Commands are `python scripts/loop/<tool>.py
--class <class> ...`; on Windows prefix `PYTHONIOENCODING=utf-8`.

| step | who | command / action |
|---|---|---|
| 1. Pick the site | operator | `sites.py --list` — the top unsampled site by `sharpness_vs_train` at or above the floor. New candidates: scan a batch, then `sites.py --score`. |
| 2. Scan | operator | `scan.py --site "<unique substring>" --model vN --conf 0.25` — low threshold on purpose; the reviewer is the filter. |
| 3. Sweep | operator → reviewer | `pages.py sweep --site ... --model vN --windows 60`; publish `pages/sweep_*.html` as an Artifact with `capabilities: {db: {}, downloads: true}`. Reviewer draws a polygon on every real object **without** a green proposal and leaves proposals alone. |
| 4. Collect | operator | Read the page's store (Artifact `read_db`, collection `reviews`) or take the downloaded JSON; save as `reviews/<class>-sweep-<site>-vN.json`. |
| 5. Triage | operator → reviewer | `pages.py triage --site ... --model vN --min-conf 0.25 --swept-by <sweep json>`; publish the same way. Reviewer answers yes / no / unsure for each proposal. Collect as `reviews/<class>-triage-<site>-vN-swept.json`. |
| 5b. Triage the rest (large sites) | operator → reviewer | When the site has many more windows than were swept, `pages.py triage ... --swept-by <sweep json> --outside-sweep` shows the proposals in the unswept windows. Judge them the same way; collect as `reviews/<class>-triage-<site>-vN-outside.json`. These verdicts become samples and negatives but are **never** ground truth — they have no matching sweep. |
| 6. Measure | operator | `coverage.py --site ... --review <sweep> --extra-truth <triage> --models vN` — the fresh-site coverage of the incumbent. Record it in `loop.md`. |
| 7. Ingest | operator | `apply.py --review <sweep json>` then `apply.py --review <triage json>` (idempotent). Polygons and yeses become samples; noes become disabled negatives. |
| 8. Package | operator | `python scripts/obb.py --class <class> --hard-negatives` — the flag is what includes the enabled negatives. Check the printed train/val counts. |
| 9. Train two candidates | operator | A low-rate fine-tune of the incumbent: `python scripts/train_obb.py --class <class> --version vN+1 --base-model models/<class>_obb_vN.pt --epochs 20 --lr0 0.0002 --data-dir experiments/classes/<class>/dataset_obb`; and a from-scratch run on the same package (same command without `--base-model`/`--lr0`, ~10 min). Fine-tunes tend to win while the incumbent is young and lose once it is several fine-tunes deep (`distillation-column` v34–v36 lost, then the from-scratch v37 tripled the reliable detections); the gate decides, not the recipe. |
| 10. Gate | operator | `benchmark.py --models vN,vN+1` (section 4). Adopt or reject; log the table in `loop.md`. |
| 11. Add the site to the benchmark | operator | Append the sweep/triage pair to `benchmark.json` so every future candidate is measured on it too. |

The reviewer's share is steps 3 and 5: roughly ten minutes for 60 windows and 30–50 proposals.

## 4. The gate: `benchmark.py`

`benchmark.json` lists two things: **positives** — every swept site with its sweep and triage
files, so ground truth grows by one site per round — and **negatives** — sites of the kind the
class must *not* fire on. For each model version it prints, per positive site, hits and false
positives at ≥ 0.5 and the count at ≥ 0.7; per negative site, the count at ≥ 0.5 and ≥ 0.7 and
the maximum confidence.

The table that decides is the last one, **reliable detections**: for each version and each
false-positive budget (0, 2, 5 wrong boxes across all refineries), the lowest threshold that
silences every negative site and stays within the budget, and how many true detections survive
above it. This is the end game as the user set it on 2026-09-21 — *the
detections must be reliable at the chosen threshold; missing some is acceptable, a wrong
confident box is not.* A candidate is **adopted** when it keeps more true detections than the
incumbent in the FP ≤ 0 column (FP ≤ 2 as the tie-break); the threshold in that cell becomes its
operating threshold. Confidence scale does not matter here, only ranking — `v33` fires far more
than `v32` at 0.5 and lights three look-alikes, yet its top 18 detections are all real columns
on refineries (v32's top 4), so it wins. The fixed-threshold tables above are for seeing *where*
a version changed. `--dump <json>` saves the raw detections so other operating points can be
read off without re-running.

Negative layers, in the order they are being added:

1. **Factories** (`industrial=factory` OSM polygons) — in place. Six sites; no version other than
   the incumbent has kept all six under 0.7.
2. **Ports, tank terminals, power stations** — in place (`lookalikes` layer, ten sites from an
   Overpass export of `power=plant`, `landuse=harbour`, `industrial=port|oil|oil_storage`).
   These share storage tanks and chimneys with refineries, so only the columns tell them apart;
   this is exactly the discrimination the class is for. Check an export by name before merging
   — `industrial=oil` also tags refineries and crackers. `sites.py --geojson <file> --layer
   <name>` merges without touching the refinery list; scan once, add to `benchmark.json`.
3. **The site classifier itself** — in place: `scripts/eval_sites.py` runs the server's
   own detection and classification over every benchmark site and prints the verdict per site.
   This is the final word: when it and the object gate disagree, the site test wins (it did for
   `v46`, which scored lower on clean object detections but took the site test from 12/18 to
   18/18 refineries at 0/31 look-alikes). Detections are cached per tile, so graph parameters
   (`--floor`, `--count`, `--min-types`, `--max-distance-m`) are swept in seconds; only a new
   model needs a fresh ~45-minute detect. Always finish with a batch of look-alikes the model
   has never seen — the ones whose rejections it trained on are memorised, not generalised.

## 4b. The look-alike round

When the site test leaks on specific look-alikes, run the loop's negative mechanism at them:
scan the look-alike sites with the incumbent, put every proposal ≥ 0.4 on one multi-site triage
page (`pages.py triage --site <first> --also <others> --page-slug lookalikes`), enable all the
"no" verdicts, train both candidates, and judge by the site test. Chemical plants and crackers
are *not* look-alikes for a column class — their columns are real. A "yes" on a look-alike is a
sample like any other, and that site leaves the negatives list. Where the graph is concerned: a
refinery is all its components (4-of-4), a component can carry a `min_count` (fans ≥ 3), and
the radius is 300 m; those came out of this round and are the defaults now.

## 5. Adding a class from zero

Before the loop can run, a class needs a seed:

1. Draw 20–40 samples in `/manual` on port 8001 (`run-experiments.ps1`) across at least three
   sites; body only, generous crop margin. Add `classes/<class>/subclass_graph.json` with
   `{"nodes": {"<class>": {"normalize_sample_crop": true}}, "edges": []}` so every training crop
   is a fixed 80 m window at the same ground resolution.
2. `obb.py --class <class>` and `train_obb.py --version v1` from `yolo11n-obb.pt`.
3. Build `sites.json` from an OSM export of the sites the class lives on:
   `sites.py --geojson <file>`. Scan a batch, `--score`, and start the queue.
4. Create `benchmark.json` with an empty positives list and the negative sites, then run round
   one. From round two on, the previous round's sweep is a positive site.

Expect the first few rounds to look bad — coverage of 20–40 % and versions that swing. That is
the class finding its feet, not a verdict on the class.

## 6. What "the model is usable" looks like

For `distillation-column` after ten rounds: the incumbent finds half to two-thirds of the
columns on a refinery it has never seen, about half of its confident boxes are right, and it
puts at least one detection above 0.7 on every refinery scanned while no factory reaches 0.7.
That is enough to act as a "this looks like a refinery" signal in the site classifier, which
is the job. Per-object recall keeps improving through further rounds; it is not the bar for
wiring the class in.
