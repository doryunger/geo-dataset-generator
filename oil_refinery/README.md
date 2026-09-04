# Oil refinery site detection

Goal: highlight oil refineries on a map using the compositional approach from the
`fence-face` discussion — detect discrete objects, cluster them spatially, and classify
a cluster as a refinery based on which objects co-occur, rather than trying to recognize
"refinery" as one end-to-end scene label.

## Candidate object list

Grouped by how feasible detection is at typical satellite resolution (large/high-contrast
objects are reliable; thin/low-contrast ones hit the same resolution ceiling as fences).

### Reliable — large, high-contrast, close to existing detector classes
- Storage tank (fixed-roof and floating-roof) — circular, large, distinctive; floating-roof
  tanks can also reveal fill level via shadow analysis
- Flare stack — tall thin vertical structure, often with a visible flame/smoke plume
- Distillation column / process tower — tall cylindrical vessels, clustered in the process unit
- Fan unit — grid-arranged circular fan housings (assumed air-cooled heat exchanger / "fin-fan"
  banks — confirm this is the intended structure before labeling)
- Cooling tower — large hyperboloid/rectangular structure, sometimes with a visible vapor plume
- Oil tanker (ship) — for refineries with a marine terminal
- Tanker truck / rail tank car — loading rack activity
- Crane — present during construction/turnarounds
- Generic vehicle (car/truck) — parking density as an activity proxy
- Building (admin/office)

### Out of scope for now — thin/linear features
Decided not to chase these at this stage: same legibility ceiling as fence-face, and no
existing dataset covers them anyway. Revisit only if the compositional approach (discrete
objects + counts) turns out to need them.
- Pipe racks / interconnecting piping
- Perimeter fencing

### Needs segmentation, not object detection
- Cooling water / wastewater treatment ponds — irregular water bodies
- Paved process-unit area vs. open ground — land-cover contrast

## Composition rule (draft)

Two separate parameters, not one blended score:

1. **Proximity threshold** — a global max distance between detections for them to count
   as "same facility" (e.g. DBSCAN `eps` over detection centroids). A hangar-like building
   1000m from a storage tank shouldn't cluster with it even if both are individually
   "refinery-plausible" objects.
2. **Per-class count range** — once a cluster exists, its object *counts* (not just
   presence/absence) decide whether the composition matches a refinery profile. E.g. a
   cluster with 10 hangars and 1 storage tank looks like a different site type even though
   "storage tank" is on the list; a cluster with 6+ storage tanks is a much stronger signal.

Both the distance threshold and the count ranges below are placeholders — they need to be
calibrated against a handful of real, known refineries (pull imagery, manually count/measure)
rather than guessed. Don't treat these numbers as anything but a starting point to test.

| Object | Plausible count range (draft, uncalibrated) |
|---|---|
| Storage tank | 6+ |
| Flare stack | 1+ |
| Distillation column / process tower | 1+ |
| Cooling tower | 0+ |
| Oil tanker (ship) | 0+ (only if marine terminal) |
| Tanker truck / rail tank car | 0+ |
| Crane | 0-few (not a defining object, just corroborating) |
| Building (admin/hangar-like) | low count relative to tanks — a high building:tank ratio should count against "refinery" |

## xView — tried and dropped (2026-09-04)

`oil_refinery/train_xview.py` trained a YOLOv3 model on the xView dataset (60 native classes,
later narrowed to a merged/reduced set), aiming to cover refinery-relevant classes beyond what
DOTAv1/DIOR already detect. Dropped: of the 8 xView classes actually relevant here (Truck
w/Liquid, Crane Truck, Tank car, Oil Tanker, Tower crane, Container Crane, Mobile Crane, Storage
Tank), only **Storage Tank** is a class this project actually needs — and DOTAv1 and DIOR already
detect it (`storage tank` id 2 / `storagetank` id 15, see `semantic_graph.md`'s pretrained-overlap
table). Reducing to just that one class and merging classes down from 60 both produced worse
results than the full 60-class baseline, and even a fully successful xView model would add no
class coverage this project doesn't already have. `train_xview.py`/`eval_xview_checkpoint.py` are
left in place but not part of the active plan.

## Detection sourcing (as of 2026-09-04)

No public pretrained model (DOTAv1, DIOR, xView, FAIR1M, NWPU VHR-10) covers distillation columns
or fan units — those are refinery-specific shapes no aerial-detection benchmark labels. Given
that, this project now does custom-label and train those two classes after all, using the same
`classes/<class>/` + `scripts/obb.py`/`train_obb.py` pipeline as everything else (see root
`CLAUDE.md`'s "compact/tactical classes" note). This reverses the earlier "no custom training for
oil-refinery components" stance. `chimney` is the one exception: a custom-trained
`classes/chimney/` model already exists and performs well, but this project deliberately keeps
using DIOR's pretrained `chimney` class for it in production instead, imperfect as DIOR's chimney
detections are.

## Next steps
1. Storage tank, oil tanker (ship), and vehicle already have usable pretrained coverage via
   DOTAv1/DIOR. Distillation column and fan unit are being custom-trained (see "Detection
   sourcing" above). Flare stack, cooling tower, tanker truck/rail tank car, crane, and building
   still have no confirmed detector — check each against DOTAv1/DIOR's own class lists before
   assuming a gap.
2. Pick 2-3 known real refineries, pull imagery, and manually catalog object counts/spacing
   to calibrate the proximity threshold and count ranges above instead of guessing them.
3. Implement clustering (proximity) + composition check (count ranges) as two separate,
   inspectable steps.
