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

## Next steps
1. Confirm which of the "reliable" objects already have a usable pretrained detector
   (xView-style classes) vs. need custom labeled samples.
2. Pick 2-3 known real refineries, pull imagery, and manually catalog object counts/spacing
   to calibrate the proximity threshold and count ranges above instead of guessing them.
3. Implement clustering (proximity) + composition check (count ranges) as two separate,
   inspectable steps.
