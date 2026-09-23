# geo-dataset-generator

Finding a kind of *site* (here, oil refineries) in satellite imagery. The site is not detected as
a whole. Instead, the repo detects the objects a site is made of and reasons about how they sit
together. It has two parts:

1. **A semantic graph that turns detections into a site verdict.** Object detectors become
   nodes, a site type is a rule over them, and you tune the rule without retraining.
2. **A faster way to train a detector for a new object class.** The graph needed objects that no
   existing model detects, so we had to build our own. The model proposes, a person judges, and
   each change is kept or dropped based on a measured number.

The demo map in [`app/`](app/) shows both parts working together. This is a proof of concept.
It shows that the approach and its tooling work end to end. It is not a production refinery
detector.

## Part 1: giving detections a meaning

A detector says "tank at 0.9 here". It does not say "this is a refinery". Storage tanks are also
found at ports, tank farms and power stations. The semantic graph
([`app/server/semantic_graph.json`](app/server/semantic_graph.json)) holds that knowledge as
data:

```json
"oil_refinery": { "kind": "site", "min_types_present": 3, "default_max_distance_m": 300 },
"edges": [
  { "relation": "requires", "from": "oil_refinery", "to": "storage tank",        "min_confidence": 0.75 },
  { "relation": "requires", "from": "oil_refinery", "to": "fan-unit",            "min_confidence": 0.7 },
  { "relation": "requires", "from": "oil_refinery", "to": "distillation-column", "min_confidence": 0.65, "min_count": 3 }
]
```

- **Components** are nodes. Each one is fed by any detector, pretrained (storage tank, from
  DOTAv1) or custom-trained (fan-unit, distillation-column, from Part 2).
- A **site** is a rule over its components: which ones, at what confidence, how many, and how
  close together. `fan-unit` also counts only in groups (`group_within_m: 20`), because a
  single fan is not evidence.
- The site's outline is not drawn by hand. It is the region where the required components
  cluster within the distance threshold.
- The rule is tuned **offline and without retraining**. Detections are cached per tile once,
  and every benchmark site is then re-scored under new floors, counts or distances in seconds.

**Evidence.** The same detectors first called almost every industrial site a refinery. Changing
only the graph, plus one targeted round of look-alike negatives, brought that to **16 of 18
refineries identified and 0 of 39 look-alikes** (power stations, ports, tank farms,
steelworks, chemical plants) on the 57-site benchmark. Design history:
[docs/semantic-graph.md](docs/semantic-graph.md). Threshold decisions:
[app/server/context/server.md](app/server/context/server.md).

## Part 2: training the classes we were missing

The graph only works if every component it needs has a detector. The pretrained aerial models
(DOTA, DIOR, xView) cover storage tanks, ships and vehicles, but they don't cover the objects
that actually separate a refinery from a tank farm or a power station: distillation columns and
fin-fan cooler banks. No public model detects them, so we had to create our own.

Hand-labelling enough examples of a new class from scratch is the usual cost. The loop cuts
that cost down to about ten minutes of review per round:

```
seed ~20-40 hand-drawn samples -> train v1
  └─ each round, one unseen refinery site:
       scan      model proposes boxes over the whole site (low threshold, on purpose)
       sweep     reviewer draws what the model missed        -> measures coverage
       triage    reviewer marks each proposal yes / no       -> yes = sample, no = hard negative
       apply     judgements become training data (nothing is ever dropped on suspicion)
       train     a candidate model on the grown dataset
       gate      candidate replaces the incumbent only if it wins on the benchmark
```

The **gate** is the measurement built into the loop. Every version is adopted or rejected on a
recorded number: coverage on a site it has never seen, and whether its confident detections
land on refineries and not on look-alikes. The number is never a feeling about one example.
Hard negatives are added in proportion to positives, because too many collapsed a small model
(the evidence is in [loop.md](scripts/loop/context/loop.md)).

**Evidence.** `distillation-column` started from 22 hand-drawn samples. After 21 rounds it has
476 samples, and most of them came from the model's own proposals. For example, in round 6
(Scholven, a site the model had never seen) it found 29 of the 32 columns there, with 79 %
precision at confidence ≥ 0.5. `fan-unit` was built the same way (382 samples).

Full process: [docs/training-a-new-class.md](docs/training-a-new-class.md). Round-by-round log
and every measurement: [scripts/loop/context/loop.md](scripts/loop/context/loop.md).

## The demo app: both parts on one map

- **Site panel** (top left): seven refineries and seven look-alikes. Pick one and the map fits
  the site. Every zoom-17 tile inside it goes through all three detectors live on the GPU,
  about 15 s for a large refinery. Nothing is precomputed or cached between runs, because the
  point is to show the pipeline working.
- **Boxes on the map** show Part 2: the custom classes firing next to the pretrained one.
  Detections that pass their confidence floor but don't count toward a rule (a lone fan) are
  drawn dashed.
- **Graph widget** (bottom) shows Part 1. A component node turns yellow when it fires and
  green when its requirement is met. The *oil refinery* node turns green only when the whole
  rule holds. The outlined area comes from the detections themselves.
- **Verdict**: each site in the panel turns green ("oil refinery") or red ("not a refinery").
  Look-alikes light up individual components but not the parent, and that difference is the
  point of the demo.

A guided tour runs after the first site. Free panning also works: at zoom ≥ 16 the view is
classified live.
