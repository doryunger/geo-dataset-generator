# search.py

Core discovery loop, shared by the CLI (`find_candidates.py`) and the web API (`api.py`). Fetches
tiles outward (ring expansion) from a seed location, embeds each, and collects the first N whose
similarity clears a threshold against the seed AND/OR (for an existing class) every previously
confirmed example of that class — whichever similarity is higher wins, so distinct visual variants
of a class each stay matchable instead of blurring into one average. Never re-fetches/re-suggests
a tile already recorded in that class's registry.

Raw tiles and their DINOv2 embeddings are cached globally (`common.TILE_IMAGES_DIR` /
`common.INDEX_PATH`) and shared across every class — a tile already embedded for one class costs
nothing extra when a different class's search reaches it. What stays per-class is the
accept/reject registry and each accepted candidate's guessed label (see `auto_labeler.py`), since
both depend on that class's own seed/exemplars, not on the tile itself.

Each accepted candidate also gets an auto-guessed label polygon from DINOv2's own per-patch tokens
— a real guess, not ground truth, which is why review stays in the loop. The seed itself needs no
guessing: its polygon is exactly what the user drew, so it's saved straight into the dataset
without going through review at all.

`PERSIST_EVERY = 20`: flush registry/index every N tile fetches, not just at the end, so a long
run doesn't lose everything on a crash/interrupt.

## _ring_search

Accepts a tile once both gates clear: whole-tile CLS similarity >= threshold, AND the
patch-labeler finds a confident, spatially-concentrated match. Whole-tile similarity alone is
dominated by broad scene content at these tile sizes, not by whether the object is present.
Shared by `run_search` (production rounds) and `run_validation` (read-only assessment) — they
differ only in where `query_matrix`/`is_excluded`/`stage_candidate` come from, not in this loop.

`stage_candidate`: optional callable(tid, candidate_path, label_polygons) -> None, for side
effects on accept (review-folder staging, labels.jsonl, etc.). `on_evaluated`: optional
callable(tid, accepted: bool) -> None, called for every tile that clears the exclusion check and
gets embedded+compared, whether accepted or not.

## _save_seed_to_dataset

The drawn shape is exact ground truth, not a guess — saved straight into the dataset, no review
needed. Always `train` split: it's a single unique example, not part of a batch of same-region
tiles where geographic leakage between train/val would be a concern.

## run_search

`bbox`: when given, crops to exactly what was drawn rather than using the whole grid tile that
happens to contain its center (precise reference image) — still fetches+registers the containing
grid tile too, purely so ring search never re-offers it as a candidate later.

Multi-seed matching: every previously confirmed tile of this class becomes an additional query
vector, so later rounds benefit from everything confirmed so far, not just the new seed click.

`get_or_embed` GSD-normalizes (`common.resample_to_target_gsd`) before embedding — this cache is
shared with `run_validation`'s own `get_or_embed`, so both must embed a given tile_id identically.

## run_validation

Read-only assessment (see the `/manual` page): how good are this class's current hand-drawn
samples at finding more of the same thing nearby? Uses every sample's embedding as the query set
instead of one seed, and never writes to `registry.jsonl` (no round/seed bookkeeping) — repeatable
any time without side effects on production search state. Still reads/writes the shared global
tile+embedding cache, since that's reusable infra regardless of purpose.
