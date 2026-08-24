# train_obb_kfold.py

K-fold cross-validation over an OBB class's *original samples* (not pieces) — trains K models,
each with a different 1/K of samples held out as val, and reports mean±std of each metric across
the K runs instead of one run's number.

Exists because a single train/val split is noisy at this sample count — two runs on identical
data landed at precision 0.32 vs 0.045 (`fence_obb_v2` vs `v3`) purely from random init. That's a
statement about *variance*, and a single held-out split can't measure variance; only repeated
runs over different splits can.

Folding happens over `samples.jsonl`'s ids, not `dataset_obb` pieces — `obb.generate_obb_package`
already keeps every piece of one sample on the same side of a split (adjacent pieces share
near-identical background, so splitting them would leak). Fold assignment is a deterministic
shuffle (fixed seed) so reruns are reproducible and comparable.

The S3 pull/push happens only at this file's `main()` and `train_obb.py`'s `main()`, not inside
`run_kfold`/`train_obb_class` themselves — `run_kfold` calls `generate_obb_package`/
`train_obb_class` once per fold against a fold-specific local split, and pulling/pushing mid-fold
would defeat the fold split entirely.
