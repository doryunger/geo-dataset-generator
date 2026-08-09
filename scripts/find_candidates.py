#!/usr/bin/env python3
"""
CLI wrapper over search.run_search(): fetch tiles outward from a seed tile URL, copy accepted
candidates into review/round_NNN/ for delete-to-reject review.

Usage:
    python scripts/find_candidates.py --class fence --seed seeds/seed_001.yaml --round 1
"""
import argparse
import json
import shutil

import yaml

import common
import search
from auto_labeler import PatchLabeler
from embedder import Embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--seed", required=True, help="Path to a seed_*.yaml file (tile_url)")
    parser.add_argument("--round", required=True, type=int, help="Review round number, e.g. 1")
    parser.add_argument("--n", type=int, default=10, help="Number of candidates to collect (default 10)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Cosine similarity acceptance bar")
    parser.add_argument(
        "--max-fetches", type=int, default=3000,
        help="Safety cap on tiles fetched this run, to bound API cost if the area has no matches "
             "(default 3000; raise if you want it to search further)",
    )
    args = parser.parse_args()

    seed = yaml.safe_load(open(args.seed))
    embedder = Embedder()
    auto_labeler = PatchLabeler(embedder)

    result = search.run_search(
        args.class_name, embedder, auto_labeler, args.round,
        tile_url=seed["tile_url"], n=args.n, threshold=args.threshold, max_fetches=args.max_fetches,
    )

    print(f"Seed tile z={result.z} x={result.x0} y={result.y0}  (~lon={result.lon:.5f}, lat={result.lat:.5f})")
    print(f"Effective resolution at this zoom/latitude: ~{result.meters_per_pixel:.2f} m/pixel")
    print(f"Using {result.exemplar_count} query exemplar(s) (seed + previously confirmed for this class)")
    for c in result.candidates:
        has_label = "auto-labeled" if c.get("label_polygon") else "no auto-label (needs manual)"
        print(f"  accepted {c['tile_id']}  sim={c['similarity']:.3f}  [{has_label}]")

    if result.seed_added_to_dataset:
        print("Seed shape saved straight to the dataset (exact label, no review needed).")

    if result.stopped_reason == "max_fetches":
        print(f"\nHit --max-fetches={args.max_fetches} without finding {args.n} candidates "
              f"(found {len(result.candidates)}). Area may be sparse, or lower --threshold.")

    if not result.candidates:
        print("No candidates found.")
        return

    round_dir = common.review_dir(args.class_name) / f"round_{args.round:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in result.candidates:
        src = next(common.TILE_IMAGES_DIR.glob(f"{c['tile_id']}.*"))
        dst = round_dir / src.name
        shutil.copy(src, dst)
        manifest.append({"tile_id": c["tile_id"], "filename": src.name, "similarity": c["similarity"]})
    (round_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(result.candidates)} candidates copied to {round_dir}")
    print("Review them now: delete the ones that are NOT the target object, leave the rest.")
    print(f"Then run: python scripts/reconcile_review.py --class {args.class_name} --round {args.round}")


if __name__ == "__main__":
    main()
