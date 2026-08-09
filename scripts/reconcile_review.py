#!/usr/bin/env python3
"""
CLI wrapper over reconcile.reconcile(): diffs review/round_NNN/ against its manifest (files the
user deleted are rejects, files remaining are keeps) and applies the result.

Usage:
    python scripts/reconcile_review.py --class fence --round 1
"""
import argparse
import json

import common
import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--round", required=True, type=int)
    args = parser.parse_args()

    round_dir = common.review_dir(args.class_name) / f"round_{args.round:03d}"
    manifest_path = round_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest found at {manifest_path} — did you run find_candidates.py for this round?")

    manifest = json.loads(manifest_path.read_text())
    kept_tile_ids = [entry["tile_id"] for entry in manifest if (round_dir / entry["filename"]).exists()]

    result = reconcile.reconcile(args.class_name, args.round, kept_tile_ids)

    print(f"Round {args.round}: {len(result['confirmed'])} confirmed, {len(result['rejected'])} rejected.")
    n_labeled = len(result["labeled_paths"])
    n_unlabeled = len(result["copied_paths"]) - n_labeled
    if result["copied_paths"]:
        print(f"\n{n_labeled} confirmed image(s) got an auto-generated label (FastSAM guess) — spot-check these, "
              f"don't blindly trust them.")
        if n_unlabeled:
            print(f"{n_unlabeled} confirmed image(s) got NO auto-label (FastSAM found nothing) — these still need "
                  f"manual tracing in CVAT, exported into the matching dataset/labels/{{train,val}}/ path.")


if __name__ == "__main__":
    main()
