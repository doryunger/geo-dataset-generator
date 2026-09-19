"""
See and control which groups of samples and hard negatives train the next model. A group is
everything that shares a provenance key `<source>:<site>:<model>` (hand-drawn samples are `hand`).
Disabling is reversible and only affects the next package generation; nothing is deleted.

Usage:
    python scripts/loop/groups.py --class distillation-column
    python scripts/loop/groups.py --class distillation-column --disable "loop-sweep:refineria_exolum_de_puer:v13"
    python scripts/loop/groups.py --class distillation-column --negatives --enable "loop-triage-rejected:bp_raffinaderij_rotterdam:v16"
    python scripts/loop/groups.py --class distillation-column --versions v15,v16,v17
"""
import argparse
import json

import loop_common  # noqa: F401
import common
import obb


def _rows(class_name: str, negatives: bool) -> list[dict]:
    return common.load_hard_negatives(class_name) if negatives else common.load_samples(class_name)


def _save(class_name: str, negatives: bool, rows: list[dict]) -> None:
    common.rewrite_jsonl(common.hard_negatives_path(class_name) if negatives else common.samples_path(class_name), rows)


def set_enabled(class_name: str, negatives: bool, key: str, enabled: bool, limit: int | None = None) -> int:
    rows = _rows(class_name, negatives)
    hit = 0
    for r in rows:
        if obb.group_key(r) != key:
            continue
        if limit is not None and hit >= limit:
            break
        r["enabled"] = enabled
        hit += 1
    if not hit:
        raise SystemExit(f"no {'negatives' if negatives else 'samples'} in group {key!r}")
    _save(class_name, negatives, rows)
    return hit


def show(class_name: str) -> None:
    g = obb.data_groups(class_name)
    for kind in ("samples", "negatives"):
        print(f"{kind}:")
        for key, c in sorted(g[kind].items(), key=lambda kv: -kv[1]["total"]):
            flag = "" if c["enabled"] == c["total"] else ("  (off)" if c["enabled"] == 0 else f"  ({c['enabled']} on)")
            print(f"  {c['total']:>4}  {key}{flag}")
        tot = sum(c["total"] for c in g[kind].values())
        on = sum(c["enabled"] for c in g[kind].values())
        print(f"  {'':>4}  {on}/{tot} enabled")


def compare(class_name: str, versions: list[str]) -> None:
    slug = common.class_slug(class_name)
    rows = {}
    for v in versions:
        p = common.MODELS_DIR / f"{slug}_obb_{v}_metrics.json"
        g = (json.loads(p.read_text()).get("groups") if p.exists() else None) or {}
        for kind in ("samples", "negatives"):
            for key, c in (g.get(kind) or {}).items():
                rows.setdefault((kind, key), {})[v] = c["enabled"]
    print(f"{'group':<58}" + "".join(f"{v:>7}" for v in versions))
    for (kind, key), per in sorted(rows.items()):
        print(f"{kind[:3] + ' ' + key:<58}" + "".join(f"{per.get(v, '-'):>7}" for v in versions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--negatives", action="store_true", help="act on hard negatives instead of samples")
    parser.add_argument("--enable", metavar="GROUP")
    parser.add_argument("--disable", metavar="GROUP")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows of the group")
    parser.add_argument("--versions", help="comma-separated: show enabled counts per group per trained version")
    args = parser.parse_args()
    if args.enable or args.disable:
        key = args.enable or args.disable
        n = set_enabled(args.class_name, args.negatives, key, bool(args.enable), args.limit)
        print(f"{'enabled' if args.enable else 'disabled'} {n} in {key}")
    if args.versions:
        compare(args.class_name, args.versions.split(","))
        return
    show(args.class_name)


if __name__ == "__main__":
    main()
