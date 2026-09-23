import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "loop"))
sys.path.insert(0, str(REPO_ROOT / "app" / "server"))

for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

import common  # noqa: E402
import classifier  # noqa: E402
import site_graph  # noqa: E402
import tile_server  # noqa: E402
import loop_common as L  # noqa: E402
from shapely.geometry import shape  # noqa: E402

Z = tile_server.DETECT_ZOOM


def site_tiles(site: dict) -> list[tuple[int, int, int]]:
    geom = shape(site["geometry"])
    w, s, e, n = geom.bounds
    x0, y0 = common.lonlat_to_tile(w, n, Z)
    x1, y1 = common.lonlat_to_tile(e, s, Z)
    out = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            b = common.tile_bounds(Z, x, y)
            from shapely.geometry import box
            if geom.intersects(box(b["west"], b["south"], b["east"], b["north"])):
                out.append((Z, x, y))
    return out


def detect(tiles: list[tuple[int, int, int]], batch: int, cache: dict) -> dict:
    by_tile = {}
    todo = []
    for t in tiles:
        key = common.tile_id(*t)
        if key in cache:
            if cache[key]:
                by_tile[t] = cache[key]
        else:
            todo.append(t)
    tiles = todo
    for i in range(0, len(tiles), batch):
        jobs = []
        for z, x, y in tiles[i:i + batch]:
            jobs.append(tile_server.Job(
                tile_id=common.tile_id(z, x, y), z=z, x=x, y=y,
                image_bytes=common.fetch_tile(z, x, y).read_bytes(), request=None,
                has_interactive_request=False, fetch_ms=0.0, enqueued_at=0.0, future=None,
            ))
        for job, (_, dets) in zip(jobs, tile_server._run_detection_batch(jobs)):
            cache[job.tile_id] = dets
            if dets:
                by_tile[(job.z, job.x, job.y)] = dets
    return by_tile


def verdict(by_tile: dict, graph: dict) -> dict:
    if not by_tile:
        return {"identified": False, "matched": [], "sites": 0}
    lats = []
    for (z, x, y) in by_tile:
        b = common.tile_bounds(z, x, y)
        lats.append((b["north"] + b["south"]) / 2)
    results = classifier.classify(by_tile, Z, sum(lats) / len(lats), graph)
    matched = sorted({t for r in results for t in r["matched_types"]})
    return {"identified": bool(results), "matched": matched, "sites": len(results)}


def without(graph: dict, component: str) -> dict:
    g = json.loads(json.dumps(graph))
    g["edges"] = [e for e in g["edges"] if e.get("to") != component and e.get("from") != component]
    g["nodes"].pop(component, None)
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="loop class whose benchmark.json names the sites")
    parser.add_argument("--without", default=None, help="component to remove for the comparison column")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--only", default=None, help="substring filter on site names")
    parser.add_argument("--site", action="append", default=[], metavar="NAME", help="evaluate this site from the loop's sites.json instead of the benchmark list (repeatable); reported as kind 'probe'")
    parser.add_argument("--cache", type=Path, default=None, help="JSON file of per-tile detections; reused when present so graph changes re-classify without re-detecting")
    parser.add_argument("--max-distance-m", type=float, default=None, help="override every site's default_max_distance_m and merge_distance_m")
    parser.add_argument("--floor", action="append", default=[], metavar="COMPONENT=CONF", help="override a required component's min_confidence (repeatable); the cache must have been built with a floor at or below it")
    parser.add_argument("--min-types", type=int, default=None, help="override min_types_present")
    parser.add_argument("--count", action="append", default=[], metavar="COMPONENT=N", help="override a required component's min_count (repeatable)")
    args = parser.parse_args()

    graph = site_graph.load_graph()
    for spec in args.floor:
        comp, _, conf = spec.partition("=")
        for e in graph["edges"]:
            if e.get("to") == comp:
                e["min_confidence"] = float(conf)
    for spec in args.count:
        comp, _, n = spec.partition("=")
        for e in graph["edges"]:
            if e.get("to") == comp:
                e["min_count"] = int(n)
    if args.min_types is not None:
        for cfg_node in graph["nodes"].values():
            if cfg_node["kind"] == "site":
                cfg_node["min_types_present"] = args.min_types
    if args.max_distance_m is not None:
        for cfg_node in graph["nodes"].values():
            if cfg_node["kind"] == "site":
                cfg_node["default_max_distance_m"] = args.max_distance_m
                cfg_node["merge_distance_m"] = args.max_distance_m
    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}
    cfg = json.loads((L.loop_dir(args.class_name) / "benchmark.json").read_text(encoding="utf-8"))
    names = [("refinery", p["site"]) for p in cfg["positives"]] + [("look-alike", n) for n in cfg["negatives"]]
    if args.site:
        names = [("probe", n) for n in args.site]
    if args.only:
        names = [(k, n) for k, n in names if args.only.lower() in n.lower()]

    async def load():
        if all(common.tile_id(*t) in cache for _, n in names for t in site_tiles(L.find_site(args.class_name, n))):
            run()
            return
        async with tile_server.lifespan():
            run()

    def run():
        print(f"{'kind':<10}{'site':<32}{'tiles':>6}  {'verdict':<11}{'matched types':<58}" + (f"{'without ' + args.without:<12}" if args.without else ""))
        for kind, name in names:
            site = L.find_site(args.class_name, name)
            tiles = site_tiles(site)
            by_tile = detect(tiles, args.batch, cache)
            if args.cache:
                args.cache.write_text(json.dumps(cache))
            v = verdict(by_tile, graph)
            best = {}
            for dets in by_tile.values():
                for d in dets:
                    best[d["class_name"]] = max(best.get(d["class_name"], 0.0), d["confidence"])
            row = f"{kind:<10}{site['name'][:31]:<32}{len(tiles):>6}  {('REFINERY' if v['identified'] else '-'):<11}{', '.join(v['matched']):<58}"
            if args.without:
                w = verdict(by_tile, without(graph, args.without))
                row += f"{('REFINERY' if w['identified'] else '-'):<12}"
            print(row + "  max: " + ", ".join(f"{k}={c:.2f}" for k, c in sorted(best.items())), flush=True)

    asyncio.run(load())


if __name__ == "__main__":
    main()
