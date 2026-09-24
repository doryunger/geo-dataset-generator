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
import sites as app_sites  # noqa: E402
import tile_server  # noqa: E402
import ws_server  # noqa: E402
import loop_common as L  # noqa: E402

Z = tile_server.DETECT_ZOOM


def site_tiles(site: dict) -> list[tuple[int, int, int]]:
    return app_sites.site_tiles(site)


async def detect(tiles: list[tuple[int, int, int]], cache: dict) -> dict:
    todo = [t for t in tiles if common.tile_id(*t) not in cache]
    if todo:
        tile_server.forget(todo)
        await ws_server._prefetch_with_ring(todo)
        results = await asyncio.gather(
            *(tile_server.get_or_process_detections(z, x, y, force_all_models=True) for z, x, y in todo)
        )
        for t, dets in zip(todo, results):
            cache[common.tile_id(*t)] = dets or []
    return {t: cache[common.tile_id(*t)] for t in tiles if cache[common.tile_id(*t)]}


def verdict(by_tile: dict, tiles: list[tuple[int, int, int]], graph: dict) -> dict:
    if not by_tile:
        return {"identified": False, "matched": [], "sites": 0}
    results = classifier.classify(by_tile, Z, ws_server._ref_lat(set(tiles)), graph)
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
    parser.add_argument("--only", default=None, help="substring filter on site names")
    parser.add_argument("--held-out", action="store_true", help="evaluate only benchmark.json's held_out refineries, which are never labelled")
    parser.add_argument("--site", action="append", default=[], metavar="NAME", help="evaluate this site from the loop's sites.json instead of the benchmark list (repeatable); reported as kind 'probe'")
    parser.add_argument("--cache", type=Path, default=None, help="JSON file of per-tile detections; reused when present so graph changes re-classify without re-detecting")
    parser.add_argument("--max-distance-m", type=float, default=None, help="override every site's default_max_distance_m and merge_distance_m")
    parser.add_argument("--floor", action="append", default=[], metavar="COMPONENT=CONF", help="override a required component's min_confidence (repeatable), also when detecting; a cache built at a higher floor is missing the detections in between")
    parser.add_argument("--min-types", type=int, default=None, help="override min_types_present")
    parser.add_argument("--count", action="append", default=[], metavar="COMPONENT=N", help="override a required component's min_count (repeatable)")
    args = parser.parse_args()

    graph = site_graph.load_graph()
    for spec in args.floor:
        comp, _, conf = spec.partition("=")
        for e in graph["edges"]:
            if e.get("to") == comp:
                e["min_confidence"] = float(conf)
        tile_server._COMPONENT_MIN_CONFIDENCE[comp] = min(tile_server._COMPONENT_MIN_CONFIDENCE.get(comp, 1.0), float(conf))
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
    by_id = {s["osm_id"]: s for s in L.load_sites(args.class_name)}
    held_out = [("held-out", by_id[h["osm_id"]]) for h in cfg.get("held_out", [])]
    named = [("refinery", p["site"]) for p in cfg["positives"]] + [("look-alike", n) for n in cfg["negatives"]]
    if args.site:
        named = [("probe", n) for n in args.site]
    targets = [(k, L.find_site(args.class_name, n)) for k, n in named]
    if args.held_out:
        targets = held_out
    elif not args.site:
        targets = targets + held_out
    if args.only:
        targets = [(k, s) for k, s in targets if args.only.lower() in s["name"].lower()]

    async def run():
        if not all(common.tile_id(*t) in cache for _, s in targets for t in site_tiles(s)):
            async with tile_server.lifespan():
                while not tile_server._state.get("warm"):
                    await asyncio.sleep(0.5)
                await report()
        else:
            await report()

    async def report():
        print(f"{'kind':<10}{'site':<32}{'tiles':>6}  {'verdict':<11}{'matched types':<58}" + (f"{'without ' + args.without:<12}" if args.without else ""))
        tally: dict[str, list[int]] = {}
        for kind, site in targets:
            tiles = site_tiles(site)
            by_tile = await detect(tiles, cache)
            if args.cache:
                args.cache.write_text(json.dumps(cache))
            v = verdict(by_tile, tiles, graph)
            hit = tally.setdefault(kind, [0, 0])
            hit[0] += v["identified"]
            hit[1] += 1
            best = {}
            for dets in by_tile.values():
                for d in dets:
                    best[d["class_name"]] = max(best.get(d["class_name"], 0.0), d["confidence"])
            row = f"{kind:<10}{site['name'][:31]:<32}{len(tiles):>6}  {('REFINERY' if v['identified'] else '-'):<11}{', '.join(v['matched']):<58}"
            if args.without:
                w = verdict(by_tile, tiles, without(graph, args.without))
                row += f"{('REFINERY' if w['identified'] else '-'):<12}"
            print(row + "  max: " + ", ".join(f"{k}={c:.2f}" for k, c in sorted(best.items())), flush=True)
        print("identified: " + ", ".join(f"{k} {n}/{total}" for k, (n, total) in tally.items()))

    asyncio.run(run())

if __name__ == "__main__":
    main()
