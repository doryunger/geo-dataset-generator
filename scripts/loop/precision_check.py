import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "app" / "server"))

import loop_common as L
import common
import geometry
import sites as app_sites


def _crop(z: int, x: int, y: int, corners: list) -> str:
    tile = Image.open(common.fetch_tile(z, x, y))
    scale = 256 / tile.size[0]
    big = Image.new("RGB", (768, 768))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            big.paste(Image.open(common.fetch_tile(z, x + dx, y + dy)).convert("RGB").resize((256, 256)), ((dx + 1) * 256, (dy + 1) * 256))
    pts = [(256 + px * scale, 256 + py * scale) for px, py in corners]
    cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
    ImageDraw.Draw(big).line(pts + [pts[0]], fill=(255, 214, 64), width=2)
    crop = big.crop((int(cx - 75), int(cy - 75), int(cx + 75), int(cy + 75))).resize((300, 300))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _global_box(z: int, x: int, y: int, corners: list) -> Polygon:
    return Polygon([geometry.global_pixel(x, y, px, py) for px, py in corners])


def check_sites(class_name: str, n_sites: int) -> list[dict]:
    cfg = json.loads((L.loop_dir(class_name) / "benchmark.json").read_text(encoding="utf-8"))
    by_id = {s["osm_id"]: s for s in L.load_sites(class_name)}
    ids = sorted(cfg.get("held_out_eval") or [h["osm_id"] for h in cfg.get("held_out", [])])
    return [by_id[i] for i in random.Random(0).sample(ids, min(n_sites, len(ids)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--models", required=True, help="comma-separated version tags, e.g. v55,v59")
    parser.add_argument("--caches", required=True, help="comma-separated eval_sites.py detection caches, one per model")
    parser.add_argument("--floor", type=float, default=0.65)
    parser.add_argument("--sites", type=int, default=9, help="held-out sites sampled (fixed seed, so every model sees the same ones)")
    parser.add_argument("--per-site", type=int, default=4, help="most confident detections kept per site and model")
    parser.add_argument("--summary", type=Path, default=None, help="the page's saved decisions JSON: print precision per model instead of building")
    args = parser.parse_args()
    models = args.models.split(",")
    slug = "-".join(models)
    meta_path = L.loop_dir(args.class_name) / "pages" / f"check_heldout_{slug}.json"

    if args.summary:
        decisions = json.loads(args.summary.read_text(encoding="utf-8"))["decisions"]
        items = json.loads(meta_path.read_text(encoding="utf-8"))
        for m in models:
            judged = [decisions.get(i["id"]) for i in items if m in i["models"] and decisions.get(i["id"]) in ("yes", "no", "neg")]
            yes = sum(1 for v in judged if v == "yes")
            print(f"{m}: {yes} real / {len(judged)} judged -> precision {yes / max(1, len(judged)):.2f}")
        return

    caches = [json.loads(Path(c).read_text()) for c in args.caches.split(",")]
    cands, meta = [], []
    for site in check_sites(args.class_name, args.sites):
        picked = []
        for model, cache in zip(models, caches):
            found = [
                (d, (z, x, y)) for z, x, y in app_sites.site_tiles(site) for d in cache.get(common.tile_id(z, x, y), [])
                if d["class_name"] == args.class_name and d["confidence"] >= args.floor
            ]
            for d, (z, x, y) in sorted(found, key=lambda f: -f[0]["confidence"])[: args.per_site]:
                box = _global_box(z, x, y, d["corners"])
                same = next((p for p in picked if L.iou(p["box"], box) >= 0.3), None)
                if same:
                    same["models"][model] = round(d["confidence"], 3)
                else:
                    picked.append({"box": box, "tile": (z, x, y), "corners": d["corners"], "models": {model: round(d["confidence"], 3)}})
        for p in picked:
            cid = f"{L.slug(site['name'])}_{len(cands):04d}"
            cands.append({
                "id": cid, "source_sample": site["name"], "conf": max(p["models"].values()), "size_m": 0.0,
                "lat": site["lat"], "lon": site["lon"], "polygon": [], "img": _crop(*p["tile"], p["corners"]),
            })
            meta.append({"id": cid, "site": site["name"], "models": p["models"]})
    label = args.class_name.replace("-", " ")
    html = L.fill("triage.html", {
        "TITLE": f"{label} held-out precision check {', '.join(models)}",
        "EYEBROW": f"{args.class_name} &middot; models {', '.join(models)} &middot; {args.sites} held-out refineries &middot; top {args.per_site} per site &middot; conf &ge; {args.floor:.2f}",
        "HEADING": "Is this a real column?",
        "LEDE": "The most confident detections of each model on a fixed sample of held-out refineries; a spot both models found appears once. Mark each real or not. This only measures precision: nothing here is ever labelled or trained on, so the held-out sites stay unseen.",
        "YES": label.capitalize(), "NO": f"Not a {label}", "YES_SHORT": label, "NO_SHORT": "not",
        "DOC": f"reviews/{args.class_name}-check-heldout-{slug}", "LS": f"{args.class_name}-check-heldout-{slug}",
        "META": f'kind: "precision-check", class: "{args.class_name}", model: "{args.models}", threshold: {args.floor}',
        "FILENAME": f"{args.class_name}-check-heldout-{slug}.json",
        "POOL_ON": "0", "POSITIVES": "0", "TRAINING_ONLY": "hidden",
    }, cands)
    out = L.loop_dir(args.class_name) / "pages" / f"check_heldout_{slug}.html"
    out.write_text(html, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    per_model = {m: sum(1 for x in meta if m in x["models"]) for m in models}
    print(f"precision check: {len(cands)} spots on {len({x['site'] for x in meta})} sites, per model {per_model} -> {out}")


if __name__ == "__main__":
    main()
