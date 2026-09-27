"""
Build the two review pages of the loop as self-contained HTML files.

    sweep:  one window per screen with the model's proposals drawn; polygon every real object the
            model has NO proposal on (the misses)
    triage: one proposal per screen, yes/no; with --swept-by <sweep json> only the proposals inside
            the swept windows that are not on a drawn polygon (the found-or-not)

    coverage of the model on that site = yeses / (yeses + polygons); precision = yeses / (yeses + noes)

Usage:
    python scripts/loop/pages.py sweep  --class distillation-column --site scholven --model v16 --windows 60 --show-conf 0.25
    python scripts/loop/pages.py triage --class distillation-column --site scholven --model v16 --min-conf 0.25 --swept-by ~/Downloads/<sweep>.json

Publish the resulting file (scripts/loop/pages/<name>.html) as an Artifact, review it, then use the
page's "Download JSON" button and feed the file to apply.py / coverage.py.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import loop_common as L
import common


def _candidates(class_name: str, site: dict, version: str) -> list[dict]:
    p = L.loop_dir(class_name) / "candidates" / f"{L.slug(site['name'])}_{version}.json"
    if not p.exists():
        raise SystemExit(f"no scan for {site['name']} with {version}; run scan.py first")
    return json.loads(p.read_text())


def _restrict_to_sweep(class_name: str, site: dict, cands: list[dict], sweep_review: Path, outside: bool = False) -> list[dict]:
    from shapely.geometry import Point, Polygon
    from apply import _sweep_polygons

    review = json.loads(sweep_review.read_text(encoding="utf-8"))
    windows = {w["n"]: w for w in L.site_windows(site)}
    swept = [windows[int(k.rsplit("_w", 1)[1])] for k in review["checked"]]
    polys = [Polygon(p) for p in _sweep_polygons(review)]
    out = []
    for c in cands:
        inside = any(w["west"] <= c["lon"] <= w["east"] and w["south"] <= c["lat"] <= w["north"] for w in swept)
        if inside == outside:
            continue
        box = Polygon(c["polygon"])
        if any(p.contains(Point(c["lon"], c["lat"])) or L.iou(box, p) >= 0.2 for p in polys):
            continue
        out.append(c)
    return out


def build_triage(class_name: str, site: dict, version: str, min_conf: float, label: str, swept_by: Path | None = None, outside: bool = False, extra_sites: list[dict] | None = None, page_slug: str | None = None) -> tuple:
    cands = [c for c in _candidates(class_name, site, version) if c["conf"] >= min_conf and not c.get("labelled")]
    for other in extra_sites or []:
        cands += [c for c in _candidates(class_name, other, version) if c["conf"] >= min_conf and not c.get("labelled")]
    if swept_by:
        cands = _restrict_to_sweep(class_name, site, cands, swept_by, outside)
    if extra_sites:
        cands.sort(key=lambda c: (c["source_sample"], -c["conf"]))
    site_slug = page_slug or L.slug(site["name"])
    suffix = "-outside" if outside else ("-swept" if swept_by else "")
    html = L.fill("triage.html", {
        "TITLE": f"{(page_slug or site['name'])} {label} triage",
        "EYEBROW": f"{class_name} &middot; {site['name']} &middot; model {version} &middot; conf &ge; {min_conf:.2f}",
        "HEADING": "What did the model find here?",
        "LEDE": f"Every crop is a {version} detection at this site with no matching label, highest confidence first{(' -- only those outside the windows you swept' if outside else ' -- only those inside the windows you swept, and not on a polygon you drew') if swept_by else ''}. Mark whether it is a real {label}. Yes becomes a sample; no is stored as a hard negative but left out of training; the third button also trains on it, and the pool line shows what that does to the balance.",
        "YES": label.capitalize(), "NO": f"Not a {label}", "YES_SHORT": label, "NO_SHORT": "not",
        "DOC": f"reviews/{class_name}-triage-{site_slug[:20]}-{version}{suffix}", "LS": f"{class_name}-triage-{site_slug[:20]}-{version}{suffix}",
        "META": f'kind: "triage", class: "{class_name}", site: "{site_slug}", model: "{version}", threshold: {min_conf}',
        "FILENAME": f"{class_name}-triage-{site_slug}-{version}.json",
        "POOL_ON": str(sum(1 for r in common.load_hard_negatives(class_name) if r.get("enabled", True))),
        "POSITIVES": str(sum(1 for r in common.load_samples(class_name) if r.get("enabled", True))),
        "TRAINING_ONLY": "",
    }, cands)
    return html, len(cands)


def _structure(im: Image.Image) -> float:
    g = im.convert("L").resize((240, 240))
    lap = np.asarray(g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)), dtype=np.float32)
    return float(lap.var())


def build_sweep(class_name: str, site: dict, version: str, n_windows: int, show_conf: float, label: str) -> tuple:
    dets = [c for c in _candidates(class_name, site, version) if c["conf"] >= show_conf]
    windows = L.site_windows(site)
    images = {w["n"]: L.window_image(class_name, site, w) for w in windows}
    for w in windows:
        w["structure"] = _structure(images[w["n"]])
    chosen = sorted(sorted(windows, key=lambda w: -w["structure"])[:n_windows], key=lambda w: (-w["iy"], w["ix"]))
    site_slug = L.slug(site["name"])
    out = []
    for k, w in enumerate(chosen, 1):
        im = images[w["n"]].copy()
        W, H = im.size
        d = ImageDraw.Draw(im)
        shown, labelled = [], 0
        for c in dets:
            if not (w["west"] <= c["lon"] <= w["east"] and w["south"] <= c["lat"] <= w["north"]):
                continue
            colour = (80, 150, 255) if c.get("labelled") else (64, 220, 110)
            pts = [L.to_px(w, x, y, W, H) for x, y in c["polygon"]]
            d.line(pts, fill=colour, width=3)
            d.text((pts[0][0] + 3, pts[0][1] + 3), f"{c['conf']:.2f}", fill=colour)
            if c.get("labelled"):
                labelled += 1
            else:
                shown.append({"lon": c["lon"], "lat": c["lat"], "conf": c["conf"]})
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=68, optimize=True)
        out.append({
            "id": f"{site_slug}_w{w['n']}", "n": w["n"], "area": site["name"], "cluster": 1, "grid": f"{k}/{len(chosen)}",
            "west": w["west"], "south": w["south"], "east": w["east"], "north": w["north"], "lat": w["lat"], "lon": w["lon"],
            "w": W, "h": H, "labels": labelled, "detections": len(shown), "dets": shown,
            "img": base64.b64encode(buf.getvalue()).decode(),
        })
    meta_path = L.loop_dir(class_name) / "sweeps" / f"{site_slug}_{version}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps([{k: v for k, v in o.items() if k != "img"} for o in out]))
    html = L.fill("sweep.html", {
        "TITLE": f"{site['name']} {label} sweep",
        "EYEBROW": f"{class_name} &middot; model {version} &middot; {site['name']} &middot; {len(out)} densest windows &middot; proposals shown at &ge; {show_conf:.2f}",
        "HEADING": f"Find the {label}s the model missed",
        "LEDE": f"The model's proposals are in <b>green</b> with their confidence, including weak ones; they get judged in a separate yes/no pass, so <b>leave them alone here</b>. Proposals in <b>blue</b> sit on an object that is already labelled; leave those too. <b>Draw a polygon around each real {label} that has no green outline</b>: click its corners, then click the first point again or press Enter to close. Click inside a finished polygon to remove it. Windows with nothing missing: just go to the next.",
        "OBJECT": label, "OBJECTS": f"{label}s",
        "DOC": f"reviews/{class_name}-sweep-{site_slug[:20]}-{version}", "LS": f"{class_name}-sweep-{site_slug[:20]}",
        "META": f'kind: "sweep", class: "{class_name}", site: "{site_slug}", model: "{version}", threshold: {show_conf}',
        "FILENAME": f"{class_name}-sweep-{site_slug}-{version}.json",
    }, out)
    return html, len(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("triage", "sweep"))
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None, help="what to call one object in the UI (default: class name)")
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--show-conf", type=float, default=0.25)
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--swept-by", type=Path, default=None, help="triage only: a sweep review JSON; keep proposals inside its windows that are not on its polygons")
    parser.add_argument("--outside-sweep", action="store_true", help="triage only, with --swept-by: the complement -- proposals outside the swept windows, judged for samples/negatives but never used as ground truth")
    parser.add_argument("--also", default=None, help="triage only: comma-separated extra site substrings whose proposals join the page (one page for several look-alike sites)")
    parser.add_argument("--page-slug", default=None, help="triage only: name the page/document after this instead of the site (use with --also)")
    args = parser.parse_args()
    site = L.find_site(args.class_name, args.site)
    label = args.label or args.class_name.replace("-", " ")
    if args.kind == "triage":
        extra = [L.find_site(args.class_name, s.strip()) for s in args.also.split(",")] if args.also else None
        html, n = build_triage(args.class_name, site, args.model, args.min_conf, label, args.swept_by, args.outside_sweep, extra, args.page_slug)
    else:
        html, n = build_sweep(args.class_name, site, args.model, args.windows, args.show_conf, label)
    tag = "_outside" if args.outside_sweep else ("_swept" if args.swept_by else "")
    out = L.loop_dir(args.class_name) / "pages" / f"{args.kind}_{args.page_slug or L.slug(site['name'])}_{args.model}{tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{args.kind}: {n} {'candidates' if args.kind == 'triage' else 'windows'} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
