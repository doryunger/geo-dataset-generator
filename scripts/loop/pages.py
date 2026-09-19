"""
Build the two review pages of the loop as self-contained HTML files.

    triage: one candidate per screen, yes/no
    sweep:  one window per screen, draw a polygon around every real object (coverage ground truth)

Usage:
    python scripts/loop/pages.py triage --class distillation-column --site puertollano --model v13 --min-conf 0.4
    python scripts/loop/pages.py sweep  --class distillation-column --site puertollano --model v13 --windows 60 --show-conf 0.25

Publish the resulting file (scripts/loop/pages/<name>.html) as an Artifact, review it, then use the
page's "Download JSON" button and feed the file to apply.py / coverage.py.
"""
import argparse
import base64
import io
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import loop_common as L


def _candidates(class_name: str, site: dict, version: str) -> list[dict]:
    p = L.loop_dir(class_name) / "candidates" / f"{L.slug(site['name'])}_{version}.json"
    if not p.exists():
        raise SystemExit(f"no scan for {site['name']} with {version}; run scan.py first")
    return json.loads(p.read_text())


def build_triage(class_name: str, site: dict, version: str, min_conf: float, label: str) -> tuple:
    cands = [c for c in _candidates(class_name, site, version) if c["conf"] >= min_conf]
    site_slug = L.slug(site["name"])
    html = L.fill("triage.html", {
        "TITLE": f"{site['name']} {label} triage",
        "EYEBROW": f"{class_name} &middot; {site['name']} &middot; model {version} &middot; conf &ge; {min_conf:.2f}",
        "HEADING": "What did the model find here?",
        "LEDE": f"Every crop is a {version} detection at this site with no matching label, highest confidence first. Mark whether it is a real {label}. Yes becomes a sample; no is stored as a hard negative.",
        "YES": label.capitalize(), "NO": f"Not a {label}", "YES_SHORT": label, "NO_SHORT": "not",
        "DOC": f"reviews/{class_name}-triage-{site_slug[:20]}-{version}", "LS": f"{class_name}-triage-{site_slug[:20]}-{version}",
        "META": f'kind: "triage", class: "{class_name}", site: "{site_slug}", model: "{version}", threshold: {min_conf}',
        "FILENAME": f"{class_name}-triage-{site_slug}-{version}.json",
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
        shown = []
        for c in dets:
            if not (w["west"] <= c["lon"] <= w["east"] and w["south"] <= c["lat"] <= w["north"]):
                continue
            pts = [L.to_px(w, x, y, W, H) for x, y in c["polygon"]]
            d.line(pts, fill=(64, 220, 110), width=3)
            d.text((pts[0][0] + 3, pts[0][1] + 3), f"{c['conf']:.2f}", fill=(64, 220, 110))
            shown.append({"lon": c["lon"], "lat": c["lat"], "conf": c["conf"]})
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=68, optimize=True)
        out.append({
            "id": f"{site_slug}_w{w['n']}", "n": w["n"], "area": site["name"], "cluster": 1, "grid": f"{k}/{len(chosen)}",
            "west": w["west"], "south": w["south"], "east": w["east"], "north": w["north"], "lat": w["lat"], "lon": w["lon"],
            "w": W, "h": H, "labels": 0, "detections": len(shown), "dets": shown,
            "img": base64.b64encode(buf.getvalue()).decode(),
        })
    meta_path = L.loop_dir(class_name) / "sweeps" / f"{site_slug}_{version}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps([{k: v for k, v in o.items() if k != "img"} for o in out]))
    html = L.fill("sweep.html", {
        "TITLE": f"{site['name']} {label} sweep",
        "EYEBROW": f"{class_name} &middot; model {version} &middot; {site['name']} &middot; {len(out)} densest windows &middot; proposals shown at &ge; {show_conf:.2f}",
        "HEADING": f"Mark every {label}, whether or not the model saw it",
        "LEDE": f"The model's proposals are in <b>green</b> with their confidence, including weak ones. <b>Draw a polygon around each real {label}: click its corners, then click the first point again or press Enter to close.</b> Mark every one you can see, green or not; that is what measures coverage. Click inside a finished polygon to remove it. Windows with none: just go to the next.",
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
    args = parser.parse_args()
    site = L.find_site(args.class_name, args.site)
    label = args.label or args.class_name.replace("-", " ")
    if args.kind == "triage":
        html, n = build_triage(args.class_name, site, args.model, args.min_conf, label)
    else:
        html, n = build_sweep(args.class_name, site, args.model, args.windows, args.show_conf, label)
    out = L.loop_dir(args.class_name) / "pages" / f"{args.kind}_{L.slug(site['name'])}_{args.model}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{args.kind}: {n} {'candidates' if args.kind == 'triage' else 'windows'} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
