#!/usr/bin/env python3
"""
Run a trained YOLO-seg model over the tiles around a hardcoded position and save its predictions
as browsable files.

Usage:
    python scripts/predict_area.py --class fence --model models/fence_v1.pt \
        --seed seeds/seed_001.yaml --radius 2
"""
import argparse

import yaml
from PIL import Image
from ultralytics import YOLO

import common


def predict_area(class_name: str, model_path: str, tile_url: str, radius: int, conf: float, imgsz: int = 1280) -> dict:
    z, x0, y0, tileset, ext = common.parse_tile_url(tile_url)
    model = YOLO(model_path)
    run_name = f"{common.tile_id(z, x0, y0)}_r{radius}"

    tile_ids, paths, positions = [], [], []
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            paths.append(common.fetch_tile(z, x, y, tileset, ext))
            tile_ids.append(common.tile_id(z, x, y))
            positions.append((x - (x0 - radius), y - (y0 - radius)))

    CHUNK = 8
    results = []
    for i in range(0, len(paths), CHUNK):
        chunk = [str(p) for p in paths[i:i + CHUNK]]
        results.extend(model.predict(chunk, conf=conf, imgsz=imgsz, verbose=False))

    out_dir = common.class_dir(class_name) / "predictions" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    side = 2 * radius + 1
    mosaic = Image.new("RGB", (side * common.TILE_PX, side * common.TILE_PX))
    found = []
    for tid, path, result, (col, row) in zip(tile_ids, paths, results, positions):
        raw_dst = out_dir / f"{tid}{path.suffix}"
        if not raw_dst.exists():
            raw_dst.symlink_to(path.resolve())
        tile_for_mosaic = path

        if result.masks is not None and len(result.masks) > 0:
            polygons = [m.tolist() for m in result.masks.xyn]
            confs = [float(c) for c in result.boxes.conf]
            (out_dir / f"{tid}.txt").write_text(common.yolo_seg_lines(polygons))
            labeled_path = out_dir / f"{tid}_labeled.jpg"
            common.draw_polygon_overlay(path, polygons, labeled_path)
            found.append((tid, confs))
            tile_for_mosaic = labeled_path

        with Image.open(tile_for_mosaic) as tile_img:
            mosaic.paste(tile_img.convert("RGB"), (col * common.TILE_PX, row * common.TILE_PX))

    mosaic_path = out_dir / "mosaic.jpg"
    mosaic.save(mosaic_path, format="JPEG")

    return {"scanned": len(paths), "found": found, "out_dir": out_dir, "mosaic_path": mosaic_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--model", required=True, help="Path to a trained .pt (see train.py)")
    parser.add_argument(
        "--seed", required=True,
        help="Path to a seed_*.yaml file (tile_url) -- reused here purely as a hardcoded test "
             "position, unrelated to whichever seed(s) trained the model",
    )
    parser.add_argument("--radius", type=int, default=1, help="Tiles to each side of the seed tile to scan (0 = just that one tile)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference resolution -- match whatever the model was trained with (train.py's "
             "default is also 1280; a thin object like a fence can get squeezed down to a few "
             "px wide at 640 once letterboxed, well below what the network can represent)",
    )
    args = parser.parse_args()

    seed = yaml.safe_load(open(args.seed))
    result = predict_area(args.class_name, args.model, seed["tile_url"], args.radius, args.conf, args.imgsz)

    print(f"Scanned {result['scanned']} tile(s), model found '{args.class_name}' in {len(result['found'])}.")
    for tid, confs in result["found"]:
        print(f"  {tid}  conf={['%.2f' % c for c in confs]}")
    print(f"\nAll {result['scanned']} scanned tiles (raw) + any labeled detections saved to {result['out_dir']}")


if __name__ == "__main__":
    main()
