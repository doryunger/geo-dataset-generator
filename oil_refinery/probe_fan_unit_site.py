import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

env_path = REPO_ROOT / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

import common  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

Z = 17


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--half-km", type=float, default=0.75)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "oil_refinery" / "probe" / args.slug / f"fan_unit_scan_{Path(args.model).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x0, y0 = common.lonlat_to_tile(args.lon, args.lat, Z)
    m_per_tile = common.meters_per_pixel(Z, args.lat) * common.TILE_PX
    radius = round((args.half_km * 1000) / m_per_tile)
    side = 2 * radius + 1
    print(f"center tile ({x0},{y0}) z{Z}, {m_per_tile:.1f}m/tile, radius={radius} -> {side}x{side} = {side * side} tiles")

    model = YOLO(args.model)

    ids, xs, ys, paths = [], [], [], []
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            paths.append(common.fetch_tile(Z, x, y, common.DEFAULT_TILESET, common.DEFAULT_FORMAT))
            ids.append(common.tile_id(Z, x, y))
            xs.append(x)
            ys.append(y)

    CHUNK = 16
    hits = []
    for i in range(0, len(paths), CHUNK):
        chunk_ids = ids[i:i + CHUNK]
        chunk_paths = paths[i:i + CHUNK]
        chunk_xs, chunk_ys = xs[i:i + CHUNK], ys[i:i + CHUNK]

        resampled = []
        for path, x, y in zip(chunk_paths, chunk_xs, chunk_ys):
            bounds = common.tile_bounds(Z, x, y)
            lat = (bounds["north"] + bounds["south"]) / 2
            with Image.open(path) as img:
                resampled.append(common.resample_to_target_gsd(img.convert("RGB"), common.meters_per_pixel(Z, lat)))

        batch_imgsz = max(32, math.ceil(max(img.size[d] for img in resampled for d in (0, 1)) / 32) * 32)
        results = model.predict(resampled, conf=args.conf, iou=0.4, imgsz=batch_imgsz, verbose=False)
        for tid, path, result in zip(chunk_ids, chunk_paths, results):
            if result.obb is not None and len(result.obb) > 0:
                rects = result.obb.xyxyxyxyn.cpu().numpy().tolist()
                confs = [float(c) for c in result.obb.conf]
                labeled_path = out_dir / f"{tid}_labeled.jpg"
                common.draw_polygon_overlay(path, rects, labeled_path)
                hits.append((tid, confs, str(labeled_path)))
        print(f"scanned {min(i + CHUNK, len(paths))}/{len(paths)}, {len(hits)} hit(s) so far", flush=True)

    print(f"\n{len(hits)} tile(s) with a detection out of {len(paths)} scanned")
    for tid, confs, p in hits:
        print(f"  {tid}  conf={['%.2f' % c for c in confs]}  -> {p}")


if __name__ == "__main__":
    main()
