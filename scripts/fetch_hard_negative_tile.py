import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
env_path = REPO_ROOT / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

import common  # noqa: E402

DETECT_ZOOM = 17


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--zoom", type=int, default=DETECT_ZOOM)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    x, y = common.lonlat_to_tile(args.lon, args.lat, args.zoom)
    tile_id = common.tile_id(args.zoom, x, y)
    path = common.fetch_tile(args.zoom, x, y, common.DEFAULT_TILESET, common.DEFAULT_FORMAT)

    preview_path = common.SCRATCH_DIR / "hard_negative_previews" / f"{tile_id}.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, preview_path)

    comment = f"  # {args.label}" if args.label else ""
    print(f"tile: {tile_id}")
    print(f"preview: {preview_path}")
    print(f'\n    "{tile_id}",{comment}')


if __name__ == "__main__":
    main()
