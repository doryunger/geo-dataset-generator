import json
import math
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402

WINDOW_M = 120.0
PAD_M = 60.0
FETCH_ZOOM = 18
DUPLICATE_IOU = 0.5
MATCH_M = 8.0
CROP_PX = 300
TEMPLATES = Path(__file__).resolve().parent / "templates"


def loop_dir(class_name: str) -> Path:
    d = common.WORKSPACE_ROOT / "loop" / class_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def sites_path(class_name: str) -> Path:
    return loop_dir(class_name) / "sites.json"


def iou(a, b) -> float:
    return a.intersection(b).area / a.union(b).area if a.intersects(b) else 0.0


def held_out_ids(class_name: str) -> set[str]:
    p = loop_dir(class_name) / "benchmark.json"
    return {h["osm_id"] for h in json.loads(p.read_text(encoding="utf-8")).get("held_out", [])} if p.exists() else set()


def load_sites(class_name: str) -> list[dict]:
    p = sites_path(class_name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_sites(class_name: str, sites: list[dict]) -> None:
    sites_path(class_name).write_text(json.dumps(sites, ensure_ascii=False, indent=1), encoding="utf-8")


def find_site(class_name: str, name_substr: str) -> dict:
    sites = load_sites(class_name)
    hits = [s for s in sites if s.get("osm_id") == name_substr] or [s for s in sites if name_substr.lower() in s["name"].lower()]
    if not hits:
        raise SystemExit(f"no site matching {name_substr!r} in {sites_path(class_name)}")
    if len(hits) > 1:
        raise SystemExit(f"{name_substr!r} matches {len(hits)} sites: {[h['name'] for h in hits]}")
    return hits[0]


def slug(name: str) -> str:
    ascii_name = "".join(ch for ch in unicodedata.normalize("NFKD", name) if ord(ch) < 128)
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:32]


def metres_per_deg(lat: float) -> tuple[float, float]:
    return 111_320 * math.cos(math.radians(lat)), 111_320


def dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    mlon, mlat = metres_per_deg((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * mlon, (a[1] - b[1]) * mlat)


def site_windows(site: dict) -> list[dict]:
    from shapely.geometry import box, shape

    geom = shape(site["geometry"])
    mlon, mlat = metres_per_deg(site["lat"])
    pad_lon, pad_lat = PAD_M / mlon, PAD_M / mlat
    w0, s0, e0, n0 = geom.bounds
    w0, s0, e0, n0 = w0 - pad_lon, s0 - pad_lat, e0 + pad_lon, n0 + pad_lat
    sl, st = WINDOW_M / mlon, WINDOW_M / mlat
    nx, ny = math.ceil((e0 - w0) / sl), math.ceil((n0 - s0) / st)
    padded = geom.buffer(pad_lon)
    out = []
    n = 0
    for iy in range(ny):
        for ix in range(nx):
            west, south = w0 + ix * sl, s0 + iy * st
            east, north = west + sl, south + st
            if padded.intersects(box(west, south, east, north)):
                n += 1
                out.append({"n": n, "ix": ix, "iy": iy, "west": west, "south": south, "east": east, "north": north,
                            "lat": (south + north) / 2, "lon": (west + east) / 2})
    return out


def window_image(class_name: str, site: dict, w: dict):
    from PIL import Image

    path = common.fetch_and_crop_bbox(
        FETCH_ZOOM, w["west"], w["south"], w["east"], w["north"], common.DEFAULT_TILESET, common.DEFAULT_FORMAT,
        loop_dir(class_name) / "scans" / slug(site["name"]) / f"w{w['n']}.jpg",
    )
    with Image.open(path) as raw:
        return common.resample_to_target_gsd(raw.convert("RGB"), common.meters_per_pixel(FETCH_ZOOM, w["lat"]))


def to_geo(w: dict, x_px: float, y_px: float, W: int, H: int) -> tuple[float, float]:
    return w["west"] + x_px / W * (w["east"] - w["west"]), w["north"] - y_px / H * (w["north"] - w["south"])


def to_px(w: dict, lon: float, lat: float, W: int, H: int) -> tuple[float, float]:
    return (lon - w["west"]) / (w["east"] - w["west"]) * W, (w["north"] - lat) / (w["north"] - w["south"]) * H


def fill(template_name: str, values: dict, data) -> str:
    t = (TEMPLATES / template_name).read_text(encoding="utf-8")
    for k, v in values.items():
        t = t.replace(f"__{k}__", v)
    return t.replace("__DATA__", json.dumps(data))
