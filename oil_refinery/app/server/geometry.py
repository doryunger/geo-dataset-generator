"""Global-pixel-space distance math for detections. Not wired into any endpoint yet -- kept ready
for the composition/clustering work described in oil_refinery/README.md's "Composition rule"
(a proximity threshold over detection centroids to decide which detections belong to the same
facility). Deliberately pixel-only: no detection point is ever converted to lon/lat here, since at
refinery-site scale a single reference latitude's meters-per-pixel is accurate enough (same
locally-constant-scale assumption common.py already makes in resample_to_target_gsd/bbox_crop_px).
Would need full lon/lat + haversine instead for points far enough apart that Mercator's
latitude-dependent scale distortion starts to matter -- out of scope here."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402


def global_pixel(tile_x: int, tile_y: int, local_px: float, local_py: float) -> tuple[float, float]:
    return tile_x * common.TILE_PX + local_px, tile_y * common.TILE_PX + local_py


def distance_m(a: tuple[float, float], b: tuple[float, float], zoom: int, ref_lat: float) -> float:
    pixel_distance = math.hypot(a[0] - b[0], a[1] - b[1])
    return pixel_distance * common.meters_per_pixel(zoom, ref_lat)
