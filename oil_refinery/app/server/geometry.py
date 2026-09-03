"""Global-pixel-space distance math for detections, used by the classifier (see
oil_refinery/semantic_graph.md's "Pipeline: model router, fuser, classifier"). Distance math itself
stays pixel-only on purpose: at refinery-site scale a single reference latitude's meters-per-pixel is
accurate enough (same locally-constant-scale assumption common.py already makes in
resample_to_target_gsd/bbox_crop_px), so no detection point is converted to lon/lat just to measure
between two of them. Would need full lon/lat + haversine instead for points far enough apart that
Mercator's latitude-dependent scale distortion starts to matter -- out of scope here.

global_pixel_to_lonlat() is the one exception, and it's an output-shaping step, not part of the
distance math above: once the classifier has decided a cluster's boundary in pixel space (see
classifier.polygon_for()), that boundary has to become real lon/lat coordinates before it can go out
as GeoJSON to the frontend -- pixel coordinates mean nothing to a map."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402


def global_pixel(tile_x: int, tile_y: int, local_px: float, local_py: float) -> tuple[float, float]:
    return tile_x * common.TILE_PX + local_px, tile_y * common.TILE_PX + local_py


def global_pixel_to_lonlat(px: float, py: float, zoom: int) -> tuple[float, float]:
    """Inverse of global_pixel() (composed with the tile origin), via common.tile_to_lonlat's own
    continuous (non-floored) math -- the exact inverse of common.lonlat_to_tile_float, just taking
    global pixel coordinates instead of a lon/lat in the first place."""
    return common.tile_to_lonlat(zoom, px / common.TILE_PX, py / common.TILE_PX)


def distance_m(a: tuple[float, float], b: tuple[float, float], zoom: int, ref_lat: float) -> float:
    pixel_distance = math.hypot(a[0] - b[0], a[1] - b[1])
    return pixel_distance * common.meters_per_pixel(zoom, ref_lat)
