import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402


def global_pixel(tile_x: int, tile_y: int, local_px: float, local_py: float) -> tuple[float, float]:
    return tile_x * common.TILE_PX + local_px, tile_y * common.TILE_PX + local_py


def global_pixel_to_lonlat(px: float, py: float, zoom: int) -> tuple[float, float]:
    return common.tile_to_lonlat(zoom, px / common.TILE_PX, py / common.TILE_PX)


def distance_m(a: tuple[float, float], b: tuple[float, float], zoom: int, ref_lat: float) -> float:
    pixel_distance = math.hypot(a[0] - b[0], a[1] - b[1])
    return pixel_distance * common.meters_per_pixel(zoom, ref_lat)
