import json
from pathlib import Path

from fastapi import APIRouter

FLAGS_PATH = Path(__file__).resolve().parent / "flags.json"
TILE_SCHEMES = ("tms", "xyz")


def _load() -> dict:
    raw = json.loads(FLAGS_PATH.read_text())
    tiles = raw.get("detection_tiles", {})
    enabled = tiles.get("enabled", False)
    scheme = tiles.get("tile_scheme", "tms")
    if not isinstance(enabled, bool):
        raise ValueError(f"flags.json: detection_tiles.enabled must be true or false, got {enabled!r}")
    if scheme not in TILE_SCHEMES:
        raise ValueError(f"flags.json: detection_tiles.tile_scheme must be one of {TILE_SCHEMES}, got {scheme!r}")
    return {"detection_tiles": {"enabled": enabled, "tile_scheme": scheme}}


FLAGS: dict = _load()
DETECTION_TILES: bool = FLAGS["detection_tiles"]["enabled"]
TILE_SCHEME: str = FLAGS["detection_tiles"]["tile_scheme"]

router = APIRouter()


@router.get("/api/flags")
def get_flags() -> dict:
    return FLAGS
