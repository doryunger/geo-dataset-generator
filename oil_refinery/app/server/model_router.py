import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config = json.loads(CONFIG_PATH.read_text())

MIN_DETECT_ZOOM: int = _config["min_detect_zoom"]
MODELS: list[str] = _config["models"]
CANONICAL_MODEL: str = _config["canonical_model"]


def models_for_tile(z: int) -> list[str]:
    if z < MIN_DETECT_ZOOM:
        return []
    return list(MODELS)
