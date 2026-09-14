import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config = json.loads(CONFIG_PATH.read_text())

MIN_DETECT_ZOOM: int = _config["min_detect_zoom"]
MODELS: list[str] = _config["models"]
CANONICAL_MODEL: str = _config["canonical_model"]
MODEL_GSD_M: dict[str, float | None] = {key: _config.get("model_gsd_m", {}).get(key) for key in MODELS}
GATED_MODELS: set[str] = set(_config.get("gated_models", []))
INFERENCE_BACKEND: str = _config.get("inference_backend", "pytorch")
EARLY_EXIT: bool = bool(_config.get("early_exit", False))


def models_for_tile(z: int) -> list[str]:
    if z < MIN_DETECT_ZOOM:
        return []
    return list(MODELS)


def gsd_for(model_key: str) -> float | None:
    return MODEL_GSD_M.get(model_key)


def is_gated(model_key: str) -> bool:
    return model_key in GATED_MODELS
