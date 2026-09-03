"""Decides which models run against an incoming tile -- never which classes within a model to look
for (see oil_refinery/semantic_graph.md's "Pipeline: model router, fuser, classifier"). Every
triggered model runs unfiltered, returning whatever classes it detects; nothing here (or anywhere
downstream) restricts a model's own class list -- that was config.json's old per-target `class_id`
filter, dropped along with this module.

Today's only routing criterion is the zoom gate `server.py` already used: below MIN_DETECT_ZOOM,
detection doesn't run at all, so no model is worth triggering; at or above it, every configured
model runs. Room for additional per-tile routing criteria later, but nothing beyond zoom exists yet.
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config = json.loads(CONFIG_PATH.read_text())

MIN_DETECT_ZOOM: int = _config["min_detect_zoom"]
MODELS: list[str] = _config["models"]
CANONICAL_MODEL: str = _config["canonical_model"]  # the fuser's fixed naming/tie-break convention,
# see semantic_graph.md's "Pipeline" -- read from config here since this is the module that already
# owns config.json, but only fuser.py actually uses the value.


def models_for_tile(z: int) -> list[str]:
    """Model paths (relative to REPO_ROOT) to run against a tile at zoom `z`. Empty below
    MIN_DETECT_ZOOM -- matches the existing gate in server.py's /api/detections endpoint."""
    if z < MIN_DETECT_ZOOM:
        return []
    return list(MODELS)
