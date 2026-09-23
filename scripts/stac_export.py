import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import stac_geoparquet.arrow
from PIL import Image
from pyproj import Transformer

import common
import obb

logger = logging.getLogger(__name__)

STAC_VERSION = "1.1.0"
LABEL_EXT = "https://stac-extensions.github.io/label/v1.0.1/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
NEGATIVE_LABEL = "negative"

_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def stac_dir(class_name: str) -> Path:
    return common.class_dir(class_name) / "stac"


def samples_parquet_path(class_name: str) -> Path:
    return stac_dir(class_name) / "samples.parquet"


def hard_negatives_parquet_path(class_name: str) -> Path:
    return stac_dir(class_name) / "hard_negatives.parquet"


def _collection_href(parquet_path: Path) -> str:
    return f"./{parquet_path.stem}.collection.json"


def summary_line(result: dict) -> str:
    return f"STAC: {result['samples']} samples, {result['hard_negatives']} hard negatives -> {result['dir']}"


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z") if ts else None


def _polygon_geometry(ring: list[list[float]]) -> dict:
    ring = [list(p) for p in ring]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _label_properties(class_name: str, label: str, description: str) -> dict:
    return {
        "label:type": "vector",
        "label:properties": ["class"],
        "label:classes": [{"name": "class", "classes": [label]}],
        "label:tasks": ["detection"],
        "label:methods": ["manual"],
        "label:description": description,
    }


def _proj_properties(row: dict, image_path: Path) -> dict:
    with Image.open(image_path) as im:
        width, height = im.size
    x0, y0 = _to_mercator.transform(row["west"], row["south"])
    x1, y1 = _to_mercator.transform(row["east"], row["north"])
    return {
        "proj:code": "EPSG:3857",
        "proj:bbox": [x0, y0, x1, y1],
        "proj:shape": [height, width],
        "proj:transform": [(x1 - x0) / width, 0.0, x0, 0.0, -(y1 - y0) / height, y1],
    }


def _base_item(row: dict, collection_id: str, created: float | None, collection_href: str) -> dict:
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [LABEL_EXT],
        "id": row["id"],
        "collection": collection_id,
        "geometry": _polygon_geometry(row["polygon"]),
        "bbox": [row["west"], row["south"], row["east"], row["north"]],
        "properties": {"datetime": _iso(created), "created": _iso(created)},
        "links": [{"rel": "collection", "href": collection_href, "type": "application/json"}],
        "assets": {},
    }


def sample_item(class_name: str, row: dict, collection_id: str) -> dict:
    item = _base_item(row, collection_id, row.get("created_at"), _collection_href(samples_parquet_path(class_name)))
    props = item["properties"]
    props.update(_label_properties(class_name, class_name, f"Hand-labeled {class_name} outline"))
    props["gdg:zoom"] = row.get("zoom")
    props["gdg:label_polygon"] = row.get("label_polygon")
    image_path = common.samples_dir(class_name) / f"{row['id']}.{row.get('ext', 'jpg')}"
    if image_path.exists():
        item["stac_extensions"].append(PROJ_EXT)
        props.update(_proj_properties(row, image_path))
        item["assets"]["image"] = {
            "href": f"../samples/{image_path.name}",
            "type": "image/jpeg",
            "roles": ["data"],
            "title": "Mapbox satellite crop (Web Mercator, bbox-aligned)",
        }
    return item


def hard_negative_item(class_name: str, row: dict, collection_id: str) -> dict:
    item = _base_item(
        row, collection_id, row.get("added_at"), _collection_href(hard_negatives_parquet_path(class_name)),
    )
    props = item["properties"]
    props.update(_label_properties(
        class_name, NEGATIVE_LABEL, f"Region confirmed to contain no {class_name} (hard negative)",
    ))
    props["gdg:enabled"] = row.get("enabled", True)
    if row.get("origin"):
        props["gdg:origin"] = row["origin"]
    image_path = common.hard_negative_review_dir(class_name) / f"{row['id']}.jpg"
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        common.fetch_and_crop_bbox(
            obb.SAMPLE_FETCH_ZOOM, row["west"], row["south"], row["east"], row["north"],
            common.DEFAULT_TILESET, common.DEFAULT_FORMAT, image_path,
        )
    item["stac_extensions"].append(PROJ_EXT)
    props.update(_proj_properties(row, image_path))
    item["assets"]["image"] = {
        "href": f"../{image_path.parent.name}/{image_path.name}",
        "type": "image/jpeg",
        "roles": ["data"],
        "title": "Mapbox satellite crop (Web Mercator, bbox-aligned)",
    }
    return item


def _collection(collection_id: str, description: str, items: list[dict]) -> dict:
    boxes = [i["bbox"] for i in items]
    times = sorted(i["properties"]["datetime"] for i in items if i["properties"]["datetime"])
    return {
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "id": collection_id,
        "description": description,
        "license": "other",
        "extent": {
            "spatial": {"bbox": [[
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ]] if boxes else [[-180, -90, 180, 90]]},
            "temporal": {"interval": [[times[0], times[-1]] if times else [None, None]]},
        },
        "links": [],
    }


def _write(items: list[dict], path: Path, collection: dict) -> None:
    collection_path = path.with_name(f"{path.stem}.collection.json")
    path.unlink(missing_ok=True)
    collection_path.unlink(missing_ok=True)
    if not items:
        return
    collection["links"] = [{"rel": "item", "href": f"./{path.name}", "type": "application/vnd.apache.parquet"}]
    collection_path.write_text(json.dumps(collection, indent=2))
    table = stac_geoparquet.arrow.parse_stac_items_to_arrow(items).read_all()
    stac_geoparquet.arrow.to_parquet(table, path, collections={collection["id"]: collection})


def export_class(class_name: str) -> dict:
    slug = common.class_slug(class_name)
    stac_dir(class_name).mkdir(parents=True, exist_ok=True)

    samples_id = slug
    samples = [sample_item(class_name, r, samples_id) for r in common.read_jsonl(common.samples_path(class_name))]
    _write(samples, samples_parquet_path(class_name), _collection(
        samples_id, f"Hand-labeled {class_name} samples", samples,
    ))

    negatives_id = f"{slug}-hard-negatives"
    negatives = [hard_negative_item(class_name, r, negatives_id) for r in common.load_hard_negatives(class_name)]
    _write(negatives, hard_negatives_parquet_path(class_name), _collection(
        negatives_id, f"Hard-negative regions for {class_name}", negatives,
    ))

    result = {"samples": len(samples), "hard_negatives": len(negatives), "dir": str(stac_dir(class_name))}
    logger.info(f"[{class_name}] {summary_line(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Export a class's samples and hard negatives as stac-geoparquet")
    parser.add_argument("--class", dest="class_name", default=None, help="Object class name (omit for every class)")
    args = parser.parse_args()
    common.setup_logging()
    for name in [args.class_name] if args.class_name else common.list_classes():
        result = export_class(name)
        print(f"{name}: {summary_line(result)}")
        for path in (samples_parquet_path(name), hard_negatives_parquet_path(name)):
            if path.exists():
                print(f"  {path.name}: {pq.read_metadata(path).num_rows} rows")


if __name__ == "__main__":
    main()
