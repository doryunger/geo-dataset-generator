"""One-off: fetch a real refinery image and see what several pretrained models
actually detect on it, before assuming dataset docs match real-world output.
Produces one annotated comparison image per model (title, per-class colors, labeled boxes)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# .env isn't loaded by python-dotenv anywhere in this repo -- run.bat parses it
# manually into the process env before launching. Mirror that here.
env_path = REPO_ROOT / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

from common import fetch_and_crop_bbox  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# HOLBORN Europa Raffinerie GmbH, Hamburg, Germany -- ~1.5km x 1.5km box around the site
SITE_SLUG = "holborn_hamburg"
CENTER_LAT, CENTER_LON = 53.4770211, 9.9517431
HALF_KM = 0.75
DLAT = HALF_KM / 111.0
DLON = HALF_KM / (111.0 * 0.5978)  # cos(53.48 deg)

OUT_DIR = Path(__file__).resolve().parent / "probe" / SITE_SLUG
OUT_DIR.mkdir(parents=True, exist_ok=True)

west, east = CENTER_LON - DLON, CENTER_LON + DLON
south, north = CENTER_LAT - DLAT, CENTER_LAT + DLAT

site_image_path = fetch_and_crop_bbox(
    z=18, west=west, south=south, east=east, north=north,
    tileset="mapbox.satellite", ext="jpg", output_path=OUT_DIR / "site.jpg",
)
print(f"Fetched site image: {site_image_path}")

# --- color palette, stable per class name across all models ---
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
]
_class_colors: dict[str, tuple[int, int, int]] = {}


def color_for(cls_name: str) -> tuple[int, int, int]:
    if cls_name not in _class_colors:
        _class_colors[cls_name] = PALETTE[len(_class_colors) % len(PALETTE)]
    return _class_colors[cls_name]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


LABEL_FONT = load_font(16)
TITLE_FONT = load_font(28)


def annotate(image_path: Path, model_title: str, detections: list[dict], out_path: Path) -> None:
    """detections: list of {"cls": str, "conf": float, "points": [(x,y), ...4]}"""
    base = Image.open(image_path).convert("RGB")
    title_h = 44
    canvas = Image.new("RGB", (base.width, base.height + title_h), (20, 20, 20))
    canvas.paste(base, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), model_title, font=TITLE_FONT, fill=(255, 255, 255))

    for det in detections:
        color = color_for(det["cls"])
        pts = [(x, y + title_h) for x, y in det["points"]]
        draw.polygon(pts, outline=color, width=3)
        label = f'{det["cls"]} {det["conf"]:.2f}'
        tx, ty = pts[0]
        text_bbox = draw.textbbox((tx, ty), label, font=LABEL_FONT)
        pad = 2
        draw.rectangle(
            (text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad),
            fill=color,
        )
        draw.text((tx, ty), label, font=LABEL_FONT, fill=(0, 0, 0))

    if not detections:
        draw.text((10, title_h + 10), "No detections", font=LABEL_FONT, fill=(255, 80, 80))

    canvas.save(out_path, format="PNG")
    print(f"Wrote {out_path} ({len(detections)} detections)")


PREDICT_IMGSZ = 1536  # default 640 downsamples this 5994px image enough to lose small objects


def run_obb_model(weights: str, model_title: str, conf: float = 0.15) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(source=str(site_image_path), conf=conf, imgsz=PREDICT_IMGSZ, verbose=False)
    r = results[0]
    dets = []
    if r.obb is not None and len(r.obb) > 0:
        for cls_id, cf, xy in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
            dets.append({"cls": r.names[int(cls_id)], "conf": cf, "points": [(p[0], p[1]) for p in xy]})
    slug = model_title.lower().replace(" ", "_").replace("/", "_")
    annotate(site_image_path, model_title, dets, OUT_DIR / f"{slug}.png")


def run_axis_aligned_model(weights: str, model_title: str, conf: float = 0.15) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(source=str(site_image_path), conf=conf, imgsz=PREDICT_IMGSZ, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for cls_id, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            x1, y1, x2, y2 = xyxy
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            dets.append({"cls": r.names[int(cls_id)], "conf": cf, "points": pts})
    slug = model_title.lower().replace(" ", "_").replace("/", "_")
    annotate(site_image_path, model_title, dets, OUT_DIR / f"{slug}.png")


if __name__ == "__main__":
    run_obb_model("yolo11n-obb.pt", "Ultralytics YOLO11-OBB (DOTAv1)")
    run_axis_aligned_model("yolo11x.pt", "Ultralytics YOLO11x (COCO, generic baseline)")
    # community fine-tune on DIOR (storage tank, chimney, harbor, ship, vehicle, ...)
    # unverified provenance/quality -- included for coverage, flag results accordingly
    # README calls this "oriented object detection" -- try OBB head first, fall
    # back to axis-aligned if the checkpoint turns out to be a plain detector.
    from ultralytics import YOLO as _YOLO
    _probe = _YOLO(
        "https://huggingface.co/pauhidalgoo/yolov8-DIOR/resolve/main/DIOR_yolov8s_backbone.pt"
    )
    if getattr(_probe.model, "task", None) == "obb":
        run_obb_model(
            "https://huggingface.co/pauhidalgoo/yolov8-DIOR/resolve/main/DIOR_yolov8s_backbone.pt",
            "Community YOLOv8-DIOR (unverified)",
        )
    else:
        run_axis_aligned_model(
            "https://huggingface.co/pauhidalgoo/yolov8-DIOR/resolve/main/DIOR_yolov8s_backbone.pt",
            "Community YOLOv8-DIOR (unverified)",
        )
