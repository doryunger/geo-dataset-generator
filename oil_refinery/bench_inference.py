"""Reality check: how long does a scan of ~10 z18 tiles actually take on this
CPU-only machine, split into one-time model load vs. steady-state per-request cost."""
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# 10 Mapbox @2x tiles (512px each) arranged 5x2, matching a plausible map-page viewport
TILE_PX = 512
composite_w, composite_h = TILE_PX * 5, TILE_PX * 2

# Crop a real refinery region at that pixel size from the Hamburg probe image
# instead of a blank canvas, so timing reflects real image content/complexity.
src = Image.open(REPO_ROOT / "oil_refinery" / "probe" / "holborn_hamburg" / "site.jpg")
crop = src.resize((composite_w, composite_h))
test_image_path = REPO_ROOT / "oil_refinery" / "probe" / "bench_10tiles.jpg"
crop.save(test_image_path)
print(f"Test image: {composite_w}x{composite_h}px ({test_image_path})")

from ultralytics import YOLO  # noqa: E402

for weights, label in [
    (str(REPO_ROOT / "weights" / "yolo11n-obb.pt"), "YOLO11n-OBB (nano, DOTAv1)"),
    (str(REPO_ROOT / "weights" / "DIOR_yolov8s_backbone.pt"), "YOLOv8s-DIOR (small)"),
]:
    t0 = time.perf_counter()
    model = YOLO(weights)
    load_s = time.perf_counter() - t0

    # warm-up call (first call includes graph/backend setup, exclude from steady-state avg)
    model.predict(source=str(test_image_path), conf=0.15, imgsz=1536, verbose=False)

    n_runs = 5
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(source=str(test_image_path), conf=0.15, imgsz=1536, verbose=False)
    avg_s = (time.perf_counter() - t0) / n_runs

    print(f"{label}: load={load_s:.2f}s, steady-state inference={avg_s:.2f}s/scan (avg of {n_runs})")
