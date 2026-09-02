"""Run any time, without asking Claude first: loads whatever xView training
checkpoint currently exists and runs it on our real (unseen -- never part of
xView's training set) refinery test images, producing annotated comparison
images filtered to just the 8 refinery-relevant classes. Doesn't touch
training in any way -- purely reads the checkpoint file for inference.

Runs on the same GPU training uses, so it competes for resources while it's
running -- use --site to check just one image instead of always paying for
both (Hamburg in particular is large enough that a full run takes several
minutes even without training competing for the GPU at the same time).

Usage: python eval_xview_checkpoint.py [--conf CONF] [--site {bazan,hamburg,all}]
  --conf  detection confidence threshold (default 0.2 -- lower than this
          repo's usual 0.99 convention, since a model this early in training
          won't yet produce very confident predictions)
  --site  which test image to run (default: all)
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / "oil_refinery" / "xview-yolov3"
sys.path.insert(0, str(SCRATCH))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from models import Darknet  # noqa: E402
from utils.utils import load_classes, non_max_suppression  # noqa: E402

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--conf", type=float, default=0.2)
_arg_parser.add_argument("--site", choices=["bazan", "hamburg", "all"], default="all")
_args = _arg_parser.parse_args()

CONF_THRES = _args.conf
NMS_THRES = 0.4
IMG_SIZE = 800  # matches training's -img_size, so the model sees familiar-scale input
CFG_PATH = SCRATCH / "cfg" / "c60_a30symmetric.cfg"
CLASS_NAMES_PATH = SCRATCH / "data" / "xview.names"
PRIORS_PATH = SCRATCH / "utils" / "targets_c60.mat"  # class_mu/sigma shape priors -- same either mat file

# same 8 classes train.py tracks per-class metrics for -- see TARGET_CLASSES there
TARGET_CLASSES = {
    "Truck w/Liquid": 15, "Crane Truck": 16, "Tank car": 21, "Oil Tanker": 32,
    "Tower crane": 34, "Container Crane": 35, "Mobile Crane": 38, "Storage Tank": 55,
}
TARGET_IDX = set(TARGET_CLASSES.values())

TEST_IMAGES = {
    "bazan": REPO_ROOT / "oil_refinery" / "probe" / "site.jpg",
    "hamburg": REPO_ROOT / "oil_refinery" / "probe" / "holborn_hamburg" / "site.jpg",
}

OUT_DIR = REPO_ROOT / "oil_refinery" / "checkpoint_eval"
OUT_DIR.mkdir(exist_ok=True)

RGB_MEAN = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
RGB_STD = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230),
]
CLASS_COLOR = dict(zip(TARGET_CLASSES.keys(), PALETTE))


def find_checkpoint() -> Path:
    for name in ("best.pt", "latest.pt"):
        p = SCRATCH / "weights" / name
        if p.exists():
            return p
    raise SystemExit("No checkpoint found in weights/ -- has training completed at least one epoch?")


def load_model(checkpoint_path: Path, device: torch.device):
    model = Darknet(str(CFG_PATH), IMG_SIZE)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt.get("epoch")


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


LABEL_FONT = load_font(16)
TITLE_FONT = load_font(28)


def run_inference(model, image_path: Path, device: torch.device, mat_priors: dict) -> list[tuple]:
    """Tile the (large) image into IMG_SIZE chips, run the model on each,
    offset back to full-image coordinates, then NMS across everything --
    same approach as this repo's own detect.py."""
    img0 = cv2.imread(str(image_path))
    img = img0[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img -= RGB_MEAN
    img /= RGB_STD

    length = IMG_SIZE
    ni = int(np.ceil(img.shape[1] / length))
    nj = int(np.ceil(img.shape[2] / length))
    preds = []
    with torch.no_grad():
        for i in range(ni):
            for j in range(nj):
                y2 = min((i + 1) * length, img.shape[1])
                y1 = y2 - length
                x2 = min((j + 1) * length, img.shape[2])
                x1 = x2 - length
                chip = torch.from_numpy(img[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                pred = model(chip)
                pred = pred[pred[:, :, 4] > CONF_THRES]
                if len(pred) > 0:
                    pred[:, 0] += x1
                    pred[:, 1] += y1
                    preds.append(pred.unsqueeze(0))

    if not preds:
        return []

    detections = non_max_suppression(torch.cat(preds, 1), CONF_THRES, NMS_THRES, mat_priors, img0, None, device)
    if not detections or detections[0] is None:
        return []

    out = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
        cls_pred = int(cls_pred)
        if cls_pred in TARGET_IDX:
            out.append((float(x1), float(y1), float(x2), float(y2), float(conf * cls_conf), cls_pred))
    return out


def annotate(image_path: Path, title: str, dets: list[tuple], class_names: list[str], out_path: Path) -> None:
    idx_to_name = {v: k for k, v in TARGET_CLASSES.items()}
    base = Image.open(image_path).convert("RGB")
    title_h = 44
    canvas = Image.new("RGB", (base.width, base.height + title_h), (20, 20, 20))
    canvas.paste(base, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, font=TITLE_FONT, fill=(255, 255, 255))

    for x1, y1, x2, y2, conf, cls_pred in dets:
        name = idx_to_name[cls_pred]
        color = CLASS_COLOR[name]
        pts = [(x1, y1 + title_h), (x2, y1 + title_h), (x2, y2 + title_h), (x1, y2 + title_h)]
        draw.polygon(pts, outline=color, width=3)
        label = f"{name} {conf:.2f}"
        tb = draw.textbbox((x1, y1 + title_h), label, font=LABEL_FONT)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
        draw.text((x1, y1 + title_h), label, font=LABEL_FONT, fill=(0, 0, 0))

    if not dets:
        draw.text((10, title_h + 10), f"No target-class detections at conf_thres={CONF_THRES}",
                   font=LABEL_FONT, fill=(255, 80, 80))

    canvas.save(out_path, format="PNG")
    print(f"Wrote {out_path} ({len(dets)} target-class detections)")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_checkpoint()
    model, epoch = load_model(checkpoint_path, device)
    class_names = load_classes(str(CLASS_NAMES_PATH))
    mat_priors = scipy.io.loadmat(str(PRIORS_PATH))

    epoch_label = f"epoch {epoch}" if epoch is not None else "unknown epoch"
    print(f"Loaded {checkpoint_path.name} ({epoch_label}) on {device}")

    sites = TEST_IMAGES if _args.site == "all" else {_args.site: TEST_IMAGES[_args.site]}
    for site_name, image_path in sites.items():
        if not image_path.exists():
            print(f"Skipping {site_name}: {image_path} not found")
            continue
        t0 = time.time()
        dets = run_inference(model, image_path, device, mat_priors)
        print(f"{site_name}: {len(dets)} detections in {time.time() - t0:.1f}s")
        title = f"Our xView checkpoint ({epoch_label}, conf>={CONF_THRES}) -- {site_name}"
        out_path = OUT_DIR / f"{site_name}_ep{epoch}_conf{CONF_THRES}.png"
        annotate(image_path, title, dets, class_names, out_path)


if __name__ == "__main__":
    main()
