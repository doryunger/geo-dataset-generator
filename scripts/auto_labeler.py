"""Auto-labeling from DINOv2's own patch tokens -- no separate segmentation model."""
import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

UPSCALE = 16
MORPH_KERNEL_PX = 16
SIMPLIFY_EPSILON_FRAC = 0.03
OUTLIER_MAD_MULTIPLIER = 2.5
SHARP_ANGLE_DEG = 60
SHARP_ANGLE_PULL_FRAC = 0.9
SHARP_ANGLE_PASSES = 5
EDGE_DENSITY_PERCENTILE = 40


class PatchLabeler:
    def __init__(self, embedder):
        self.embedder = embedder

    def label(
        self, image_path, query_matrix: np.ndarray, *,
        percentile: float = 75.0, min_contrast: float = 0.05, min_area_frac: float = 0.01,
    ) -> list[list[list[float]]] | None:
        _, patch_grid = self.embedder.embed_with_patches(image_path)
        h, w, d = patch_grid.shape
        flat = patch_grid.reshape(-1, d)
        sims = (flat @ query_matrix.T).max(axis=1)

        thresh = np.percentile(sims, percentile)
        hot = sims >= thresh
        contrast = sims[hot].mean() - float(np.median(sims))
        if contrast < min_contrast:
            return None

        edge_density = _patch_edge_density(image_path, h, w).flatten()
        edge_thresh = np.percentile(edge_density, EDGE_DENSITY_PERCENTILE)
        hot = hot & (edge_density >= edge_thresh)
        if not hot.any():
            return None

        mask = hot.reshape(h, w).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((MORPH_KERNEL_PX, MORPH_KERNEL_PX), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mh, mw = mask.shape

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) / (mh * mw) < min_area_frac:
                continue

            epsilon = SIMPLIFY_EPSILON_FRAC * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue

            poly = ShapelyPolygon([pt[0] for pt in simplified])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.geom_type != "Polygon" or poly.exterior is None:
                continue

            coords = list(poly.exterior.coords)[:-1]
            if len(coords) < 3:
                continue
            for _ in range(SHARP_ANGLE_PASSES):
                coords = _smooth_sharp_corners(coords)
            coords = _pull_in_outlier_vertices(coords)
            polygons.append([[x / mw, y / mh] for x, y in coords])

        return polygons or None


def _patch_edge_density(image_path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.full((h, w), np.inf)
    edges = np.abs(cv2.Laplacian(img, cv2.CV_32F))
    img_h, img_w = img.shape
    density = np.zeros((h, w), dtype=np.float32)
    cell_h, cell_w = img_h / h, img_w / w
    for i in range(h):
        y0, y1 = int(i * cell_h), int((i + 1) * cell_h)
        for j in range(w):
            x0, x1 = int(j * cell_w), int((j + 1) * cell_w)
            density[i, j] = edges[y0:y1, x0:x1].mean()
    return density


def _smooth_sharp_corners(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.array(coords, dtype=float)
    n = len(pts)
    if n < 4:
        return coords
    out = pts.copy()
    for i in range(n):
        prev_pt, curr, next_pt = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = prev_pt - curr, next_pt - curr
        len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg < SHARP_ANGLE_DEG:
            midpoint = (prev_pt + next_pt) / 2
            out[i] = curr + (midpoint - curr) * SHARP_ANGLE_PULL_FRAC
    return [tuple(p) for p in out]


def _pull_in_outlier_vertices(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.array(coords, dtype=float)
    centroid = pts.mean(axis=0)
    offsets = pts - centroid
    dists = np.linalg.norm(offsets, axis=1)
    median = np.median(dists)
    mad = np.median(np.abs(dists - median)) or 1e-6
    cap = median + OUTLIER_MAD_MULTIPLIER * mad
    scale = np.minimum(1.0, cap / np.maximum(dists, 1e-6))
    pulled = centroid + offsets * scale[:, None]
    return [tuple(p) for p in pulled]
