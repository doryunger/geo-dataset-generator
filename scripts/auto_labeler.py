"""
Auto-labeling from DINOv2's own patch tokens — no separate segmentation model.

A candidate tile was accepted because its whole-image (CLS) embedding resembles the seed's.
That same forward pass also produced per-patch tokens that retain rough spatial position
(discarded for the accept/reject decision, which only needs the whole-tile summary). Comparing
each patch to the seed/confirmed-exemplar vectors gives a rough heatmap of *where* the matching
content sits — the same signal that caused the match, just not thrown away this time.

This replaced an earlier FastSAM-based approach: FastSAM is a generic "what object is at this
point" segmenter trained on ground-level photos, prompted blindly at each tile's center — a poor
fit for thin aerial features like fences, and blind to what the seed actually looked like. This
approach is tied directly to the seed by construction and needs no extra model or dependency.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

UPSCALE = 16  # the patch grid (e.g. 16x16) is far too coarse to trace a polygon from directly --
              # tracing contours straight off it gives a blocky staircase shape, and thin
              # single-patch bridges between blobs produce self-touching ("bowtie") contours that
              # aren't valid simple polygons. Upscaling first gives morphological cleanup and
              # simplification room to work with. Tuned empirically against real candidates
              # (fence round 7): smaller kernels/epsilon left 20+ jagged vertices even after the
              # shapely validity fix -- technically simple, but not "a proper polygon" visually.
MORPH_KERNEL_PX = 16
SIMPLIFY_EPSILON_FRAC = 0.03
OUTLIER_MAD_MULTIPLIER = 2.5   # a vertex further from the polygon's centroid than
                                # median + this*MAD gets pulled in to that distance instead of
                                # deleted -- keeps the vertex count/winding intact.
SHARP_ANGLE_DEG = 60           # a vertex whose interior angle is tighter than this is a "peak" --
                                # usually a single stray/misaligned raster point, not real shape --
                                # and gets pulled toward its neighbors' midpoint instead of deleted.
SHARP_ANGLE_PULL_FRAC = 0.9
SHARP_ANGLE_PASSES = 5         # one pass often just softens a peak rather than resolving it --
                                # repeating lets it converge. Tuned empirically: 1 pass left visible
                                # zigzags, 5 passes cleaned them up without flattening genuine
                                # right-angle corners (e.g. a plus-shaped region stayed intact).
EDGE_DENSITY_PERCENTILE = 40   # a patch DINOv2 calls a strong match still gets dropped if its
                                # pixel-level edge density falls below this percentile within the
                                # tile -- i.e. it looks visually flat, like a shadow interior.


class PatchLabeler:
    def __init__(self, embedder):
        self.embedder = embedder

    def label(
        self, image_path, query_matrix: np.ndarray, *,
        percentile: float = 75.0, min_contrast: float = 0.05, min_area_frac: float = 0.01,
    ) -> list[list[list[float]]] | None:
        """query_matrix: [Q, D] normalized reference vectors — the same seed + confirmed-exemplar
        set used to accept this candidate in the first place. Returns a list of normalized [0,1]
        polygons, one per separate region of patches similar to those references (a tile can
        legitimately contain more than one instance), or None if nothing localized stands out
        (e.g. the match is diffuse/uniform across the whole tile)."""
        _, patch_grid = self.embedder.embed_with_patches(image_path)
        h, w, d = patch_grid.shape
        flat = patch_grid.reshape(-1, d)
        sims = (flat @ query_matrix.T).max(axis=1)  # best match to any reference, per patch

        thresh = np.percentile(sims, percentile)
        hot = sims >= thresh
        contrast = sims[hot].mean() - float(np.median(sims))
        if contrast < min_contrast:
            return None  # no clear local concentration -- roughly uniform match everywhere

        # A cast shadow is a smooth, uniform, elongated dark region -- in DINOv2's embedding
        # space that can look deceptively similar to the seed's own linear structure, even though
        # visually it has none of the actual texture (posts, mesh, hard edges) a real fence has.
        # Cross-checking against genuine pixel-level edge density (computed straight from the
        # image, no model needed) and requiring hot patches to clear both filters out flat/shadow
        # regions that only "matched" on smoothness. Confirmed directly: on a real false-positive
        # tile, the patches this drops line up exactly with a roof's cast shadow.
        edge_density = _patch_edge_density(image_path, h, w).flatten()
        edge_thresh = np.percentile(edge_density, EDGE_DENSITY_PERCENTILE)
        hot = hot & (edge_density >= edge_thresh)
        if not hot.any():
            return None

        mask = hot.reshape(h, w).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((MORPH_KERNEL_PX, MORPH_KERNEL_PX), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # bridge small gaps between blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop single-patch noise specks
        mh, mw = mask.shape

        # RETR_EXTERNAL gives one contour per separate blob -- every one clearing the area
        # threshold becomes its own label, instead of only ever keeping the single largest.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) / (mh * mw) < min_area_frac:
                continue

            # Simplify away the raster staircase, then let shapely fix any remaining topology
            # issues (self-intersections/pinch points) rather than handing back an invalid
            # polygon as-is.
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

            coords = list(poly.exterior.coords)[:-1]  # drop the closing duplicate of point 0
            if len(coords) < 3:
                continue
            for _ in range(SHARP_ANGLE_PASSES):
                coords = _smooth_sharp_corners(coords)
            coords = _pull_in_outlier_vertices(coords)
            polygons.append([[x / mw, y / mh] for x, y in coords])

        return polygons or None


def _patch_edge_density(image_path, h: int, w: int) -> np.ndarray:
    """Mean absolute Laplacian (a standard edge/texture-richness measure) within each patch
    cell, computed at the image's real resolution -- not the coarse patch grid -- so a smooth
    shadow interior scores low even though it sits inside a "hot" DINOv2 patch."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.full((h, w), np.inf)  # fail open -- don't block labeling on a decode error
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
    """A vertex forming a very tight angle with its neighbors is usually a single stray raster
    point creating a needle-like 'peak', not genuine shape -- pull it toward the midpoint of its
    neighbors (partial smoothing) instead of deleting it, so vertex count/winding stay intact."""
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
