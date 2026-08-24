# auto_labeler.py

Auto-labeling from DINOv2's own patch tokens, no separate segmentation model. A candidate tile
was accepted because its whole-image (CLS) embedding resembles the seed's; that same forward pass
also produced per-patch tokens with rough spatial position. Comparing each patch to the seed/
confirmed-exemplar vectors gives a rough heatmap of *where* the matching content sits — the same
signal that caused the match, just not discarded this time.

Replaced an earlier FastSAM-based approach: FastSAM is a generic ground-level-photo segmenter
prompted blindly at each tile's center — a poor fit for thin aerial features like fences, and
blind to what the seed actually looked like. This approach is tied directly to the seed by
construction and needs no extra model/dependency.

## Constants (all tuned empirically against real candidates, fence round 7)

- `UPSCALE = 16`: the raw patch grid (e.g. 16x16) is far too coarse to trace a polygon from
  directly — contours straight off it are blocky, and thin single-patch bridges between blobs
  produce self-touching ("bowtie") contours that aren't valid simple polygons. Upscaling first
  gives morphological cleanup/simplification room to work with. Smaller kernels/epsilon left 20+
  jagged vertices even after the shapely validity fix.
- `OUTLIER_MAD_MULTIPLIER = 2.5` (`_pull_in_outlier_vertices`): a vertex further from the
  polygon's centroid than median + this*MAD gets pulled in to that distance instead of deleted —
  keeps vertex count/winding intact.
- `SHARP_ANGLE_DEG = 60` (`_smooth_sharp_corners`): a vertex whose interior angle is tighter than
  this is treated as a stray/misaligned raster point, not genuine shape, and gets pulled toward
  its neighbors' midpoint instead of deleted.
- `SHARP_ANGLE_PASSES = 5`: one pass often just softens a peak rather than resolving it. 1 pass
  left visible zigzags; 5 cleaned them up without flattening genuine right-angle corners (e.g. a
  plus-shaped region stayed intact).
- `EDGE_DENSITY_PERCENTILE = 40`: a patch DINOv2 calls a strong match still gets dropped if its
  pixel-level edge density falls below this percentile within the tile — i.e. it looks visually
  flat, like a shadow interior.

## label()

`sims`: best match to any reference vector, per patch. `contrast < min_contrast` means no clear
local concentration — roughly uniform match everywhere, so return None.

The edge-density cross-check exists because a cast shadow is a smooth, uniform, elongated dark
region — in DINOv2's embedding space that can look deceptively similar to the seed's own linear
structure, even though visually it has none of the actual texture (posts, mesh, hard edges) a real
fence has. Requiring hot patches to also clear a genuine pixel-level edge-density threshold
filters out flat/shadow regions that only "matched" on smoothness — confirmed directly: on a real
false-positive tile, the patches this drops line up exactly with a roof's cast shadow.

`cv2.RETR_EXTERNAL` gives one contour per separate blob, so every one clearing the area threshold
becomes its own label instead of only ever keeping the single largest (a tile can legitimately
contain more than one instance).

Contour cleanup order: simplify away the raster staircase (`cv2.approxPolyDP`) -> let shapely fix
any remaining topology issues (self-intersections/pinch points) -> smooth sharp corners -> pull in
outlier vertices.
