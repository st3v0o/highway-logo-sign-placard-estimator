"""
fitting.py

Placard fitting logic.

For each empty region detected by the empty-space model, greedily places
placard rectangles inside the region mask, respecting outer margin and
inter-placard spacing.

Perspective-aware mode: when a Blue-Logo-Sign detection is available, its
polygon defines a homography that warps all empty regions to flat space,
fits rectangles there, then warps the corners back as quadrilaterals that
respect the camera angle.
"""

import numpy as np

from geometry import (
    prediction_to_mask,
    compute_region_homography,
    simplify_polygon_to_quad,
    sign_quad_from_pred,
)


# ---------------------------------------------------------------------------
# Region scaling
# ---------------------------------------------------------------------------

def scale_region_prediction(pred: dict, scale: float) -> dict:
    """Shrink a region prediction from its center by *scale* (0.0–1.0)."""
    if scale >= 1.0:
        return pred
    new_pred = dict(pred)
    cx, cy = pred["x"], pred["y"]
    new_pred["width"]  = pred["width"]  * scale
    new_pred["height"] = pred["height"] * scale
    points = pred.get("points")
    if points and len(points) >= 3:
        new_pred["points"] = [
            {"x": cx + (p["x"] - cx) * scale, "y": cy + (p["y"] - cy) * scale}
            for p in points
        ]
    return new_pred


# ---------------------------------------------------------------------------
# Mask / erosion helpers
# ---------------------------------------------------------------------------

def _safe_erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.erode(mask, kernel, iterations=1)


def build_region_mask(
    pred: dict, img_h: int, img_w: int, margin: int = 0
) -> tuple[np.ndarray, bool]:
    """Build a binary region mask with optional margin erosion."""
    mask, used_polygon = prediction_to_mask(pred, img_h, img_w)
    if margin > 0:
        kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
        mask = _safe_erode(mask, kernel)
    return mask, used_polygon


# ---------------------------------------------------------------------------
# Rectangle placement
# ---------------------------------------------------------------------------

def can_place_rectangle(
    mask: np.ndarray, x: int, y: int, pw: int, ph: int
) -> bool:
    """Return True if a (pw × ph) placard fits entirely inside the mask at (x, y)."""
    img_h, img_w = mask.shape
    if x < 0 or y < 0 or x + pw > img_w or y + ph > img_h:
        return False
    return bool(np.all(mask[y:y + ph, x:x + pw] == 1))


def place_rectangles_in_region(
    mask: np.ndarray,
    placard_w: int,
    placard_h: int,
    spacing: int = 4,
    global_reserved: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Greedily place non-overlapping placard rectangles inside the region mask.
    Scans left-to-right, top-to-bottom.
    Returns a list of (x1, y1, x2, y2) tuples.
    """
    if placard_w <= 0 or placard_h <= 0:
        return []

    img_h, img_w = mask.shape
    reserved = (global_reserved if global_reserved is not None
                else np.zeros((img_h, img_w), dtype=np.uint8))
    placements: list[tuple[int, int, int, int]] = []
    step_x = max(1, placard_w + spacing)

    y = 0
    while y + placard_h <= img_h:
        x = 0
        while x + placard_w <= img_w:
            if (np.all(reserved[y:y + placard_h, x:x + placard_w] == 0)
                    and can_place_rectangle(mask, x, y, placard_w, placard_h)):
                placements.append((x, y, x + placard_w, y + placard_h))
                rx1 = max(0, x - spacing)
                ry1 = max(0, y - spacing)
                rx2 = min(img_w, x + placard_w + spacing)
                ry2 = min(img_h, y + placard_h + spacing)
                reserved[ry1:ry2, rx1:rx2] = 1
                x += step_x
            else:
                x += 1
        y += 1

    return placements


# ---------------------------------------------------------------------------
# Perspective-aware placement
# ---------------------------------------------------------------------------

def place_rectangles_perspective_aware(
    region_pred: dict,
    placard_w: int,
    placard_h: int,
    img_h: int,
    img_w: int,
    spacing: int = 4,
    margin: int = 0,
    global_reserved: np.ndarray | None = None,
    sign_homography: tuple | None = None,
) -> list[list[list[float]]]:
    """
    Perspective-aware placement for a single empty region.

    Uses the sign homography (derived from the Blue-Logo-Sign polygon) to warp
    the region to flat space, fits rectangles greedily, then warps corners back.
    Returns a list of quads — each quad is 4 [x, y] pairs in image space.
    """
    import cv2

    if sign_homography is not None:
        H, H_inv, dst_w, dst_h = sign_homography
    else:
        raw_points = region_pred.get("points", [])
        if len(raw_points) < 3:
            return []
        points = (raw_points if len(raw_points) == 4
                  else simplify_polygon_to_quad(raw_points))
        if points is None:
            return []
        H, H_inv, dst_w, dst_h = compute_region_homography(points)

    orig_mask, _ = prediction_to_mask(region_pred, img_h, img_w)
    warped_mask = cv2.warpPerspective(orig_mask, H, (dst_w, dst_h),
                                      flags=cv2.INTER_NEAREST)

    if margin > 0:
        kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
        warped_mask = _safe_erode(warped_mask, kernel)

    flat_placements = place_rectangles_in_region(
        warped_mask, placard_w, placard_h, spacing=spacing
    )

    quads: list[list[list[float]]] = []
    for (x1, y1, x2, y2) in flat_placements:
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
        quads.append(warped_corners.tolist())
        if global_reserved is not None:
            cv2.fillConvexPoly(global_reserved, warped_corners.astype(np.int32), 1)

    return quads


# ---------------------------------------------------------------------------
# Top-level estimator
# ---------------------------------------------------------------------------

def estimate_total_capacity(
    empty_space_predictions: list[dict],
    img_h: int,
    img_w: int,
    default_placard_w: int = 90,
    default_placard_h: int = 70,
    margin: int = 8,
    spacing: int = 4,
    min_confidence: float = 0.4,
    placard_scale: float = 1.0,
    empty_space_scale: float = 1.0,
    sign_prediction: dict | None = None,
) -> dict:
    """
    Estimate how many new placards can fit in the empty regions.

    sign_prediction: the highest-confidence Blue-Logo-Sign detection.
        Its polygon (or bbox) defines the perspective quad for all empty regions.
        When None, falls back to per-region polygon perspective or flat fitting.

    Returns:
        total_fit           – total placard slots across all regions
        per_region          – list of per-region result dicts
        placard_w/h         – dimensions used (after scaling)
        used_perspective_mode – bool
        sign_quad           – the 4-corner sign boundary (list[dict] | None)
    """
    placard_w = max(1, int(default_placard_w * placard_scale))
    placard_h = max(1, int(default_placard_h * placard_scale))

    valid_regions = [
        p for p in empty_space_predictions
        if p.get("confidence", 0) >= min_confidence
    ]

    # Build the global sign homography once
    sign_quad: list[dict] | None = None
    sign_homography: tuple | None = None
    if sign_prediction is not None:
        sign_quad = sign_quad_from_pred(sign_prediction)
        H_s, H_s_inv, dst_w_s, dst_h_s = compute_region_homography(sign_quad)
        sign_homography = (H_s, H_s_inv, dst_w_s, dst_h_s)

    global_reserved = np.zeros((img_h, img_w), dtype=np.uint8)

    per_region: list[dict] = []
    total_fit = 0
    any_perspective = False

    for idx, region_pred in enumerate(valid_regions):
        region_pred = scale_region_prediction(region_pred, empty_space_scale)
        points = region_pred.get("points", [])

        use_perspective = sign_homography is not None or len(points) >= 3
        used_perspective = False
        quad_placements: list[list[list[float]]] = []
        placements: list[tuple[int, int, int, int]] = []

        if use_perspective:
            quad_placements = place_rectangles_perspective_aware(
                region_pred, placard_w, placard_h,
                img_h=img_h, img_w=img_w,
                spacing=spacing, margin=margin,
                global_reserved=global_reserved,
                sign_homography=sign_homography,
            )
            count = len(quad_placements)
            used_perspective = True
            any_perspective  = True
        else:
            mask, _ = build_region_mask(region_pred, img_h, img_w, margin=margin)
            placements = place_rectangles_in_region(
                mask, placard_w, placard_h, spacing=spacing,
                global_reserved=global_reserved,
            )
            count = len(placements)

        total_fit += count
        per_region.append({
            "region_index":    idx,
            "count":           count,
            "placements":      placements,
            "quad_placements": quad_placements,
            "used_perspective": used_perspective,
        })

    return {
        "total_fit":             total_fit,
        "per_region":            per_region,
        "placard_w":             placard_w,
        "placard_h":             placard_h,
        "used_perspective_mode": any_perspective,
        "sign_quad":             sign_quad,
    }
