"""
fitting.py

Placard fitting logic.

For each empty region, scans left-to-right, top-to-bottom and greedily places
full placard rectangles inside the region mask, respecting outer margin and
inter-placard spacing.
"""

from typing import Optional

import numpy as np

from geometry import bbox_to_xyxy, prediction_to_mask


# ---------------------------------------------------------------------------
# Placard size estimation
# ---------------------------------------------------------------------------

def estimate_placard_size_from_detections(
    predictions: list[dict],
    min_confidence: float = 0.0,
) -> Optional[tuple[int, int]]:
    """
    Estimate a standard placard size (width, height) from existing detections.

    Uses the median width and height of confident predictions.
    Returns None if no valid predictions are available.
    """
    valid = [
        p for p in predictions
        if p.get("confidence", 0) >= min_confidence
    ]
    if not valid:
        return None

    widths = [int(round(p["width"])) for p in valid]
    heights = [int(round(p["height"])) for p in valid]
    w = int(np.median(widths))
    h = int(np.median(heights))
    return (max(w, 1), max(h, 1))


# ---------------------------------------------------------------------------
# Region scaling helper
# ---------------------------------------------------------------------------

def scale_region_prediction(pred: dict, scale: float) -> dict:
    """
    Shrink a region prediction from its center by *scale* (0.0–1.0).

    Works for both bbox-only and polygon predictions:
      - bbox: width and height are multiplied by scale (center x/y unchanged).
      - polygon: each point is moved toward the centroid by scale.

    Returns a shallow-copy of pred with modified fields.
    """
    if scale >= 1.0:
        return pred

    new_pred = dict(pred)
    cx = pred["x"]
    cy = pred["y"]

    # Scale bbox dimensions
    new_pred["width"] = pred["width"] * scale
    new_pred["height"] = pred["height"] * scale

    # Scale polygon points if present
    points = pred.get("points")
    if points and len(points) >= 3:
        new_pred["points"] = [
            {"x": cx + (p["x"] - cx) * scale, "y": cy + (p["y"] - cy) * scale}
            for p in points
        ]

    return new_pred


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------

def build_region_mask(
    pred: dict,
    img_h: int,
    img_w: int,
    margin: int = 0,
) -> tuple[np.ndarray, bool]:
    """
    Build a binary mask for an empty-space prediction.

    Applies an optional inner erosion (margin) to keep placements away from
    the region boundary.

    Returns (mask, used_polygon).
    """
    mask, used_polygon = prediction_to_mask(pred, img_h, img_w)

    if margin > 0:
        kernel_size = 2 * margin + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = _safe_erode(mask, kernel)

    return mask, used_polygon


def _safe_erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Erode a binary mask; return zeros if result is all zeros."""
    import cv2
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded


# ---------------------------------------------------------------------------
# Rectangle placement helpers
# ---------------------------------------------------------------------------

def can_place_rectangle(
    mask: np.ndarray,
    x: int,
    y: int,
    pw: int,
    ph: int,
) -> bool:
    """
    Return True if a placard rectangle (pw x ph) can be placed at (x, y)
    such that the entire rectangle falls within the region mask.

    (x, y) is the top-left corner of the proposed placement.
    """
    img_h, img_w = mask.shape
    if x < 0 or y < 0 or x + pw > img_w or y + ph > img_h:
        return False
    region = mask[y: y + ph, x: x + pw]
    return bool(np.all(region == 1))


def place_rectangles_in_region(
    mask: np.ndarray,
    placard_w: int,
    placard_h: int,
    spacing: int = 4,
    global_reserved: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Greedily place non-overlapping placard rectangles inside the region mask.

    Scans left-to-right, top-to-bottom. After placing each placard, reserves
    spacing around it before trying the next position.

    global_reserved: optional shared array (same shape as mask) tracking
    positions already occupied by placements from other regions. Mutated
    in-place so that subsequent regions respect these placements too.

    Returns a list of (x1, y1, x2, y2) tuples for each proposed placement.
    """
    if placard_w <= 0 or placard_h <= 0:
        return []

    img_h, img_w = mask.shape

    # Use the shared global reserved array if provided, otherwise a local one.
    reserved = global_reserved if global_reserved is not None else np.zeros((img_h, img_w), dtype=np.uint8)
    placements: list[tuple[int, int, int, int]] = []

    step_x = max(1, placard_w + spacing)

    y = 0
    while y + placard_h <= img_h:
        x = 0
        while x + placard_w <= img_w:
            # Check the ENTIRE proposed rectangle footprint is unreserved
            # across all regions (not just the top-left corner).
            footprint_clear = np.all(reserved[y:y + placard_h, x:x + placard_w] == 0)
            if footprint_clear and can_place_rectangle(mask, x, y, placard_w, placard_h):
                placements.append((x, y, x + placard_w, y + placard_h))
                # Mark reserved area (placard + spacing on all sides)
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
# Top-level estimator
# ---------------------------------------------------------------------------

def estimate_total_capacity(
    empty_space_predictions: list[dict],
    placard_predictions: list[dict],
    img_h: int,
    img_w: int,
    default_placard_w: int = 90,
    default_placard_h: int = 70,
    margin: int = 8,
    spacing: int = 4,
    min_confidence: float = 0.4,
    estimate_size_from_detections: bool = True,
    placard_scale: float = 1.0,
    empty_space_scale: float = 1.0,
) -> dict:
    """
    Estimate how many new placards can fit in the detected empty regions.

    placard_scale: multiplier applied to the final placard size (0.0–1.0).
    Values below 1.0 allow placards to fit into tighter spaces.

    empty_space_scale: multiplier applied to each detected empty region's
    dimensions from its center (0.0–1.0). Values below 1.0 shrink the
    usable area, reducing how many placards fit per region.

    Returns a dict with:
      - total_fit: int
      - per_region: list of {"region_index": int, "count": int, "placements": [...]}
      - placard_w: int (width used after scaling)
      - placard_h: int (height used after scaling)
      - used_polygon: bool (True if at least one region used polygon mode)
    """
    # Determine placard size
    placard_w = default_placard_w
    placard_h = default_placard_h
    if estimate_size_from_detections and placard_predictions:
        estimated = estimate_placard_size_from_detections(
            placard_predictions, min_confidence=min_confidence
        )
        if estimated:
            placard_w, placard_h = estimated

    # Apply scale factor
    placard_w = max(1, int(placard_w * placard_scale))
    placard_h = max(1, int(placard_h * placard_scale))

    # Filter empty regions by confidence
    valid_regions = [
        p for p in empty_space_predictions
        if p.get("confidence", 0) >= min_confidence
    ]

    per_region = []
    total_fit = 0
    any_polygon = False

    # Single reserved map shared across ALL regions — prevents placements
    # from different detected regions from overlapping each other.
    global_reserved = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx, region_pred in enumerate(valid_regions):
        region_pred = scale_region_prediction(region_pred, empty_space_scale)
        region_mask, used_polygon = build_region_mask(
            region_pred, img_h, img_w, margin=margin
        )
        if used_polygon:
            any_polygon = True

        placements = place_rectangles_in_region(
            region_mask, placard_w, placard_h, spacing=spacing,
            global_reserved=global_reserved,
        )

        count = len(placements)
        total_fit += count
        per_region.append({
            "region_index": idx,
            "count": count,
            "placements": placements,
            "used_polygon": used_polygon,
        })

    return {
        "total_fit": total_fit,
        "per_region": per_region,
        "placard_w": placard_w,
        "placard_h": placard_h,
        "used_polygon_mode": any_polygon,
    }
