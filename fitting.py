"""
fitting.py

Placard fitting logic.

For each empty region, scans left-to-right, top-to-bottom and greedily places
full placard rectangles inside the region mask, respecting outer margin and
inter-placard spacing.

Perspective-aware mode: when a region has exactly 4 polygon points, the region
is warped to a flat rectangle via homography, rectangles are fitted there, and
the corners are warped back to quadrilaterals that respect the camera angle.
"""

from typing import Optional

import numpy as np

from geometry import (
    bbox_to_xyxy,
    prediction_to_mask,
    compute_region_homography,
    simplify_polygon_to_quad,
    detect_sign_quad_from_blue,
)


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

    reserved = global_reserved if global_reserved is not None else np.zeros((img_h, img_w), dtype=np.uint8)
    placements: list[tuple[int, int, int, int]] = []

    step_x = max(1, placard_w + spacing)

    y = 0
    while y + placard_h <= img_h:
        x = 0
        while x + placard_w <= img_w:
            footprint_clear = np.all(reserved[y:y + placard_h, x:x + placard_w] == 0)
            if footprint_clear and can_place_rectangle(mask, x, y, placard_w, placard_h):
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
    Perspective-aware placement for empty regions.

    Algorithm:
      1. Compute (or use a provided) homography from the sign's perspective quad to flat.
      2. Warp the region mask into the flat rectangle space.
      3. Fit standard rectangular placards in the warped (flat) space.
      4. Warp each fitted rectangle's corners back through the inverse homography.
      5. Return the resulting quadrilaterals (as lists of 4 [x, y] pairs).

    sign_homography: optional pre-computed (H, H_inv, dst_w, dst_h) derived from
        the actual blue sign pixels. When provided, this is used instead of
        computing a local homography from the region polygon's corners. This gives
        correct perspective scaling based on the real sign geometry.

    Falls back and returns [] if no usable homography can be determined.
    """
    import cv2

    if sign_homography is not None:
        H, H_inv, dst_w, dst_h = sign_homography
    else:
        # Fall back: compute local homography from the region polygon corners
        raw_points = region_pred.get("points", [])
        if len(raw_points) < 3:
            return []
        if len(raw_points) == 4:
            points = raw_points
        else:
            points = simplify_polygon_to_quad(raw_points)
            if points is None:
                return []
        H, H_inv, dst_w, dst_h = compute_region_homography(points)

    # Build region mask in original image space and warp to flat rectangle
    orig_mask, _ = prediction_to_mask(region_pred, img_h, img_w)
    warped_mask = cv2.warpPerspective(orig_mask, H, (dst_w, dst_h),
                                      flags=cv2.INTER_NEAREST)

    # Apply margin in the warped (flat) space
    if margin > 0:
        kernel_size = 2 * margin + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        warped_mask = _safe_erode(warped_mask, kernel)

    # Fit rectangles in the flat space (no global_reserved here — handled below)
    flat_placements = place_rectangles_in_region(
        warped_mask, placard_w, placard_h, spacing=spacing
    )

    # Warp each fitted rectangle back to image space as a quadrilateral
    quads: list[list[list[float]]] = []
    for (x1, y1, x2, y2) in flat_placements:
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
        quad = warped_corners.tolist()
        quads.append(quad)

        # Mark the quad footprint on the global reserved array (image space)
        if global_reserved is not None:
            pts_int = warped_corners.astype(np.int32)
            cv2.fillConvexPoly(global_reserved, pts_int, 1)

    return quads


# ---------------------------------------------------------------------------
# Grid-based placement helpers
# ---------------------------------------------------------------------------

def _cluster_1d(values: list[float], gap_threshold: float) -> list[float]:
    """
    Group a sorted list of 1D values into clusters separated by gaps > gap_threshold.
    Returns the mean of each cluster.
    """
    if not values:
        return []
    sv = sorted(values)
    clusters: list[list[float]] = [[sv[0]]]
    for v in sv[1:]:
        if v - clusters[-1][-1] > gap_threshold:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [float(np.mean(c)) for c in clusters]


def infer_and_extend_grid(
    placard_predictions: list[dict],
    placard_w: int,
    placard_h: int,
    spacing: int,
    img_w: int,
    img_h: int,
) -> tuple[list[float], list[float]]:
    """
    Infer a uniform row/column grid from existing placard center positions,
    then extend it to fill the full image dimensions.

    The grid pitch is the median gap between adjacent row/column centers.
    If only one placard is detected, pitch falls back to placard size + spacing.

    Returns (row_centers, col_centers) — y and x coordinates of grid lines.
    """
    centers = [(float(p["x"]), float(p["y"])) for p in placard_predictions]
    if not centers:
        # No reference placards — return a simple default grid
        row_pitch = float(placard_h + spacing)
        col_pitch = float(placard_w + spacing)
        rows = [placard_h / 2 + i * row_pitch for i in range(int(img_h / row_pitch) + 1)]
        cols = [placard_w / 2 + i * col_pitch for i in range(int(img_w / col_pitch) + 1)]
        return rows, cols

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]

    # Cluster into distinct rows and columns
    row_centers = _cluster_1d(ys, placard_h * 0.5)
    col_centers = _cluster_1d(xs, placard_w * 0.5)

    # Estimate pitch from adjacent cluster gaps (fall back to placard size + spacing)
    def pitch_from(vals: list[float], fallback: float) -> float:
        if len(vals) < 2:
            return fallback
        gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return max(float(np.median(gaps)), fallback)

    row_pitch = pitch_from(row_centers, float(placard_h + spacing))
    col_pitch = pitch_from(col_centers, float(placard_w + spacing))

    # Extend rows upward and downward
    half_h = placard_h / 2.0
    while row_centers[0] - row_pitch >= half_h:
        row_centers.insert(0, row_centers[0] - row_pitch)
    while row_centers[-1] + row_pitch <= img_h - half_h:
        row_centers.append(row_centers[-1] + row_pitch)

    # Extend columns left and right
    half_w = placard_w / 2.0
    while col_centers[0] - col_pitch >= half_w:
        col_centers.insert(0, col_centers[0] - col_pitch)
    while col_centers[-1] + col_pitch <= img_w - half_w:
        col_centers.append(col_centers[-1] + col_pitch)

    return row_centers, col_centers


def place_on_grid(
    row_centers: list[float],
    col_centers: list[float],
    placard_w: int,
    placard_h: int,
    mask: np.ndarray,
    global_reserved: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Propose new placards at grid intersections that fall within the region mask
    and are not already reserved.

    Returns (x1, y1, x2, y2) tuples — the same format as place_rectangles_in_region.
    """
    img_h, img_w = mask.shape
    if global_reserved is None:
        global_reserved = np.zeros((img_h, img_w), dtype=np.uint8)

    half_w = placard_w // 2
    half_h = placard_h // 2
    placements: list[tuple[int, int, int, int]] = []

    for ry in row_centers:
        for cx in col_centers:
            x1 = int(round(cx)) - half_w
            y1 = int(round(ry)) - half_h
            x2 = x1 + placard_w
            y2 = y1 + placard_h

            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                continue
            footprint_clear = np.all(global_reserved[y1:y2, x1:x2] == 0)
            if footprint_clear and can_place_rectangle(mask, x1, y1, placard_w, placard_h):
                placements.append((x1, y1, x2, y2))
                global_reserved[y1:y2, x1:x2] = 1

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
    perspective_mode: bool = False,
    image_bgr: np.ndarray | None = None,
    grid_mode: bool = False,
) -> dict:
    """
    Estimate how many new placards can fit in the detected empty regions.

    placard_scale: multiplier applied to the final placard size (0.0–1.0).
    empty_space_scale: multiplier applied to each region's dimensions from its center.
    perspective_mode: if True, detects the sign's blue quad from pixel colors and
        applies a single global perspective transform to all empty regions, then
        warps fitted placard corners back as quads that match the camera angle.
    image_bgr: raw image as a BGR numpy array; required for blue-based perspective
        detection. If not provided, falls back to region-polygon-derived perspective.
    grid_mode: if True, infers a uniform row/column grid from existing placard
        positions and proposes new placards only at grid intersections that fall
        within empty regions. More realistic than the greedy scan.

    Returns a dict with:
      - total_fit: int
      - per_region: list of region dicts, each containing:
          "region_index", "count", "placements" (rects), "quad_placements" (quads),
          "detected_quad", "used_polygon", "used_perspective"
      - placard_w, placard_h: ints (dimensions used after scaling)
      - used_polygon_mode: bool
      - used_perspective_mode: bool
      - sign_quad: the detected 4-corner sign quad (list[dict] | None)
      - grid: {"row_centers": [...], "col_centers": [...]} if grid_mode else None
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
    any_perspective = False

    # ---------------------------------------------------------------
    # Detect the sign's real perspective from blue pixel colors — once,
    # globally.  This gives a single homography based on the actual sign
    # geometry rather than any model-output polygon.
    # ---------------------------------------------------------------
    sign_quad: list[dict] | None = None
    sign_homography: tuple | None = None
    if perspective_mode:
        if image_bgr is not None:
            sign_quad = detect_sign_quad_from_blue(image_bgr)
        if sign_quad is not None:
            H_s, H_s_inv, dst_w_s, dst_h_s = compute_region_homography(sign_quad)
            sign_homography = (H_s, H_s_inv, dst_w_s, dst_h_s)

    # ---------------------------------------------------------------
    # Infer the grid from existing placard positions (once, globally).
    # ---------------------------------------------------------------
    grid_info: dict | None = None
    grid_row_centers: list[float] = []
    grid_col_centers: list[float] = []
    if grid_mode:
        grid_row_centers, grid_col_centers = infer_and_extend_grid(
            placard_predictions, placard_w, placard_h, spacing, img_w, img_h
        )
        grid_info = {"row_centers": grid_row_centers, "col_centers": grid_col_centers}

    global_reserved = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx, region_pred in enumerate(valid_regions):
        region_pred = scale_region_prediction(region_pred, empty_space_scale)

        points = region_pred.get("points", [])
        # Trigger perspective if we have a blue-detected sign quad OR a polygon
        use_perspective = perspective_mode and (
            sign_homography is not None or len(points) >= 3
        )
        used_perspective = False
        quad_placements: list[list[list[float]]] = []
        placements: list[tuple[int, int, int, int]] = []
        detected_quad = sign_quad  # for visualisation — show the blue-detected quad

        if use_perspective:
            quad_placements = place_rectangles_perspective_aware(
                region_pred,
                placard_w,
                placard_h,
                img_h=img_h,
                img_w=img_w,
                spacing=spacing,
                margin=margin,
                global_reserved=global_reserved,
                sign_homography=sign_homography,  # None → falls back to polygon
            )
            if quad_placements:
                used_perspective = True
                any_perspective = True
                count = len(quad_placements)
                used_polygon = bool(points)
                if used_polygon:
                    any_polygon = True
            else:
                # Fall back to grid or greedy fitting
                region_mask, used_polygon = build_region_mask(
                    region_pred, img_h, img_w, margin=margin
                )
                if used_polygon:
                    any_polygon = True
                if grid_mode:
                    placements = place_on_grid(
                        grid_row_centers, grid_col_centers,
                        placard_w, placard_h, region_mask,
                        global_reserved=global_reserved,
                    )
                else:
                    placements = place_rectangles_in_region(
                        region_mask, placard_w, placard_h, spacing=spacing,
                        global_reserved=global_reserved,
                    )
                count = len(placements)
        else:
            region_mask, used_polygon = build_region_mask(
                region_pred, img_h, img_w, margin=margin
            )
            if used_polygon:
                any_polygon = True
            if grid_mode:
                placements = place_on_grid(
                    grid_row_centers, grid_col_centers,
                    placard_w, placard_h, region_mask,
                    global_reserved=global_reserved,
                )
            else:
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
            "quad_placements": quad_placements,
            "detected_quad": detected_quad,
            "used_polygon": used_polygon,
            "used_perspective": used_perspective,
        })

    return {
        "total_fit": total_fit,
        "per_region": per_region,
        "placard_w": placard_w,
        "placard_h": placard_h,
        "used_polygon_mode": any_polygon,
        "used_perspective_mode": any_perspective,
        "sign_quad": sign_quad,
        "grid": grid_info,
    }
