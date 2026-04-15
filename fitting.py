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
    sign_quad_from_polygon,
    sign_quad_for_homography,
    sign_polygon_from_pred,
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

def _grid_lines(origin: float, step: int, limit: int) -> list[float]:
    """Return all grid line positions within [0, limit] anchored at origin."""
    lines: list[float] = []
    pos = origin
    while pos <= limit:
        lines.append(pos)
        pos += step
    pos = origin - step
    while pos >= 0:
        lines.append(pos)
        pos -= step
    return sorted(lines)


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
    placard_predictions: list[dict] | None = None,
    grid_w: int | None = None,
    grid_h: int | None = None,
) -> list[list[list[float]]]:
    """
    Perspective-aware placement for a single empty region.

    When placard_predictions are provided, placements are snapped to the grid
    defined by the median detected-placard centre (warped to flat space) and
    the placard_w / placard_h step size.  Only positions where a vertical AND a
    horizontal grid line cross through the centre of the proposed placard are
    attempted (satisfying the "at least one grid line through the middle"
    requirement with exact alignment).

    Falls back to a greedy dense scan when no placard anchor is available.
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

    # Build grid anchor from detected placard centres (warped to flat space)
    grid_xs: list[float] | None = None
    grid_ys: list[float] | None = None
    if placard_predictions and placard_w > 0 and placard_h > 0:
        # Anchor the grid at the median centre of existing placard detections
        # (warped to flat space) so a grid line always passes through each
        # existing placard.
        centres = np.array(
            [[p["x"], p["y"]] for p in placard_predictions], dtype=np.float32
        ).reshape(-1, 1, 2)
        flat_centres = cv2.perspectiveTransform(centres, H).reshape(-1, 2)
        x0 = float(np.median(flat_centres[:, 0]))
        y0 = float(np.median(flat_centres[:, 1]))
        # Grid pitch = DETECTED (unscaled) placard size + fixed 8 px gap.
        # No slider (scale, spacing, margin) affects the grid line positions.
        _GRID_GAP = 8
        base_x = grid_w if grid_w and grid_w > 0 else placard_w
        base_y = grid_h if grid_h and grid_h > 0 else placard_h
        step_x = max(1, base_x + _GRID_GAP)
        step_y = max(1, base_y + _GRID_GAP)
        grid_xs = _grid_lines(x0, step_x, dst_w)
        grid_ys = _grid_lines(y0, step_y, dst_h)

    quads: list[list[list[float]]] = []

    if grid_xs is not None and grid_ys is not None:
        # Grid-snapped placement: try every grid intersection.
        # available = full sign flat space (all 1s) with existing placards and
        # their spacing buffers punched out.  Outer margin is enforced via
        # per-slot boundary checks.
        available = np.ones((dst_h, dst_w), dtype=np.uint8)
        if placard_predictions:
            for pp in placard_predictions:
                px, py = pp.get("x", 0), pp.get("y", 0)
                pw2, ph2 = pp.get("width", 0) / 2, pp.get("height", 0) / 2
                # Expand each existing placard by the spacing buffer so proposed
                # slots cannot land within `spacing` px of an occupied slot.
                pw2s = pw2 + spacing
                ph2s = ph2 + spacing
                box = np.array([
                    [px - pw2s, py - ph2s],
                    [px + pw2s, py - ph2s],
                    [px + pw2s, py + ph2s],
                    [px - pw2s, py + ph2s],
                ], dtype=np.float32).reshape(-1, 1, 2)
                flat_box = cv2.perspectiveTransform(box, H).reshape(-1, 2).astype(np.int32)
                cv2.fillConvexPoly(available, flat_box, 0)
        for y_line in grid_ys:
            y1 = int(round(y_line - placard_h / 2))
            y2 = y1 + placard_h
            # Outer margin: keep proposed slots away from the sign boundary
            if y1 < margin or y2 > dst_h - margin:
                continue
            for x_line in grid_xs:
                x1 = int(round(x_line - placard_w / 2))
                x2 = x1 + placard_w
                if x1 < margin or x2 > dst_w - margin:
                    continue
                if not can_place_rectangle(available, x1, y1, placard_w, placard_h):
                    continue
                corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                ).reshape(-1, 1, 2)
                warped_corners = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
                quads.append(warped_corners.tolist())
                # Reserve the footprint PLUS the spacing buffer so no other
                # proposed slot can land within `spacing` px of this one.
                sy1 = max(0, y1 - spacing)
                sx1 = max(0, x1 - spacing)
                sy2 = min(dst_h, y2 + spacing)
                sx2 = min(dst_w, x2 + spacing)
                available[sy1:sy2, sx1:sx2] = 0
                if global_reserved is not None:
                    cv2.fillConvexPoly(global_reserved, warped_corners.astype(np.int32), 1)
    else:
        # Fallback: greedy dense scan
        flat_placements = place_rectangles_in_region(
            warped_mask, placard_w, placard_h, spacing=spacing
        )
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
# Placard-size estimation from detections
# ---------------------------------------------------------------------------

def estimate_placard_size_from_detections(
    placard_predictions: list[dict],
    sign_quad: list[dict] | None,
    min_confidence: float = 0.3,
) -> tuple[int, int] | None:
    """
    Derive the placard template size (width, height) from existing placard detections.

    When a sign quad is provided the placard bboxes are first warped into
    perspective-corrected flat space so the measurement is view-angle independent.
    Falls back to raw pixel dimensions when no homography is available.

    Returns (median_w, median_h) in pixels, or None if no usable detections.
    """
    import cv2

    valid = [
        p for p in placard_predictions
        if p.get("confidence", 0) >= min_confidence
    ]
    if not valid:
        return None

    # Build sign axis unit vectors and flat-space scale from the sign quad.
    # We project bbox corners directly onto the sign's own X/Y axes (in image
    # space) to get sign-space extents, then scale to flat-space pixels.
    # This avoids the non-uniform pixel-scale artefact that flat-space
    # homography warping introduces (the left-side vertical stretch).
    sign_x_unit: np.ndarray | None = None
    sign_y_unit: np.ndarray | None = None
    flat_w_per_proj: float = 1.0
    flat_h_per_proj: float = 1.0

    if sign_quad and len(sign_quad) == 4:
        from geometry import order_points
        pts = np.array([[p["x"], p["y"]] for p in sign_quad], dtype=np.float32)
        src = order_points(pts)  # [TL, TR, BR, BL]
        tl, tr, br, bl = src

        w_top   = float(np.linalg.norm(tr - tl))
        w_bot   = float(np.linalg.norm(br - bl))
        h_left  = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        sign_w_px = max(w_top, w_bot)      # apparent sign width in image pixels
        sign_h_px = max(h_left, h_right)   # apparent sign height in image pixels

        if sign_w_px > 0 and sign_h_px > 0:
            sign_x_unit = (tr - tl) / w_top         # unit vector along sign's horizontal axis
            sign_y_unit = (bl - tl) / h_left        # unit vector along sign's vertical axis
            dst_w = max(1, int(sign_w_px))
            dst_h = max(1, int(sign_h_px))
            flat_w_per_proj = dst_w / sign_w_px      # flat-px per image-proj-unit (horiz)
            flat_h_per_proj = dst_h / sign_h_px      # flat-px per image-proj-unit (vert)

    widths: list[float] = []
    heights: list[float] = []

    for p in valid:
        cx, cy = p["x"], p["y"]
        pw, ph = p["width"], p["height"]
        x1, y1 = cx - pw / 2, cy - ph / 2
        x2, y2 = cx + pw / 2, cy + ph / 2

        if sign_x_unit is not None and sign_y_unit is not None:
            # Project all four bbox corners onto sign's horizontal and vertical axes.
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            proj_x = corners @ sign_x_unit  # dot product with sign-x unit
            proj_y = corners @ sign_y_unit  # dot product with sign-y unit
            # Extent along each sign axis gives the true sign-space dimension.
            mw = float((proj_x.max() - proj_x.min()) * flat_w_per_proj)
            mh = float((proj_y.max() - proj_y.min()) * flat_h_per_proj)
        else:
            mw, mh = float(pw), float(ph)

        if mw > 5 and mh > 5:
            widths.append(mw)
            heights.append(mh)

    if not widths:
        return None

    return int(round(float(np.median(widths)))), int(round(float(np.median(heights))))


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
    placard_predictions: list[dict] | None = None,
    estimate_from_detections: bool = True,
) -> dict:
    """
    Estimate how many new placards can fit in the empty regions.

    sign_prediction: the highest-confidence Blue-Logo-Sign detection.
        Its polygon (or bbox) defines the perspective quad for all empty regions.
        When None, falls back to per-region polygon perspective or flat fitting.

    placard_predictions: detections from the placard model.
        When estimate_from_detections is True and detections are available,
        the median perspective-corrected placard size overrides the default size.

    Returns:
        total_fit             – total placard slots across all regions
        per_region            – list of per-region result dicts
        placard_w/h           – dimensions used (after scaling)
        used_perspective_mode – bool
        sign_quad             – the 4-corner sign boundary (list[dict] | None)
        sign_polygon          – raw model polygon for outline drawing (list[dict] | None)
        size_from_detections  – bool indicating whether detected size was used
    """
    # Build the sign quad first so we can use it for size estimation
    sign_quad: list[dict] | None = None
    sign_polygon: list[dict] | None = None
    sign_homography: tuple | None = None
    if sign_prediction is not None:
        sign_polygon = sign_polygon_from_pred(sign_prediction)
        # Use spike-removal-only (no Laplacian smoothing) for the homography quad
        # so sign corners stay at their true pixel positions.
        # Laplacian smoothing rounds corners inward, corrupting the warp.
        sign_quad = sign_quad_for_homography(sign_prediction)
        H_s, H_s_inv, dst_w_s, dst_h_s = compute_region_homography(sign_quad)
        sign_homography = (H_s, H_s_inv, dst_w_s, dst_h_s)

    # Resolve placard size — use detected size when available
    size_from_detections = False
    if estimate_from_detections and placard_predictions:
        detected = estimate_placard_size_from_detections(
            placard_predictions, sign_quad, min_confidence=min_confidence
        )
        if detected is not None:
            default_placard_w, default_placard_h = detected
            size_from_detections = True

    # Keep the raw (unscaled) detected size for grid anchoring.
    # The grid is fixed to the real placard footprint; placard_scale only
    # changes the size of the *proposed* cyan rectangles, not the grid step.
    detected_placard_w = default_placard_w
    detected_placard_h = default_placard_h

    placard_w = max(1, int(default_placard_w * placard_scale))
    placard_h = max(1, int(default_placard_h * placard_scale))

    valid_regions = [
        p for p in empty_space_predictions
        if p.get("confidence", 0) >= min_confidence
    ]

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
                placard_predictions=placard_predictions if placard_predictions else None,
                grid_w=detected_placard_w,
                grid_h=detected_placard_h,
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
        "detected_placard_w":    detected_placard_w,
        "detected_placard_h":    detected_placard_h,
        "used_perspective_mode": any_perspective,
        "sign_quad":             sign_quad,
        "sign_polygon":          sign_polygon,
        "size_from_detections":  size_from_detections,
        "sign_homography":       sign_homography,
    }
