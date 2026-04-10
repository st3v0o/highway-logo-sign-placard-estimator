"""
geometry.py

Utility functions for converting prediction formats into masks and coordinates.
Uses NumPy and OpenCV.
"""

from typing import Optional

import cv2
import numpy as np


def bbox_to_xyxy(pred: dict) -> tuple[int, int, int, int]:
    """
    Convert a Roboflow center-format bounding box to (x1, y1, x2, y2).

    Roboflow bboxes use: x=center_x, y=center_y, width, height
    """
    cx = pred["x"]
    cy = pred["y"]
    w = pred["width"]
    h = pred["height"]
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def clip_to_image_bounds(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Clip coordinates so they don't exceed image boundaries."""
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def polygon_to_mask(
    points: list[dict],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Create a binary mask (uint8) from a list of polygon points.

    points: list of {"x": ..., "y": ...} dicts (Roboflow polygon format)
    Returns a (img_h, img_w) array where 1 = inside polygon.
    """
    pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def bbox_to_mask(
    x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
) -> np.ndarray:
    """
    Create a binary mask (uint8) for a rectangular bounding box.

    Returns a (img_h, img_w) array where 1 = inside box.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def prediction_to_mask(pred: dict, img_h: int, img_w: int) -> tuple[np.ndarray, bool]:
    """
    Convert a single prediction (placard or empty-space) to a binary mask.

    Returns (mask, used_polygon) where used_polygon=True if polygon data was used,
    False if fallback bbox was used.
    """
    points = pred.get("points")
    if points and len(points) >= 3:
        mask = polygon_to_mask(points, img_h, img_w)
        return mask, True
    else:
        x1, y1, x2, y2 = bbox_to_xyxy(pred)
        x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
        mask = bbox_to_mask(x1, y1, x2, y2, img_h, img_w)
        return mask, False


# ---------------------------------------------------------------------------
# Perspective / homography helpers
# ---------------------------------------------------------------------------

def detect_sign_quad_from_blue(
    img_bgr: np.ndarray,
    s_min: int = 140,
    v_max: int = 180,
    open_k: int = 12,
    outlier_mult: float = 1.3,
    _work_width: int = 1000,
) -> list[dict] | None:
    """
    Detect the 4 corners of the blue highway sign panel directly from pixel colors.

    Parameters
    ----------
    img_bgr      : BGR image array
    s_min        : HSV saturation floor (0–255); raise to exclude sky
    v_max        : HSV value ceiling (0–255); lower to exclude bright sky
    open_k       : morphological opening kernel size at the internal working width
    outlier_mult : hull points farther than (outlier_mult × median_dist) are pruned
    _work_width  : image is resized to this width internally so kernel sizes stay
                   consistent regardless of the camera's megapixel count; corners
                   are scaled back to original resolution before returning.

    Returns a list of 4 {"x", "y"} dicts in [TL, TR, BR, BL] order,
    or None if the blue region cannot be found.
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # ── Work at a fixed internal resolution so kernel sizes are predictable ──
    scale = _work_width / orig_w
    work_h = int(orig_h * scale)
    if scale < 0.99:          # only resize if meaningfully different
        img_work = cv2.resize(img_bgr, (_work_width, work_h),
                              interpolation=cv2.INTER_AREA)
    else:
        img_work = img_bgr
        scale = 1.0

    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)

    # Highway sign blue: very saturated, moderately dark royal blue.
    # Clear sky: similar hue but less saturated (S < 140) and brighter (V > 180).
    lower_blue = np.array([90, s_min, 40],    dtype=np.uint8)
    upper_blue = np.array([130, 255, v_max],  dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Keep a copy of the raw (pre-morphology) mask so the gap-detection step
    # can see structural air-gaps between separate sign structures (EXIT sign vs
    # main sign) that morphological closing would otherwise bridge over.
    mask_raw = mask.copy()

    # Smaller open (k=12) preserves the narrow blue strips between logo panels in
    # multi-panel signs, which a k=20 opening would erase.
    # Larger close (k=40) bridges inter-row gaps within the same sign body.
    k_open  = np.ones((open_k, open_k), np.uint8)
    k_close = np.ones((40, 40),         np.uint8)
    mask_open_only = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask_open_only, cv2.MORPH_CLOSE, k_close)

    # ── Sky-from-top suppression ──────────────────────────────────────────────
    # On images taken against a vivid blue sky the HSV ranges can overlap with
    # sign blue.  Remove sky rows by scanning downward from the image top.
    # Guard: only activate if high-density blue appears within the TOPMOST 5 %
    # of the image.  On close-up / telephoto shots the sign fills the frame and
    # its own blue body rises above 50 % density; those sign-top rows must NOT
    # be cropped.  Real sky always appears from the very first rows.
    _top_scan   = int(0.18 * work_h)
    _sky_guard  = max(1, int(0.05 * work_h))   # blue must appear THIS early
    _sky_end    = 0
    _sky_active = False
    for _ri in range(_top_scan):
        _d = float(mask[_ri, :].sum()) / 255.0 / _work_width
        if _d > 0.50:
            if _ri < _sky_guard:        # blue appears very early → real sky
                _sky_active = True
            if _sky_active:
                _sky_end = _ri + 1      # extend the crop boundary
        elif _d < 0.20 and _sky_end > 0:
            break                       # gap/white-border row → stop
    if _sky_end > 0:
        mask[:_sky_end, :]           = 0
        mask_open_only[:_sky_end, :] = 0
        mask_raw[:_sky_end, :]       = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h_w, img_w_w = img_work.shape[:2]
    img_area = img_h_w * img_w_w
    min_area = 0.03 * img_area   # 3 % (was 5 %) to catch sub-panels in large signs

    # Collect all large-enough contours with their centroids.
    # Prefer contours that do NOT touch the very top of the image (sky-touching).
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cy = M["m01"] / M["m00"]
        candidates.append((area, cy, cnt))

    if not candidates:
        return None

    # Sort largest-first; the biggest contour anchors the sign cluster.
    candidates.sort(key=lambda x: -x[0])
    anchor_cy = candidates[0][1]

    # Group contours whose centroid_y is within 25 % of the working image height
    # from the anchor.  Sign panels in the same horizontal row share almost the
    # same centroid_y (e.g. three side-by-side RaceWay panels).  A header panel
    # sitting well above the main sign (like EXIT 771B) is excluded.
    cluster = [cnt for area, cy, cnt in candidates
               if abs(cy - anchor_cy) < 0.25 * img_h_w]

    if len(cluster) == 1:
        best_cnt = cluster[0]
    else:
        # Merge all co-planar panels into one combined convex hull so Hough
        # lines see the full sign width, not just one panel's edges.
        all_pts = np.vstack([c.reshape(-1, 2) for c in cluster])
        best_cnt = cv2.convexHull(all_pts)

    # ── Per-edge Hough line detection ────────────────────────────────────────
    # Detect each edge independently so that a header panel (EXIT sign) attached
    # above the main sign body cannot pull the top edge upward.
    #
    # Strategy for the TOP edge:
    #   Only consider horizontal segments whose y-midpoint falls in the range
    #   [bbox_y + 15 % × bbox_h,  bbox_y + 60 % × bbox_h].
    #   This band skips the EXIT-panel top (< 15 %) while staying above the
    #   sign mid-line (> 60 %), reliably finding the main panel's top edge.
    #
    # All other edges use the standard extreme-quartile approach.

    bx, by, bw, bh = cv2.boundingRect(best_cnt)

    # Filled contour mask → Canny edges → Hough segments
    cnt_mask = np.zeros(img_work.shape[:2], dtype=np.uint8)
    cv2.drawContours(cnt_mask, [best_cnt], 0, 255, -1)
    edges = cv2.Canny(cnt_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=30, minLineLength=30, maxLineGap=15)

    def _minAreaRect_fallback():
        hull_pts = cv2.convexHull(best_cnt).reshape(-1, 2).astype(np.float32)
        if len(hull_pts) > 4:
            centroid = hull_pts.mean(axis=0)
            dists    = np.linalg.norm(hull_pts - centroid, axis=1)
            threshold_d = float(np.median(dists)) * outlier_mult
            cleaned  = hull_pts[dists <= threshold_d]
            if len(cleaned) >= 4:
                hull_pts = cleaned
        rect = cv2.minAreaRect(hull_pts)
        return _four_extreme_hull_points(np.float32(cv2.boxPoints(rect)))

    if lines is None:
        quad = _minAreaRect_fallback()
    else:
        segs = lines.reshape(-1, 4)

        # Classify each segment as horizontal or vertical by its angle
        h_segs, v_segs = [], []
        for x1, y1, x2, y2 in segs:
            dx, dy = float(x2 - x1), float(y2 - y1)
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle < 30 or angle > 150:
                h_segs.append((x1, y1, x2, y2))
            elif 60 < angle < 120:
                v_segs.append((x1, y1, x2, y2))

        if len(h_segs) < 2 or len(v_segs) < 2:
            quad = _minAreaRect_fallback()
        else:
            # Sort by mid-y (horizontal) and mid-x (vertical)
            h_sorted = sorted(h_segs, key=lambda s: (s[1] + s[3]) / 2)
            v_sorted = sorted(v_segs, key=lambda s: (s[0] + s[2]) / 2)

            k_h = max(1, len(h_segs) // 4)
            k_v = max(1, len(v_segs) // 4)

            def _band_pick(segs, lo, hi, sort_key, take_top):
                """Filter segs to [lo,hi] band, return top or bottom quartile."""
                band = [s for s in segs if lo <= sort_key(s) <= hi]
                if not band:
                    return None
                band_sorted = sorted(band, key=sort_key)
                k = max(1, len(band) // 4)
                return band_sorted[:k] if take_top else band_sorted[-k:]

            mid_y = lambda s: (s[1] + s[3]) / 2  # noqa: E731
            mid_x = lambda s: (s[0] + s[2]) / 2  # noqa: E731

            # TOP: gap-aware adaptive strategy.
            #
            # Scan the OPEN-ONLY mask (before closing) in the top 15 % of the
            # bounding box looking for a structural gap — a row-band where blue
            # density drops to less than 25 % of the local peak.  Such a gap
            # signals a separate EXIT/GAS header sign above the main panel.
            # (The close step fills that gap, so we must look at mask_open_only.)
            #
            # • If a header gap exists → use the 15–60 % banded approach to
            #   skip the header and find the main panel's true top edge.
            # • If no gap → the contour is a unified sign (single-panel or
            #   multi-row); use the topmost Hough segments directly.
            x_lo_top = max(0, bx)
            x_hi_top = min(mask.shape[1], bx + bw)
            scan_top_r = by
            scan_15_r  = min(by + int(0.15 * bh), mask.shape[0] - 1)

            has_header_gap = False
            if scan_15_r > scan_top_r and x_hi_top > x_lo_top:
                col_w = x_hi_top - x_lo_top
                # Use the RAW mask (before morphology) so the physical air-gap
                # between an EXIT sign and the main sign body is still visible.
                # Tighten the ratio to < 10 % of peak: logo-panel interiors still
                # have border pixels (≥ 10 % density) whereas a true sky/air gap
                # between separate sign structures is essentially zero.
                raw_dens = [
                    float(mask_raw[r, x_lo_top:x_hi_top].sum())
                    / 255.0 / col_w
                    for r in range(scan_top_r, scan_15_r)
                ]
                if raw_dens:
                    peak_d = max(raw_dens)
                    min_d  = min(raw_dens)
                    if peak_d > 0.20 and min_d < 0.10 * peak_d:
                        has_header_gap = True

            # Proximity-based cluster of topmost h_segs.
            # Problem: on close-up shots (e.g. lodging sign filling the frame)
            # nearly ALL h_segs are at the sign's bottom edge; blindly taking
            # the top-N pulls bottom segments into the fit.  Solution: only
            # include segments within 15 % of bbox height of the topmost one.
            _top_prox = max(30.0, 0.15 * bh)
            if h_sorted:
                _t0 = mid_y(h_sorted[0])
                _top_cluster = [s for s in h_sorted
                                if mid_y(s) <= _t0 + _top_prox]
                _top_cluster = (_top_cluster or h_sorted)[:max(k_h, 3)]
            else:
                _top_cluster = []

            if has_header_gap:
                # EXIT/GAS header above main sign → skip header, find main top
                top_segs = _band_pick(h_sorted,
                                      by + 0.15 * bh, by + 0.60 * bh,
                                      mid_y, take_top=True)
                if top_segs is None:
                    top_segs = _top_cluster
            else:
                # Unified sign: use topmost proximity cluster
                top_segs = _top_cluster

            # BOTTOM: use a density-minimum scan on the blue mask to locate the
            # sign panel's actual bottom edge.
            #
            # Background: morphological close merges sign panels with any stray
            # blue below the sign (mounting hardware, reflections) into one blob.
            # The Hough boundary then lies at the stray blue's lower edge, not
            # the sign panel's bottom.
            #
            # Solution: scan column-slice of the blue mask row-by-row between
            # 50 % and 92 % of bbox height.  The sign panel bottom is the row
            # of minimum blue density in that scan (the thin gap between sign
            # and below-sign blue).  If no clear gap exists, fall back to Hough.
            x_lo = max(0, bx)
            x_hi = min(mask.shape[1], bx + bw)
            # Start at 70 % of bbox height so sign logos / text (which live in
            # the upper-to-mid sign body and create false density minima) are
            # skipped.  The actual panel bottom and any below-sign gap always
            # live in the lower 30 % of the sign panel range.
            scan_lo = by + int(0.70 * bh)
            scan_hi = by + int(0.92 * bh)
            scan_lo = max(0, min(scan_lo, mask.shape[0] - 1))
            scan_hi = max(0, min(scan_hi, mask.shape[0] - 1))

            # Compute Hough-based bottom first
            hough_bot = _band_pick(h_sorted,
                                    by + 0.55 * bh, by + 0.98 * bh,
                                    mid_y, take_top=False)
            if hough_bot is None:
                hough_bot = h_sorted[-k_h:]

            # Check if Hough bottom is dragged to the very edge of the bbox.
            # If so, the closed mask has merged the sign with below-sign blue,
            # so use a density-minimum scan on the open-only mask to find the
            # real panel bottom.  Otherwise the Hough result is reliable.
            hough_bot_y = max(mid_y(s) for s in hough_bot) if hough_bot else 0
            use_density_scan = (hough_bot_y > by + 0.90 * bh)

            bot_segs = None
            if use_density_scan and x_hi > x_lo and scan_hi > scan_lo:
                col_width = x_hi - x_lo
                densities = [
                    int(mask_open_only[row, x_lo:x_hi].sum()) / 255 / col_width
                    for row in range(scan_lo, scan_hi + 1)
                ]
                peak_d = max(densities) if densities else 0.0
                min_d  = min(densities) if densities else 0.0
                if peak_d > 0 and min_d < 0.82 * peak_d:
                    min_idx = int(np.argmin(densities))
                    # Require the density to bounce back AFTER the minimum —
                    # sign→gap→below-sign-blue pattern.  If the minimum is at
                    # the end of the scan (just the sign face tapering off), it
                    # is not a real below-sign gap and should be ignored.
                    if min_idx < len(densities) - 3:
                        recovery = float(np.mean(densities[min_idx+1:min_idx+4]))
                        if recovery > min_d * 1.10:   # 10 % bounce-back
                            gap_y = scan_lo + min_idx
                            bot_segs = [(x_lo, gap_y, x_hi, gap_y)]

            if bot_segs is None:
                bot_segs = hough_bot

            # LEFT: vertical segs in the leftmost 35 % horizontal band.
            # Only the actual left edge of the sign lives here.
            left_segs = _band_pick(v_sorted,
                                    bx, bx + 0.35 * bw,
                                    mid_x, take_top=True)
            if left_segs is None:
                left_segs = v_sorted[:k_v]

            # RIGHT: vertical segs in the rightmost 35 % horizontal band.
            right_segs = _band_pick(v_sorted,
                                     bx + 0.65 * bw, bx + bw,
                                     mid_x, take_top=False)
            if right_segs is None:
                right_segs = v_sorted[-k_v:]

            # Hull-based top edge: fit a line through the topmost convex hull
            # points.  This captures the sign's perspective top edge correctly
            # even when Hough top segments are concentrated on one horizontal
            # side of the sign (common on angled/telephoto multi-panel shots).
            hull_pts_arr = cv2.convexHull(best_cnt).reshape(-1, 2).astype(np.float32)
            hull_top_y0  = float(hull_pts_arr[:, 1].min())
            hull_top_pts = hull_pts_arr[
                hull_pts_arr[:, 1] <= hull_top_y0 + max(20.0, 0.10 * bh)]
            hull_top_line = None
            if len(hull_top_pts) >= 2:
                fl_h = cv2.fitLine(hull_top_pts.reshape(-1, 1, 2),
                                   cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                vx_h, vy_h = float(fl_h[0]), float(fl_h[1])
                cx_h, cy_h = float(fl_h[2]), float(fl_h[3])
                a_h, b_h   = -vy_h, vx_h
                c_h        = vy_h * cx_h - vx_h * cy_h
                n_h        = max(float(np.sqrt(a_h*a_h + b_h*b_h)), 1e-9)
                hull_top_line = (a_h/n_h, b_h/n_h, c_h/n_h)

            # Decide whether to trust Hough top or defer to hull top.
            # Use hull top when the Hough cluster is spatially biased
            # (all segments concentrated in the left or right half only).
            _use_hull_top = False
            if top_segs and hull_top_line is not None:
                _top_xs = [(s[0]+s[2])/2 for s in top_segs]
                _mid_bx = bx + 0.5 * bw
                # Biased if every segment is on one side of the sign's midline
                _use_hull_top = (
                    all(x < _mid_bx for x in _top_xs) or
                    all(x > _mid_bx for x in _top_xs)
                )

            # Fit one line per edge and intersect opposite pairs
            def fit_abc_local(s_list):
                pts = np.array([[(s[0]+s[2])/2, (s[1]+s[3])/2] for s in s_list],
                               dtype=np.float32)
                if len(pts) == 1:
                    x1, y1, x2, y2 = s_list[0]
                    a, b = float(y2-y1), float(x1-x2)
                    c = float(x2*y1 - x1*y2)
                else:
                    fl = cv2.fitLine(pts.reshape(-1, 1, 2),
                                     cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                    vx, vy, cx, cy = float(fl[0]), float(fl[1]), float(fl[2]), float(fl[3])
                    a, b = -vy, vx
                    c = vy * cx - vx * cy
                n = max(float(np.sqrt(a*a + b*b)), 1e-9)
                return a/n, b/n, c/n

            def intersect_local(l1, l2):
                a1, b1, c1 = l1;  a2, b2, c2 = l2
                det = a1*b2 - a2*b1
                if abs(det) < 1e-6:
                    return None
                return ((-c1*b2 + c2*b1) / det,
                        (-a1*c2 + a2*c1) / det)

            top   = hull_top_line if _use_hull_top else fit_abc_local(top_segs)
            if top is None:
                top = fit_abc_local(top_segs)
            bot   = fit_abc_local(bot_segs)
            left  = fit_abc_local(left_segs)
            right = fit_abc_local(right_segs)

            tl = intersect_local(top, left)
            tr = intersect_local(top, right)
            br = intersect_local(bot, right)
            bl = intersect_local(bot, left)

            if any(pt is None for pt in [tl, tr, br, bl]):
                quad = _minAreaRect_fallback()
            else:
                pts_arr = np.array([
                    [tl[0], tl[1]], [tr[0], tr[1]],
                    [br[0], br[1]], [bl[0], bl[1]],
                ], dtype=np.float32)
                # Sanity 1: reject implausibly huge values (near-parallel lines)
                if any(abs(p[0]) > 1e4 or abs(p[1]) > 1e4 for p in pts_arr):
                    quad = _minAreaRect_fallback()
                else:
                    # Sanity 2: all corners must lie within the contour bounding
                    # box plus a 12 % margin.  Line extrapolation outside the
                    # segment range can push intersections well beyond the sign body.
                    mx, my = 0.12 * bw, 0.12 * bh
                    inside = all(
                        bx - mx <= p[0] <= bx + bw + mx and
                        by - my <= p[1] <= by + bh + my
                        for p in pts_arr
                    )
                    if inside:
                        quad = _four_extreme_hull_points(pts_arr)
                    else:
                        quad = _minAreaRect_fallback()

    if quad is None:
        return None

    # Scale corners back to original (full-resolution) coordinate space
    if scale != 1.0:
        for pt in quad:
            pt["x"] = pt["x"] / scale
            pt["y"] = pt["y"] / scale
    return quad


def detect_sign_quad_from_detections(
    img_bgr: np.ndarray,
    placard_predictions: list[dict],
    empty_predictions: list[dict],
) -> list[dict] | None:
    """
    Three-phase approach: ROI → best blue contour → Hough-line quad.

    Phase 1 — generous ROI (15 % of image height upward)
        Captures the sign header row above the detections.  Using image-relative
        padding avoids exploding when detections already fill most of the frame.

    Phase 2 — blue contour selection
        Finds all blue connected components and picks the one with the highest
        overlap with the detection bounding box.  This naturally discards
        adjacent panels (e.g. an EXIT sign above Popeyes) whose blue area
        doesn't overlap with any detected placard or empty region.

    Phase 3 — Canny + Hough line intersection
        Runs Canny on the selected contour's filled mask, applies the
        probabilistic Hough transform to find line segments, clusters them into
        top / bottom / left / right groups, fits a single representative line
        per group, and intersects each pair to get the 4 exact sign corners.
        Produces perspective-correct trapezoid corners for angled shots.
        Falls back to approxPolyDP → minAreaRect if Hough doesn't converge.

    Returns 4 corner dicts in [TL, TR, BR, BL] order, or None on failure.
    """
    img_h, img_w = img_bgr.shape[:2]
    all_preds = placard_predictions + empty_predictions
    if not all_preds:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for pred in all_preds:
        polygon = pred.get("points", [])
        if polygon:
            for p in polygon:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
        else:
            x1, y1, x2, y2 = bbox_to_xyxy(pred)
            xs += [float(x1), float(x2)]
            ys += [float(y1), float(y2)]

    if not xs:
        return None

    det_x1, det_x2 = min(xs), max(xs)
    det_y1, det_y2 = min(ys), max(ys)

    # --- Phase 1: generous ROI (image-relative) ---
    pad_x  = img_w * 0.08
    pad_yu = img_h * 0.15   # enough to include header row above detections
    pad_yd = img_h * 0.06

    roi_x1 = max(0,     int(det_x1 - pad_x))
    roi_y1 = max(0,     int(det_y1 - pad_yu))
    roi_x2 = min(img_w, int(det_x2 + pad_x))
    roi_y2 = min(img_h, int(det_y2 + pad_yd))

    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return _bbox_quad(det_x1, det_y1, det_x2, det_y2, img_w, img_h)

    roi_h, roi_w = roi.shape[:2]

    # --- Phase 2: blue mask → select best contour by detection overlap ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([90,  70,  25], dtype=np.uint8),
        np.array([135, 255, 220], dtype=np.uint8),
    )
    kc = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _bbox_quad(det_x1, det_y1, det_x2, det_y2, img_w, img_h)

    # Detection bbox in ROI-local coordinates
    dlx1 = det_x1 - roi_x1
    dly1 = det_y1 - roi_y1
    dlx2 = det_x2 - roi_x1
    dly2 = det_y2 - roi_y1

    def _overlap_score(cnt):
        area = cv2.contourArea(cnt)
        if area < 0.02 * roi_h * roi_w:
            return -1.0
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        ix1 = max(cx, dlx1);  ix2 = min(cx + cw, dlx2)
        iy1 = max(cy, dly1);  iy2 = min(cy + ch, dly2)
        if ix2 <= ix1 or iy2 <= iy1:
            return -1.0
        return (ix2 - ix1) * (iy2 - iy1) / max(cw * ch, 1)

    scored = sorted(contours, key=_overlap_score, reverse=True)
    best_cnt = scored[0] if _overlap_score(scored[0]) >= 0.15 else max(contours, key=cv2.contourArea)

    # --- Phase 3: Canny + Hough lines → intersect to get 4 corners ---
    cnt_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.drawContours(cnt_mask, [best_cnt], 0, 255, -1)
    edges = cv2.Canny(cnt_mask, 50, 150)

    min_len = min(roi_h, roi_w) * 0.15
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=20,
    )

    if lines is not None and len(lines) >= 4:
        quad = _hough_to_quad(lines.reshape(-1, 4), roi_x1, roi_y1)
        if quad is not None:
            return quad

    # Fallback: approxPolyDP on convex hull
    hull = cv2.convexHull(best_cnt)
    peri = cv2.arcLength(hull, True)
    for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += roi_x1;  pts[:, 1] += roi_y1
            return _four_extreme_hull_points(pts)

    rect = cv2.minAreaRect(hull)
    pts = cv2.boxPoints(rect).astype(np.float32)
    pts[:, 0] += roi_x1;  pts[:, 1] += roi_y1
    return _four_extreme_hull_points(pts)


def _hough_to_quad(
    lines: np.ndarray,
    roi_x1: int,
    roi_y1: int,
) -> list[dict] | None:
    """
    Given an (N, 4) array of Hough line segments (x1,y1,x2,y2 in ROI coords),
    cluster into top/bottom/left/right groups, fit one line per group via
    cv2.fitLine, intersect pairs to get 4 corners, return full-image coords.
    """
    h_segs, v_segs = [], []
    for x1, y1, x2, y2 in lines:
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle < 30 or angle > 150:
            h_segs.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            v_segs.append((x1, y1, x2, y2))

    if len(h_segs) < 2 or len(v_segs) < 2:
        return None

    def mid_y(s): return (s[1] + s[3]) / 2.0
    def mid_x(s): return (s[0] + s[2]) / 2.0

    h_sorted = sorted(h_segs, key=mid_y)
    v_sorted = sorted(v_segs, key=mid_x)

    k_h = max(1, len(h_sorted) // 4)
    k_v = max(1, len(v_sorted) // 4)

    top_segs   = h_sorted[:k_h]
    bot_segs   = h_sorted[-k_h:]
    left_segs  = v_sorted[:k_v]
    right_segs = v_sorted[-k_v:]

    def fit_abc(segs):
        """Return (a, b, c) of normalised line  ax + by + c = 0.

        cv2.fitLine returns (vx, vy, cx, cy) where (vx,vy) is the unit
        direction and (cx,cy) is a point on the line.
        Normal form:  a = -vy,  b = vx,  c = vy*cx - vx*cy
        (verified:  a*cx + b*cy + c = -vy*cx + vx*cy + vy*cx - vx*cy = 0 ✓)
        """
        pts = np.array([[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in segs],
                       dtype=np.float32)
        if len(pts) == 1:
            x1, y1, x2, y2 = segs[0]
            a, b = float(y2 - y1), float(x1 - x2)
            c = float(x2 * y1 - x1 * y2)
        else:
            _fl = cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            vx, vy, cx, cy = float(_fl[0]), float(_fl[1]), float(_fl[2]), float(_fl[3])
            a, b = -vy, vx
            c = vy * cx - vx * cy   # NOT cy*vy - cx*vx (that swaps cx/cy roles)
        n = max(float(np.sqrt(a * a + b * b)), 1e-9)
        return a / n, b / n, c / n

    def intersect(l1, l2):
        a1, b1, c1 = l1;  a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        return (-c1 * b2 + c2 * b1) / det, (-a1 * c2 + a2 * c1) / det

    top   = fit_abc(top_segs)
    bot   = fit_abc(bot_segs)
    left  = fit_abc(left_segs)
    right = fit_abc(right_segs)

    tl = intersect(top, left)
    tr = intersect(top, right)
    br = intersect(bot, right)
    bl = intersect(bot, left)

    if any(pt is None for pt in [tl, tr, br, bl]):
        return None

    pts = np.array([
        [tl[0] + roi_x1, tl[1] + roi_y1],
        [tr[0] + roi_x1, tr[1] + roi_y1],
        [br[0] + roi_x1, br[1] + roi_y1],
        [bl[0] + roi_x1, bl[1] + roi_y1],
    ], dtype=np.float32)

    # Sanity: near-parallel line pairs produce intersections at huge coordinates.
    # Reject the result if any corner is implausibly far from the image.
    if any(abs(p[0]) > 1e4 or abs(p[1]) > 1e4 for p in pts):
        return None

    return _four_extreme_hull_points(pts)


def _bbox_quad(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> list[dict]:
    """Plain detection-union rectangle — last-resort fallback."""
    pad = min(img_w, img_h) * 0.04
    corners = np.array([
        [max(0.0, x1 - pad),          max(0.0, y1 - pad)],
        [min(float(img_w), x2 + pad), max(0.0, y1 - pad)],
        [min(float(img_w), x2 + pad), min(float(img_h), y2 + pad)],
        [max(0.0, x1 - pad),          min(float(img_h), y2 + pad)],
    ], dtype=np.float32)
    return _four_extreme_hull_points(corners)


def simplify_polygon_to_quad(points: list[dict]) -> list[dict] | None:
    """
    Reduce an arbitrary polygon to exactly 4 corner points.

    Strategy:
      1. Compute convex hull of all points.
      2. Try progressively larger epsilon values with approxPolyDP until we get 4 vertices.
      3. If that fails, fall back to picking the 4 most extreme points from the hull.

    Returns a list of 4 {"x", "y"} dicts, or None if reduction is not possible.
    """
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    hull = cv2.convexHull(pts)

    peri = cv2.arcLength(hull, True)
    for scale in [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25]:
        epsilon = scale * peri
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            return [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]

    # Fall back: pick the 4 most extreme directions from the hull
    hull_pts = hull.reshape(-1, 2)
    if len(hull_pts) >= 4:
        return _four_extreme_hull_points(hull_pts)

    return None


def _four_extreme_hull_points(pts: np.ndarray) -> list[dict]:
    """
    Select 4 representative corner points from a convex hull using coordinate
    sums/differences (same heuristic as order_points).

    Returns them in [TL, TR, BR, BL] order.
    """
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return [
        {"x": float(tl[0]), "y": float(tl[1])},
        {"x": float(tr[0]), "y": float(tr[1])},
        {"x": float(br[0]), "y": float(br[1])},
        {"x": float(bl[0]), "y": float(bl[1])},
    ]


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order an array of 4 (x, y) points as [top-left, top-right, bottom-right, bottom-left].

    Uses coordinate-sum and coordinate-difference heuristics:
      TL = smallest x+y,  BR = largest x+y
      TR = smallest x-y,  BL = largest x-y
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def compute_region_homography(
    points: list[dict],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute the perspective homography that maps a 4-point polygon to a flat rectangle.

    The destination rectangle dimensions are set to the maximum width and height
    of the source quadrilateral so no area is lost.

    Returns:
        H       – 3×3 homography matrix (source → destination)
        H_inv   – inverse homography (destination → source)
        dst_w   – width of the flat destination rectangle (pixels)
        dst_h   – height of the flat destination rectangle (pixels)
    """
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    src = order_points(pts)  # (TL, TR, BR, BL)

    # Destination rectangle: width = max of top/bottom edges, height = max of left/right edges
    w_top = float(np.linalg.norm(src[1] - src[0]))
    w_bot = float(np.linalg.norm(src[2] - src[3]))
    h_left = float(np.linalg.norm(src[3] - src[0]))
    h_right = float(np.linalg.norm(src[2] - src[1]))

    dst_w = max(1, int(max(w_top, w_bot)))
    dst_h = max(1, int(max(h_left, h_right)))

    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv, dst_w, dst_h
