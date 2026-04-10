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
    s_min: int = 120,
    v_max: int = 200,
    open_k: int = 20,
    outlier_mult: float = 1.3,
) -> list[dict] | None:
    """
    Detect the 4 corners of the blue highway sign panel directly from pixel colors.

    Parameters
    ----------
    img_bgr      : BGR image array
    s_min        : HSV saturation floor (0–255); raise to exclude sky
    v_max        : HSV value ceiling (0–255); lower to exclude bright sky
    open_k       : morphological opening kernel size; larger severs wider sky bridges
    outlier_mult : hull points farther than (outlier_mult × median_dist) are pruned

    Returns a list of 4 {"x", "y"} dicts in [TL, TR, BR, BL] order,
    or None if the blue region cannot be found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Highway blue signs: highly saturated, moderately bright blue.
    # Clear sky is low-saturation and very bright (S ≈ 60-90, V > 200).
    lower_blue = np.array([90, s_min, 40],       dtype=np.uint8)
    upper_blue = np.array([130, 255,  v_max],    dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Open first (remove thin sky bridges connecting sign to sky blobs),
    # then close (fill internal holes / gaps in the sign body).
    k_open  = np.ones((open_k, open_k), np.uint8)
    k_close = np.ones((25, 25),         np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    min_area = 0.05 * img_area

    # Among candidates large enough, prefer the most "solid" one —
    # sign panels are dense rectangles (fill-ratio ≈ 0.8–1.0); scattered
    # sky blobs have a much lower fill-ratio.
    best_score = -1.0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        bbox_area = bw * bh
        fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
        if fill_ratio > best_score:
            best_score = fill_ratio
            best_cnt = cnt

    if best_cnt is None:
        return None

    # Build the convex hull of the winning contour, then prune outlier hull
    # points before fitting minAreaRect.  Stray sky pixels that survived
    # morphological cleanup sit far from the sign's centre and will be cut.
    hull_pts = cv2.convexHull(best_cnt).reshape(-1, 2).astype(np.float32)

    if len(hull_pts) > 4:
        centroid  = hull_pts.mean(axis=0)
        dists     = np.linalg.norm(hull_pts - centroid, axis=1)
        med_dist  = float(np.median(dists))
        # Allow points up to outlier_mult × the median distance — sign corners
        # are roughly equidistant from the centroid; sky outliers are farther.
        threshold = med_dist * outlier_mult
        cleaned   = hull_pts[dists <= threshold]
        if len(cleaned) >= 4:
            hull_pts = cleaned

    # minAreaRect is anchored to the dense contour mass; even a few outlier
    # edge pixels can't pull a corner far off the sign.
    rect = cv2.minAreaRect(hull_pts)
    box  = cv2.boxPoints(rect)           # 4 corners, arbitrary order
    box  = np.float32(box)
    return _four_extreme_hull_points(box)


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
            vx, vy, cx, cy = [float(v) for v in
                               cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)]
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
