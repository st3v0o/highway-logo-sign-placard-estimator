"""
geometry.py

Utility functions for converting prediction formats into masks and coordinates.
Uses NumPy and OpenCV.
"""

import cv2
import numpy as np


def bbox_to_xyxy(pred: dict) -> tuple[int, int, int, int]:
    """
    Convert a Roboflow center-format bounding box to (x1, y1, x2, y2).
    Roboflow bboxes: x=center_x, y=center_y, width, height
    """
    cx = pred["x"]
    cy = pred["y"]
    w  = pred["width"]
    h  = pred["height"]
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def clip_to_image_bounds(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Clip coordinates so they don't exceed image boundaries."""
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def polygon_to_mask(points: list[dict], img_h: int, img_w: int) -> np.ndarray:
    """
    Create a binary mask (uint8) from a list of polygon points.
    points: list of {"x": ..., "y": ...} dicts
    """
    pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def bbox_to_mask(
    x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
) -> np.ndarray:
    """Create a binary mask (uint8) for a rectangular bounding box."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def prediction_to_mask(pred: dict, img_h: int, img_w: int) -> tuple[np.ndarray, bool]:
    """
    Convert a single prediction to a binary mask.
    Returns (mask, used_polygon).
    """
    points = pred.get("points")
    if points and len(points) >= 3:
        return polygon_to_mask(points, img_h, img_w), True
    x1, y1, x2, y2 = bbox_to_xyxy(pred)
    x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
    return bbox_to_mask(x1, y1, x2, y2, img_h, img_w), False


def _remove_polygon_spikes(pts: np.ndarray, min_angle_deg: float = 50.0) -> np.ndarray:
    """
    Remove vertices that form sharp spikes.

    For each vertex, compute the interior angle formed by its two neighbours.
    If the angle is below min_angle_deg, the vertex is a spike and is dropped.
    Iterates until no more spikes are found (handles consecutive spikes).
    """
    changed = True
    while changed and len(pts) >= 4:
        changed = False
        n = len(pts)
        keep: list[int] = []
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-8:
                keep.append(i)
                continue
            cos_a = np.dot(v1, v2) / norm
            angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if angle < min_angle_deg:
                changed = True  # skip — this is a spike
            else:
                keep.append(i)
        pts = pts[keep]
    return pts


def sign_polygon_from_pred(pred: dict) -> list[dict]:
    """
    Return the Blue-Logo-Sign polygon with sharp spike vertices removed.

    Uses angle-based spike detection: any vertex whose interior angle is below
    50° is treated as a spike and dropped. Smooth curves and real corners (which
    have wider angles) are left completely untouched.
    Falls back to bbox corners when no polygon is present.
    """
    points = pred.get("points")
    if points and len(points) >= 3:
        pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        pts = _remove_polygon_spikes(pts, min_angle_deg=50.0)
        return [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    x1, y1, x2, y2 = bbox_to_xyxy(pred)
    return [
        {"x": float(x1), "y": float(y1)},
        {"x": float(x2), "y": float(y1)},
        {"x": float(x2), "y": float(y2)},
        {"x": float(x1), "y": float(y2)},
    ]


def sign_quad_from_pred(pred: dict) -> list[dict]:
    """
    Convert a Blue-Logo-Sign detection to a [TL, TR, BR, BL] quad of exactly
    4 corners, required for the perspective homography.

    Extracts the 4 extreme corners from the model's polygon (or bbox).
    """
    points = pred.get("points")
    if points and len(points) >= 3:
        pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        return _four_extreme_hull_points(pts)

    x1, y1, x2, y2 = bbox_to_xyxy(pred)
    return [
        {"x": float(x1), "y": float(y1)},
        {"x": float(x2), "y": float(y1)},
        {"x": float(x2), "y": float(y2)},
        {"x": float(x1), "y": float(y2)},
    ]


def simplify_polygon_to_quad(points: list[dict]) -> list[dict] | None:
    """
    Reduce an arbitrary polygon to exactly 4 corner points via convex hull + approxPolyDP.
    Returns a list of 4 {"x", "y"} dicts, or None if reduction fails.
    """
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for scale in [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25]:
        approx = cv2.approxPolyDP(hull, scale * peri, True)
        if len(approx) == 4:
            return [{"x": float(p[0][0]), "y": float(p[0][1])} for p in approx]
    hull_pts = hull.reshape(-1, 2)
    if len(hull_pts) >= 4:
        return _four_extreme_hull_points(hull_pts)
    return None


def _four_extreme_hull_points(pts: np.ndarray) -> list[dict]:
    """
    Select 4 representative corner points in [TL, TR, BR, BL] order
    using coordinate sums/differences.
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
    Order 4 (x, y) points as [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def compute_region_homography(
    points: list[dict],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute the perspective homography that maps a 4-point quad to a flat rectangle.

    Returns:
        H       – 3×3 homography (source → flat)
        H_inv   – inverse homography (flat → source)
        dst_w   – width of the flat rectangle
        dst_h   – height of the flat rectangle
    """
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    src = order_points(pts)

    w_top  = float(np.linalg.norm(src[1] - src[0]))
    w_bot  = float(np.linalg.norm(src[2] - src[3]))
    h_left = float(np.linalg.norm(src[3] - src[0]))
    h_right= float(np.linalg.norm(src[2] - src[1]))

    dst_w = max(1, int(max(w_top, w_bot)))
    dst_h = max(1, int(max(h_left, h_right)))

    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    H     = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv, dst_w, dst_h
