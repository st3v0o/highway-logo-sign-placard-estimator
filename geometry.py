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
