"""
visualize.py

Overlay drawing functions using OpenCV.
All functions work on BGR numpy arrays (as loaded by OpenCV).
render_annotated_image returns a PIL Image for use with st.image().
"""

import numpy as np
import cv2
from PIL import Image

from geometry import bbox_to_xyxy, clip_to_image_bounds


# Color constants (BGR for OpenCV)
COLOR_PLACARD = (0, 200, 0)        # Green — existing placards
COLOR_EMPTY_REGION = (0, 140, 255) # Orange — empty regions (bbox fallback)
COLOR_EMPTY_POLYGON = (0, 100, 255) # Orange-red — empty regions (polygon mode)
COLOR_PROPOSED = (220, 200, 0)     # Cyan — proposed new placements


def draw_placards(
    img: np.ndarray,
    predictions: list[dict],
    min_confidence: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw existing placard detections in red on the image.
    """
    img_h, img_w = img.shape[:2]
    for pred in predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        x1, y1, x2, y2 = bbox_to_xyxy(pred)
        x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PLACARD, thickness)
        label = f"placard {pred.get('confidence', 0):.0%}"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COLOR_PLACARD, 1, cv2.LINE_AA)
    return img


def draw_empty_regions(
    img: np.ndarray,
    predictions: list[dict],
    min_confidence: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw empty-space detections. Uses polygon outline (green) if points are
    available, otherwise draws a bounding box (yellow).
    """
    img_h, img_w = img.shape[:2]
    for pred in predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        points = pred.get("points")
        if points and len(points) >= 3:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=COLOR_EMPTY_POLYGON, thickness=thickness)
            label = f"empty {pred.get('confidence', 0):.0%} (poly)"
            cv2.putText(img, label, (pts[0][0], max(pts[0][1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_EMPTY_POLYGON, 1, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = bbox_to_xyxy(pred)
            x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_EMPTY_REGION, thickness)
            label = f"empty {pred.get('confidence', 0):.0%} (bbox)"
            cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COLOR_EMPTY_REGION, 1, cv2.LINE_AA)
    return img


def draw_proposed_placements(
    img: np.ndarray,
    per_region: list[dict],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw proposed new placard placements in cyan.

    per_region: list of region result dicts from estimate_total_capacity()
    """
    img_h, img_w = img.shape[:2]
    for region in per_region:
        for x1, y1, x2, y2 in region.get("placements", []):
            x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PROPOSED, thickness)
    return img


def render_annotated_image(
    pil_image: Image.Image,
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    per_region: list[dict],
    min_confidence: float = 0.0,
) -> Image.Image:
    """
    Compose all overlays onto the image and return a PIL Image.

    Drawing order: empty regions → existing placards → proposed placements
    so that proposals appear on top.
    """
    # Convert PIL -> OpenCV BGR
    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    img = draw_empty_regions(img, empty_space_predictions, min_confidence)
    img = draw_placards(img, placard_predictions, min_confidence)
    img = draw_proposed_placements(img, per_region)

    # Convert back to PIL RGB
    img_rgb_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_out)
