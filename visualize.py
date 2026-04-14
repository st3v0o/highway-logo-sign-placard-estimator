"""
visualize.py

Overlay drawing functions using OpenCV.
All functions work on BGR numpy arrays (as loaded by OpenCV).
render_annotated_image returns a PIL Image for use with st.image().
"""

import cv2
import numpy as np
from PIL import Image

from geometry import bbox_to_xyxy, clip_to_image_bounds


# Color constants (BGR for OpenCV)
COLOR_PLACARD      = (0, 200, 0)      # Green  — existing placards
COLOR_EMPTY_REGION = (0, 140, 255)    # Orange — empty regions
COLOR_PROPOSED     = (220, 200, 0)    # Cyan   — proposed new placements
COLOR_SIGN_QUAD    = (0, 230, 255)    # Yellow — sign boundary outline
COLOR_GRID         = (255, 220, 0)    # Yellow-cyan — grid lines


def draw_sign_quad(img: np.ndarray, sign_quad: list[dict]) -> np.ndarray:
    """Draw the detected sign boundary quad in yellow."""
    pts = np.array(
        [[int(round(p["x"])), int(round(p["y"]))] for p in sign_quad],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=COLOR_SIGN_QUAD, thickness=2, lineType=cv2.LINE_AA)
    return img


def draw_placards(
    img: np.ndarray,
    predictions: list[dict],
    min_confidence: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw existing placard detections in green."""
    img_h, img_w = img.shape[:2]
    for pred in predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        x1, y1, x2, y2 = bbox_to_xyxy(pred)
        x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PLACARD, thickness)
    return img


def draw_empty_regions(
    img: np.ndarray,
    predictions: list[dict],
    min_confidence: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw empty-space detections in orange."""
    img_h, img_w = img.shape[:2]
    for pred in predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        points = pred.get("points")
        if points and len(points) >= 3:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=COLOR_EMPTY_REGION, thickness=thickness)
        else:
            x1, y1, x2, y2 = bbox_to_xyxy(pred)
            x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_EMPTY_REGION, thickness)
    return img


def draw_proposed_placements(
    img: np.ndarray,
    per_region: list[dict],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw proposed new placard placements in cyan.
    Perspective regions draw quadrilaterals; fallback draws rectangles.
    """
    img_h, img_w = img.shape[:2]
    for region in per_region:
        if region.get("used_perspective") and region.get("quad_placements"):
            for quad in region["quad_placements"]:
                pts = np.array(
                    [[int(round(p[0])), int(round(p[1]))] for p in quad],
                    dtype=np.int32,
                )
                cv2.polylines(img, [pts], isClosed=True, color=COLOR_PROPOSED, thickness=thickness)
        else:
            for x1, y1, x2, y2 in region.get("placements", []):
                x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PROPOSED, thickness)
    return img


def draw_sign_corner_grid(
    img: np.ndarray,
    sign_quad: list[dict],
    cols: int = 10,
    rows: int = 6,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw a perspective-corrected slot grid over the sign panel.

    Pipeline:
      1. Warp the sign quad to a flat rectangle via getPerspectiveTransform.
      2. Draw a uniform grid across the full flat extent.
      3. Warp the grid overlay back to the original perspective.
      4. Blend onto the original image inside the sign quad footprint.

    sign_quad: [TL, TR, BR, BL] dicts with 'x' and 'y' keys.
    """
    h_img, w_img = img.shape[:2]
    src = np.float32([[q["x"], q["y"]] for q in sign_quad])

    w_top  = float(np.linalg.norm(src[1] - src[0]))
    w_bot  = float(np.linalg.norm(src[2] - src[3]))
    h_left = float(np.linalg.norm(src[3] - src[0]))
    h_right= float(np.linalg.norm(src[2] - src[1]))
    fw = min(int(max(w_top, w_bot)), 1200)
    fh = min(int(max(h_left, h_right)), 900)
    if fw < 20 or fh < 20:
        return img

    dst   = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    flat    = cv2.warpPerspective(img, M, (fw, fh))
    overlay = flat.copy()

    for i in range(cols + 1):
        x = int(i * fw / cols)
        cv2.line(overlay, (x, 0), (x, fh), COLOR_GRID, 2, cv2.LINE_AA)
    for j in range(rows + 1):
        y = int(j * fh / rows)
        cv2.line(overlay, (0, y), (fw, y), COLOR_GRID, 2, cv2.LINE_AA)

    flat_blended = cv2.addWeighted(overlay, alpha, flat, 1.0 - alpha, 0)
    warped = cv2.warpPerspective(flat_blended, M_inv, (w_img, h_img))

    # Build a mask for the sign quad footprint and blend
    quad_mask_flat = np.ones((fh, fw), dtype=np.float32) * 255
    quad_mask_orig = cv2.warpPerspective(quad_mask_flat, M_inv, (w_img, h_img))
    blend_mask = (quad_mask_orig > 0.5).astype(np.uint8)

    out = img.copy()
    out[blend_mask > 0] = warped[blend_mask > 0]
    return out


def render_annotated_image(
    pil_image: Image.Image,
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    per_region: list[dict],
    min_confidence: float = 0.0,
    sign_quad: list[dict] | None = None,
) -> Image.Image:
    """
    Compose all overlays onto the image and return a PIL Image.

    Drawing order:
      1. Perspective grid (if sign_quad detected)
      2. Sign boundary outline
      3. Empty regions
      4. Existing placards
      5. Proposed placements (on top)
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if sign_quad:
        img = draw_sign_corner_grid(img, sign_quad)
        img = draw_sign_quad(img, sign_quad)

    img = draw_empty_regions(img, empty_space_predictions, min_confidence)
    img = draw_placards(img, placard_predictions, min_confidence)
    img = draw_proposed_placements(img, per_region)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
