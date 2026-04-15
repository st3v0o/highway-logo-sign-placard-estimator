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
    placard_predictions: list[dict] | None = None,
    placard_w: int | None = None,
    placard_h: int | None = None,
    spacing: int = 0,
    grid_w: int | None = None,
    grid_h: int | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw a perspective-corrected slot grid over the sign panel.

    When placard_predictions, placard_w, and placard_h are supplied the grid is
    anchored to the median centre of the detected placards (warped to flat space)
    and lines are spaced at placard_w / placard_h intervals expanding outward.
    Falls back to a uniform 10 × 6 grid when no placard data is available.

    Pipeline:
      1. Warp the sign quad to a flat rectangle via getPerspectiveTransform.
      2. Draw the grid on the flat canvas.
      3. Warp the overlay back to the original perspective.
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

    # Simple uniform grid — step = placard size + spacing gap.
    _pw = max(1, placard_w or 1)
    _ph = max(1, placard_h or 1)
    step_x = max(1, _pw + spacing)
    step_y = max(1, _ph + spacing)
    xs = list(range(0, fw + 1, step_x))
    ys = list(range(0, fh + 1, step_y))

    for x in xs:
        cv2.line(overlay, (x, 0), (x, fh), COLOR_GRID, 2, cv2.LINE_AA)
    for y in ys:
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


def render_flat_annotated_image(
    pil_image: Image.Image,
    sign_homography: tuple,
    per_region: list[dict],
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    placard_w: int | None = None,
    placard_h: int | None = None,
    spacing: int = 0,
    grid_w: int | None = None,
    grid_h: int | None = None,
    min_confidence: float = 0.0,
) -> Image.Image:
    """
    Render the perspective-corrected (flat) sign view with overlays drawn
    directly in flat space:
      - Grid lines anchored to detected placard centres
      - Orange polygons for detected empty regions
      - Green boxes for detected existing placards
      - Cyan rectangles for proposed new placard slots

    sign_homography: (H, H_inv, dst_w, dst_h) from compute_region_homography.
    quad_placements in per_region are in image space; they are warped back to
    flat space via H before drawing.
    """
    H, H_inv, dst_w, dst_h = sign_homography

    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    flat = cv2.warpPerspective(img_bgr, H, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)

    # --- Grid lines ---
    # Simple uniform grid — step = placard size + spacing gap.
    _pw = max(1, placard_w or 1)
    _ph = max(1, placard_h or 1)
    step_x = max(1, _pw + spacing)
    step_y = max(1, _ph + spacing)
    xs = list(range(0, dst_w + 1, step_x))
    ys = list(range(0, dst_h + 1, step_y))

    overlay = flat.copy()
    for x in xs:
        cv2.line(overlay, (x, 0), (x, dst_h), COLOR_GRID, 2, cv2.LINE_AA)
    for y in ys:
        cv2.line(overlay, (0, y), (dst_w, y), COLOR_GRID, 2, cv2.LINE_AA)
    flat = cv2.addWeighted(overlay, 0.45, flat, 0.55, 0)

    # --- Empty regions (warp polygon to flat space) ---
    for pred in empty_space_predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        pts = pred.get("points", [])
        if not pts:
            continue
        pts_arr = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32).reshape(-1, 1, 2)
        flat_pts = cv2.perspectiveTransform(pts_arr, H).reshape(-1, 2).astype(np.int32)
        cv2.polylines(flat, [flat_pts], isClosed=True, color=COLOR_EMPTY_REGION, thickness=2, lineType=cv2.LINE_AA)

    # --- Existing placards (warp bbox corners to flat space) ---
    for pred in placard_predictions:
        if pred.get("confidence", 0) < min_confidence:
            continue
        x_c, y_c = pred.get("x", 0), pred.get("y", 0)
        pw, ph   = pred.get("width", 0), pred.get("height", 0)
        corners  = np.array([
            [x_c - pw / 2, y_c - ph / 2],
            [x_c + pw / 2, y_c - ph / 2],
            [x_c + pw / 2, y_c + ph / 2],
            [x_c - pw / 2, y_c + ph / 2],
        ], dtype=np.float32).reshape(-1, 1, 2)
        flat_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2).astype(np.int32)
        cv2.polylines(flat, [flat_corners], isClosed=True, color=COLOR_PLACARD, thickness=2, lineType=cv2.LINE_AA)

    # --- Proposed placard slots (warp quad corners to flat space) ---
    for region in per_region:
        if not region.get("used_perspective"):
            continue
        for quad in region.get("quad_placements", []):
            pts_arr = np.array(quad, dtype=np.float32).reshape(-1, 1, 2)
            flat_pts = cv2.perspectiveTransform(pts_arr, H).reshape(-1, 2).astype(np.int32)
            cv2.polylines(flat, [flat_pts], isClosed=True, color=COLOR_PROPOSED, thickness=2, lineType=cv2.LINE_AA)

    flat_rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(flat_rgb)


def render_annotated_image(
    pil_image: Image.Image,
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    per_region: list[dict],
    min_confidence: float = 0.0,
    sign_quad: list[dict] | None = None,
    sign_polygon: list[dict] | None = None,
    placard_w: int | None = None,
    placard_h: int | None = None,
    spacing: int = 0,
    grid_w: int | None = None,
    grid_h: int | None = None,
) -> Image.Image:
    """
    Compose all overlays onto the image and return a PIL Image.

    sign_quad    – exactly 4 corners [TL, TR, BR, BL], used for the perspective grid
    sign_polygon – raw model polygon (all points), used for the yellow outline
    placard_w/h  – scaled slot dimensions; grid_w/h – fixed unscaled grid step

    Drawing order:
      1. Perspective grid (if sign_quad detected)
      2. Sign boundary outline (raw model polygon)
      3. Empty regions
      4. Existing placards
      5. Proposed placements (on top)
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if sign_quad:
        img = draw_sign_corner_grid(
            img, sign_quad,
            placard_predictions=placard_predictions if placard_predictions else None,
            placard_w=placard_w,
            placard_h=placard_h,
            spacing=spacing,
            grid_w=grid_w,
            grid_h=grid_h,
        )

    outline_pts = sign_quad or sign_polygon
    if outline_pts:
        img = draw_sign_quad(img, outline_pts)

    img = draw_empty_regions(img, empty_space_predictions, min_confidence)
    img = draw_placards(img, placard_predictions, min_confidence)
    img = draw_proposed_placements(img, per_region)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
