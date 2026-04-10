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
COLOR_PLACARD = (0, 200, 0)         # Green  — existing placards
COLOR_EMPTY_REGION = (0, 140, 255)  # Orange — empty regions (bbox fallback)
COLOR_EMPTY_POLYGON = (0, 100, 255) # Orange-red — empty regions (polygon mode)
COLOR_PROPOSED = (220, 200, 0)      # Cyan  — proposed new placements
COLOR_PERSP_QUAD = (0, 230, 255)    # Yellow — detected perspective quad corners


def draw_placards(
    img: np.ndarray,
    predictions: list[dict],
    min_confidence: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw existing placard detections in green on the image."""
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
    Draw empty-space detections. Uses polygon outline if points are available,
    otherwise draws a bounding box.
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

    For perspective-aware regions (used_perspective=True) draws quadrilaterals
    from quad_placements. Falls back to drawing rectangles from placements.
    """
    img_h, img_w = img.shape[:2]

    for region in per_region:
        # (sign quad is now drawn at the top level in render_annotated_image)

        if region.get("used_perspective") and region.get("quad_placements"):
            # Draw perspective-corrected quadrilaterals
            for quad in region["quad_placements"]:
                pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in quad],
                               dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=COLOR_PROPOSED, thickness=thickness)
        else:
            # Draw standard rectangles
            for x1, y1, x2, y2 in region.get("placements", []):
                x1, y1, x2, y2 = clip_to_image_bounds(x1, y1, x2, y2, img_w, img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PROPOSED, thickness)

    return img


COLOR_GRID = (255, 220, 0)  # Yellow-cyan — grid lines


def _project_line_through_H_inv(
    H_inv: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Project two flat-space endpoints through H_inv to get image-space endpoints.
    Returns a pair of integer (x, y) tuples.
    """
    def proj(pt):
        v = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        out = H_inv @ v
        return (int(round(out[0] / out[2])), int(round(out[1] / out[2])))
    return proj(p0), proj(p1)


def draw_grid_lines(
    img: np.ndarray,
    row_centers: list[float],
    col_centers: list[float],
    alpha: float = 0.45,
    flat_row_centers: list[float] | None = None,
    flat_col_centers: list[float] | None = None,
    H_inv: np.ndarray | None = None,
    dst_w: int | None = None,
    dst_h: int | None = None,
) -> np.ndarray:
    """
    Draw the inferred row/column grid as semi-transparent lines.

    When flat_row_centers/flat_col_centers and H_inv are provided (perspective mode),
    each grid line is drawn by projecting its flat-space endpoints through H_inv so the
    lines follow the sign's real perspective angle (converging toward the vanishing point).

    Otherwise falls back to straight horizontal/vertical lines in image space.
    """
    img_h, img_w = img.shape[:2]
    overlay = img.copy()

    use_perspective = (
        flat_row_centers is not None
        and flat_col_centers is not None
        and H_inv is not None
        and dst_w is not None
        and dst_h is not None
    )

    if use_perspective:
        # Row lines: horizontal lines in flat space → projected to image space
        for ry in flat_row_centers:
            pt_a, pt_b = _project_line_through_H_inv(H_inv, (0.0, ry), (float(dst_w), ry))
            cv2.line(overlay, pt_a, pt_b, COLOR_GRID, 1, cv2.LINE_AA)

        # Column lines: vertical lines in flat space → projected to image space
        for cx in flat_col_centers:
            pt_a, pt_b = _project_line_through_H_inv(H_inv, (cx, 0.0), (cx, float(dst_h)))
            cv2.line(overlay, pt_a, pt_b, COLOR_GRID, 1, cv2.LINE_AA)
    else:
        # Non-perspective: straight horizontal/vertical lines in image space
        for ry in row_centers:
            y = int(round(ry))
            if 0 <= y < img_h:
                cv2.line(overlay, (0, y), (img_w - 1, y), COLOR_GRID, 1, cv2.LINE_AA)

        for cx in col_centers:
            x = int(round(cx))
            if 0 <= x < img_w:
                cv2.line(overlay, (x, 0), (x, img_h - 1), COLOR_GRID, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    return img


def draw_sign_quad(
    img: np.ndarray,
    sign_quad: list[dict],
) -> np.ndarray:
    """
    Draw the detected sign boundary quad (4 corners) in yellow.
    Always drawn when perspective mode is on, regardless of whether any
    placard placements fit inside it.
    """
    quad_pts = np.array(
        [[int(round(p["x"])), int(round(p["y"]))] for p in sign_quad],
        dtype=np.int32,
    )
    return img


def _detect_blue_bounds_flat(flat_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the sign's blue panel in a perspective-corrected
    (flat) image.  Uses strict saturation to separate vivid sign-blue from
    washed-out sky-blue, then falls back to the full image if nothing is found.

    Returns (x1, y1, x2, y2) in flat-image pixel coords.
    """
    fh, fw = flat_bgr.shape[:2]
    hsv = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV)

    # Strict blue: sign-panel blue (saturation ≥ 80 distinguishes it from sky)
    blue = cv2.inRange(hsv, np.array([85, 80, 40]), np.array([140, 255, 230]))
    ker_c = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    ker_o = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, ker_c)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN,  ker_o)

    n, _, stats, _ = cv2.connectedComponentsWithStats(blue)
    if n < 2:
        return 0, 0, fw, fh  # fallback: whole flat image

    # Largest non-background component
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area_frac = stats[best, cv2.CC_STAT_AREA] / (fw * fh)
    if area_frac < 0.08:
        return 0, 0, fw, fh  # tiny blob → fallback

    x1 = int(stats[best, cv2.CC_STAT_LEFT])
    y1 = int(stats[best, cv2.CC_STAT_TOP])
    x2 = x1 + int(stats[best, cv2.CC_STAT_WIDTH])
    y2 = y1 + int(stats[best, cv2.CC_STAT_HEIGHT])
    return x1, y1, x2, y2


def draw_sign_corner_grid(
    img: np.ndarray,
    sign_quad: list[dict],
    cols: int = 10,
    rows: int = 6,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw a perspective-corrected grid over the sign panel.

    Pipeline:
      1. Warp the sign quad to a flat (head-on) rectangle via
         cv2.getPerspectiveTransform + cv2.warpPerspective.
      2. Detect the actual blue-panel bounds in the flat image
         (strict HSV mask → largest connected component bounding box).
      3. Draw a simple horizontal/vertical grid in flat space.
      4. Warp the flat grid overlay back to the original perspective.
      5. Blend it onto the original image clipped to the sign quad.

    sign_quad: [TL, TR, BR, BL] dicts with 'x' and 'y' keys.
    """
    h_img, w_img = img.shape[:2]

    src = np.float32([[q["x"], q["y"]] for q in sign_quad])  # TL TR BR BL

    # Flat output size from quad edge lengths (capped for performance)
    w_top  = float(np.linalg.norm(src[1] - src[0]))
    w_bot  = float(np.linalg.norm(src[2] - src[3]))
    h_left = float(np.linalg.norm(src[3] - src[0]))
    h_right= float(np.linalg.norm(src[2] - src[1]))
    fw = min(int(max(w_top, w_bot)), 1200)
    fh = min(int(max(h_left, h_right)), 900)
    if fw < 20 or fh < 20:
        return img  # degenerate quad → skip

    dst = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])

    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # ── Step 1: warp to flat ─────────────────────────────────────────────────
    flat = cv2.warpPerspective(img, M, (fw, fh))

    # ── Step 2: find actual blue bounds in flat space ────────────────────────
    x1, y1, x2, y2 = _detect_blue_bounds_flat(flat)
    gw, gh = x2 - x1, y2 - y1
    if gw < 10 or gh < 10:
        x1, y1, x2, y2 = 0, 0, fw, fh
        gw, gh = fw, fh

    # ── Step 3: draw grid in flat space ──────────────────────────────────────
    overlay_flat = flat.copy()
    for i in range(cols + 1):
        x = x1 + int(i * gw / cols)
        cv2.line(overlay_flat, (x, y1), (x, y2), COLOR_GRID, 2, cv2.LINE_AA)
    for j in range(rows + 1):
        y = y1 + int(j * gh / rows)
        cv2.line(overlay_flat, (x1, y), (x2, y), COLOR_GRID, 2, cv2.LINE_AA)

    flat_blended = cv2.addWeighted(overlay_flat, alpha, flat, 1.0 - alpha, 0)

    # ── Step 4: warp flat grid back to original perspective ──────────────────
    warped = cv2.warpPerspective(flat_blended, M_inv, (w_img, h_img))

    # ── Step 5: blend — only inside the detected blue region (mapped back) ───
    # Build a mask in flat space for the refined blue region and warp it back.
    grid_mask_flat = np.zeros((fh, fw), np.uint8)
    cv2.rectangle(grid_mask_flat, (x1, y1), (x2, y2), 255, -1)
    grid_mask_orig = cv2.warpPerspective(
        grid_mask_flat.astype(np.float32), M_inv, (w_img, h_img)
    )
    blend_mask = (grid_mask_orig > 0.5).astype(np.uint8)

    out = img.copy()
    out[blend_mask > 0] = warped[blend_mask > 0]
    return out


def render_annotated_image(
    pil_image: Image.Image,
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    per_region: list[dict],
    min_confidence: float = 0.0,
    grid: dict | None = None,
    sign_quad: list[dict] | None = None,
    draw_sign_grid: bool = False,
) -> Image.Image:
    """
    Compose all overlays onto the image and return a PIL Image.

    draw_sign_grid: when True and sign_quad is available, draws the same
        bilinear 10×6 perspective grid as the Generate Grid button — derived
        from the sign boundary corners rather than from detected placard positions.

    Drawing order: grid lines → sign quad → empty regions → existing placards
    → proposed placements, so proposals always appear on top.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if draw_sign_grid and sign_quad:
        img = draw_sign_corner_grid(img, sign_quad)
    elif grid:
        img = draw_grid_lines(
            img,
            grid["row_centers"],
            grid["col_centers"],
            flat_row_centers=grid.get("flat_row_centers"),
            flat_col_centers=grid.get("flat_col_centers"),
            H_inv=grid.get("H_inv"),
            dst_w=grid.get("dst_w"),
            dst_h=grid.get("dst_h"),
        )

    # Draw the detected sign quad unconditionally — visible even when no
    # placards fit (e.g. large placard scale) so the user can verify alignment.
    if sign_quad:
        img = draw_sign_quad(img, sign_quad)

    img = draw_proposed_placements(img, per_region)

    img_rgb_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_out)
