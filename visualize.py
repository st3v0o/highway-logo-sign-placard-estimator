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
        # Draw the detected perspective quad (4-corner approximation) in yellow
        detected_quad = region.get("detected_quad")
        if detected_quad and region.get("used_perspective"):
            quad_pts = np.array(
                [[int(round(p["x"])), int(round(p["y"]))] for p in detected_quad],
                dtype=np.int32,
            )
            cv2.polylines(img, [quad_pts], isClosed=True,
                          color=COLOR_PERSP_QUAD, thickness=max(thickness, 4))
            # Mark each corner with a filled circle
            for pt in quad_pts:
                cv2.circle(img, tuple(pt), 10, COLOR_PERSP_QUAD, -1)

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


def render_annotated_image(
    pil_image: Image.Image,
    placard_predictions: list[dict],
    empty_space_predictions: list[dict],
    per_region: list[dict],
    min_confidence: float = 0.0,
    grid: dict | None = None,
) -> Image.Image:
    """
    Compose all overlays onto the image and return a PIL Image.

    Drawing order: grid lines → empty regions → existing placards → proposed placements
    so that proposals always appear on top.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if grid:
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

    img = draw_empty_regions(img, empty_space_predictions, min_confidence)
    img = draw_placards(img, placard_predictions, min_confidence)
    img = draw_proposed_placements(img, per_region)

    img_rgb_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_out)
