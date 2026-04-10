"""
Standalone debug script: run sign-quad detection + perspective grid on a
single image without Streamlit or Roboflow.

Usage:
    python debug_quad.py
Output:
    /tmp/debug_quad_result.jpg
"""

import cv2
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from geometry import detect_sign_quad_from_detections

IMG_PATH   = "/tmp/debug_sign.jpg"
OUT_PATH   = "/tmp/debug_quad_result.jpg"
SCALE      = 0.25          # work at 25 % to keep things fast; quad is up-scaled back

img_full = cv2.imread(IMG_PATH)
if img_full is None:
    raise FileNotFoundError(IMG_PATH)

h_full, w_full = img_full.shape[:2]
print(f"Full image: {w_full}×{h_full}")

img = cv2.resize(img_full, (int(w_full * SCALE), int(h_full * SCALE)))
h, w = img.shape[:2]
print(f"Working at: {w}×{h}")

# ─── Mock predictions (pixel coords at SCALED size) ───────────────────────
# Estimated from visual inspection of the image (4032×3024 full → 1008×756)
#
#  Main sign panel   x: 52–975,  y: 188–737
#  Popeyes placard   x: 568–838, y: 225–420
#  Empty region      x: 58–568,  y: 188–737

def make_pred(x1, y1, x2, y2, cls="object"):
    return {
        "x": (x1 + x2) / 2,
        "y": (y1 + y2) / 2,
        "width":  x2 - x1,
        "height": y2 - y1,
        "confidence": 0.9,
        "class": cls,
    }

placard_preds = [make_pred(568, 225, 838, 420, "placard")]
empty_preds   = [make_pred( 58, 188, 568, 737, "empty")]

# ─── Run quad detection ────────────────────────────────────────────────────
quad = detect_sign_quad_from_detections(img, placard_preds, empty_preds)
print("quad:", quad)

# ─── Draw on image ─────────────────────────────────────────────────────────
vis = img.copy()

# Draw prediction bboxes
for p in placard_preds:
    x1=int(p['x']-p['width']/2); y1=int(p['y']-p['height']/2)
    x2=int(p['x']+p['width']/2); y2=int(p['y']+p['height']/2)
    cv2.rectangle(vis, (x1,y1),(x2,y2), (0,255,0), 2)
    cv2.putText(vis, "placard", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

for p in empty_preds:
    x1=int(p['x']-p['width']/2); y1=int(p['y']-p['height']/2)
    x2=int(p['x']+p['width']/2); y2=int(p['y']+p['height']/2)
    cv2.rectangle(vis, (x1,y1),(x2,y2), (0,165,255), 2)
    cv2.putText(vis, "empty", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

if quad:
    pts = np.array([[int(p['x']), int(p['y'])] for p in quad], dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0,255,255), 3)
    for i, p in enumerate(quad):
        cv2.circle(vis, (int(p['x']), int(p['y'])), 8, (0,0,255), -1)
        cv2.putText(vis, ["TL","TR","BR","BL"][i],
                    (int(p['x'])+5, int(p['y'])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ── perspective grid (10×6 cells) ──────────────────────────────────────
    tl = np.float32([quad[0]['x'], quad[0]['y']])
    tr = np.float32([quad[1]['x'], quad[1]['y']])
    br = np.float32([quad[2]['x'], quad[2]['y']])
    bl = np.float32([quad[3]['x'], quad[3]['y']])

    COLS, ROWS = 10, 6
    for i in range(COLS + 1):
        t = i / COLS
        top_pt  = tl + t * (tr - tl)
        bot_pt  = bl + t * (br - bl)
        cv2.line(vis, tuple(top_pt.astype(int)), tuple(bot_pt.astype(int)),
                 (0, 255, 255), 1)

    for j in range(ROWS + 1):
        t = j / ROWS
        left_pt  = tl + t * (bl - tl)
        right_pt = tr + t * (br - tr)
        cv2.line(vis, tuple(left_pt.astype(int)), tuple(right_pt.astype(int)),
                 (0, 255, 255), 1)
else:
    print("⚠  quad is None — fallback not reached a quad")
    cv2.putText(vis, "QUAD DETECTION FAILED", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

cv2.imwrite(OUT_PATH, vis)
print(f"Saved → {OUT_PATH}")
