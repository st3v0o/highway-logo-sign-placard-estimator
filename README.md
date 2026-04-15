# Highway Logo Sign — Placard Capacity Estimator

Estimates how many new business placards can fit in the open space on a highway blue logo sign. Given one or more sign photos, it:

1. Detects the sign boundary and corrects for camera perspective.
2. Detects existing placards to measure the standard placard size and anchor the slot grid.
3. Detects empty space regions on the sign.
4. Lays out a grid anchored to the existing placards and counts how many new placards fit in each open region.

The standalone Streamlit app (`app.py`) lets a user upload images or supply a spreadsheet of image URLs, tune parameters with live sliders, and view annotated results. The core fitting and geometry logic is written as plain Python modules with no Streamlit dependency so it can be embedded in any other application.

---

## Repository structure

```
├── app.py               # Streamlit UI — upload, sidebar controls, results display
├── fitting.py           # Core capacity estimation (no UI dependency)
├── visualize.py         # OpenCV overlay drawing — annotated + flat views
├── geometry.py          # Homography, masking, polygon utilities
├── roboflow_client.py   # Roboflow API calls + mock responses
├── settings.json        # Persisted sidebar settings (written by the UI on Save)
├── requirements.txt     # Python dependencies
└── response_cache/      # Disk cache of API responses keyed by image MD5 (auto-created)
```

---

## The three Roboflow models

All three models are called via the Roboflow hosted inference REST API (`https://detect.roboflow.com/<project>/<version>`). The helper functions in `roboflow_client.py` handle image encoding and HTTP.

| Purpose | Project ID | Version | What it returns |
|---|---|---|---|
| Sign boundary | `blue-logo-sign` | 1 | A polygon or bbox for the sign panel. Used to build the perspective homography so all downstream work happens in flat (de-perspected) space. |
| Existing placards | `placard-condition-assessment` | 12 | Bounding boxes for every brand panel already on the sign. Used to measure the standard placard size and anchor the slot grid. |
| Empty space | `empty-blue-space` | 2 | Polygon regions that are open and eligible for new placards. |

### Roboflow response format (same for all three models)

```json
{
  "predictions": [
    {
      "class": "Blue-Logo-Sign",
      "x": 320,
      "y": 285,
      "width": 590,
      "height": 410,
      "confidence": 0.97,
      "points": [{"x": 28, "y": 82}, {"x": 612, "y": 78}, ...]
    }
  ],
  "image": {"width": 640, "height": 480}
}
```

- `x` and `y` are the **center** of the bounding box.
- `points` (polygon) is used for masks and homography when present; the bbox is used as a fallback.

---

## Data flow

```
Image
  │
  ├─► call_sign_model()        →  sign_resp
  ├─► call_placard_model()     →  placard_resp
  └─► call_empty_space_model() →  empty_resp
                                       │
                         estimate_total_capacity(...)
                                       │
              ┌────────────────────────┤
              │                        │
    render_annotated_image()    render_flat_annotated_image()
    (original perspective)      (perspective-corrected, with grid)
```

### Step 1 — Sign boundary → perspective homography

`geometry.sign_quad_for_homography(sign_pred)` extracts a clean 4-corner quad from the model polygon (spike removal only — no Laplacian smoothing, so real corners stay at their exact pixel positions). `geometry.compute_region_homography(sign_quad)` returns `(H, H_inv, dst_w, dst_h)` — the 3×3 matrix that maps image pixels to flat-sign pixels and back.

### Step 2 — Placard size estimation

`fitting.estimate_placard_size_from_detections(placard_predictions, sign_quad)` projects each detected placard's bounding box corners onto the sign's own horizontal/vertical axes in image space, then scales to flat-space pixels. This compensates for viewing angle distortion. Returns `(median_width, median_height)` in flat-space pixels.

### Step 3 — Grid anchoring

The grid is anchored to the **leftmost detected placard** in flat space. Column pitch is derived from the **actual center-to-center distance between adjacent detected placards** — not from the placard width alone. This ensures every existing placard falls cleanly within its own column and no grid line bisects an existing placard. Row pitch is similarly derived from vertical spacing between detected placards. When only one placard is detected, the pitch falls back to the measured placard size.

`grid_w` / `grid_h` = detected placard size × `grid_cell_scale` slider.

### Step 4 — Slot fitting

`fitting.place_rectangles_perspective_aware()` works entirely in flat sign space:

1. Warps the empty-space mask into flat space.
2. Punches out existing placard footprints (no extra spacing — spacing only applies between proposed slots).
3. Iterates over every grid intersection; at each one, attempts to place a placard rectangle centered on the intersection.
4. Accepts the placement only if the rectangle falls entirely within the warped empty-region mask and does not overlap a reserved zone.
5. Warps the accepted rectangle's four corners back to image space as a quadrilateral.

Returns a list of quads (each a 4-element list of `[x, y]` image-space coordinates) per empty region.

### Step 5 — Rendering

Two rendered views are produced:

- **Annotated view** (`render_annotated_image`): original image with overlays — yellow sign boundary, orange empty regions, green existing placards, cyan proposed slots. No grid (grid is only in the flat view).
- **Flat / perspective-corrected view** (`render_flat_annotated_image`): the sign warped to a rectangle with the slot grid drawn directly in flat space, plus all overlays warped into flat coordinates.

---

## Integrating the core logic into another app

Only three files are needed for headless use: `fitting.py`, `geometry.py`, and `roboflow_client.py`, plus the packages in `requirements.txt` minus `streamlit`.

### Minimal integration example

```python
from PIL import Image
from roboflow_client import call_sign_model, call_empty_space_model, call_placard_model
from fitting import estimate_total_capacity

API_KEY         = "your-roboflow-api-key"
SIGN_PROJECT    = "blue-logo-sign"
EMPTY_PROJECT   = "empty-blue-space"
PLACARD_PROJECT = "placard-condition-assessment"

pil_image = Image.open("sign_photo.jpg").convert("RGB")
img_w, img_h = pil_image.size

# 1. Run inference against the three models
sign_resp    = call_sign_model(pil_image, API_KEY, SIGN_PROJECT,    version=1)
empty_resp   = call_empty_space_model(pil_image, API_KEY, EMPTY_PROJECT,  version=2)
placard_resp = call_placard_model(pil_image, API_KEY, PLACARD_PROJECT, version=12)

# 2. Extract predictions
sign_pred = max(
    (p for p in sign_resp["predictions"] if p["class"] == "Blue-Logo-Sign"),
    key=lambda p: p["confidence"],
    default=None,
)
empty_predictions   = [p for p in empty_resp["predictions"]   if p["confidence"] >= 0.4]
placard_predictions = [p for p in placard_resp["predictions"] if p["confidence"] >= 0.3]

# 3. Run capacity estimation
results = estimate_total_capacity(
    empty_space_predictions=empty_predictions,
    img_h=img_h,
    img_w=img_w,
    default_placard_w=90,       # fallback width (px) if no placards are detected
    default_placard_h=70,       # fallback height (px)
    margin=8,                   # min gap between proposed placard and region edge (px)
    spacing=4,                  # min gap between two proposed placards (px)
    min_confidence=0.4,
    placard_scale=1.0,          # scale proposed placard rectangle size (1.0 = detected size)
    grid_cell_scale=1.0,        # scale grid cell pitch (1.0 = detected placard size)
    empty_space_scale=1.0,      # shrink empty regions from center (1.0 = no shrink)
    sign_prediction=sign_pred,
    placard_predictions=placard_predictions,
    estimate_from_detections=True,
)

print(f"New placard slots available: {results['total_fit']}")
print(f"Placard size used: {results['placard_w']} × {results['placard_h']} px")
for region in results["per_region"]:
    print(f"  Region {region['region_index'] + 1}: {region['count']} slots")
    # region["quad_placements"] — list of quads in image space (perspective mode)
    # region["placements"]      — list of (x1,y1,x2,y2) tuples (fallback flat mode)
```

---

## `estimate_total_capacity` — parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `empty_space_predictions` | `list[dict]` | required | Raw predictions from the empty-space model. |
| `img_h`, `img_w` | `int` | required | Image dimensions in pixels. |
| `default_placard_w/h` | `int` | 90 / 70 | Fallback placard template size when no detections are available. |
| `margin` | `int` | 8 | Min gap (px) between proposed placard and region boundary. |
| `spacing` | `int` | 4 | Min gap (px) between two proposed placards. |
| `min_confidence` | `float` | 0.4 | Detection threshold applied to empty-space predictions. |
| `placard_scale` | `float` | 1.0 | Multiplier on the template size for the drawn proposed rectangle. Does **not** affect grid pitch. |
| `grid_cell_scale` | `float` | 1.0 | Multiplier on the template size for the grid cell pitch. Does **not** affect the drawn rectangle size. |
| `empty_space_scale` | `float` | 1.0 | Shrink each empty region from its center before fitting (e.g. 0.9 removes 10% of the region edge). |
| `sign_prediction` | `dict \| None` | None | Best sign-boundary detection. Enables perspective correction when provided. |
| `placard_predictions` | `list[dict] \| None` | None | Existing placard detections. Used for size estimation and grid anchoring. |
| `estimate_from_detections` | `bool` | True | When True, use detected placard size instead of `default_placard_w/h`. |

---

## `estimate_total_capacity` — return value

```python
{
    "total_fit":             int,        # total new placard slots across all regions
    "per_region":            list[dict], # one entry per empty region (see below)
    "placard_w":             int,        # proposed placard width actually used (px)
    "placard_h":             int,        # proposed placard height actually used (px)
    "detected_placard_w":    int,        # raw detected size before placard_scale
    "detected_placard_h":    int,
    "grid_w":                int,        # grid cell pitch width used (px)
    "grid_h":                int,        # grid cell pitch height used (px)
    "used_perspective_mode": bool,
    "sign_quad":             list[dict] | None,  # 4-corner sign boundary [TL,TR,BR,BL]
    "sign_polygon":          list[dict] | None,  # smoothed outline polygon (for drawing)
    "size_from_detections":  bool,       # True if placard size came from model detections
    "sign_homography":       tuple | None,  # (H, H_inv, dst_w, dst_h)
}
```

Each `per_region` entry:

```python
{
    "region_index":    int,
    "count":           int,                      # placard slots placed in this region
    "quad_placements": list[list[list[float]]],  # [[x,y],…] quads — perspective mode
    "placements":      list[tuple],              # (x1,y1,x2,y2) tuples — flat fallback
    "used_perspective": bool,
}
```

`quad_placements` are the coordinates to use for display or downstream logic. Each entry is a list of four `[x, y]` pairs in the **original image coordinate space**.

---

## Adding visualizations

```python
from visualize import render_annotated_image, render_flat_annotated_image

# Original-perspective view with detection overlays (PIL Image)
annotated = render_annotated_image(
    pil_image=pil_image,
    placard_predictions=placard_predictions,
    empty_space_predictions=empty_predictions,
    per_region=results["per_region"],
    sign_quad=results.get("sign_quad"),
    sign_polygon=results.get("sign_polygon"),
    placard_w=results["placard_w"],
    placard_h=results["placard_h"],
    grid_w=results["grid_w"],
    grid_h=results["grid_h"],
)

# Perspective-corrected flat view with grid overlay (PIL Image)
if results.get("sign_homography"):
    flat = render_flat_annotated_image(
        pil_image=pil_image,
        sign_homography=results["sign_homography"],
        per_region=results["per_region"],
        placard_predictions=placard_predictions,
        empty_space_predictions=empty_predictions,
        placard_w=results["placard_w"],
        placard_h=results["placard_h"],
        grid_w=results["grid_w"],
        grid_h=results["grid_h"],
    )
```

### Overlay color legend

| Color | Meaning |
|---|---|
| Cyan | Proposed new placard slots |
| Orange | Detected empty regions |
| Green | Detected existing placards |
| Yellow | Sign boundary outline |
| Yellow-cyan grid | Slot grid (flat view only) |

---

## Disk response cache

`app.py` caches every Roboflow API response to `response_cache/<md5>.json` so the same image is never re-sent to the API. The cache key is the MD5 of the JPEG-encoded image (resized to max 1200 px) and all three model responses are stored together in one file. This logic lives entirely in `app.py` and is not a dependency of the core modules — implement your own caching strategy as needed.

---

## Mock mode (development / testing)

`roboflow_client.py` exports three ready-made sample responses:

```python
from roboflow_client import MOCK_SIGN_RESPONSE, MOCK_PLACARD_RESPONSE, MOCK_EMPTY_SPACE_RESPONSE
```

Pass these directly to `estimate_total_capacity` instead of making live API calls. The Streamlit app exposes a **Mock Mode** checkbox in the sidebar that does exactly this.

---

## Running the standalone Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 5000
```

Open `http://localhost:5000`. Settings (API key, model IDs, slider defaults) are stored in `settings.json` via the **Save Settings** button in the sidebar.

The main area shows two tabs per image:
- **Perspective-Corrected Sign** — flat view with slot grid and all overlays.
- **Annotated** — original-perspective view with overlays but no grid.

---

## Dependencies

```
streamlit>=1.32.0        # UI only — not needed for headless integration
requests>=2.31.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.2.0
```

For headless integration you only need `fitting.py`, `geometry.py`, and `roboflow_client.py`, which require only `numpy`, `opencv-python-headless`, `Pillow`, and `requests`.

---

## Key design decisions

**Perspective-first fitting.** All placard placement is done in flat (perspective-corrected) sign space, then results are projected back to image space as quadrilaterals. This ensures proposed placards are correctly sized and aligned regardless of the camera angle.

**Grid pitch from actual placard spacing.** The column pitch is the measured center-to-center distance between adjacent detected placards — not just the placard width. When two existing placards sit side-by-side, their real center gap is larger than one placard width; using the placard width as the pitch would put a column boundary through the second placard.

**Slider independence.** `placard_scale` changes only the drawn cyan rectangle size. `grid_cell_scale` changes only the grid cell pitch. Neither shifts the grid anchor or requires a new API call — re-fitting after any slider change reuses cached API responses.

**Spike removal without smoothing for the homography quad.** The sign polygon from the model often has exit-panel protrusions. Angle-based spike removal eliminates these. Laplacian smoothing is intentionally skipped for the homography quad because it rounds corners inward and corrupts the perspective transform.
