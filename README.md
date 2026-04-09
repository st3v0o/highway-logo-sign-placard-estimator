# Highway Logo Sign — Placard Capacity Estimator

A Python Streamlit app for estimating how many new business placards can fit into open blue space on a highway logo sign.

---

## How to Install

1. Make sure Python 3.11+ is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py --server.port 5000
```

Then open `http://localhost:5000` in your browser.

---

## Using Mock Mode (Recommended for Quick Testing)

**Mock Mode is enabled by default.** When on, the app uses hardcoded sample prediction data instead of making real API calls — no Roboflow account or credentials needed.

1. Upload any image (even a blank one will work in Mock Mode).
2. Click **Run Inference**.
3. The app will show the annotated result using sample placard and empty-space detections.

To disable Mock Mode, uncheck the "Mock Mode" checkbox in the sidebar and enter your Roboflow credentials.

---

## Entering Roboflow Settings (Live Mode)

To use your real Roboflow models:

1. Uncheck **Mock Mode** in the sidebar.
2. Enter your **Roboflow API Key**.
3. For the **Placard Detection Model**:
   - Workspace ID (your Roboflow workspace slug)
   - Project ID (your project slug)
   - Version number
4. For the **Empty-Space Detection Model**:
   - Check "Use same workspace for both models" if they share a workspace.
   - Enter Project ID and Version.
5. Adjust confidence threshold and layout parameters as needed.
6. Upload your image and click **Run Inference**.

---

## How the Placard Count Is Estimated

1. **Empty-space detection**: The empty-blue-space model identifies regions where placards could be installed.
2. **Placard size**: Either estimated from the median size of existing detected placards, or set manually via the sidebar.
3. **Fitting algorithm**: For each empty region:
   - A binary mask is created from the polygon outline (or bounding box if no polygon is available).
   - An outer margin is eroded from the mask boundary.
   - The algorithm scans left-to-right, top-to-bottom and greedily places placard rectangles.
   - A rectangle is only counted if the **full placard** fits inside the region mask.
   - After placing one, spacing is reserved around it before trying the next position.
4. **Result**: Total count and a per-region breakdown are displayed.

This approach counts physical fit — not just total area divided by placard area.

---

## Adapting the Roboflow Endpoint URL

The URL builder is in **`roboflow_client.py`**, in the `build_inference_url()` function:

```python
def build_inference_url(workspace_id: str, project_id: str, version: int) -> str:
    return f"https://detect.roboflow.com/{project_id}/{version}"
```

If Roboflow changes their hosted inference URL format, edit only this one function. All model calls use it automatically.

---

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI and main workflow |
| `roboflow_client.py` | Roboflow API calls and URL builder |
| `geometry.py` | Bbox/polygon → mask conversions |
| `fitting.py` | Placard fitting algorithm |
| `visualize.py` | OpenCV overlay drawing |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Color Legend

| Color | Meaning |
|---|---|
| Red | Existing installed placards |
| Yellow | Empty region (bounding box mode) |
| Green | Empty region (polygon mode) |
| Cyan | Proposed new placard placement |
