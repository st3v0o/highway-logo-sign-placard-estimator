"""
app.py

Streamlit GUI for the Highway Logo Sign Placard Capacity Estimator.

Workflow:
  1. Upload an image of a highway logo sign
  2. Configure Roboflow model credentials in the sidebar (or enable Mock Mode)
  3. Click "Run Inference" — calls both models once and stores predictions
  4. Adjust the Placard Scale slider to live-update placements without re-running inference
"""

import json
import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from roboflow_client import (
    call_placard_model,
    call_empty_space_model,
    MOCK_PLACARD_RESPONSE,
    MOCK_EMPTY_SPACE_RESPONSE,
)
from fitting import estimate_total_capacity
from geometry import detect_sign_quad_from_blue
from visualize import render_annotated_image, draw_sign_quad


# ---------------------------------------------------------------------------
# Settings persistence helpers
# ---------------------------------------------------------------------------

SETTINGS_FILE = "settings.json"

DEFAULTS = {
    "api_key": "",
    "placard_workspace": "",
    "placard_project": "",
    "placard_version": 1,
    "same_workspace": True,
    "empty_workspace": "",
    "empty_project": "",
    "empty_version": 1,
    "mock_mode": True,
    "estimate_from_detections": True,
    "default_placard_w": 90,
    "default_placard_h": 70,
    "outer_margin": 8,
    "spacing": 4,
    "min_confidence": 0.4,
    "placard_scale": 100,
    "empty_space_scale": 100,
    "perspective_mode": False,
    "grid_mode": False,
}


def load_settings() -> dict:
    """Load saved settings from disk, falling back to defaults for any missing keys."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_settings(values: dict) -> None:
    """Write current settings to disk."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=2)


# Load saved settings once per session
if "settings_loaded" not in st.session_state:
    saved = load_settings()
    for k, v in saved.items():
        st.session_state[k] = v
    st.session_state["settings_loaded"] = True


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Placard Capacity Estimator",
    page_icon="🛣️",
    layout="wide",
)


st.title("Highway Logo Sign — Placard Capacity Estimator")
st.caption(
    "Upload a sign image, configure Roboflow models, and estimate how many new "
    "business placards can fit in the open blue space."
)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    mock_mode = st.checkbox(
        "Mock Mode (no API calls)",
        key="mock_mode",
        help="Uses hardcoded sample predictions so you can test without a Roboflow account.",
    )

    st.divider()
    st.subheader("Roboflow API Key")
    api_key = st.text_input("API Key", key="api_key", type="password", disabled=mock_mode)

    st.divider()
    st.subheader("Placard Detection Model")
    placard_workspace = st.text_input("Workspace ID", key="placard_workspace", disabled=mock_mode)
    placard_project = st.text_input("Project ID", key="placard_project", disabled=mock_mode)
    placard_version = st.number_input("Version", min_value=1, step=1, key="placard_version", disabled=mock_mode)

    same_workspace = st.checkbox("Use same workspace for both models", key="same_workspace", disabled=mock_mode)

    st.divider()
    st.subheader("Empty-Space Detection Model")
    if same_workspace:
        empty_workspace = placard_workspace
        st.text_input("Workspace ID (same as above)", value=placard_workspace, disabled=True)
    else:
        empty_workspace = st.text_input("Workspace ID", key="empty_workspace", disabled=mock_mode)

    empty_project = st.text_input("Project ID", key="empty_project", disabled=mock_mode)
    empty_version = st.number_input("Version", min_value=1, step=1, key="empty_version", disabled=mock_mode)

    st.divider()
    st.subheader("Placard Size")
    estimate_from_detections = st.checkbox(
        "Estimate placard size from detected placards",
        key="estimate_from_detections",
        help="Uses the median size of existing detected placards as the template size.",
    )
    default_placard_w = st.number_input("Default width (px)", min_value=10, step=5, key="default_placard_w")
    default_placard_h = st.number_input("Default height (px)", min_value=10, step=5, key="default_placard_h")
    placard_scale = st.slider(
        "Placard scale (%)",
        min_value=30,
        max_value=100,
        step=5,
        key="placard_scale",
        help="Scales the fitting rectangle. Lower values fit placards into tighter spaces. Updates live after inference.",
    )
    empty_space_scale = st.slider(
        "Empty space scale (%)",
        min_value=30,
        max_value=100,
        step=5,
        key="empty_space_scale",
        help="Shrinks each detected empty region from its center. Lower values reduce the usable area, fitting fewer placards per region. Updates live after inference.",
    )
    perspective_mode = st.checkbox(
        "Perspective-aware fitting",
        key="perspective_mode",
        help=(
            "When enabled and a detected region has exactly 4 polygon corners, "
            "the region is un-warped to a flat rectangle, placards are fitted there, "
            "and their outlines are warped back as trapezoids matching the camera angle. "
            "Requires your detection model to return polygon (not just bounding-box) outputs."
        ),
    )
    grid_mode = st.checkbox(
        "Grid-aligned placement",
        key="grid_mode",
        help=(
            "Infers a uniform row/column grid from the positions of existing detected "
            "placards, extends that grid across the whole sign, and proposes new placards "
            "only at grid intersections inside empty regions. "
            "The inferred grid lines are drawn in yellow. "
            "Produces realistic placements that stay in line with the existing signs."
        ),
    )

    st.divider()
    st.subheader("Layout Parameters")
    outer_margin = st.number_input(
        "Outer margin (px)",
        min_value=0,
        step=2,
        key="outer_margin",
        help="Minimum gap between a placard and the edge of the empty region.",
    )
    spacing = st.number_input(
        "Spacing between placards (px)",
        min_value=0,
        step=2,
        key="spacing",
        help="Gap between adjacent proposed placards.",
    )
    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="min_confidence",
    )

    st.divider()
    if st.button("Save Settings", use_container_width=True):
        save_settings({
            "api_key": st.session_state.api_key,
            "placard_workspace": st.session_state.placard_workspace,
            "placard_project": st.session_state.placard_project,
            "placard_version": int(st.session_state.placard_version),
            "same_workspace": st.session_state.same_workspace,
            "empty_workspace": st.session_state.get("empty_workspace", ""),
            "empty_project": st.session_state.empty_project,
            "empty_version": int(st.session_state.empty_version),
            "mock_mode": st.session_state.mock_mode,
            "estimate_from_detections": st.session_state.estimate_from_detections,
            "default_placard_w": int(st.session_state.default_placard_w),
            "default_placard_h": int(st.session_state.default_placard_h),
            "outer_margin": int(st.session_state.outer_margin),
            "spacing": int(st.session_state.spacing),
            "min_confidence": float(st.session_state.min_confidence),
            "placard_scale": int(st.session_state.placard_scale),
            "empty_space_scale": int(st.session_state.empty_space_scale),
            "perspective_mode": bool(st.session_state.perspective_mode),
            "grid_mode": bool(st.session_state.grid_mode),
        })
        st.success("Settings saved!")


# ---------------------------------------------------------------------------
# Image uploader + Run button
# ---------------------------------------------------------------------------

st.divider()
uploaded_file = st.file_uploader(
    "Upload a sign image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag and drop or click to browse.",
)

# ---------------------------------------------------------------------------
# Sign Quad Debug Panel — no API calls needed, works instantly on any upload
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    with st.expander("🔍 Sign Outline Debug (no API calls needed)", expanded=False):
        st.caption(
            "Preview the sign boundary on your image instantly — no API call needed. "
            "This panel uses colour-only detection on the full image (sky and trees can interfere). "
            "**After you run inference**, the full run uses a much more reliable method: it builds "
            "the sign boundary directly from the polygon corners returned by the model — those "
            "points already lie on the sign surface in perspective, so no colour guessing is needed. "
            "The sliders here are useful for a rough sanity check before your first API call."
        )
        col_s, col_v, col_k, col_o = st.columns(4)
        with col_s:
            dbg_s_min = st.slider("S min (saturation floor)", 60, 200, 120, 5,
                                  help="Raise to exclude sky (sky S ≈ 60–100)")
        with col_v:
            dbg_v_max = st.slider("V max (brightness ceiling)", 100, 255, 200, 5,
                                  help="Lower to exclude bright sky (sky V > 200)")
        with col_k:
            dbg_open_k = st.slider("Open kernel", 4, 40, 20, 2,
                                   help="Larger breaks wider sky-to-sign pixel bridges")
        with col_o:
            dbg_outlier = st.slider("Outlier cutoff ×", 1.0, 2.5, 1.3, 0.05,
                                    help="Hull points beyond this × median distance are pruned")

        _dbg_pil = Image.open(uploaded_file).convert("RGB")
        _dbg_bgr = cv2.cvtColor(np.array(_dbg_pil), cv2.COLOR_RGB2BGR)
        _dbg_quad = detect_sign_quad_from_blue(
            _dbg_bgr,
            s_min=dbg_s_min,
            v_max=dbg_v_max,
            open_k=dbg_open_k,
            outlier_mult=dbg_outlier,
        )
        _dbg_img = cv2.cvtColor(_dbg_bgr.copy(), cv2.COLOR_BGR2RGB)
        if _dbg_quad:
            _dbg_img_drawn = draw_sign_quad(cv2.cvtColor(_dbg_img, cv2.COLOR_RGB2BGR), _dbg_quad)
            _dbg_img_drawn = cv2.cvtColor(_dbg_img_drawn, cv2.COLOR_BGR2RGB)
            st.image(_dbg_img_drawn, caption="Detected sign outline", use_container_width=True)
            st.success(
                f"Quad detected — corners: "
                + ", ".join(f"({p['x']:.0f}, {p['y']:.0f})" for p in _dbg_quad)
            )
        else:
            st.image(_dbg_img, caption="Original image", use_container_width=True)
            st.warning("No blue sign region found with current settings. Try lowering S min or raising V max.")


run_button = st.button("Run Inference", type="primary", disabled=uploaded_file is None)

# ---------------------------------------------------------------------------
# Inference — runs only when the button is clicked.
# Stores raw predictions + image bytes in session_state so that slider
# changes can re-run fitting without hitting the API again.
# ---------------------------------------------------------------------------

if run_button and uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = pil_image.size

    with st.spinner("Running inference..."):
        if mock_mode:
            placard_resp = MOCK_PLACARD_RESPONSE
            empty_resp = MOCK_EMPTY_SPACE_RESPONSE
        else:
            missing = []
            if not api_key:
                missing.append("API Key")
            if not placard_project:
                missing.append("Placard Project ID")
            if not empty_project:
                missing.append("Empty-Space Project ID")

            if missing:
                st.error(f"Please fill in the following fields: {', '.join(missing)}")
                st.stop()

            try:
                placard_resp = call_placard_model(
                    pil_image=pil_image,
                    api_key=api_key,
                    workspace_id=placard_workspace,
                    project_id=placard_project,
                    version=int(placard_version),
                    confidence=min_confidence,
                )
            except Exception as exc:
                st.error(f"Placard model call failed: {exc}")
                st.stop()

            try:
                empty_resp = call_empty_space_model(
                    pil_image=pil_image,
                    api_key=api_key,
                    workspace_id=empty_workspace,
                    project_id=empty_project,
                    version=int(empty_version),
                    confidence=min_confidence,
                )
            except Exception as exc:
                st.error(f"Empty-space model call failed: {exc}")
                st.stop()

    # Filter by confidence and store everything in session_state
    placard_preds = placard_resp.get("predictions", [])
    empty_preds = empty_resp.get("predictions", [])
    conf = min_confidence
    st.session_state["last_placard_preds"] = [p for p in placard_preds if p.get("confidence", 0) >= conf]
    st.session_state["last_empty_preds"] = [p for p in empty_preds if p.get("confidence", 0) >= conf]
    st.session_state["last_placard_resp"] = placard_resp
    st.session_state["last_empty_resp"] = empty_resp
    st.session_state["last_mock_mode"] = mock_mode

    # Store image as bytes so it survives reruns
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    st.session_state["last_img_bytes"] = buf.getvalue()
    st.session_state["last_img_w"] = img_w
    st.session_state["last_img_h"] = img_h


# ---------------------------------------------------------------------------
# Results — runs on every rerun (including slider changes) if predictions exist.
# This means scale/margin/spacing sliders live-update without re-calling the API.
# ---------------------------------------------------------------------------

if "last_placard_preds" in st.session_state:
    placard_preds_filtered = st.session_state["last_placard_preds"]
    empty_preds_filtered = st.session_state["last_empty_preds"]
    placard_resp = st.session_state["last_placard_resp"]
    empty_resp = st.session_state["last_empty_resp"]
    img_w = st.session_state["last_img_w"]
    img_h = st.session_state["last_img_h"]
    pil_image = Image.open(BytesIO(st.session_state["last_img_bytes"]))

    if st.session_state.get("last_mock_mode"):
        st.info("Mock Mode is ON — using sample predictions (no API call made).")

    # Convert PIL image to BGR for blue-based perspective detection
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) if perspective_mode else None

    # Re-run fitting with current slider values every time
    results = estimate_total_capacity(
        empty_space_predictions=empty_preds_filtered,
        placard_predictions=placard_preds_filtered,
        img_h=img_h,
        img_w=img_w,
        default_placard_w=int(default_placard_w),
        default_placard_h=int(default_placard_h),
        margin=int(outer_margin),
        spacing=int(spacing),
        min_confidence=0.0,  # already pre-filtered
        estimate_size_from_detections=estimate_from_detections,
        placard_scale=placard_scale / 100.0,
        empty_space_scale=empty_space_scale / 100.0,
        perspective_mode=perspective_mode,
        image_bgr=image_bgr,
        grid_mode=grid_mode,
    )

    # Re-render annotated image with current placements
    annotated = render_annotated_image(
        pil_image=pil_image,
        placard_predictions=placard_preds_filtered,
        empty_space_predictions=empty_preds_filtered,
        per_region=results["per_region"],
        min_confidence=0.0,
        grid=results.get("grid"),
        sign_quad=results.get("sign_quad"),
    )

    # -------------------------------------------------------------------------
    # Results display
    # -------------------------------------------------------------------------

    st.divider()
    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.subheader("Annotated Image")
        st.caption(
            "**Green** = existing placards   "
            "**Orange** = empty regions   "
            "**Cyan** = proposed new placements"
        )
        st.image(annotated, use_container_width=True)

    with col_stats:
        st.subheader("Results")
        n_regions = len([r for r in results["per_region"] if r.get("count", 0) > 0])
        st.metric(
            "Estimated New Placard Slots",
            results["total_fit"],
            delta=f"across {n_regions} region(s)" if n_regions else None,
            delta_color="off",
            help="Total number of new placard positions that fit across all detected empty regions.",
        )
        st.metric("Placard Width Used (px)", results["placard_w"])
        st.metric("Placard Height Used (px)", results["placard_h"])

    # -------------------------------------------------------------------------
    # Debug panel
    # -------------------------------------------------------------------------

    st.divider()
    st.subheader("Debug Panel")

    debug_cols = st.columns(4)
    debug_cols[0].metric("Placards Detected", len(placard_preds_filtered))
    debug_cols[1].metric("Empty Regions Detected", len(empty_preds_filtered))
    debug_cols[2].info(
        "Perspective mode" if results["used_perspective_mode"]
        else ("Polygon mode" if results["used_polygon_mode"] else "Bbox fallback")
    )
    debug_cols[3].metric("Regions with fits", sum(1 for r in results["per_region"] if r["count"] > 0))

    if results["per_region"]:
        st.write("**Count per empty region:**")
        region_data = [
            {
                "Region": f"Region {r['region_index'] + 1}",
                "Placards fit": r["count"],
                "Mode": (
                    "perspective" if r.get("used_perspective")
                    else ("polygon" if r.get("used_polygon") else "bbox")
                ),
            }
            for r in results["per_region"]
        ]
        st.table(region_data)

    # -------------------------------------------------------------------------
    # Raw JSON
    # -------------------------------------------------------------------------

    st.divider()
    with st.expander("Raw JSON — Placard Model Response"):
        st.json(placard_resp)

    with st.expander("Raw JSON — Empty-Space Model Response"):
        st.json(empty_resp)

elif uploaded_file is None and "last_placard_preds" not in st.session_state:
    st.info("Upload an image to get started.")
