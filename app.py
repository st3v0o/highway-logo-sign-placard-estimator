"""
app.py

Streamlit GUI for the Highway Logo Sign Placard Capacity Estimator.

Workflow:
  1. Upload an image of a highway logo sign
  2. Configure Roboflow model credentials in the sidebar (or enable Mock Mode)
  3. Click "Run Inference"
  4. View annotated image, fit count, and debug info
"""

import json
import os

import streamlit as st
from PIL import Image

from roboflow_client import (
    call_placard_model,
    call_empty_space_model,
    MOCK_PLACARD_RESPONSE,
    MOCK_EMPTY_SPACE_RESPONSE,
)
from fitting import estimate_total_capacity
from visualize import render_annotated_image


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
}


def load_settings() -> dict:
    """Load saved settings from disk, falling back to defaults for any missing keys."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            # Merge with defaults so new keys added in future versions are included
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_settings(values: dict) -> None:
    """Write current settings to disk."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=2)


# Load saved settings once per session (not on every rerun)
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
        })
        st.success("Settings saved!")


# ---------------------------------------------------------------------------
# Image uploader
# ---------------------------------------------------------------------------

st.divider()
uploaded_file = st.file_uploader(
    "Upload a sign image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag and drop or click to browse.",
)

run_button = st.button("Run Inference", type="primary", disabled=uploaded_file is None)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

if run_button and uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = pil_image.size

    with st.spinner("Running inference..."):
        if mock_mode:
            placard_resp = MOCK_PLACARD_RESPONSE
            empty_resp = MOCK_EMPTY_SPACE_RESPONSE
            st.info("Mock Mode is ON — using sample predictions (no API call made).")
        else:
            # Validate credentials before calling
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

    placard_preds = placard_resp.get("predictions", [])
    empty_preds = empty_resp.get("predictions", [])

    # Filter by confidence
    placard_preds_filtered = [p for p in placard_preds if p.get("confidence", 0) >= min_confidence]
    empty_preds_filtered = [p for p in empty_preds if p.get("confidence", 0) >= min_confidence]

    # Run fitting
    results = estimate_total_capacity(
        empty_space_predictions=empty_preds_filtered,
        placard_predictions=placard_preds_filtered,
        img_h=img_h,
        img_w=img_w,
        default_placard_w=int(default_placard_w),
        default_placard_h=int(default_placard_h),
        margin=int(outer_margin),
        spacing=int(spacing),
        min_confidence=min_confidence,
        estimate_size_from_detections=estimate_from_detections,
    )

    # Render annotated image
    annotated = render_annotated_image(
        pil_image=pil_image,
        placard_predictions=placard_preds_filtered,
        empty_space_predictions=empty_preds_filtered,
        per_region=results["per_region"],
        min_confidence=0.0,  # already pre-filtered
    )

    # ---------------------------------------------------------------------------
    # Results display
    # ---------------------------------------------------------------------------

    st.divider()
    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.subheader("Annotated Image")
        st.caption(
            "**Red** = existing placards   "
            "**Yellow/Green** = empty regions (bbox/polygon)   "
            "**Cyan** = proposed new placements"
        )
        st.image(annotated, use_container_width=True)

    with col_stats:
        st.subheader("Results")
        st.metric("Estimated Available Positions", results["total_fit"])
        st.metric("Placard Width Used (px)", results["placard_w"])
        st.metric("Placard Height Used (px)", results["placard_h"])

    # ---------------------------------------------------------------------------
    # Debug panel
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("Debug Panel")

    debug_cols = st.columns(4)
    debug_cols[0].metric("Placards Detected", len(placard_preds_filtered))
    debug_cols[1].metric("Empty Regions Detected", len(empty_preds_filtered))
    debug_cols[2].info(
        "Polygon mode" if results["used_polygon_mode"] else "Bbox fallback"
    )
    debug_cols[3].metric("Regions with fits", sum(1 for r in results["per_region"] if r["count"] > 0))

    if results["per_region"]:
        st.write("**Count per empty region:**")
        region_data = [
            {
                "Region": f"Region {r['region_index'] + 1}",
                "Placards fit": r["count"],
                "Mode": "polygon" if r.get("used_polygon") else "bbox",
            }
            for r in results["per_region"]
        ]
        st.table(region_data)

    # ---------------------------------------------------------------------------
    # Raw JSON
    # ---------------------------------------------------------------------------

    st.divider()
    with st.expander("Raw JSON — Placard Model Response"):
        st.json(placard_resp)

    with st.expander("Raw JSON — Empty-Space Model Response"):
        st.json(empty_resp)

elif uploaded_file is None:
    st.info("Upload an image to get started.")
