"""
app.py

Streamlit GUI for the Highway Logo Sign Placard Capacity Estimator.

Workflow:
  1. Upload one or more images of highway logo signs
  2. Configure Roboflow model credentials in the sidebar (or enable Mock Mode)
  3. Click "Run Inference" — calls both models for each image and stores predictions
  4. Adjust sliders to live-update placements without re-running inference
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
from geometry import detect_sign_quad_from_detections
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
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_settings(values: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=2)


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
    "Upload one or more sign images, configure Roboflow models, and estimate how many new "
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
# Image uploader — accepts multiple files
# ---------------------------------------------------------------------------

st.divider()
uploaded_files = st.file_uploader(
    "Upload sign image(s)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    help="Drag and drop or click to browse. You can select multiple images at once.",
)

run_button = st.button(
    "Run Inference",
    type="primary",
    disabled=not uploaded_files,
)

# ---------------------------------------------------------------------------
# Inference — runs only when the button is clicked.
# Stores results per image in session_state["results_list"].
# ---------------------------------------------------------------------------

if run_button and uploaded_files:
    if not mock_mode:
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

    results_list = []
    progress = st.progress(0, text="Running inference…")

    for idx, uf in enumerate(uploaded_files):
        progress.progress(
            (idx) / len(uploaded_files),
            text=f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})…",
        )

        pil_image = Image.open(uf).convert("RGB")
        img_w, img_h = pil_image.size

        if mock_mode:
            placard_resp = MOCK_PLACARD_RESPONSE
            empty_resp = MOCK_EMPTY_SPACE_RESPONSE
        else:
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
                st.error(f"{uf.name}: Placard model failed — {exc}")
                continue

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
                st.error(f"{uf.name}: Empty-space model failed — {exc}")
                continue

        conf = min_confidence
        placard_preds = [p for p in placard_resp.get("predictions", []) if p.get("confidence", 0) >= conf]
        empty_preds = [p for p in empty_resp.get("predictions", []) if p.get("confidence", 0) >= conf]

        buf = BytesIO()
        pil_image.save(buf, format="PNG")

        results_list.append({
            "filename": uf.name,
            "img_bytes": buf.getvalue(),
            "img_w": img_w,
            "img_h": img_h,
            "placard_preds": placard_preds,
            "empty_preds": empty_preds,
            "placard_resp": placard_resp,
            "empty_resp": empty_resp,
            "mock_mode": mock_mode,
        })

    progress.progress(1.0, text="Done!")
    st.session_state["results_list"] = results_list


# ---------------------------------------------------------------------------
# Results — re-rendered on every rerun (slider changes update live).
# ---------------------------------------------------------------------------

if "results_list" in st.session_state and st.session_state["results_list"]:
    for item in st.session_state["results_list"]:
        pil_image = Image.open(BytesIO(item["img_bytes"]))
        img_w = item["img_w"]
        img_h = item["img_h"]
        placard_preds_filtered = item["placard_preds"]
        empty_preds_filtered = item["empty_preds"]
        placard_resp = item["placard_resp"]
        empty_resp = item["empty_resp"]

        st.divider()
        st.subheader(item["filename"])

        if item.get("mock_mode"):
            st.info("Mock Mode is ON — using sample predictions (no API call made).")

        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) if perspective_mode else None

        results = estimate_total_capacity(
            empty_space_predictions=empty_preds_filtered,
            placard_predictions=placard_preds_filtered,
            img_h=img_h,
            img_w=img_w,
            default_placard_w=int(default_placard_w),
            default_placard_h=int(default_placard_h),
            margin=int(outer_margin),
            spacing=int(spacing),
            min_confidence=0.0,
            estimate_size_from_detections=estimate_from_detections,
            placard_scale=placard_scale / 100.0,
            empty_space_scale=empty_space_scale / 100.0,
            perspective_mode=perspective_mode,
            image_bgr=image_bgr,
            grid_mode=grid_mode,
        )

        annotated = render_annotated_image(
            pil_image=pil_image,
            placard_predictions=placard_preds_filtered,
            empty_space_predictions=empty_preds_filtered,
            per_region=results["per_region"],
            min_confidence=0.0,
            grid=results.get("grid"),
            sign_quad=results.get("sign_quad"),
        )

        col_img, col_stats = st.columns([3, 1])

        with col_img:
            st.caption(
                "**Green** = existing placards   "
                "**Orange** = empty regions   "
                "**Cyan** = proposed new placements"
            )
            st.image(annotated, use_container_width=True)

        with col_stats:
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

        # --- Sign quad debug (no API calls — uses cached detections) ---
        with st.expander("🔍 Sign Outline Preview"):
            st.caption(
                "Shows the sign boundary box computed from the detected placard and empty-region "
                "positions. No extra API call is made — this reuses the predictions already fetched above."
            )
            _bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            _quad = detect_sign_quad_from_detections(_bgr, placard_preds_filtered, empty_preds_filtered)
            if _quad:
                _preview = draw_sign_quad(_bgr.copy(), _quad)
                _preview = cv2.cvtColor(_preview, cv2.COLOR_BGR2RGB)
                st.image(_preview, use_container_width=True)
                st.success(
                    "Sign boundary corners: "
                    + ", ".join(f"({p['x']:.0f}, {p['y']:.0f})" for p in _quad)
                )
            else:
                st.image(np.array(pil_image), use_container_width=True)
                st.warning("Could not compute sign boundary — no detections available.")

        # --- Debug metrics ---
        with st.expander("📊 Debug Details"):
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

        # --- Raw JSON ---
        with st.expander("Raw JSON — Placard Model Response"):
            st.json(placard_resp)
        with st.expander("Raw JSON — Empty-Space Model Response"):
            st.json(empty_resp)

elif not uploaded_files and "results_list" not in st.session_state:
    st.info("Upload one or more images to get started.")
