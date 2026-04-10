"""
app.py

Streamlit GUI for the Highway Logo Sign Placard Capacity Estimator.

Workflow:
  1. Upload one or more images directly (or supply an Excel/CSV with sign IDs + image URLs)
  2. Configure Roboflow model credentials in the sidebar (or enable Mock Mode)
  3. Click "Run Inference" or "Generate Grid" — results are stored and re-rendered live
  4. Adjust sliders to live-update placements without re-running inference
"""

import json
import os
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from roboflow_client import (
    call_placard_model,
    call_empty_space_model,
    MOCK_PLACARD_RESPONSE,
    MOCK_EMPTY_SPACE_RESPONSE,
)
from fitting import estimate_total_capacity
from geometry import detect_sign_quad
from visualize import render_annotated_image, draw_sign_quad, draw_sign_corner_grid


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
    "Upload sign images directly or supply an Excel/CSV with sign IDs and image URLs. "
    "The app estimates how many new business placards can fit in the open blue space."
)


# ---------------------------------------------------------------------------
# Sidebar — settings
# (All variables set here are module-level, accessible by the helpers below)
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
    placard_project   = st.text_input("Project ID", key="placard_project", disabled=mock_mode)
    placard_version   = st.number_input("Version", min_value=1, step=1, key="placard_version", disabled=mock_mode)

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
        "Placard scale (%)", min_value=30, max_value=100, step=5, key="placard_scale",
        help="Scales the fitting rectangle. Lower values fit placards into tighter spaces.",
    )
    empty_space_scale = st.slider(
        "Empty space scale (%)", min_value=30, max_value=100, step=5, key="empty_space_scale",
        help="Shrinks each detected empty region from its center.",
    )

    st.divider()
    st.subheader("Layout Parameters")
    outer_margin = st.number_input(
        "Outer margin (px)", min_value=0, step=2, key="outer_margin",
        help="Minimum gap between a placard and the edge of the empty region.",
    )
    spacing = st.number_input(
        "Spacing between placards (px)", min_value=0, step=2, key="spacing",
        help="Gap between adjacent proposed placards.",
    )
    min_confidence = st.slider(
        "Minimum confidence", min_value=0.0, max_value=1.0, step=0.05, key="min_confidence",
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
        })
        st.success("Settings saved!")


# ---------------------------------------------------------------------------
# Shared helper functions
# (Defined after the sidebar so sidebar variables are in module scope)
# ---------------------------------------------------------------------------

_MAX_DISPLAY_PX = 1200   # cap for any image stored in session state


def _resize_for_display(pil_image: Image.Image) -> Image.Image:
    """Resize to at most _MAX_DISPLAY_PX on the long edge, preserving aspect ratio."""
    w, h = pil_image.size
    if max(w, h) > _MAX_DISPLAY_PX:
        ratio = _MAX_DISPLAY_PX / max(w, h)
        pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return pil_image


def _to_jpeg_bytes(img, quality: int = 82) -> bytes:
    """Convert a PIL Image or numpy RGB array to JPEG bytes."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _current_fitting_params() -> dict:
    """Snapshot of all sidebar fitting parameters that affect placement count."""
    return {
        "placard_scale":            int(st.session_state.placard_scale),
        "empty_space_scale":        int(st.session_state.empty_space_scale),
        "default_placard_w":        int(st.session_state.default_placard_w),
        "default_placard_h":        int(st.session_state.default_placard_h),
        "outer_margin":             int(st.session_state.outer_margin),
        "spacing":                  int(st.session_state.spacing),
        "min_confidence":           float(st.session_state.min_confidence),
        "estimate_from_detections": bool(st.session_state.estimate_from_detections),
    }


def _refit_result_item(item: dict, params: dict) -> dict:
    """
    Re-run fitting + rendering using cached predictions and image.
    No API call is made.  Returns an updated copy of the item.
    """
    conf = params["min_confidence"]
    placard_preds = [p for p in item["placard_resp"].get("predictions", [])
                     if p.get("confidence", 0) >= conf]
    empty_preds   = [p for p in item["empty_resp"].get("predictions", [])
                     if p.get("confidence", 0) >= conf]

    pil_image  = Image.open(BytesIO(item["_img_bytes"])).convert("RGB")
    img_w, img_h = pil_image.size
    image_bgr  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    results = estimate_total_capacity(
        empty_space_predictions=empty_preds,
        placard_predictions=placard_preds,
        img_h=img_h,
        img_w=img_w,
        default_placard_w=params["default_placard_w"],
        default_placard_h=params["default_placard_h"],
        margin=params["outer_margin"],
        spacing=params["spacing"],
        min_confidence=0.0,
        estimate_size_from_detections=params["estimate_from_detections"],
        placard_scale=params["placard_scale"] / 100.0,
        empty_space_scale=params["empty_space_scale"] / 100.0,
        perspective_mode=True,
        image_bgr=image_bgr,
        grid_mode=True,
    )

    annotated_rgb = render_annotated_image(
        pil_image=pil_image,
        placard_predictions=placard_preds,
        empty_space_predictions=empty_preds,
        per_region=results["per_region"],
        min_confidence=0.0,
        grid=results.get("grid"),
        sign_quad=results.get("sign_quad"),
        draw_sign_grid=True,
    )
    annotated_bytes = _to_jpeg_bytes(annotated_rgb)

    sign_quad = results.get("sign_quad")
    if sign_quad:
        quad_bgr     = draw_sign_quad(image_bgr.copy(), sign_quad)
        quad_bytes   = _to_jpeg_bytes(cv2.cvtColor(quad_bgr, cv2.COLOR_BGR2RGB))
        quad_corners = ", ".join(f"({p['x']:.0f}, {p['y']:.0f})" for p in sign_quad)
    else:
        quad_bytes   = None
        quad_corners = None

    region_rows = [
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

    return {
        **item,
        "annotated_bytes": annotated_bytes,
        "quad_bytes":      quad_bytes,
        "quad_corners":    quad_corners,
        "total_fit":       results["total_fit"],
        "placard_w":       results["placard_w"],
        "placard_h":       results["placard_h"],
        "n_placards_det":  len(placard_preds),
        "n_empty_det":     len(empty_preds),
        "n_regions_fit":   sum(1 for r in results["per_region"] if r["count"] > 0),
        "mode_label": (
            "Perspective" if results["used_perspective_mode"]
            else ("Polygon" if results["used_polygon_mode"] else "Bbox fallback")
        ),
        "region_rows":     region_rows,
        "_fitting_params": params,
    }


def _run_inference_on_image(pil_image: Image.Image, sign_id: str) -> dict:
    """
    Run the full inference + rendering pipeline on a single PIL image.
    All expensive CV work happens here once; the returned dict contains only
    pre-rendered JPEG bytes and scalar metrics so _render_results() is
    just a display call with no recomputation.
    """
    # Resize before heavy processing to keep memory bounded
    pil_image = _resize_for_display(pil_image)
    img_w, img_h = pil_image.size
    # Cache the resized image bytes for live re-fitting without a new API call
    img_bytes = _to_jpeg_bytes(pil_image)

    if mock_mode:
        placard_resp = MOCK_PLACARD_RESPONSE
        empty_resp   = MOCK_EMPTY_SPACE_RESPONSE
        used_mock    = True
    else:
        placard_resp = call_placard_model(
            pil_image=pil_image,
            api_key=api_key,
            workspace_id=placard_workspace,
            project_id=placard_project,
            version=int(placard_version),
            confidence=min_confidence,
        )
        empty_resp = call_empty_space_model(
            pil_image=pil_image,
            api_key=api_key,
            workspace_id=empty_workspace,
            project_id=empty_project,
            version=int(empty_version),
            confidence=min_confidence,
        )
        used_mock = False

    conf = min_confidence
    placard_preds = [p for p in placard_resp.get("predictions", []) if p.get("confidence", 0) >= conf]
    empty_preds   = [p for p in empty_resp.get("predictions",   []) if p.get("confidence", 0) >= conf]

    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run fitting once
    results = estimate_total_capacity(
        empty_space_predictions=empty_preds,
        placard_predictions=placard_preds,
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
        perspective_mode=True,
        image_bgr=image_bgr,
        grid_mode=True,
    )

    # Render annotated image once → store as JPEG
    annotated_rgb = render_annotated_image(
        pil_image=pil_image,
        placard_predictions=placard_preds,
        empty_space_predictions=empty_preds,
        per_region=results["per_region"],
        min_confidence=0.0,
        grid=results.get("grid"),
        sign_quad=results.get("sign_quad"),
        draw_sign_grid=True,
    )
    annotated_bytes = _to_jpeg_bytes(annotated_rgb)

    # Render sign-outline preview once → store as JPEG
    sign_quad = results.get("sign_quad")
    if sign_quad:
        quad_bgr     = draw_sign_quad(image_bgr.copy(), sign_quad)
        quad_bytes   = _to_jpeg_bytes(cv2.cvtColor(quad_bgr, cv2.COLOR_BGR2RGB))
        quad_corners = ", ".join(f"({p['x']:.0f}, {p['y']:.0f})" for p in sign_quad)
    else:
        quad_bytes   = None
        quad_corners = None

    # Build per-region table rows (scalars only — no arrays)
    region_rows = [
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

    return {
        "sign_id":         sign_id,
        # Pre-rendered images (JPEG bytes — cheap to store and display)
        "annotated_bytes": annotated_bytes,
        "quad_bytes":      quad_bytes,
        "quad_corners":    quad_corners,
        # Scalar metrics
        "total_fit":       results["total_fit"],
        "placard_w":       results["placard_w"],
        "placard_h":       results["placard_h"],
        "n_placards_det":  len(placard_preds),
        "n_empty_det":     len(empty_preds),
        "n_regions_fit":   sum(1 for r in results["per_region"] if r["count"] > 0),
        "mode_label": (
            "Perspective" if results["used_perspective_mode"]
            else ("Polygon" if results["used_polygon_mode"] else "Bbox fallback")
        ),
        "region_rows":     region_rows,
        # Raw JSON (kept for inspection, but lightweight)
        "placard_resp":    placard_resp,
        "empty_resp":      empty_resp,
        "mock_mode":       used_mock,
        # Cached inputs for live re-fitting (no API call needed on slider change)
        "_img_bytes":      img_bytes,
        "_fitting_params": _current_fitting_params(),
    }


def _run_grid_on_image(pil_image: Image.Image, sign_id: str) -> dict:
    """
    Run grid-only (no API) sign boundary detection on a single PIL image.
    The visualisation is pre-rendered to JPEG bytes so reruns are cheap.
    """
    pil_image = _resize_for_display(pil_image)
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    quad = detect_sign_quad(bgr)
    vis_bgr = bgr.copy()
    if quad:
        vis_bgr = draw_sign_corner_grid(vis_bgr, quad)
        status = "Corners: " + ", ".join(f"({p['x']:.0f},{p['y']:.0f})" for p in quad)
    else:
        status = None
    vis_rgb   = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis_bytes = _to_jpeg_bytes(vis_rgb)
    return {
        "sign_id":   sign_id,
        "vis_bytes": vis_bytes,   # JPEG bytes — no large arrays in session state
        "has_quad":  quad is not None,
        "status":    status,
    }


def _render_results(results_list: list) -> None:
    """
    Display pre-rendered inference results.
    All images are stored as JPEG bytes — no CV recomputation on each Streamlit rerun.
    """
    for item in results_list:
        st.divider()
        st.subheader(item["sign_id"])

        if item.get("mock_mode"):
            st.info("Mock Mode is ON — using sample predictions (no API call made).")

        col_img, col_stats = st.columns([3, 1])
        with col_img:
            st.caption(
                "**Green** = existing placards   "
                "**Orange** = empty regions   "
                "**Cyan** = proposed new placements"
            )
            st.image(item["annotated_bytes"], use_container_width=True)

        with col_stats:
            n = item["n_regions_fit"]
            st.metric(
                "Estimated New Placard Slots",
                item["total_fit"],
                delta=f"across {n} region(s)" if n else None,
                delta_color="off",
            )
            st.metric("Placard Width Used (px)",  item["placard_w"])
            st.metric("Placard Height Used (px)", item["placard_h"])

        with st.expander("🔍 Sign Outline Preview"):
            if item["quad_bytes"]:
                st.image(item["quad_bytes"], use_container_width=True)
                st.success(f"Sign boundary corners: {item['quad_corners']}")
            else:
                st.warning("Could not detect a blue sign boundary in this image.")

        with st.expander("📊 Debug Details"):
            debug_cols = st.columns(4)
            debug_cols[0].metric("Placards Detected",    item["n_placards_det"])
            debug_cols[1].metric("Empty Regions Detected", item["n_empty_det"])
            debug_cols[2].info(item["mode_label"])
            debug_cols[3].metric("Regions with fits",    item["n_regions_fit"])
            if item["region_rows"]:
                st.write("**Count per empty region:**")
                st.table(item["region_rows"])

        with st.expander("Raw JSON — Placard Model Response"):
            st.json(item["placard_resp"])
        with st.expander("Raw JSON — Empty-Space Model Response"):
            st.json(item["empty_resp"])


def _render_grid_items(grid_items: list) -> None:
    """Display pre-rendered grid-only results (JPEG bytes — no recomputation)."""
    for item in grid_items:
        st.markdown(f"**{item['sign_id']}**")
        st.image(item["vis_bytes"], use_container_width=True)
        if item["has_quad"]:
            st.success(item["status"])
        else:
            st.warning(
                "Could not detect a blue sign boundary. "
                "Try a clearer image or one with more visible blue sign area."
            )
        st.divider()


# ---------------------------------------------------------------------------
# Tabs — Direct Upload | Excel / CSV Batch
# ---------------------------------------------------------------------------

st.divider()
tab_upload, tab_excel = st.tabs(["📂 Direct Image Upload", "📊 Excel / CSV Batch"])


# ===========================================================================
# TAB 1 — Direct Image Upload
# ===========================================================================

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload sign image(s)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Drag and drop or click to browse. You can select multiple images at once.",
    )

    _btn_col1, _btn_col2 = st.columns([2, 1])
    with _btn_col1:
        run_button = st.button(
            "Run Inference",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )
    with _btn_col2:
        grid_button = st.button(
            "Generate Grid",
            disabled=not uploaded_files,
            use_container_width=True,
            help="Detects the sign boundary from blue-pixel analysis — no API call needed.",
        )

    # Grid preview
    if grid_button and uploaded_files:
        grid_items = []
        for uf in uploaded_files:
            pil_img = Image.open(uf).convert("RGB")
            grid_items.append(_run_grid_on_image(pil_img, uf.name))
        st.session_state["grid_preview_list"] = grid_items

    if st.session_state.get("grid_preview_list"):
        st.subheader("Grid Preview (colour-based, no API)")
        _render_grid_items(st.session_state["grid_preview_list"])

    # Inference
    if run_button and uploaded_files:
        if not mock_mode:
            missing = []
            if not api_key:        missing.append("API Key")
            if not placard_project: missing.append("Placard Project ID")
            if not empty_project:   missing.append("Empty-Space Project ID")
            if missing:
                st.error(f"Please fill in: {', '.join(missing)}")
                st.stop()

        results_list = []
        progress = st.progress(0, text="Running inference…")
        for idx, uf in enumerate(uploaded_files):
            progress.progress(idx / len(uploaded_files),
                              text=f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})…")
            try:
                pil_image = Image.open(uf).convert("RGB")
                results_list.append(_run_inference_on_image(pil_image, uf.name))
            except Exception as exc:
                st.error(f"{uf.name}: failed — {exc}")
        progress.progress(1.0, text="Done!")
        st.session_state["results_list"] = results_list

    if st.session_state.get("results_list"):
        # Live re-fit: if any sidebar fitting param changed since the last run,
        # re-run estimate_total_capacity + render for each cached item without
        # calling the Roboflow API again.
        current_params = _current_fitting_params()
        results_list_cached = st.session_state["results_list"]
        needs_save = False
        for i, item in enumerate(results_list_cached):
            if item.get("_fitting_params") != current_params:
                if "_img_bytes" not in item:
                    # Legacy cached result from a pre-update run — cannot
                    # re-fit without the stored image bytes.  Skip silently;
                    # the user just needs to re-run inference once.
                    continue
                with st.spinner(f"Updating {item['sign_id']}…"):
                    results_list_cached[i] = _refit_result_item(item, current_params)
                needs_save = True
        if needs_save:
            st.session_state["results_list"] = results_list_cached
        _render_results(st.session_state["results_list"])
    elif not uploaded_files and not st.session_state.get("results_list"):
        st.info("Upload one or more images to get started.")


# ===========================================================================
# TAB 2 — Excel / CSV Batch
# ===========================================================================

with tab_excel:
    st.subheader("Batch Processing from Excel or CSV")
    st.caption(
        "Upload a spreadsheet with a sign identifier column and an image URL column. "
        "The app processes one image per step so it never times out on large batches."
    )

    excel_file = st.file_uploader(
        "Upload Excel (.xlsx) or CSV (.csv)",
        type=["xlsx", "xls", "csv"],
        key="excel_uploader",
    )

    # -----------------------------------------------------------------------
    # File parsing + column mapping (only shown when NOT actively processing)
    # -----------------------------------------------------------------------

    batch_active = st.session_state.get("batch_active", False)

    if excel_file is not None and not batch_active:
        try:
            if excel_file.name.lower().endswith(".csv"):
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file, engine="openpyxl")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            df = None

        if df is not None and not df.empty:
            st.write(f"**Preview** — {len(df)} row(s), {len(df.columns)} column(s)")
            st.dataframe(df.head(10), use_container_width=True)

            cols = list(df.columns)

            st.subheader("Map Columns")
            _mc1, _mc2 = st.columns(2)
            with _mc1:
                sign_id_col = st.selectbox(
                    "Sign identifier column",
                    options=cols,
                    index=0,
                    help="Unique sign number or name shown in the results.",
                    key="sign_id_col",
                )
            with _mc2:
                url_col = st.selectbox(
                    "Image URL column",
                    options=cols,
                    index=min(1, len(cols) - 1),
                    help="Full URL to download the sign image.",
                    key="url_col",
                )

            st.subheader("Processing Mode")
            batch_mode = st.radio(
                "Choose what to run for each image:",
                options=["Generate Grid only (no API)", "Run Inference + Grid (calls AI models)"],
                key="batch_mode",
                horizontal=True,
            )

            process_batch = st.button("Process Batch", type="primary", use_container_width=True)

            if process_batch:
                do_inference = st.session_state["batch_mode"].startswith("Run Inference")

                if do_inference and not mock_mode:
                    missing = []
                    if not api_key:         missing.append("API Key")
                    if not placard_project: missing.append("Placard Project ID")
                    if not empty_project:   missing.append("Empty-Space Project ID")
                    if missing:
                        st.error(f"Please fill in: {', '.join(missing)}")
                        st.stop()

                rows = df[[sign_id_col, url_col]].dropna().values.tolist()
                rows = [(str(r[0]).strip(), str(r[1]).strip()) for r in rows]

                # Initialise state for one-row-per-rerun processing
                st.session_state["batch_rows"]         = rows
                st.session_state["batch_idx"]          = 0
                st.session_state["batch_do_inference"] = do_inference
                st.session_state["batch_results_list"] = []
                st.session_state["batch_grid_items"]   = []
                st.session_state["batch_errors"]       = []
                st.session_state["batch_active"]       = True
                st.rerun()

    # -----------------------------------------------------------------------
    # One-row-per-rerun processing loop
    # -----------------------------------------------------------------------

    if st.session_state.get("batch_active"):
        rows         = st.session_state["batch_rows"]
        idx          = st.session_state["batch_idx"]
        total        = len(rows)
        do_inference = st.session_state["batch_do_inference"]

        # Progress UI
        frac = idx / total if total else 1.0
        st.progress(frac, text=f"Processing sign {idx + 1} of {total}…")
        st.caption(f"**{rows[idx][0]}** — {rows[idx][1]}" if idx < total else "")

        # Stop button
        if st.button("Stop Batch", key="stop_batch"):
            st.session_state["batch_active"] = False
            st.rerun()

        # Process the current row
        if idx < total:
            sign_id, url = rows[idx]
            try:
                resp = requests.get(url, timeout=25)
                resp.raise_for_status()
                pil_image = Image.open(BytesIO(resp.content)).convert("RGB")

                if do_inference:
                    item = _run_inference_on_image(pil_image, sign_id)
                    st.session_state["batch_results_list"].append(item)
                else:
                    item = _run_grid_on_image(pil_image, sign_id)
                    st.session_state["batch_grid_items"].append(item)

            except Exception as exc:
                st.session_state["batch_errors"].append(f"**{sign_id}**: {exc}")

            # Advance to next row and immediately rerun
            st.session_state["batch_idx"] = idx + 1

            if st.session_state["batch_idx"] >= total:
                st.session_state["batch_active"] = False

            st.rerun()

    # -----------------------------------------------------------------------
    # Completion banner + any per-row errors
    # -----------------------------------------------------------------------

    if (not st.session_state.get("batch_active")
            and (st.session_state.get("batch_results_list")
                 or st.session_state.get("batch_grid_items"))):
        done_count = (len(st.session_state.get("batch_results_list", [])) +
                      len(st.session_state.get("batch_grid_items", [])))
        st.success(f"Batch complete — {done_count} sign(s) processed successfully.")

    for err in st.session_state.get("batch_errors", []):
        st.warning(err)

    # -----------------------------------------------------------------------
    # Display accumulated results
    # -----------------------------------------------------------------------

    if st.session_state.get("batch_grid_items"):
        st.divider()
        st.subheader("Batch Grid Results")
        _render_grid_items(st.session_state["batch_grid_items"])

    if st.session_state.get("batch_results_list"):
        st.divider()
        st.subheader("Batch Inference Results")
        _render_results(st.session_state["batch_results_list"])
