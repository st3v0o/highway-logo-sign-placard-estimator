"""
app.py

Streamlit GUI for the Highway Logo Sign Placard Capacity Estimator.

Workflow:
  1. Upload one or more sign images (or supply an Excel/CSV with image URLs).
  2. Configure the two Roboflow model credentials in the sidebar.
  3. Click "Run Inference" — the sign-boundary model finds the sign polygon
     for perspective correction; the empty-space model finds open slots.
  4. Adjust sliders to live-update placard placements without re-running inference.
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
    call_sign_model,
    call_empty_space_model,
    MOCK_SIGN_RESPONSE,
    MOCK_EMPTY_SPACE_RESPONSE,
    SIGN_CLASS,
    EMPTY_CLASS,
)
from fitting import estimate_total_capacity
from visualize import render_annotated_image


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

SETTINGS_FILE = "settings.json"

DEFAULTS = {
    "api_key":          "",
    "sign_project":     "blue-logo-sign",
    "sign_version":     1,
    "empty_project":    "empty-blue-space",
    "empty_version":    2,
    "mock_mode":        True,
    "default_placard_w": 90,
    "default_placard_h": 70,
    "placard_scale":    100,
    "empty_space_scale": 100,
    "outer_margin":     8,
    "spacing":          4,
    "min_confidence":   0.4,
}


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return {**DEFAULTS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULTS)


def save_settings(values: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(values, f, indent=2)


if "settings_loaded" not in st.session_state:
    for k, v in load_settings().items():
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
    "Upload sign images or supply an Excel/CSV with image URLs. "
    "Estimates how many new business placards can fit in the open sign space."
)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    mock_mode = st.checkbox(
        "Mock Mode (no API calls)",
        key="mock_mode",
        help="Uses built-in sample predictions so you can test without a Roboflow account.",
    )

    st.divider()
    st.subheader("Roboflow API Key")
    api_key = st.text_input("API Key", key="api_key", type="password", disabled=mock_mode)

    st.divider()
    st.subheader("Sign Boundary Model")
    st.caption("Detects the Blue-Logo-Sign polygon for perspective correction.")
    sign_project = st.text_input("Project ID", key="sign_project", disabled=mock_mode)
    sign_version = st.number_input("Version", min_value=1, step=1,
                                   key="sign_version", disabled=mock_mode)

    st.divider()
    st.subheader("Empty-Space Model")
    st.caption("Detects open slots where new placards can be placed.")
    empty_project = st.text_input("Project ID", key="empty_project", disabled=mock_mode)
    empty_version = st.number_input("Version", min_value=1, step=1,
                                    key="empty_version", disabled=mock_mode)

    st.divider()
    st.subheader("Placard Size")
    default_placard_w = st.number_input("Default width (px)",  min_value=10, step=5,
                                        key="default_placard_w")
    default_placard_h = st.number_input("Default height (px)", min_value=10, step=5,
                                        key="default_placard_h")
    placard_scale = st.slider(
        "Placard scale (%)", min_value=30, max_value=100, step=5, key="placard_scale",
        help="Scales the fitting rectangle relative to the default size.",
    )
    empty_space_scale = st.slider(
        "Empty-space scale (%)", min_value=30, max_value=100, step=5, key="empty_space_scale",
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
    )
    min_confidence = st.slider(
        "Minimum confidence", min_value=0.0, max_value=1.0, step=0.05,
        key="min_confidence",
    )

    st.divider()
    if st.button("Save Settings", use_container_width=True):
        save_settings({
            "api_key":           st.session_state.api_key,
            "sign_project":      st.session_state.sign_project,
            "sign_version":      int(st.session_state.sign_version),
            "empty_project":     st.session_state.empty_project,
            "empty_version":     int(st.session_state.empty_version),
            "mock_mode":         st.session_state.mock_mode,
            "default_placard_w": int(st.session_state.default_placard_w),
            "default_placard_h": int(st.session_state.default_placard_h),
            "placard_scale":     int(st.session_state.placard_scale),
            "empty_space_scale": int(st.session_state.empty_space_scale),
            "outer_margin":      int(st.session_state.outer_margin),
            "spacing":           int(st.session_state.spacing),
            "min_confidence":    float(st.session_state.min_confidence),
        })
        st.success("Settings saved!")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_DISPLAY_PX = 1200


def _resize_for_display(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if max(w, h) > _MAX_DISPLAY_PX:
        ratio = _MAX_DISPLAY_PX / max(w, h)
        pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return pil_image


def _to_jpeg_bytes(img, quality: int = 82) -> bytes:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _current_fitting_params() -> dict:
    return {
        "default_placard_w":  int(st.session_state.default_placard_w),
        "default_placard_h":  int(st.session_state.default_placard_h),
        "placard_scale":      int(st.session_state.placard_scale),
        "empty_space_scale":  int(st.session_state.empty_space_scale),
        "outer_margin":       int(st.session_state.outer_margin),
        "spacing":            int(st.session_state.spacing),
        "min_confidence":     float(st.session_state.min_confidence),
    }


def _best_sign_pred(sign_resp: dict, conf: float) -> dict | None:
    """Return the highest-confidence Blue-Logo-Sign detection above threshold."""
    preds = [
        p for p in sign_resp.get("predictions", [])
        if p.get("class") == SIGN_CLASS and p.get("confidence", 0) >= conf
    ]
    return max(preds, key=lambda p: p["confidence"]) if preds else None


def _empty_preds(empty_resp: dict, conf: float) -> list[dict]:
    return [
        p for p in empty_resp.get("predictions", [])
        if p.get("confidence", 0) >= conf
    ]


def _run_fitting(
    pil_image: Image.Image,
    sign_pred: dict | None,
    empty_predictions: list[dict],
    params: dict,
) -> tuple[bytes, dict]:
    """Run fitting + render. Returns (annotated_jpeg_bytes, results_dict)."""
    img_w, img_h = pil_image.size

    results = estimate_total_capacity(
        empty_space_predictions=empty_predictions,
        img_h=img_h,
        img_w=img_w,
        default_placard_w=params["default_placard_w"],
        default_placard_h=params["default_placard_h"],
        margin=params["outer_margin"],
        spacing=params["spacing"],
        min_confidence=0.0,
        placard_scale=params["placard_scale"] / 100.0,
        empty_space_scale=params["empty_space_scale"] / 100.0,
        sign_prediction=sign_pred,
    )

    annotated_rgb = render_annotated_image(
        pil_image=pil_image,
        placard_predictions=[],
        empty_space_predictions=empty_predictions,
        per_region=results["per_region"],
        min_confidence=0.0,
        sign_quad=results.get("sign_quad"),
    )
    annotated_bytes = _to_jpeg_bytes(annotated_rgb)
    return annotated_bytes, results


def _run_inference_on_image(pil_image: Image.Image, sign_id: str) -> dict:
    """Full inference pipeline for a single image. Returns a result item dict."""
    pil_image = _resize_for_display(pil_image)
    img_bytes = _to_jpeg_bytes(pil_image)
    conf = min_confidence

    if mock_mode:
        sign_resp  = MOCK_SIGN_RESPONSE
        empty_resp = MOCK_EMPTY_SPACE_RESPONSE
        used_mock  = True
    else:
        sign_resp  = call_sign_model(
            pil_image, api_key, sign_project, int(sign_version), confidence=conf)
        empty_resp = call_empty_space_model(
            pil_image, api_key, empty_project, int(empty_version), confidence=conf)
        used_mock = False

    sign_pred    = _best_sign_pred(sign_resp, conf)
    empty_pred_list = _empty_preds(empty_resp, conf)
    params       = _current_fitting_params()

    annotated_bytes, results = _run_fitting(pil_image, sign_pred, empty_pred_list, params)

    region_rows = [
        {"Region": f"Region {r['region_index'] + 1}", "Placards fit": r["count"]}
        for r in results["per_region"]
    ]

    return {
        "sign_id":         sign_id,
        "annotated_bytes": annotated_bytes,
        "total_fit":       results["total_fit"],
        "placard_w":       results["placard_w"],
        "placard_h":       results["placard_h"],
        "n_empty_det":     len(empty_pred_list),
        "n_regions_fit":   sum(1 for r in results["per_region"] if r["count"] > 0),
        "region_rows":     region_rows,
        "sign_detected":   sign_pred is not None,
        "mock_mode":       used_mock,
        "_sign_resp":      sign_resp,
        "_empty_resp":     empty_resp,
        "_sign_pred":      sign_pred,
        "_img_bytes":      img_bytes,
        "_fitting_params": params,
    }


def _refit_result_item(item: dict, params: dict) -> dict:
    """Re-run fitting with new params using cached predictions (no API call)."""
    conf = params["min_confidence"]
    sign_pred       = item["_sign_pred"]
    empty_pred_list = _empty_preds(item["_empty_resp"], conf)

    pil_image = Image.open(BytesIO(item["_img_bytes"])).convert("RGB")
    annotated_bytes, results = _run_fitting(pil_image, sign_pred, empty_pred_list, params)

    region_rows = [
        {"Region": f"Region {r['region_index'] + 1}", "Placards fit": r["count"]}
        for r in results["per_region"]
    ]

    return {
        **item,
        "annotated_bytes": annotated_bytes,
        "total_fit":       results["total_fit"],
        "placard_w":       results["placard_w"],
        "placard_h":       results["placard_h"],
        "n_regions_fit":   sum(1 for r in results["per_region"] if r["count"] > 0),
        "region_rows":     region_rows,
        "n_empty_det":     len(empty_pred_list),
        "_fitting_params": params,
    }


def _render_results(results_list: list) -> None:
    for item in results_list:
        st.divider()
        st.subheader(item["sign_id"])

        if item.get("mock_mode"):
            st.info("Mock Mode — using built-in sample predictions (no API call made).")

        if not item.get("sign_detected"):
            st.warning("No sign boundary detected — perspective correction skipped.")

        col_img, col_stats = st.columns([3, 1])
        with col_img:
            st.caption("**Cyan** = proposed new placard slots  |  **Orange** = detected empty regions  |  **Yellow** = sign boundary")
            st.image(item["annotated_bytes"], use_container_width=True)

        with col_stats:
            n = item["n_regions_fit"]
            st.metric(
                "Estimated New Placard Slots",
                item["total_fit"],
                delta=f"across {n} region(s)" if n else None,
                delta_color="off",
            )
            st.metric("Placard Width (px)",  item["placard_w"])
            st.metric("Placard Height (px)", item["placard_h"])

        with st.expander("📊 Details"):
            c1, c2 = st.columns(2)
            c1.metric("Empty Regions Detected", item["n_empty_det"])
            c2.metric("Regions with Fits",      item["n_regions_fit"])
            if item["region_rows"]:
                st.table(item["region_rows"])

        with st.expander("Raw JSON — Sign Model"):
            st.json(item["_sign_resp"])
        with st.expander("Raw JSON — Empty-Space Model"):
            st.json(item["_empty_resp"])


# ---------------------------------------------------------------------------
# Tabs
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
    )

    run_button = st.button(
        "Run Inference",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if run_button and uploaded_files:
        if not mock_mode:
            missing = []
            if not api_key:       missing.append("API Key")
            if not sign_project:  missing.append("Sign Boundary Project ID")
            if not empty_project: missing.append("Empty-Space Project ID")
            if missing:
                st.error(f"Please fill in: {', '.join(missing)}")
                st.stop()

        results_list = []
        progress = st.progress(0, text="Running inference…")
        for idx, uf in enumerate(uploaded_files):
            progress.progress(
                idx / len(uploaded_files),
                text=f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})…",
            )
            try:
                pil_image = Image.open(uf).convert("RGB")
                results_list.append(_run_inference_on_image(pil_image, uf.name))
            except Exception as exc:
                st.error(f"{uf.name}: failed — {exc}")
        progress.progress(1.0, text="Done!")
        st.session_state["results_list"] = results_list
        # Clear any stale batch state to avoid showing old results
        st.session_state.pop("batch_active", None)

    if st.session_state.get("results_list"):
        current_params = _current_fitting_params()
        results_list_cached = st.session_state["results_list"]
        needs_save = False
        for i, item in enumerate(results_list_cached):
            if item.get("_fitting_params") != current_params:
                if "_img_bytes" not in item:
                    continue
                with st.spinner(f"Updating {item['sign_id']}…"):
                    results_list_cached[i] = _refit_result_item(item, current_params)
                needs_save = True
        if needs_save:
            st.session_state["results_list"] = results_list_cached
        _render_results(st.session_state["results_list"])
    elif not uploaded_files:
        st.info("Upload one or more images to get started.")


# ===========================================================================
# TAB 2 — Excel / CSV Batch
# ===========================================================================

with tab_excel:
    st.subheader("Batch Processing from Excel or CSV")
    st.caption(
        "Upload a spreadsheet with a sign identifier column and an image URL column. "
        "Images are processed one at a time so large batches never time out."
    )

    excel_file = st.file_uploader(
        "Upload Excel (.xlsx) or CSV (.csv)",
        type=["xlsx", "xls", "csv"],
        key="excel_uploader",
    )

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
            _mc1, _mc2 = st.columns(2)
            with _mc1:
                sign_id_col = st.selectbox(
                    "Sign identifier column", options=cols, index=0, key="sign_id_col"
                )
            with _mc2:
                url_col = st.selectbox(
                    "Image URL column", options=cols,
                    index=min(1, len(cols) - 1), key="url_col"
                )

            if st.button("Process Batch", type="primary", use_container_width=True):
                if not mock_mode:
                    missing = []
                    if not api_key:       missing.append("API Key")
                    if not sign_project:  missing.append("Sign Boundary Project ID")
                    if not empty_project: missing.append("Empty-Space Project ID")
                    if missing:
                        st.error(f"Please fill in: {', '.join(missing)}")
                        st.stop()

                rows = df[[sign_id_col, url_col]].dropna().values.tolist()
                rows = [(str(r[0]).strip(), str(r[1]).strip()) for r in rows]

                st.session_state["batch_rows"]         = rows
                st.session_state["batch_idx"]          = 0
                st.session_state["batch_results_list"] = []
                st.session_state["batch_errors"]       = []
                st.session_state["batch_active"]       = True
                st.rerun()

    # ── One-row-per-rerun batch loop ─────────────────────────────────────────
    if st.session_state.get("batch_active"):
        rows  = st.session_state["batch_rows"]
        idx   = st.session_state["batch_idx"]
        total = len(rows)

        st.progress(idx / total if total else 1.0,
                    text=f"Processing sign {idx + 1} of {total}…")
        if idx < total:
            st.caption(f"**{rows[idx][0]}** — {rows[idx][1]}")

        if st.button("Stop Batch", key="stop_batch"):
            st.session_state["batch_active"] = False
            st.rerun()

        if idx < total:
            sign_id, url = rows[idx]
            try:
                resp = requests.get(url, timeout=25)
                resp.raise_for_status()
                pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
                item = _run_inference_on_image(pil_image, sign_id)
                st.session_state["batch_results_list"].append(item)
            except Exception as exc:
                st.session_state["batch_errors"].append(f"**{sign_id}**: {exc}")

            st.session_state["batch_idx"] = idx + 1
            if st.session_state["batch_idx"] >= total:
                st.session_state["batch_active"] = False
            st.rerun()

    # ── Completion + results ──────────────────────────────────────────────────
    if (not st.session_state.get("batch_active")
            and st.session_state.get("batch_results_list")):
        done = len(st.session_state["batch_results_list"])
        st.success(f"Batch complete — {done} sign(s) processed.")

    for err in st.session_state.get("batch_errors", []):
        st.warning(err)

    if st.session_state.get("batch_results_list"):
        st.divider()
        st.subheader("Batch Results")
        _render_results(st.session_state["batch_results_list"])
