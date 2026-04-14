"""
roboflow_client.py

Two-model inference pipeline:
  Model 1 — Sign boundary  (blue-logo-sign/1)
             Classes: Blue-Logo-Sign, Exit Cap
             Used to extract the sign polygon for perspective correction.

  Model 2 — Empty-space    (empty-blue-space/N)
             Classes: empty-space (with optional polygon points)
             Used to determine where new placards can be placed.
"""

import base64
from io import BytesIO
from typing import Any

import requests
from PIL import Image


SIGN_CLASS     = "Blue-Logo-Sign"
EXIT_CAP_CLASS = "Exit Cap"
EMPTY_CLASS    = "empty-space"


# ---------------------------------------------------------------------------
# Mock responses (used when Mock Mode is ON)
# ---------------------------------------------------------------------------

MOCK_SIGN_RESPONSE: dict[str, Any] = {
    "predictions": [
        {
            "class": SIGN_CLASS,
            "x": 320, "y": 285, "width": 590, "height": 410,
            "confidence": 0.97,
            "points": [
                {"x": 28,  "y": 82},
                {"x": 612, "y": 78},
                {"x": 615, "y": 492},
                {"x": 25,  "y": 488},
            ],
        },
        {
            "class": EXIT_CAP_CLASS,
            "x": 290, "y": 38, "width": 215, "height": 58,
            "confidence": 0.95,
        },
    ],
    "image": {"width": 640, "height": 480},
}

MOCK_EMPTY_SPACE_RESPONSE: dict[str, Any] = {
    "predictions": [
        {
            "class": EMPTY_CLASS,
            "x": 460, "y": 200, "width": 155, "height": 140,
            "confidence": 0.88,
            "points": [
                {"x": 383, "y": 130}, {"x": 538, "y": 130},
                {"x": 538, "y": 270}, {"x": 383, "y": 270},
            ],
        },
        {
            "class": EMPTY_CLASS,
            "x": 355, "y": 325, "width": 245, "height": 130,
            "confidence": 0.83,
        },
    ],
    "image": {"width": 640, "height": 480},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _encode_image(pil_image: Image.Image) -> str:
    """Encode a PIL image as base64 for the Roboflow API."""
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _call(pil_image: Image.Image, api_key: str, project_id: str,
          version: int, confidence: float) -> dict[str, Any]:
    url = f"https://detect.roboflow.com/{project_id}/{version}"
    resp = requests.post(
        url,
        params={"api_key": api_key, "confidence": confidence},
        data=_encode_image(pil_image),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public callers
# ---------------------------------------------------------------------------

def call_sign_model(
    pil_image: Image.Image,
    api_key: str,
    project_id: str,
    version: int,
    confidence: float = 0.4,
) -> dict[str, Any]:
    """Call the sign-boundary model (blue-logo-sign). Returns raw Roboflow JSON."""
    return _call(pil_image, api_key, project_id, version, confidence)


def call_empty_space_model(
    pil_image: Image.Image,
    api_key: str,
    project_id: str,
    version: int,
    confidence: float = 0.4,
) -> dict[str, Any]:
    """Call the empty-space detection model. Returns raw Roboflow JSON."""
    return _call(pil_image, api_key, project_id, version, confidence)
