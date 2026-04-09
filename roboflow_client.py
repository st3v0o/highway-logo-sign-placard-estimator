"""
roboflow_client.py

Handles all communication with Roboflow hosted inference endpoints.
The URL builder is in one place so it is easy to adjust if the format changes.
"""

import base64
from io import BytesIO
from typing import Any

import requests
from PIL import Image


# ---------------------------------------------------------------------------
# URL builder — edit this function if Roboflow's endpoint format ever changes
# ---------------------------------------------------------------------------

def build_inference_url(workspace_id: str, project_id: str, version: int) -> str:
    """
    Build the Roboflow hosted inference URL for a given model.

    Current format (Roboflow hosted inference):
        https://detect.roboflow.com/{project_id}/{version}

    Note: workspace_id is accepted for future flexibility but is not part of
    the hosted inference URL — Roboflow routes by project_id and version only.
    If Roboflow changes their URL format, update only this function.
    """
    return f"https://detect.roboflow.com/{project_id}/{version}"


# ---------------------------------------------------------------------------
# Mock prediction data — used when Mock Mode is enabled
# ---------------------------------------------------------------------------

MOCK_PLACARD_RESPONSE: dict[str, Any] = {
    "predictions": [
        {"class": "placard", "x": 120, "y": 180, "width": 95, "height": 70, "confidence": 0.94},
        {"class": "placard", "x": 230, "y": 180, "width": 90, "height": 68, "confidence": 0.91},
        {"class": "placard", "x": 340, "y": 180, "width": 92, "height": 69, "confidence": 0.88},
    ],
    "image": {"width": 640, "height": 480},
}

MOCK_EMPTY_SPACE_RESPONSE: dict[str, Any] = {
    "predictions": [
        {
            "class": "empty-space",
            "x": 450,
            "y": 180,
            "width": 180,
            "height": 140,
            "confidence": 0.89,
            "points": [
                {"x": 362, "y": 112},
                {"x": 538, "y": 112},
                {"x": 538, "y": 248},
                {"x": 362, "y": 248},
            ],
        },
        {
            "class": "empty-space",
            "x": 120,
            "y": 330,
            "width": 400,
            "height": 120,
            "confidence": 0.82,
        },
    ],
    "image": {"width": 640, "height": 480},
}


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image(pil_image: Image.Image) -> str:
    """Encode a PIL image as a base64 string for the Roboflow API."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------

def call_placard_model(
    pil_image: Image.Image,
    api_key: str,
    workspace_id: str,
    project_id: str,
    version: int,
    confidence: float = 0.4,
) -> dict[str, Any]:
    """
    Call the placard detection model.

    Returns the raw JSON response dict from Roboflow.
    Raises requests.HTTPError on non-2xx responses.
    """
    url = build_inference_url(workspace_id, project_id, version)
    image_b64 = _encode_image(pil_image)

    params = {
        "api_key": api_key,
        "confidence": confidence,
    }

    response = requests.post(
        url,
        params=params,
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_empty_space_model(
    pil_image: Image.Image,
    api_key: str,
    workspace_id: str,
    project_id: str,
    version: int,
    confidence: float = 0.4,
) -> dict[str, Any]:
    """
    Call the empty-blue-space detection model.

    Returns the raw JSON response dict from Roboflow.
    The response may contain polygon points or just bounding boxes.
    Raises requests.HTTPError on non-2xx responses.
    """
    url = build_inference_url(workspace_id, project_id, version)
    image_b64 = _encode_image(pil_image)

    params = {
        "api_key": api_key,
        "confidence": confidence,
    }

    response = requests.post(
        url,
        params=params,
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
