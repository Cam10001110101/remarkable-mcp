"""Unified OCR dispatcher for reMarkable handwriting recognition.

Given PNG bytes, returns ``(text, backend_used)``. This module owns the OCR
engines (tesseract, Google Vision, Ollama, and an MCP-sampling wrapper),
backend resolution, and local Ollama config/reachability. ``tools.py`` and
``extract.py`` delegate here so the engine logic lives in exactly one place.

Import direction is one-way: ``ocr`` -> ``sampling`` / ``capabilities``.
``sampling`` must never import ``ocr`` (would be circular).
"""

import base64
import os
from typing import TYPE_CHECKING, Optional, Tuple

import requests

from remarkable_mcp.capabilities import client_supports_sampling
from remarkable_mcp.sampling import (
    OCR_SYSTEM_PROMPT,
    OCR_USER_PROMPT,
    ocr_via_sampling,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_TIMEOUT = 180

_NO_TEXT_SENTINEL = "[NO TEXT DETECTED]"

# Per-process cache of the reachability probe (None = not yet probed).
_ollama_reachable: Optional[bool] = None


def get_ocr_backend() -> str:
    """Configured backend: 'auto' | 'ollama' | 'sampling' | 'google' | 'tesseract'."""
    return os.environ.get("REMARKABLE_OCR_BACKEND", "auto").lower()


def get_ollama_host() -> str:
    return (
        os.environ.get("REMARKABLE_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_OLLAMA_HOST
    )


def get_ollama_model() -> str:
    return os.environ.get("REMARKABLE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_ollama_timeout() -> int:
    try:
        return int(os.environ.get("REMARKABLE_OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT


def _normalize_host(host: str) -> str:
    """Accept '127.0.0.1:11434' or 'http://host:port'; return a scheme'd, slash-trimmed URL."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def ollama_available() -> bool:
    """True if a local Ollama server answers /api/tags. Cached per process."""
    global _ollama_reachable
    if _ollama_reachable is not None:
        return _ollama_reachable
    try:
        resp = requests.get(f"{_normalize_host(get_ollama_host())}/api/tags", timeout=2)
        _ollama_reachable = resp.status_code == 200
    except Exception:
        _ollama_reachable = False
    return _ollama_reachable


def _reset_ollama_cache() -> None:
    """Test hook: clear the cached reachability probe."""
    global _ollama_reachable
    _ollama_reachable = None


def should_use_sampling_ocr(ctx: "Context") -> bool:
    """True only when backend is explicitly 'sampling' AND the client supports sampling."""
    if get_ocr_backend() != "sampling":
        return False
    return client_supports_sampling(ctx)


def ocr_png_ollama(png: bytes) -> Optional[str]:
    """OCR a PNG via a local Ollama vision model. Returns None on any failure."""
    try:
        image_b64 = base64.b64encode(png).decode("utf-8")
        resp = requests.post(
            f"{_normalize_host(get_ollama_host())}/api/generate",
            json={
                "model": get_ollama_model(),
                "system": OCR_SYSTEM_PROMPT,
                "prompt": OCR_USER_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=get_ollama_timeout(),
        )
        if resp.status_code != 200:
            return None
        text = (resp.json().get("response") or "").strip()
        if not text or _NO_TEXT_SENTINEL in text:
            return None
        return text
    except Exception:
        return None


def ocr_png_tesseract(png: bytes) -> Optional[str]:
    """Local Tesseract OCR on PNG bytes (weak on handwriting; last-resort fallback)."""
    try:
        import io

        import pytesseract
        from PIL import Image, ImageFilter, ImageOps

        img = Image.open(io.BytesIO(png)).convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = img.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(img, config=r"--psm 11 --oem 3")
        return text.strip() or None
    except ImportError:
        return None
    except Exception:
        return None


def ocr_png_google(png: bytes) -> Optional[str]:
    """Google Cloud Vision REST OCR on PNG bytes. Requires GOOGLE_VISION_API_KEY."""
    api_key = os.environ.get("GOOGLE_VISION_API_KEY")
    if not api_key:
        return None
    try:
        image_b64 = base64.b64encode(png).decode("utf-8")
        resp = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json={
                "requests": [
                    {
                        "image": {"content": image_b64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    }
                ]
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        responses = resp.json().get("responses") or []
        if responses and "fullTextAnnotation" in responses[0]:
            text = responses[0]["fullTextAnnotation"]["text"].strip()
            return text or None
        return None
    except Exception:
        return None


async def ocr_png_sampling(ctx: "Context", png: bytes) -> Optional[str]:
    """Async wrapper over the MCP-sampling OCR primitive in sampling.py."""
    return await ocr_via_sampling(ctx, png)
