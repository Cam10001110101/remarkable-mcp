# Local Ollama/Gemma OCR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully-local Ollama/Gemma OCR backend (default `gemma4:31b`) and route all OCR through one unified `ocr.py` dispatcher, removing duplicated tesseract/google engine code.

**Architecture:** New `remarkable_mcp/ocr.py` owns OCR engines (tesseract, Google Vision, Ollama, sampling-wrapper), backend resolution, and Ollama config/reachability — all keyed on "PNG bytes → `(text, backend)`". `tools.py` and `extract.py` delegate to it. `auto` mode prefers local Ollama when reachable; `remarkable_read` does lazy per-page OCR for the page-level engines (sampling/ollama) so a 31B model only processes the requested page.

**Tech Stack:** Python, `requests` (already shipped, used by Google Vision), `cairosvg`+`rmc` (already shipped, render `.rm`→PNG), Ollama local REST API, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-ollama-gemma-ocr-design.md`

**Branch:** Implement on a feature branch (`feat/ollama-ocr`) → PR → merge when CI green (large refactor per repo CLAUDE.md).

**Regression gate:** `uv run pytest -q` (baseline = 99 passed) must stay green after every task.

---

### Task 1: Create `ocr.py` foundation + relocate backend helpers from `sampling.py`

Establishes the new module with config/reachability and **moves** `get_ocr_backend` +
`should_use_sampling_ocr` out of `sampling.py` (no circular import: `ocr.py` → `sampling.py` only).
Updates the tests that imported them so the suite stays green in this same commit.

**Files:**
- Create: `remarkable_mcp/ocr.py`
- Modify: `remarkable_mcp/sampling.py` (delete `get_ocr_backend`, `should_use_sampling_ocr`)
- Modify: `test_server.py` (repoint imports of those two symbols to `remarkable_mcp.ocr`)
- Test: `test_server.py` (new `TestOcrConfig`)

- [ ] **Step 1: Write the failing test** — append to `test_server.py`:

```python
class TestOcrConfig:
    """Config + reachability for the unified OCR module."""

    def test_get_ocr_backend_default_and_env(self):
        import os
        from remarkable_mcp.ocr import get_ocr_backend
        os.environ.pop("REMARKABLE_OCR_BACKEND", None)
        try:
            assert get_ocr_backend() == "auto"
            os.environ["REMARKABLE_OCR_BACKEND"] = "OLLAMA"
            assert get_ocr_backend() == "ollama"
        finally:
            os.environ.pop("REMARKABLE_OCR_BACKEND", None)

    def test_ollama_config_defaults(self):
        import os
        from remarkable_mcp import ocr
        for k in ("REMARKABLE_OLLAMA_MODEL", "REMARKABLE_OLLAMA_HOST",
                  "OLLAMA_HOST", "REMARKABLE_OLLAMA_TIMEOUT"):
            os.environ.pop(k, None)
        assert ocr.get_ollama_model() == "gemma4:31b"
        assert ocr.get_ollama_host() == "http://localhost:11434"
        assert ocr.get_ollama_timeout() == 180

    def test_ollama_available_true_false_and_cache(self):
        from unittest.mock import patch, MagicMock
        from remarkable_mcp import ocr
        ocr._reset_ollama_cache()
        with patch("remarkable_mcp.ocr.requests.get") as g:
            g.return_value = MagicMock(status_code=200)
            assert ocr.ollama_available() is True
            assert ocr.ollama_available() is True  # cached, no second call
            assert g.call_count == 1
        ocr._reset_ollama_cache()
        with patch("remarkable_mcp.ocr.requests.get", side_effect=Exception("refused")):
            assert ocr.ollama_available() is False
        ocr._reset_ollama_cache()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_server.py::TestOcrConfig -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'remarkable_mcp.ocr'`

- [ ] **Step 3: Create `remarkable_mcp/ocr.py`**

```python
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
```

- [ ] **Step 4: Delete the moved functions from `sampling.py`**

In `remarkable_mcp/sampling.py`, delete `def get_ocr_backend(...)` (the whole function) and
`def should_use_sampling_ocr(...)` (the whole function). Keep `OCR_SYSTEM_PROMPT`,
`OCR_USER_PROMPT`, `OCR_MODEL_PREFERENCES`, `ocr_via_sampling`, `ocr_pages_via_sampling`.
Remove the now-unused `from remarkable_mcp.capabilities import client_supports_sampling` line
inside `should_use_sampling_ocr` (it went away with the function).

- [ ] **Step 5: Repoint the affected tests in `test_server.py`**

Change these imports from `remarkable_mcp.sampling` to `remarkable_mcp.ocr`:
- in `test_get_ocr_backend_default`: `from remarkable_mcp.ocr import get_ocr_backend`
- in `test_get_ocr_backend_sampling`: `from remarkable_mcp.ocr import get_ocr_backend`
- in `test_should_use_sampling_ocr_false_when_not_configured`: `from remarkable_mcp.ocr import should_use_sampling_ocr`
- in `test_should_use_sampling_ocr_true_when_configured`: `from remarkable_mcp.ocr import should_use_sampling_ocr`
- in `test_should_use_sampling_ocr_false_when_client_doesnt_support`: `from remarkable_mcp.ocr import should_use_sampling_ocr`

In `test_sampling_imports_from_module`, remove `get_ocr_backend` and `should_use_sampling_ocr`
from the `from remarkable_mcp.sampling import (...)` tuple (keep `OCR_SYSTEM_PROMPT`,
`OCR_USER_PROMPT`, `ocr_pages_via_sampling`, `ocr_via_sampling`) and drop their `assert callable(...)`
lines for those two symbols.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest test_server.py::TestOcrConfig -q && uv run pytest -q`
Expected: `TestOcrConfig` PASS; full suite still green (99+ passed).

- [ ] **Step 7: Commit**

```bash
git add remarkable_mcp/ocr.py remarkable_mcp/sampling.py test_server.py
git commit -m "refactor: add ocr.py foundation; move backend helpers out of sampling"
```

---

### Task 2: Ollama OCR engine (`ocr_png_ollama`)

**Files:**
- Modify: `remarkable_mcp/ocr.py`
- Test: `test_server.py` (new `TestOllamaEngine`)

- [ ] **Step 1: Write the failing test**

```python
class TestOllamaEngine:
    def test_ocr_png_ollama_success(self):
        from unittest.mock import patch, MagicMock
        from remarkable_mcp import ocr
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"response": "hello world\n"}
        with patch("remarkable_mcp.ocr.requests.post", return_value=resp) as p:
            out = ocr.ocr_png_ollama(b"PNGDATA")
        assert out == "hello world"
        body = p.call_args.kwargs["json"]
        assert body["model"] == "gemma4:31b"
        assert body["images"] and isinstance(body["images"][0], str)
        assert body["stream"] is False

    def test_ocr_png_ollama_no_text_sentinel(self):
        from unittest.mock import patch, MagicMock
        from remarkable_mcp import ocr
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"response": "[NO TEXT DETECTED]"}
        with patch("remarkable_mcp.ocr.requests.post", return_value=resp):
            assert ocr.ocr_png_ollama(b"x") is None

    def test_ocr_png_ollama_error_returns_none(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.requests.post", side_effect=Exception("refused")):
            assert ocr.ocr_png_ollama(b"x") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_server.py::TestOllamaEngine -q`
Expected: FAIL — `AttributeError: module 'remarkable_mcp.ocr' has no attribute 'ocr_png_ollama'`

- [ ] **Step 3: Implement** — append to `remarkable_mcp/ocr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_server.py::TestOllamaEngine -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add remarkable_mcp/ocr.py test_server.py
git commit -m "feat: add ocr_png_ollama engine (local Gemma OCR)"
```

---

### Task 3: PNG-bytes tesseract + google engines + sampling wrapper

Ports the two PNG helpers (currently in `tools.py`) into `ocr.py` as the canonical, bytes-based
engines, plus a thin async sampling wrapper.

**Files:**
- Modify: `remarkable_mcp/ocr.py`
- Test: `test_server.py` (new `TestPngEngines`)

- [ ] **Step 1: Write the failing test**

```python
class TestPngEngines:
    def test_ocr_png_google_no_key_returns_none(self):
        import os
        from remarkable_mcp import ocr
        os.environ.pop("GOOGLE_VISION_API_KEY", None)
        assert ocr.ocr_png_google(b"x") is None

    def test_ocr_png_google_success(self):
        import os
        from unittest.mock import patch, MagicMock
        from remarkable_mcp import ocr
        os.environ["GOOGLE_VISION_API_KEY"] = "k"
        try:
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"responses": [{"fullTextAnnotation": {"text": "G text"}}]}
            with patch("remarkable_mcp.ocr.requests.post", return_value=resp):
                assert ocr.ocr_png_google(b"x") == "G text"
        finally:
            os.environ.pop("GOOGLE_VISION_API_KEY", None)

    def test_ocr_png_tesseract_missing_dep_returns_none(self):
        import builtins
        from unittest.mock import patch
        from remarkable_mcp import ocr
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "pytesseract":
                raise ImportError("no tesseract")
            return real_import(name, *a, **k)

        with patch("builtins.__import__", side_effect=fake_import):
            assert ocr.ocr_png_tesseract(b"x") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_server.py::TestPngEngines -q`
Expected: FAIL — `AttributeError: ... has no attribute 'ocr_png_google'`

- [ ] **Step 3: Implement** — append to `remarkable_mcp/ocr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_server.py::TestPngEngines -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add remarkable_mcp/ocr.py test_server.py
git commit -m "feat: PNG-bytes tesseract/google engines + sampling wrapper in ocr.py"
```

---

### Task 4: Backend resolution + dispatchers

**Files:**
- Modify: `remarkable_mcp/ocr.py`
- Test: `test_server.py` (new `TestOcrDispatcher`)

- [ ] **Step 1: Write the failing test**

```python
class TestOcrDispatcher:
    def setup_method(self):
        import os
        from remarkable_mcp import ocr
        for k in ("REMARKABLE_OCR_BACKEND", "GOOGLE_VISION_API_KEY"):
            os.environ.pop(k, None)
        ocr._reset_ollama_cache()

    def test_resolve_page_engine_auto_prefers_ollama_when_reachable(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.ollama_available", return_value=True):
            assert ocr.resolve_page_ocr_engine(ctx=None) == "ollama"
        with patch("remarkable_mcp.ocr.ollama_available", return_value=False):
            assert ocr.resolve_page_ocr_engine(ctx=None) is None

    def test_sync_dispatch_auto_ollama_then_fallback(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.ollama_available", return_value=True), \
             patch("remarkable_mcp.ocr.ocr_png_ollama", return_value=None), \
             patch("remarkable_mcp.ocr.ocr_png_google", return_value=None), \
             patch("remarkable_mcp.ocr.ocr_png_tesseract", return_value="T"):
            text, backend = ocr.ocr_png_sync(b"x")
        assert (text, backend) == ("T", "tesseract")

    def test_sync_dispatch_auto_ollama_wins(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.ollama_available", return_value=True), \
             patch("remarkable_mcp.ocr.ocr_png_ollama", return_value="O"):
            assert ocr.ocr_png_sync(b"x") == ("O", "ollama")

    def test_sync_dispatch_no_ollama_uses_tesseract(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.ollama_available", return_value=False), \
             patch("remarkable_mcp.ocr.ocr_png_tesseract", return_value="T"):
            assert ocr.ocr_png_sync(b"x") == ("T", "tesseract")

    @pytest.mark.asyncio
    async def test_async_dispatch_auto_ollama(self):
        from unittest.mock import patch
        from remarkable_mcp import ocr
        with patch("remarkable_mcp.ocr.ollama_available", return_value=True), \
             patch("remarkable_mcp.ocr.ocr_png_ollama", return_value="O"):
            assert await ocr.ocr_png(b"x", ctx=None) == ("O", "ollama")
```

(Note: `test_server.py` already uses `pytest` and async tests — confirm `import pytest` is present at top of file; it is, given existing `@pytest.mark.asyncio` tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_server.py::TestOcrDispatcher -q`
Expected: FAIL — `AttributeError: ... has no attribute 'resolve_page_ocr_engine'`

- [ ] **Step 3: Implement** — append to `remarkable_mcp/ocr.py`:

```python
def resolve_page_ocr_engine(ctx: Optional["Context"]) -> Optional[str]:
    """Which lazy, page-level engine the read path should use: 'sampling', 'ollama', or None.

    None means: no page-level engine selected -> caller uses the batch
    (extract_handwriting_ocr) path for google/tesseract.
    """
    backend = get_ocr_backend()
    if backend == "sampling" and ctx is not None and client_supports_sampling(ctx):
        return "sampling"
    if backend == "ollama" and ollama_available():
        return "ollama"
    if backend == "auto" and ollama_available():
        return "ollama"
    return None


def _sync_fallback_chain() -> list:
    """Ordered (label, engine) pairs for sync OCR, based on configured backend + reachability."""
    backend = get_ocr_backend()
    ollama = ("ollama", ocr_png_ollama)
    google = ("google", ocr_png_google)
    tess = ("tesseract", ocr_png_tesseract)
    if backend == "ollama":
        return [ollama, google, tess]
    if backend == "google":
        return [google, tess]
    if backend == "tesseract":
        return [tess]
    # "auto" (and "sampling" when called from a sync path): prefer local ollama.
    if ollama_available():
        return [ollama, google, tess]
    if os.environ.get("GOOGLE_VISION_API_KEY"):
        return [google, tess]
    return [tess]


def ocr_png_sync(png: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Sync dispatcher (no sampling). Returns (text, backend_used) or (None, None)."""
    for label, engine in _sync_fallback_chain():
        text = engine(png)
        if text:
            return text, label
    return None, None


async def ocr_png(
    png: bytes, ctx: Optional["Context"] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Async dispatcher. Tries sampling first only when explicitly configured + supported,
    then the sync engine chain (offloaded to a thread so slow inference can't block the loop).
    Returns (text, backend_used) or (None, None)."""
    import asyncio

    if (
        get_ocr_backend() == "sampling"
        and ctx is not None
        and client_supports_sampling(ctx)
    ):
        text = await ocr_png_sampling(ctx, png)
        if text:
            return text, "sampling"

    for label, engine in _sync_fallback_chain():
        text = await asyncio.to_thread(engine, png)
        if text:
            return text, label
    return None, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_server.py::TestOcrDispatcher -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add remarkable_mcp/ocr.py test_server.py
git commit -m "feat: OCR backend resolution + sync/async dispatchers (auto prefers ollama)"
```

---

### Task 5: Delegate `extract.py::extract_handwriting_ocr`; delete duplicated rm-based engines

**Files:**
- Modify: `remarkable_mcp/extract.py` (rewrite `extract_handwriting_ocr`; delete `_ocr_google_vision`, `_ocr_google_vision_rest`, `_ocr_google_vision_sdk`, `_ocr_tesseract`)
- Test: `test_server.py` (new `TestExtractHandwritingDelegation`)

- [ ] **Step 1: Write the failing test**

```python
class TestExtractHandwritingDelegation:
    def test_extract_handwriting_delegates_to_ocr(self):
        from unittest.mock import patch
        from remarkable_mcp.extract import extract_handwriting_ocr
        rm_files = ["/tmp/p1.rm", "/tmp/p2.rm"]
        with patch("remarkable_mcp.extract.render_rm_file_to_png", return_value=b"PNG"), \
             patch("remarkable_mcp.ocr.ocr_png_sync", side_effect=[("page one", "ollama"),
                                                                   ("page two", "ollama")]):
            results, backend = extract_handwriting_ocr(rm_files)
        assert results == ["page one", "page two"]
        assert backend == "ollama"

    def test_extract_handwriting_skips_unrenderable_pages(self):
        from unittest.mock import patch
        from remarkable_mcp.extract import extract_handwriting_ocr
        with patch("remarkable_mcp.extract.render_rm_file_to_png", return_value=None), \
             patch("remarkable_mcp.ocr.ocr_png_sync", return_value=("x", "tesseract")):
            results, backend = extract_handwriting_ocr(["/tmp/p1.rm"])
        assert results is None and backend is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_server.py::TestExtractHandwritingDelegation -q`
Expected: FAIL — current `extract_handwriting_ocr` doesn't call `render_rm_file_to_png` per page / signature mismatch.

- [ ] **Step 3: Replace `extract_handwriting_ocr` and delete the old engine helpers**

In `remarkable_mcp/extract.py`, replace the entire body of `extract_handwriting_ocr` (and its
docstring) with:

```python
def extract_handwriting_ocr(rm_files: List[Path]) -> tuple[Optional[List[str]], Optional[str]]:
    """Render each .rm page to PNG and OCR it via the unified ocr dispatcher.

    Returns (ocr_results, backend_used). Engine selection (auto/ollama/google/
    tesseract) and fallback live in remarkable_mcp.ocr; this function only
    handles rendering + aggregation. Empty/failed pages are skipped (parity with
    prior behavior).
    """
    from remarkable_mcp import ocr

    background = get_background_color()
    results: List[str] = []
    backend_used: Optional[str] = None
    for rm_file in rm_files:
        png = render_rm_file_to_png(Path(rm_file), background_color=background)
        if not png:
            continue
        text, used = ocr.ocr_png_sync(png)
        if text:
            results.append(text)
            backend_used = backend_used or used
    return (results if results else None, backend_used)
```

Then **delete** these now-unused functions in their entirety: `_ocr_google_vision`,
`_ocr_google_vision_rest`, `_ocr_google_vision_sdk`, `_ocr_tesseract`. (Their PNG-based
equivalents now live in `ocr.py`.) Leave `render_rm_file_to_png`, `render_rm_file_to_svg`,
`render_page_from_document_zip`, the `_render_rm_v5/v6` helpers, and all non-OCR extraction intact.

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest test_server.py::TestExtractHandwritingDelegation -q && uv run pytest -q`
Expected: new tests PASS; full suite green. If any old test referenced `_ocr_tesseract` /
`_ocr_google_vision*` directly, update it to patch `remarkable_mcp.ocr.ocr_png_sync` instead.

- [ ] **Step 5: Commit**

```bash
git add remarkable_mcp/extract.py test_server.py
git commit -m "refactor: extract_handwriting_ocr delegates to ocr dispatcher; drop dup engines"
```

---

### Task 6: Wire `tools.py` (`remarkable_image` + `remarkable_read`) to the dispatcher; remove tools OCR helpers

**Files:**
- Modify: `remarkable_mcp/tools.py` (imports; delete `_ocr_png_tesseract`, `_ocr_png_google_vision`; rewrite OCR in `remarkable_image`; generalize the page-level branch in `remarkable_read`)
- Test: `test_server.py` (full suite + the existing `test_read_notebook_empty_content_ocr_retry`)

- [ ] **Step 1: Update imports**

In `remarkable_mcp/tools.py`, replace:

```python
from remarkable_mcp.sampling import (
    get_ocr_backend,
    ocr_via_sampling,
    should_use_sampling_ocr,
)
```

with:

```python
from remarkable_mcp import ocr
```

Keep the existing `from remarkable_mcp.extract import (... get_background_color ...)` import.

- [ ] **Step 2: Delete the two PNG OCR helpers in `tools.py`**

Delete `def _ocr_png_tesseract(...)` and `def _ocr_png_google_vision(...)` in their entirety
(now provided by `ocr.ocr_png_tesseract` / `ocr.ocr_png_google`).

- [ ] **Step 3: Replace the OCR block in `remarkable_image` (PNG branch)**

Find the block that starts with `ocr_text = None` / `ocr_backend_used = None` and runs the
`sampling -> google -> tesseract` chain (the `if include_ocr:` section inside the PNG branch).
Replace that whole section with:

```python
                # OCR via the unified dispatcher (auto prefers local ollama).
                ocr_text = None
                ocr_backend_used = None
                if include_ocr:
                    ocr_text, ocr_backend_used = await ocr.ocr_png(png_data, ctx)
```

Leave the downstream `ocr_info` / hint / response construction unchanged (it already reads
`ocr_text` and `ocr_backend_used`).

- [ ] **Step 4: Generalize the page-level OCR branch in `remarkable_read`**

Replace the `use_sampling = ...` line and its two dependent branches with a `page_engine`-driven
version. Specifically:

Replace:
```python
            use_sampling = is_notebook and include_ocr and ctx and should_use_sampling_ocr(ctx)
```
with:
```python
            page_engine = (
                ocr.resolve_page_ocr_engine(ctx) if (is_notebook and include_ocr) else None
            )
```

Replace `if use_sampling:` with `if page_engine:`. Inside that branch, change the cached lookup
and store calls from the literal `"sampling"` to `page_engine`:
- `get_cached_page_ocr(target_doc.ID, page, "sampling")` -> `get_cached_page_ocr(target_doc.ID, page, page_engine)`
- in the cached branch: `ocr_backend_used = "sampling"` -> `ocr_backend_used = page_engine`
- `cache_page_ocr(target_doc.ID, page, "sampling", ocr_text)` -> `cache_page_ocr(target_doc.ID, page, page_engine, ocr_text)`

And replace the actual OCR call:
```python
                        ocr_text = await ocr_via_sampling(ctx, png_data)
                        if ocr_text:
                            cache_page_ocr(target_doc.ID, page, "sampling", ocr_text)
                            notebook_pages = [""] * total_notebook_pages
                            notebook_pages[page - 1] = ocr_text
                            ocr_backend_used = "sampling"
```
with:
```python
                        ocr_text, used_backend = await ocr.ocr_png(png_data, ctx)
                        if ocr_text:
                            cache_page_ocr(target_doc.ID, page, page_engine, ocr_text)
                            notebook_pages = [""] * total_notebook_pages
                            notebook_pages[page - 1] = ocr_text
                            ocr_backend_used = used_backend or page_engine
```

Finally, change the next guard:
```python
            if not use_sampling and is_notebook and include_ocr:
```
to:
```python
            if not page_engine and is_notebook and include_ocr:
```

Leave everything else (the `if not notebook_pages and is_notebook:` batch-extract block, grep,
pagination, hints) unchanged.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green (99+ passed). Pay attention to `test_read_notebook_empty_content_ocr_retry`
(exercises the read OCR auto-retry path) — if it patched `ocr_via_sampling` or
`should_use_sampling_ocr` on `tools`/`sampling`, update it to patch `remarkable_mcp.ocr.ocr_png`
and/or `remarkable_mcp.ocr.resolve_page_ocr_engine`.

- [ ] **Step 6: Commit**

```bash
git add remarkable_mcp/tools.py test_server.py
git commit -m "feat: route remarkable_image/remarkable_read OCR through unified dispatcher"
```

---

### Task 7: Docs + live end-to-end smoke test

**Files:**
- Modify: `README.md` (OCR section)
- Create: `docs/ollama-setup.md`
- Verify: live Ollama OCR on a real notebook page

- [ ] **Step 1: Update README OCR section**

In `README.md`, in the OCR/backends section, add `ollama` to the backend list and an env-var table:

```markdown
### Local OCR with Ollama (Gemma) — recommended

Fully local, no API key, no data leaves your machine. The default `auto` mode uses Ollama
automatically when its server is reachable.

1. Install Ollama (https://ollama.com) and pull a vision model:
   ```bash
   ollama pull gemma4:31b      # best accuracy (default); or gemma4:e4b for speed
   ```
2. That's it — with `REMARKABLE_OCR_BACKEND=auto` (the default), reMarkable-MCP detects the
   running Ollama server and uses it for handwriting OCR, falling back to Google Vision / Tesseract
   if it's not reachable.

| Env var | Default | Meaning |
|---|---|---|
| `REMARKABLE_OCR_BACKEND` | `auto` | `auto` \| `ollama` \| `sampling` \| `google` \| `tesseract` |
| `REMARKABLE_OLLAMA_MODEL` | `gemma4:31b` | Ollama vision model tag |
| `REMARKABLE_OLLAMA_HOST` | `OLLAMA_HOST` or `http://localhost:11434` | Ollama server URL |
| `REMARKABLE_OLLAMA_TIMEOUT` | `180` | Per-page OCR timeout (seconds) |
```

- [ ] **Step 2: Create `docs/ollama-setup.md`** (mirror `docs/google-vision-setup.md` brevity)

```markdown
# Local OCR with Ollama (Gemma)

reMarkable notebooks are mostly handwriting. The Ollama backend runs a local vision model
(default `gemma4:31b`) so OCR is high quality, free, and fully on-device.

## Setup
1. Install Ollama: https://ollama.com
2. Pull a vision model: `ollama pull gemma4:31b` (or `gemma4:e4b` for lower latency/RAM).
3. Ensure the server is running (`ollama serve`, or the desktop app). Verify: `curl http://localhost:11434/api/tags`.

## Usage
With the default `REMARKABLE_OCR_BACKEND=auto`, the server uses Ollama whenever it's reachable and
falls back to Google Vision (if `GOOGLE_VISION_API_KEY` is set) then Tesseract. Force it explicitly
with `REMARKABLE_OCR_BACKEND=ollama`.

## Tuning
- `REMARKABLE_OLLAMA_MODEL` — `gemma4:e2b` (fastest) … `gemma4:31b` (most accurate).
- `REMARKABLE_OLLAMA_TIMEOUT` — raise if a large model is slow on first load.
- `REMARKABLE_OLLAMA_HOST` — point at a remote Ollama if not on localhost.
```

- [ ] **Step 3: Commit docs**

```bash
git add README.md docs/ollama-setup.md
git commit -m "docs: document local Ollama/Gemma OCR backend"
```

- [ ] **Step 4: Live end-to-end smoke test (real Ollama + gemma4:31b)**

Pick a real handwritten notebook from the tablet (e.g. `Scratch Pad`). Run a one-off script that
renders page 1 and OCRs it through the live dispatcher, and **paste the actual printed output**:

```bash
uv run python - <<'PY'
import asyncio, os
os.environ["REMARKABLE_OCR_BACKEND"] = "auto"
from remarkable_mcp import ocr
print("ollama reachable:", ocr.ollama_available(), "| model:", ocr.get_ollama_model())

# Render page 1 of a real notebook to PNG via the SSH client, then OCR it.
from remarkable_mcp.api import get_rmapi, get_items_by_id
from remarkable_mcp.extract import render_page_from_document_zip, get_document_page_count
import tempfile, pathlib
client = get_rmapi()
docs = [d for d in client.get_meta_items() if not d.is_folder and d.VissibleName == "Scratch Pad"]
assert docs, "Scratch Pad not found"
raw = client.download(docs[0])
with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as t:
    t.write(raw); zp = pathlib.Path(t.name)
png = render_page_from_document_zip(zp, 1)
print("rendered PNG bytes:", len(png) if png else None)
text, backend = asyncio.run(ocr.ocr_png(png))
print("backend:", backend)
print("---- OCR TEXT ----")
print(text)
PY
```

Expected: `ollama reachable: True`, a non-zero PNG size, `backend: ollama`, and recognizable text
from the page. **Do not mark this task complete without pasting the real output.** If Ollama OCR
fails, debug with the systematic-debugging skill before claiming success.

- [ ] **Step 5: Final regression gate**

Run: `uv run pytest -q`
Expected: all green (≥99 passed). Then open the PR (`feat/ollama-ocr`) and merge when CI is green.

---

## Self-Review

**Spec coverage:**
- Ollama backend (`ocr_png_ollama`) → Task 2. ✔
- `auto` prefers Ollama when reachable → `resolve_page_ocr_engine` + `_sync_fallback_chain` (Task 4), used by read (Task 6) and image (Task 6). ✔
- Unify duplicated engines → tesseract/google ported to `ocr.py` (Task 3); deleted from `extract.py` (Task 5) and `tools.py` (Task 6). ✔
- Lazy per-page OCR for 31B in read → `page_engine` branch (Task 6). ✔
- Resources/batch path gets Ollama → `extract_handwriting_ocr` delegates to `ocr_png_sync` (Task 5). ✔
- Config env vars (model/host/timeout/backend) → Task 1. ✔
- No new deps → uses `requests`/`cairosvg`/`PIL` already present. ✔
- `asyncio.to_thread` offload → `ocr_png` (Task 4). ✔
- No circular import (prompts stay in `sampling.py`) → Task 1/Task 3 imports. ✔
- Tests + 99-test regression gate → every task. ✔
- Live smoke test → Task 7. ✔
- Docs → Task 7. ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows expected output.

**Type consistency:** `ocr_png(png, ctx) -> (text, backend)` and `ocr_png_sync(png) -> (text, backend)` used consistently across Tasks 4/5/6. `resolve_page_ocr_engine(ctx) -> Optional[str]` used in Tasks 4 and 6. `render_rm_file_to_png(path, background_color) -> Optional[bytes]` matches `extract.py:534`. `extract_handwriting_ocr(rm_files) -> (Optional[List[str]], Optional[str])` preserves the existing caller contract in `extract_text_from_document_zip`.
