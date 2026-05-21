# Design: Local Ollama/Gemma OCR via a Unified OCR Dispatcher

- **Date:** 2026-05-21
- **Status:** Approved (design); pending implementation plan
- **Feature:** Add a fully-local handwriting OCR backend (Ollama vision models, default `gemma4:31b`) and unify the currently-duplicated OCR engine code behind one dispatcher.

## Problem

reMarkable notebooks are mostly handwriting. Today OCR has three backends selected via
`REMARKABLE_OCR_BACKEND` (`auto` | `sampling` | `google` | `tesseract`):

- `sampling` — asks the MCP client's own LLM via `ctx.session`. Good quality, **but** only
  works if the client supports MCP sampling, and it ships private handwriting to that cloud model.
- `google` — Google Cloud Vision; requires an API key.
- `tesseract` — local, no key, **but poor at handwriting**.

There is no local, high-quality, key-free option. Ollama (already installed and running on the
target machine, with `gemma4` vision models pulled) fills that gap: LLM-quality handwriting OCR
that stays on-device and needs no API key or sampling-capable client.

Secondary problem: the OCR engine code is **duplicated**. Tesseract and Google Vision are each
implemented twice — once PNG-based in `tools.py` (`_ocr_png_tesseract`, `_ocr_png_google_vision`)
for `remarkable_image`, and once `.rm`-files-based in `extract.py` (`_ocr_tesseract`,
`_ocr_google_vision*`) for `remarkable_read`/resources. Adding a fourth backend to both places
would deepen the duplication.

## Goals

1. Add an `ollama` OCR backend backed by a local Ollama vision model (default `gemma4:31b`).
2. Make the default `auto` mode **prefer local Ollama when the server is reachable**, with the
   existing backends as fallback. No regression when Ollama is absent.
3. Unify the duplicated OCR engine + backend-resolution code into one module so all call sites
   share one implementation.
4. No new Python dependencies. No change to non-OCR behavior.

## Non-Goals

- Rewriting the rendering pipeline (`.rm` → SVG/PNG stays in `extract.py`).
- Changing typed-text / highlight / native-`.rm`-text extraction.
- Changing Google/Tesseract caching semantics in the read path.
- Pulling models for the user (document `ollama pull gemma4:31b` instead).

## Decisions (from brainstorming)

- **Integration:** `auto` prefers Ollama when reachable; keep sampling/google/tesseract as fallback.
- **Default model:** `gemma4:31b` (best accuracy; configurable). Implies generous timeout and
  lazy per-page OCR in the read path so a long notebook doesn't OCR every page up front.
- **Approach:** unify into one `ocr.py` dispatcher, then add Ollama once.

## Architecture

### New module: `remarkable_mcp/ocr.py`

Single responsibility: **given PNG bytes, return `(text, backend_used)`.** Owns OCR engines,
backend resolution, and Ollama config/reachability. Depends on `requests`, `PIL`,
`sampling.py` (async sampling primitive), `capabilities.py` (client sampling check).

**Config**
- `get_ocr_backend() -> str` — moved from `sampling.py`; reads `REMARKABLE_OCR_BACKEND` (default `auto`).
- `get_ollama_model() -> str` — `REMARKABLE_OLLAMA_MODEL`, default `gemma4:31b`.
- `get_ollama_host() -> str` — `REMARKABLE_OLLAMA_HOST`, else `OLLAMA_HOST`, else `http://localhost:11434`.
- `get_ollama_timeout() -> int` — `REMARKABLE_OLLAMA_TIMEOUT`, default `180` (seconds).

**Reachability**
- `ollama_available() -> bool` — `GET {host}/api/tags` with ~2s timeout; result cached in a
  module global for the process. `_reset_ollama_cache()` test hook.

**Engines (PNG bytes → `Optional[str]`)**
- `ocr_png_tesseract(png)` — port of the existing PNG tesseract helper (grayscale, autocontrast,
  sharpen, `--psm 11 --oem 3`).
- `ocr_png_google(png)` — Google Vision REST with `GOOGLE_VISION_API_KEY` (DOCUMENT_TEXT_DETECTION).
- `ocr_png_ollama(png)` — see below.
- `async ocr_png_sampling(ctx, png)` — thin wrapper over `sampling.ocr_via_sampling`.

**Prompts** — reuse `OCR_SYSTEM_PROMPT` / `OCR_USER_PROMPT`. They **stay in `sampling.py`** and
`ocr.py` imports them from there (alongside `ocr_via_sampling`). This avoids a circular import:
`ocr.py` → `sampling.py` is one-directional; `sampling.py` must not import `ocr.py`. Honor the
`[NO TEXT DETECTED]` sentinel.

**Dispatchers**
- `async ocr_png(png, ctx=None) -> tuple[Optional[str], Optional[str]]`
- `ocr_png_sync(png) -> tuple[Optional[str], Optional[str]]` (no sampling; for sync callers/resources)

Both resolve the backend, try it, then fall back through the chain, returning the engine that
actually produced text.

### Backend resolution

```
resolve order for ocr_png() / ocr_png_sync():
  backend = get_ocr_backend()
  if backend == "auto":
      if ollama_available():            -> ollama  (then google, then tesseract)
      elif GOOGLE_VISION_API_KEY:        -> google  (then tesseract)
      else:                              -> tesseract
  if backend == "ollama":                -> ollama  (fallback google/tesseract; label = actual)
  if backend == "google":                -> google  (fallback tesseract)
  if backend == "tesseract":             -> tesseract
  if backend == "sampling":
      ocr_png (async, ctx, client supports sampling) -> sampling (fallback ollama/google/tesseract)
      ocr_png_sync (no async ctx)                    -> treat as "auto"
```

For the read lazy-path decision, a helper resolves which **page-level** engine to use:
`resolve_page_ocr_engine(ctx) -> Optional["sampling"|"ollama"]`
- `sampling` if backend == `sampling` and client supports sampling.
- `ollama` if backend == `ollama` (and reachable) or backend == `auto` and reachable.
- else `None` → read uses the existing batch path (google/tesseract via `extract.py`).

### Ollama call (`ocr_png_ollama`)

```
POST {host}/api/generate
{
  "model":  get_ollama_model(),
  "system": OCR_SYSTEM_PROMPT,
  "prompt": OCR_USER_PROMPT,
  "images": [base64(png)],
  "stream": false,
  "options": {"temperature": 0}
}
-> {"response": "<text>"}   # return stripped text, or None on sentinel/empty
```
- Timeout = `get_ollama_timeout()`.
- Any exception (connection refused, model not pulled, timeout, non-200) → `None` (dispatcher
  falls back). In the async dispatcher, the sync `requests` call runs via `asyncio.to_thread`
  so a 31B inference doesn't block the event loop.
- Image is the rendered page PNG. (Resolution stays as produced by the existing renderer; a future
  knob can downscale for latency — out of scope here.)

### Call-site changes

- `tools.py::remarkable_image` — remove `_ocr_png_tesseract` / `_ocr_png_google_vision`; replace the
  inline `sampling→google→tesseract` chain with `text, backend = await ocr.ocr_png(png_data, ctx)`.
- `tools.py::remarkable_read` — generalize the current `use_sampling` lazy branch so it also fires
  when `resolve_page_ocr_engine(ctx) == "ollama"`: render **only the requested page**, call
  `await ocr.ocr_png(page_png, ctx)`, write to the per-page cache with the returned backend label.
  When `resolve_page_ocr_engine` returns `None`, fall through to the existing batch path unchanged.
- `extract.py::extract_handwriting_ocr(rm_files)` — render each `.rm` → PNG via the existing
  `render_rm_file_to_png`, call `ocr.ocr_png_sync(png)`, collect non-empty results; return
  `(results, backend_used)`. **Delete** `_ocr_tesseract`, `_ocr_google_vision`,
  `_ocr_google_vision_rest`, `_ocr_google_vision_sdk` (~250 lines) — now handled by `ocr.py`.
- `sampling.py` — keeps only `ocr_via_sampling` (+ shared prompts/model-prefs). `get_ocr_backend`
  and `should_use_sampling_ocr` move to `ocr.py`; update importers (`tools.py`).

### Module boundaries (isolation)

| Module | One job | Depends on |
|---|---|---|
| `ocr.py` | PNG → `(text, backend)`; engines, resolution, ollama config/reachability | requests, PIL, sampling, capabilities |
| `sampling.py` | MCP sampling primitive only | mcp |
| `extract.py` | render `.rm`→PNG/SVG + non-OCR text extraction; delegates OCR to `ocr.py` | ocr, rmc, cairosvg |
| `tools.py` | tool endpoints; delegates OCR to `ocr.py` | ocr |

## Configuration summary

| Env var | Default | Meaning |
|---|---|---|
| `REMARKABLE_OCR_BACKEND` | `auto` | `auto`\|`ollama`\|`sampling`\|`google`\|`tesseract` |
| `REMARKABLE_OLLAMA_MODEL` | `gemma4:31b` | Ollama vision model tag |
| `REMARKABLE_OLLAMA_HOST` | `OLLAMA_HOST` or `http://localhost:11434` | Ollama server base URL |
| `REMARKABLE_OLLAMA_TIMEOUT` | `180` | Per-page generate timeout (s) |

## Error handling / edge cases

- **Ollama down / not installed:** `ollama_available()` is false → `auto` skips it silently;
  explicit `ollama` attempts then falls back, labeling the real backend used.
- **Model not pulled:** `/api/generate` errors → `None` → fallback. README documents `ollama pull`.
- **Long notebooks + 31B:** read path is lazy per page (only the requested page is OCR'd), cached
  per page. Resources/batch path still does all pages (acceptable; different access pattern).
- **Model emits commentary/fences despite prompt:** strip whitespace; rely on prompt + sentinel.
  No heavy post-processing.
- **Event-loop blocking:** async dispatcher offloads the blocking HTTP call via `asyncio.to_thread`.

## Testing & validation

**Unit (CI, no live server — mock `requests.get`/`requests.post`):**
- `ocr_png_ollama`: success/parse; respects model + host; timeout/ConnectionError → `None`;
  `[NO TEXT DETECTED]`/empty → `None`.
- `ollama_available`: true/false on tags reachability; cached; reset hook works.
- Resolution matrix: `auto`→ollama (reachable) / google (key, no ollama) / tesseract (neither);
  explicit `ollama`; explicit `sampling` path unchanged.
- Dispatcher fallback: ollama fails → google → tesseract.
- `extract_handwriting_ocr` returns correct `(list, backend)` via mocked `ocr_png_sync`.
- Update/replace tests that reference deleted helpers.
- **Regression gate: all existing 99 tests still pass.**

**Live smoke test (manual, real Ollama + `gemma4:31b`):**
- Render a real handwritten notebook page and OCR it through local Ollama; paste the actual output
  as end-to-end evidence (per project validation discipline — no success claims without observed output).

## Docs

- README OCR section: add `ollama` backend, env-var table, `ollama pull gemma4:31b`, and the note
  that `auto` now prefers local Ollama when reachable.
- `docs/` may get a short `ollama-setup.md` mirroring `google-vision-setup.md` (optional).

## Rollout / risk

- Pure addition for `auto` users **without** Ollama (resolution falls straight through).
- For `auto` users **with** Ollama running, default OCR engine changes to local Gemma — intended,
  and the headline benefit. Documented in README.
- Largest risk is the engine-dedup refactor in `extract.py`/`tools.py`; mitigated by the
  all-tests-pass regression gate and keeping rendering + control flow untouched.
