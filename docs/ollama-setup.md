# Local OCR with Ollama (Gemma)

reMarkable notebooks are mostly handwriting. The Ollama backend runs a local vision model
(default `gemma4:31b`) so OCR is high quality, free, and fully on-device — no API key, and your
notes never leave your machine.

## Setup

1. Install Ollama: https://ollama.com
2. Pull a vision model:
   ```bash
   ollama pull gemma4:31b      # best accuracy (default)
   # or, for lower latency / RAM:
   ollama pull gemma4:e4b
   ```
3. Make sure the server is running (the desktop app, or `ollama serve`). Verify:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Usage

With the default `REMARKABLE_OCR_BACKEND=auto`, the server uses Ollama whenever its server is
reachable, and otherwise falls back to Google Vision (if `GOOGLE_VISION_API_KEY` is set) and then
Tesseract. To force it explicitly:

```json
{
  "env": {
    "REMARKABLE_OCR_BACKEND": "ollama"
  }
}
```

OCR is triggered by `remarkable_read(..., include_ocr=True)`, `remarkable_image(..., include_ocr=True)`,
and automatically for notebooks that have no typed text. In the read path, only the page you request
is OCR'd (then cached), so a large model stays usable on long notebooks.

## Tuning

| Env var | Default | Notes |
|---------|---------|-------|
| `REMARKABLE_OLLAMA_MODEL` | `gemma4:31b` | `gemma4:e2b` (fastest) … `gemma4:31b` (most accurate). |
| `REMARKABLE_OLLAMA_HOST` | `OLLAMA_HOST` or `http://localhost:11434` | Point at a remote Ollama if not on localhost. |
| `REMARKABLE_OLLAMA_TIMEOUT` | `180` | Raise if a large model is slow to load on the first call. |

> **Note:** the first OCR call after starting Ollama loads the model into memory and can take
> noticeably longer (tens of seconds for `gemma4:31b`); subsequent pages are much faster.
