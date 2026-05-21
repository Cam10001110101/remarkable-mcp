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
| `REMARKABLE_OLLAMA_MODEL` | `gemma4:31b` | See model-fidelity note below. |
| `REMARKABLE_OLLAMA_HOST` | `REMARKABLE_OLLAMA_HOST` → `OLLAMA_HOST` → `http://localhost:11434` | Local or remote Ollama; a bare `host:port` is accepted (scheme added automatically). |
| `REMARKABLE_OLLAMA_TIMEOUT` | `180` | Per-page timeout (s). Raise for big models on slower hosts — e.g. `qwen3-vl:32b` ran ~230 s/page in testing. |
| `REMARKABLE_OLLAMA_TEMPERATURE` | `0` | Sampling temperature. `0` (deterministic) is correct for OCR; only raise it if a specific model misbehaves at 0. |

## Using a remote Ollama host (swap)

Point the server at any reachable Ollama instance — your laptop, a GPU box, etc. It's a one-line
change in your MCP config's `env` block, then reconnect:

```jsonc
// local (default)
"REMARKABLE_OLLAMA_HOST": "http://localhost:11434"

// remote box
"REMARKABLE_OLLAMA_HOST": "http://192.168.1.50:11434",
"REMARKABLE_OLLAMA_MODEL": "qwen3-vl:32b",
"REMARKABLE_OLLAMA_TIMEOUT": "360"
```

> **Reconnect required:** the server probes Ollama reachability once per process and caches it, so
> after changing the host you must restart/reconnect the MCP server (e.g. `/mcp` in Claude Code).

## Model fidelity (handwriting)

For **handwriting**, a strong general vision model is the most *faithful* reader. `qwen3-vl`
(`qwen3-vl:32b` locally, or `qwen3-vl:235b-cloud`) transcribed real handwritten pages accurately in
testing. Plain LLMs without vision (`kimi`, base `qwen3.5`, `nemotron`) **cannot** OCR — they ignore
the image and hallucinate. Dedicated document-OCR models (`deepseek-ocr`, `glm-ocr`) target printed
scans and tend to hallucinate on freehand handwriting. `gemma4:31b` works but has been observed to
confidently *invent* text on hard/messy pages, so verify against the page when accuracy matters.

> **Note:** the first OCR call after starting Ollama loads the model into memory and can take
> noticeably longer (tens of seconds, or minutes for very large models); subsequent pages are faster.
