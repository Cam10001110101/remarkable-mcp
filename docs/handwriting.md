# Handwriting strokes (experimental)

Write **native reMarkable v6 pen strokes** to the tablet from text — real, selectable/erasable
strokes (true pen tool + pressure), not a flat image or typed font. Built on `rmscene` (already a
dependency); the single-line font needs the optional `spike` extra:

```bash
uv pip install Hershey-Fonts   # or: uv sync --extra spike
```

Modules: `remarkable_mcp/handwriting.py` (text → strokes → v6 `.rm`), `SSHClient.create_rm_notebook`
(packages `.rm` pages into a native notebook), `scripts/handwrite_spike.py`, `scripts/pen_sample.py`.

## Pen palette (1:1 with reMarkable tools)

`PEN_PRESETS` maps friendly names to reMarkable's actual v6 tools (`rmscene.scene_items.Pen`, the
`_2` variants the device uses today) with per-tool ink defaults:

| name | reMarkable tool | use |
|---|---|---|
| `fineliner` | Fineliner | **default** — uniform, legible handwriting |
| `ballpoint` | Ballpoint | solid, pressure-responsive |
| `marker` | Marker | broad, bold |
| `pencil` | Pencil | grainy/textured |
| `mechanical_pencil` | Mechanical pencil | fine grain |
| `paintbrush` | Paintbrush | pressure/speed dynamics |
| `calligraphy` | Calligraphy | angle-dependent width |
| `highlighter` | Highlighter | translucent emphasis (not for writing prose) |

```python
from remarkable_mcp.handwriting import text_to_rm_bytes
rm = text_to_rm_bytes("Hi Cam!", pen="fineliner", font="cursive")   # font: cursive/scripts/futural/...
```

Survey every pen on-device (one notebook = **one** restart):

```bash
uv run --extra spike python scripts/pen_sample.py --local            # render PNGs to /tmp
op run -- uv run --extra spike python scripts/pen_sample.py --push --folder "Scratch Pad"
```

> Note: the local render (`extract.render_rm_file_to_png`) only approximates color/width/highlighter;
> true pen **texture/opacity/pressure** only shows on the device.

## A note on the "reboot" when saving

Writing a native notebook (or any SSH write: upload/create/move/delete) ends with
`systemctl restart xochitl` so the change appears. This is a **brief (~3s) UI restart**, not a full
device reboot — xochitl caches its document database in memory, and reMarkable provides no
signal/dbus reload, so a restart is unavoidable for filesystem-direct writes (the USB web interface
avoids it but can't create native notebooks — PDF/EPUB only).

Mitigations in this repo:
- **Coalesce:** `create_rm_notebook(name, rm_pages=[...])` writes any number of pages in **one**
  restart (e.g. the whole pen survey).
- **Self-heal:** `SSHClient._restart_xochitl` detects systemd's start-limit ("start request repeated
  too quickly") and runs `reset-failed` + start, preventing the failed-state freeze / watchdog reboot
  that a rapid burst of writes could otherwise trigger.
