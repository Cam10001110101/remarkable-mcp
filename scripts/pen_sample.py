#!/usr/bin/env python
"""Survey the reMarkable pen/tool palette.

`--local` renders each pen's sample page to a /tmp PNG (color/width/highlighter
preview only — the device shows true texture/opacity). `--push` writes ONE
multi-page "Pen palette" notebook to the tablet (a single xochitl restart for all
pens) so you can see each tool's real look on-device.

    uv run --extra spike python scripts/pen_sample.py --local
    op run -- uv run --extra spike python scripts/pen_sample.py --push --folder "Scratch Pad"
"""

import os
import sys

# macOS: re-exec with DYLD set so cairosvg can load libcairo for --local render.
if sys.platform == "darwin" and os.environ.get("RX_REEXEC") != "1":
    _l = "/opt/homebrew/lib"
    _e = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _l not in _e.split(os.pathsep):
        os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(p for p in (_l, _e) if p)
        os.environ["RX_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse  # noqa: E402
import io  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402

from rmscene import read_blocks  # noqa: E402
from rmscene.scene_stream import SceneLineItemBlock  # noqa: E402

from remarkable_mcp.handwriting import PEN_PRESETS, pen_sample_pages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", action="store_true", help="render each pen page to /tmp PNG")
    ap.add_argument("--push", action="store_true", help="push a multi-page palette notebook")
    ap.add_argument("--folder", default=None, help="destination folder name (for --push)")
    ap.add_argument("--name", default="Pen palette", help="document title (for --push)")
    args = ap.parse_args()
    if not (args.local or args.push):
        args.local = True

    pens = list(PEN_PRESETS)
    pages = pen_sample_pages(pens)
    print(f"generated {len(pages)} pen pages: {pens}")

    if args.local:
        from remarkable_mcp.extract import render_rm_file_to_png

        for name, data in zip(pens, pages):
            nlines = sum(
                1 for b in read_blocks(io.BytesIO(data)) if isinstance(b, SceneLineItemBlock)
            )
            with tempfile.NamedTemporaryFile(suffix=".rm", delete=False) as t:
                t.write(data)
                rm_path = pathlib.Path(t.name)
            png = render_rm_file_to_png(rm_path, background_color="#FFFFFF")
            out = f"/tmp/pen_{name}.png"
            if png:
                pathlib.Path(out).write_bytes(png)
                print(f"  {name}: {nlines} strokes -> {out} ({len(png)} bytes)")
            else:
                print(f"  {name}: {nlines} strokes -> RENDER FAILED")

    if args.push:
        from remarkable_mcp.ssh import create_ssh_client

        client = create_ssh_client()
        parent_id = ""
        if args.folder:
            match = [
                d for d in client.get_meta_items()
                if d.is_folder and d.VissibleName == args.folder
            ]
            if not match:
                print(f"folder '{args.folder}' not found", file=sys.stderr)
                return 1
            parent_id = match[0].ID
        result = client.create_rm_notebook(args.name, pages, parent_id=parent_id)
        print("pushed palette notebook:", result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
