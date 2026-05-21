#!/usr/bin/env python
"""EXPERIMENTAL spike: write a hand-drawn note (native v6 pen strokes) to the tablet.

Generates real reMarkable strokes from text (single-line Hershey font + jitter/
pressure) and creates a native .rm notebook on the device — so the result is
selectable/erasable pen strokes, not a flat image or typed font.

Requires SSH mode env (REMARKABLE_SSH_HOST/PASSWORD) and the `spike` extra
(`uv pip install Hershey-Fonts`). On macOS, run under the cairo-aware wrapper if
you also render; this script only writes, so SSH env is enough:

    REMARKABLE_SSH_HOST=... REMARKABLE_SSH_PASSWORD='op://...' \
        op run -- uv run python scripts/handwrite_spike.py "Hi Cam!" --folder "Scratch Pad"
"""

import argparse
import sys

from rmscene import scene_items as si

from remarkable_mcp.handwriting import text_to_rm_bytes
from remarkable_mcp.ssh import create_ssh_client

TOOLS = {
    "fineliner": si.Pen.FINELINER_2,
    "ballpoint": si.Pen.BALLPOINT_2,
    "pencil": si.Pen.PENCIL_2,
    "marker": si.Pen.MARKER_2,
    "highlighter": si.Pen.HIGHLIGHTER_2,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", help="message to write in handwriting")
    ap.add_argument("--name", default="Claude (handwritten)", help="document title")
    ap.add_argument("--folder", default=None, help="destination folder name (default: root)")
    ap.add_argument("--font", default="cursive", help="Hershey font (cursive/scripts/futural/...)")
    ap.add_argument("--tool", default="fineliner", choices=list(TOOLS))
    args = ap.parse_args()

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

    rm_bytes = text_to_rm_bytes(args.text, font=args.font, tool=TOOLS[args.tool])
    print(f"generated {len(rm_bytes)} .rm bytes ({args.tool}, font={args.font})")
    result = client.create_rm_notebook(args.name, [rm_bytes], parent_id=parent_id)
    print("created native notebook:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
