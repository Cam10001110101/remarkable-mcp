"""EXPERIMENTAL spike: generate native reMarkable v6 pen strokes from text.

Produces real `Line` stroke scene items (selectable/erasable on-device, true pen
tool + pressure) rather than a flat image or font glyph. Uses a single-line
Hershey font for a hand-drawn look, with per-point jitter, baseline wobble, and
variable pressure/width so it reads as handwriting, not computer text.

Status: spike / proof-of-concept. Requires the optional `spike` extra
(`Hershey-Fonts`). The rmscene block scaffold mirrors
`rmscene.scene_stream.simple_text_document`, swapping the text block for one
`SceneLineItemBlock` per stroke.
"""

from __future__ import annotations

import io
import math
import random
from typing import List, Tuple
from uuid import uuid4

from rmscene import scene_items as si
from rmscene import write_blocks
from rmscene.crdt_sequence import CrdtSequenceItem
from rmscene.scene_stream import (
    AuthorIdsBlock,
    MigrationInfoBlock,
    PageInfoBlock,
    SceneGroupItemBlock,
    SceneLineItemBlock,
    SceneTreeBlock,
    TreeNodeBlock,
)
from rmscene.tagged_block_common import CrdtId, LwwValue

# reMarkable v6 canvas: origin near top-centre; x roughly [-702, 702] (1404 wide),
# y increases downward from ~0. Keep a comfortable margin.
X_LEFT = -640.0
Y_TOP = 140.0
X_RIGHT = 640.0

Stroke = List[Tuple[float, float]]  # a polyline in device coordinates


def _hershey_line_segments(line_text: str, font: str):
    """Single-line glyph segments for one line of text, via Hershey-Fonts."""
    from HersheyFonts import HersheyFonts

    h = HersheyFonts()
    h.load_default_font(font)
    return list(h.lines_for_text(line_text))  # [((x1,y1),(x2,y2)), ...]


def _chain_segments(segments, tol: float = 0.01) -> List[Stroke]:
    """Join consecutive Hershey segments that share an endpoint into polylines,
    so each pen-down..pen-up is one continuous stroke (natural, not dotty)."""
    strokes: List[Stroke] = []
    cur: Stroke = []
    last = None
    for (x1, y1), (x2, y2) in segments:
        if last is not None and (abs(x1 - last[0]) > tol or abs(y1 - last[1]) > tol):
            if len(cur) >= 2:
                strokes.append(cur)
            cur = []
        if not cur:
            cur.append((x1, y1))
        cur.append((x2, y2))
        last = (x2, y2)
    if len(cur) >= 2:
        strokes.append(cur)
    return strokes


def _wrap(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        cur = ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
    return lines


def text_to_strokes(
    text: str,
    *,
    font: str = "cursive",
    scale: float = 2.6,
    line_spacing: float = 95.0,
    max_chars: int = 30,
    jitter: float = 0.7,
    slant: float = 0.10,
    seed: int = 0,
) -> List[Stroke]:
    """Lay out `text` as hand-drawn polyline strokes in reMarkable coordinates.

    Hershey glyph space: baseline y=0, caps at negative y, x advances rightward.
    device_x = X_LEFT + (x + slant*-y)*scale ; device_y = baseline + y*scale.
    Adds gaussian jitter + a slow baseline wobble so it doesn't look computerized.
    """
    rng = random.Random(seed)
    strokes: List[Stroke] = []
    baseline = Y_TOP
    for line_text in _wrap(text, max_chars):
        if line_text.strip():
            segs = _hershey_line_segments(line_text, font)
            wobble_phase = rng.uniform(0, math.tau)
            for poly in _chain_segments(segs):
                dev: Stroke = []
                for (hx, hy) in poly:
                    dx = X_LEFT + (hx + slant * (-hy)) * scale
                    dy = baseline + hy * scale
                    # baseline wobble (low freq) + per-point jitter (high freq)
                    dy += 1.6 * math.sin(0.012 * dx + wobble_phase)
                    dx += rng.gauss(0, jitter)
                    dy += rng.gauss(0, jitter)
                    dev.append((dx, dy))
                strokes.append(dev)
        baseline += line_spacing
    return strokes


def strokes_to_lines(
    strokes: List[Stroke],
    *,
    tool: si.Pen = si.Pen.FINELINER_2,
    color: si.PenColor = si.PenColor.BLACK,
    base_width: int = 2,
    base_pressure: int = 120,
    thickness_scale: float = 1.6,
    seed: int = 1,
) -> List[si.Line]:
    """Convert polylines to rmscene Line strokes with varying pressure/width
    (pen accelerates/decelerates → thin-thick-thin) for a non-computerized feel."""
    rng = random.Random(seed)
    lines: List[si.Line] = []
    for stroke in strokes:
        n = len(stroke)
        if n < 2:
            continue
        points: List[si.Point] = []
        for i, (x, y) in enumerate(stroke):
            t = i / max(1, n - 1)
            taper = math.sin(math.pi * t)  # 0 at ends, 1 in middle
            pressure = int(max(20, min(255, base_pressure * (0.55 + 0.45 * taper)
                                       + rng.gauss(0, 8))))
            width = int(max(1, base_width + round(taper)))
            points.append(si.Point(x=float(x), y=float(y), speed=0, direction=0,
                                   width=width, pressure=pressure))
        lines.append(si.Line(color=color, tool=tool, points=points,
                             thickness_scale=thickness_scale, starting_length=0.0))
    return lines


def lines_to_rm_bytes(lines: List[si.Line], *, version: str = "3.2.2") -> bytes:
    """Serialize Line strokes into a native v6 .rm page (one layer, 'Layer 1')."""
    author = uuid4()
    blocks = [
        AuthorIdsBlock(author_uuids={1: author}),
        MigrationInfoBlock(migration_id=CrdtId(1, 1), is_device=True),
        PageInfoBlock(loads_count=1, merges_count=0, text_chars_count=0, text_lines_count=0),
        SceneTreeBlock(
            tree_id=CrdtId(0, 11), node_id=CrdtId(0, 0), is_update=True, parent_id=CrdtId(0, 1)
        ),
    ]
    # One SceneLineItemBlock per stroke, chained left->right in the layer's CRDT sequence.
    prev = CrdtId(0, 0)
    for idx, line in enumerate(lines):
        item_id = CrdtId(1, 16 + idx)
        blocks.append(
            SceneLineItemBlock(
                parent_id=CrdtId(0, 11),
                item=CrdtSequenceItem(
                    item_id=item_id,
                    left_id=prev,
                    right_id=CrdtId(0, 0),
                    deleted_length=0,
                    value=line,
                ),
            )
        )
        prev = item_id
    blocks += [
        TreeNodeBlock(si.Group(node_id=CrdtId(0, 1))),
        TreeNodeBlock(
            si.Group(
                node_id=CrdtId(0, 11),
                label=LwwValue(timestamp=CrdtId(0, 12), value="Layer 1"),
            )
        ),
        SceneGroupItemBlock(
            parent_id=CrdtId(0, 1),
            item=CrdtSequenceItem(
                item_id=CrdtId(0, 13),
                left_id=CrdtId(0, 0),
                right_id=CrdtId(0, 0),
                deleted_length=0,
                value=CrdtId(0, 11),
            ),
        ),
    ]
    buf = io.BytesIO()
    write_blocks(buf, blocks, options={"version": version})
    return buf.getvalue()


def text_to_rm_bytes(text: str, *, tool: si.Pen = si.Pen.FINELINER_2, **kw) -> bytes:
    """Convenience: text -> hand-drawn v6 .rm page bytes."""
    strokes = text_to_strokes(text, **{k: v for k, v in kw.items() if k in {
        "font", "scale", "line_spacing", "max_chars", "jitter", "slant", "seed"}})
    lines = strokes_to_lines(strokes, tool=tool)
    return lines_to_rm_bytes(lines)
