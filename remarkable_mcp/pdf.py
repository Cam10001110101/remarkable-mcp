"""
Shared PDF helpers used by both the SSH and USB-web write backends.

Keeping the blank-notebook generation in one place avoids silent drift between
the two transports: create_notebook in usb_web.py and ssh.py both call
blank_pdf_bytes so a page-size or format change happens once, not twice.
"""

from __future__ import annotations

# reMarkable native page size in points. The device renders at 1404x1872 px
# @ 226 dpi, which is ~497x663 pt. A PDF this size fills the screen 1:1.
RM_PAGE_WIDTH_PT = 497
RM_PAGE_HEIGHT_PT = 663

# Guard rail shared by both backends so a typo can't ask for a million pages.
MAX_NOTEBOOK_PAGES = 200


def blank_pdf_bytes(pages: int = 1) -> bytes:
    """Generate a blank, reMarkable-sized PDF with ``pages`` empty pages.

    We deliberately avoid constructing the binary .rm v6 stroke format (it is
    reverse-engineered and varies across firmware). A blank PDF is annotatable
    with the pen, behaves like any other PDF, and is bullet-proof across
    firmware versions.

    Args:
        pages: Number of blank pages (clamped to at least 1).

    Returns:
        The PDF file as bytes.

    Raises:
        RuntimeError: if PyMuPDF (fitz) is not importable.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "Blank-notebook creation requires PyMuPDF (already a project "
            "dependency). Run: uv sync"
        ) from e

    pdf = fitz.open()
    for _ in range(max(1, pages)):
        pdf.new_page(width=RM_PAGE_WIDTH_PT, height=RM_PAGE_HEIGHT_PT)
    pdf_bytes = pdf.tobytes()
    pdf.close()
    return pdf_bytes
