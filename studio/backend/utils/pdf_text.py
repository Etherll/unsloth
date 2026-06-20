# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared PyMuPDF4LLM text-quality guard.

pymupdf4llm rebuilds text from positioned glyphs, which mangles complex-shaping
scripts (RTL Arabic/Hebrew -> shaped Presentation Forms, Indic -> dropped matras
as U+FFFD). When its Markdown trips these signals, callers fall back to PyMuPDF's
logical-order ``get_text()``. Used by the chat document extractor and the data
recipe seed extractor so both stay correct on non-Latin documents.
"""

from __future__ import annotations

import re
from typing import Any

# A few legitimately-encoded shaped glyphs or replacement chars shouldn't trip
# the fallback; require the count to clear this floor (or ratio of total length).
_PDF_FALLBACK_MIN_BAD_GLYPHS = 5
_PDF_FALLBACK_BAD_GLYPH_RATIO = 0.0005
# One contiguous span FB1D-FDFF covers Hebrew + Arabic Presentation Forms-A;
# FE70-FEFC covers Arabic Presentation Forms-B. The B range stops at FEFC, so
# FEFF (BOM) is left out -- a stray mark alone must not force the fallback.
_SHAPED_PRESENTATION_FORMS = re.compile("[\uFB1D-\uFDFF\uFE70-\uFEFC]")


def markdown_text_is_corrupted(text: str) -> bool:
    """True when pymupdf4llm's glyph reconstruction has mangled the text layer.

    Two signals, each gated by a small floor/ratio so a stray legitimate glyph
    (e.g. a lone U+FDFD Bismillah or U+FDFC Rial sign) can't trip the fallback:
    shaped Presentation Forms (RTL Arabic/Hebrew that pymupdf4llm emits as
    visual-order shaped glyphs instead of logical base letters) and U+FFFD
    replacement characters (dropped Indic combining marks / math glyphs).
    PyMuPDF's ``get_text()`` returns logical Unicode and avoids both, so it is
    the fallback when either count clears the threshold.
    """
    if not text:
        return False
    threshold = max(
        _PDF_FALLBACK_MIN_BAD_GLYPHS,
        _PDF_FALLBACK_BAD_GLYPH_RATIO * len(text),
    )
    shaped = len(_SHAPED_PRESENTATION_FORMS.findall(text))
    replacements = text.count("\uFFFD")
    return shaped > threshold or replacements > threshold


def plain_pdf_text(doc: Any) -> str:
    """Logical-order text via PyMuPDF ``get_text()``; the corruption-safe fallback."""
    return "\n\n".join(page.get_text() for page in doc).strip()


# pymupdf4llm can also silently drop text (not just corrupt it) on heavy-RTL
# pages, keeping only a fraction of the logical text. Fall back when its Markdown
# holds far less than the raw layer. This is a coarse guard: it catches large
# drops (well under this ratio) but not partial ones near it.
_PDF_INCOMPLETE_RATIO = 0.75
_PDF_INCOMPLETE_MIN_LETTERS = 200


def markdown_text_is_incomplete(markdown: str, plain: str) -> bool:
    """True when ``markdown`` holds far fewer letters than ``plain`` (PyMuPDF's
    ``get_text()`` output) -- a coarse length check, not a completeness guarantee.
    It catches heavy-RTL pages pymupdf4llm silently drops without the shaped
    glyphs ``markdown_text_is_corrupted`` flags."""
    plain_letters = sum(1 for c in plain if c.isalnum())
    if plain_letters < _PDF_INCOMPLETE_MIN_LETTERS:
        return False
    markdown_letters = sum(1 for c in markdown if c.isalnum())
    return markdown_letters < _PDF_INCOMPLETE_RATIO * plain_letters
