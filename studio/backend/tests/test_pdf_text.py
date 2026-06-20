# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the shared PyMuPDF4LLM text-quality guard (``utils.pdf_text``).

These lock in the load-bearing count thresholds and the hand-tuned shaped-
Presentation-Form range (notably the BOM exclusion), which a future tweak could
silently regress. Non-ASCII inputs use ``\\u`` escapes so the exact codepoints
are explicit.
"""

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.pdf_text import (
    _SHAPED_PRESENTATION_FORMS,
    markdown_text_is_corrupted,
    markdown_text_is_incomplete,
    plain_pdf_text,
)


def test_empty_text_is_not_corrupted():
    assert markdown_text_is_corrupted("") is False


def test_plain_latin_with_table_is_not_corrupted():
    assert markdown_text_is_corrupted("# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |") is False


def test_many_shaped_forms_trigger():
    # A mangled RTL run is almost entirely shaped glyphs -> well over the floor.
    assert markdown_text_is_corrupted("\uFB50" * 20) is True            # Arabic-A
    assert markdown_text_is_corrupted("\uFE8E" * 20 + " text") is True  # Arabic-B
    assert markdown_text_is_corrupted("\uFB2A" * 20) is True            # Hebrew


def test_a_few_legitimate_shaped_glyphs_do_not_trigger():
    # Lone presentation forms in clean text are legitimate symbols, not mojibake:
    # U+FDFC RIAL SIGN, U+FDFD BISMILLAH. They must NOT discard the markdown.
    assert markdown_text_is_corrupted("Total: 100 \uFDFC. \uFDFD blessing. Done.") is False
    assert markdown_text_is_corrupted("\uFB50") is False  # a single shaped glyph


def test_typographic_punctuation_is_not_corrupted():
    # Curly quotes / arrow / en-dash are common in clean PDFs and far from the range.
    assert markdown_text_is_corrupted("\u2019\u201C\u201D \u2192 en\u2013dash") is False


def test_replacement_char_below_floor_is_tolerated():
    # <= _PDF_FALLBACK_MIN_BAD_GLYPHS (5) in a short doc stays clean.
    assert markdown_text_is_corrupted("ok " * 50 + "\uFFFD" * 5) is False


def test_replacement_char_above_floor_triggers():
    assert markdown_text_is_corrupted("x" * 100 + "\uFFFD" * 6) is True


def test_replacement_ratio_scales_with_length():
    # ratio 0.0005: a 100k-char doc tolerates ~50 bad glyphs, trips well above it.
    base = "x" * 100_000
    assert markdown_text_is_corrupted(base + "\uFFFD" * 40) is False
    assert markdown_text_is_corrupted(base + "\uFFFD" * 200) is True


def test_shaped_form_regex_range():
    # Lock the detection range independently of the count threshold. The span is
    # FB1D-FDFF (Hebrew + Arabic-A) and FE70-FEFC (Arabic-B); FEFF (BOM) is out.
    assert _SHAPED_PRESENTATION_FORMS.search("\uFB1C") is None      # before range
    assert _SHAPED_PRESENTATION_FORMS.search("\uFB1D") is not None  # first in range
    assert _SHAPED_PRESENTATION_FORMS.search("\uFEFC") is not None  # last in range
    assert _SHAPED_PRESENTATION_FORMS.search("\uFEFD") is None      # past range
    assert _SHAPED_PRESENTATION_FORMS.search("\uFEFF") is None      # BOM excluded


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text


def test_plain_pdf_text_joins_pages_and_strips():
    doc = [_FakePage("  page one  "), _FakePage("page two")]
    assert plain_pdf_text(doc) == "page one  \n\npage two"


def test_plain_pdf_text_empty_doc():
    assert plain_pdf_text([]) == ""


def test_incomplete_when_markdown_far_shorter_than_raw():
    # pymupdf4llm dropped heavy-RTL text: markdown holds far less than get_text.
    plain = "word " * 1000           # ~4000 letters
    assert markdown_text_is_incomplete("word " * 600, plain) is True   # 60%
    assert markdown_text_is_incomplete("word " * 980, plain) is False  # 98%


def test_incomplete_skips_tiny_documents():
    # below the min-letters floor, ratios are noise -> never flag.
    assert markdown_text_is_incomplete("a", "a b c d e") is False
    assert markdown_text_is_incomplete("", "") is False
