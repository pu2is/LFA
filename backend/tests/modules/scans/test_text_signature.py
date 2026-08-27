"""Unit tests for scans/text_signature.py (ADR-0001b D3 fuzzy recovery)."""

from app.modules.scans.text_signature import (
    SIGNATURE_BITS,
    compute_text_signature,
    similarity,
)


def test_compute_text_signature_is_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    assert compute_text_signature(text) == compute_text_signature(text)


def test_compute_text_signature_is_hex_of_expected_width():
    signature = compute_text_signature("Some reasonably long piece of document text.")
    assert signature is not None
    assert len(signature) == SIGNATURE_BITS // 4
    int(signature, 16)  # raises ValueError if not valid hex


def test_compute_text_signature_none_for_text_shorter_than_ngram():
    assert compute_text_signature("hi") is None


def test_compute_text_signature_none_for_empty_text():
    assert compute_text_signature("") is None


def test_similarity_of_identical_signatures_is_one():
    signature = compute_text_signature("Identical content produces identical signatures.")
    assert similarity(signature, signature) == 1.0


_INVOICE_TEXT = """Invoice Number 2024-001. This invoice covers consulting services rendered
during the month of January 2024 for the engineering department. Services included
system architecture review, code quality audits, and mentoring of junior developers.
The total amount due for these services is 1500 EUR, payable within thirty days of
the invoice date. Please remit payment to the account listed below."""

_RECIPE_TEXT = """Recipe for chocolate chip cookies. Cream together butter and sugar until
light and fluffy. Beat in eggs one at a time, then stir in vanilla extract. Combine
flour, baking soda, and salt; gradually blend into the creamed mixture. Fold in
chocolate chips. Drop rounded spoonfuls onto ungreased baking sheets and bake at
375 degrees for about ten minutes until golden brown around the edges."""


def test_similarity_of_near_duplicate_text_is_high():
    # A short, isolated numeric edit on a document-length text -- realistic
    # stand-in for the "light text edits" case ADR-0001b D3 targets.
    lightly_edited = _INVOICE_TEXT.replace("1500 EUR", "1550 EUR")

    score = similarity(compute_text_signature(_INVOICE_TEXT), compute_text_signature(lightly_edited))

    assert score > 0.9


def test_similarity_of_unrelated_text_is_low():
    score = similarity(compute_text_signature(_INVOICE_TEXT), compute_text_signature(_RECIPE_TEXT))

    assert score < 0.9
