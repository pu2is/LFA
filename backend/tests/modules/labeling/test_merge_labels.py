"""Tests for initial-mode _write_initial_candidates and augment-mode _append_augment_candidates.

Initial mode: pure INSERT (no existing rows expected).
Augment mode: append-only — confirmed/rejected/suggested NEVER touched;
only genuinely new names are inserted.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from sqlalchemy import delete, select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.service import (
    CatalogCandidate,
    FreetextCandidate,
    LabelSuggestionOutput,
    normalize_label_name,
    suggest_labels,
)
from app.modules.rag.models import FileChunk


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mock_llm(
    catalog: list[tuple[str, float]] | None = None,
    freetext: list[tuple[str, float]] | None = None,
) -> MagicMock:
    output = LabelSuggestionOutput(
        catalog_picks=[CatalogCandidate(name=n, confidence=c) for n, c in (catalog or [])],
        free_suggestions=[FreetextCandidate(name=n, confidence=c) for n, c in (freetext or [])],
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = output
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


# --------------------------------------------------------------------------- #
# Initial mode: _write_initial_candidates
# --------------------------------------------------------------------------- #

def test_initial_inserts_catalog_and_freetext(db):
    label_name = "wrt_t1_inv"
    path_str = "/tmp/lfa_write_t1"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t1.pdf", full_path=f"{path_str}/t1.pdf",
        file_type="pdf", file_size=100, file_hash="wrt_hash_t1",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.9)], freetext=[("wrt_t1_custom", 0.7)])
        result = suggest_labels(db, f.id, llm=llm)

        catalog_picks = [fl for fl in result if fl.label_id is not None]
        freetext_picks = [fl for fl in result if fl.label_id is None]
        assert len(catalog_picks) == 1
        assert len(freetext_picks) == 1
        assert catalog_picks[0].label_name == label_name
        assert freetext_picks[0].label_name == "wrt_t1_custom"
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_initial_deduplicates_repeated_catalog_pick(db):
    label_name = "wrt_t2_report"
    path_str = "/tmp/lfa_write_t2"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t2.pdf", full_path=f"{path_str}/t2.pdf",
        file_type="pdf", file_size=100, file_hash="wrt_hash_t2",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.9), (label_name, 0.5)])
        result = suggest_labels(db, f.id, llm=llm)
        assert len(result) == 1
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()



# --------------------------------------------------------------------------- #
# Normalization consistency
# --------------------------------------------------------------------------- #

def test_normalize_label_name_replaces_spaces_with_underscores():
    assert normalize_label_name("Bank Statement") == "bank_statement"
    assert normalize_label_name("  Tax  ") == "tax"
    assert normalize_label_name("INVOICE") == "invoice"
    assert normalize_label_name("car_rental_agreement") == "car_rental_agreement"
