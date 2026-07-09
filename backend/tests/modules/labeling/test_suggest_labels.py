"""Unit tests for labeling.service.suggest_labels().

All tests use a MagicMock LLM — Ollama is never called.
The mock wires up: llm.with_structured_output(...).invoke(...) → LabelSuggestionOutput.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import delete, select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.prompts import (
    CatalogCandidate,
    FreetextCandidate,
    LabelSuggestionOutput,
)
from app.modules.labeling.suggestion import suggest_labels
from app.modules.rag.models import FileChunk


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def seeded_db(db):
    """One file with 2 chunks and 3 labels in the DB."""
    path = RegisteredPath(path="/tmp/lfa_label_test")
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id,
        filename="invoice.pdf",
        full_path="/tmp/lfa_label_test/invoice.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="deadbeef1234",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(f)
    db.flush()

    db.add_all(
        [
            FileChunk(file_id=f.id, chunk_index=0, content="Invoice No. 123, Total: €500"),
            FileChunk(file_id=f.id, chunk_index=1, content="Payment due within 30 days."),
        ]
    )

    labels = [Label(name=n) for n in ["invoice", "contract", "report"]]
    db.add_all(labels)
    db.commit()

    yield f, labels

    # Teardown: FK order — file_labels before labels, chunks and file before path.
    db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
    db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
    for lbl in labels:
        db.delete(lbl)
    db.delete(f)
    db.delete(path)
    db.commit()


def _mock_llm(
    catalog: list[tuple[str, float]] | None = None,
    freetext: list[tuple[str, float]] | None = None,
) -> MagicMock:
    """Build a mock BaseChatModel whose structured-output chain returns the given picks."""
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
# Happy-path tests
# --------------------------------------------------------------------------- #

def test_suggest_labels_stores_catalog_picks(db, seeded_db):
    file, _ = seeded_db
    # No server-side confidence filtering -- both are stored regardless of confidence
    llm = _mock_llm(catalog=[("invoice", 0.9), ("contract", 0.3)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 2
    assert all(fl.source == "llm" for fl in result)
    assert all(fl.status == "suggested" for fl in result)
    assert all(fl.label_id is not None for fl in result)


def test_suggest_labels_stores_medium_confidence_label(db, seeded_db):
    file, _ = seeded_db
    # Below HIGH (0.75) → still stored, just not UI-preselected
    llm = _mock_llm(catalog=[("report", 0.6)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1
    assert result[0].confidence == 0.6


def test_suggest_labels_stores_freetext_suggestions(db, seeded_db):
    """LLM-invented labels (not in catalog) are stored with label_id=None."""
    file, _ = seeded_db
    llm = _mock_llm(catalog=[("invoice", 0.9)], freetext=[("car_rental_agreement", 0.8)])

    result = suggest_labels(db, file.id, llm=llm)

    catalog_picks = [fl for fl in result if fl.label_id is not None]
    freetext_picks = [fl for fl in result if fl.label_id is None]
    assert len(catalog_picks) == 1
    assert len(freetext_picks) == 1
    assert freetext_picks[0].label_name == "car_rental_agreement"


def test_suggest_labels_drops_unknown_catalog_names(db, seeded_db):
    """catalog_picks whose names are not in the label table are silently dropped."""
    file, _ = seeded_db
    # "receipt" is not in the label catalog → dropped from catalog_picks path
    llm = _mock_llm(catalog=[("invoice", 0.9), ("receipt", 0.85)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1
    assert result[0].label_name == "invoice"


def test_suggest_labels_deduplicates_repeated_label(db, seeded_db):
    file, _ = seeded_db
    # LLM returns the same catalog label twice → only one row inserted
    llm = _mock_llm(catalog=[("invoice", 0.9), ("invoice", 0.75)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1


def test_suggest_labels_case_insensitive_name_match(db, seeded_db):
    file, _ = seeded_db
    # Labels are stored lowercase; LLM might return "Invoice"
    llm = _mock_llm(catalog=[("Invoice", 0.85)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1


def test_suggest_labels_freetext_skipped_if_matches_catalog_name(db, seeded_db):
    """A free_suggestion whose name matches an existing catalog label is dropped."""
    file, _ = seeded_db
    # "invoice" exists as a catalog label → free-text path skips it (avoid duplication)
    llm = _mock_llm(freetext=[("invoice", 0.95)])

    result = suggest_labels(db, file.id, llm=llm)

    assert result == []


# --------------------------------------------------------------------------- #
# Early-exit and edge-case tests
# --------------------------------------------------------------------------- #

def test_suggest_labels_auto_populates_presets_when_catalog_empty(db):
    """When the label catalog is empty, all presets are inserted and the LLM is called."""
    path = RegisteredPath(path="/tmp/lfa_auto_preset_test")
    db.add(path)
    db.flush()
    f = File(
        path_id=path.id,
        filename="test.pdf",
        full_path="/tmp/lfa_auto_preset_test/test.pdf",
        file_type="pdf",
        file_size=500,
        file_hash="abc123preset",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="Test content for labeling"))
    db.commit()

    assert db.scalars(select(Label)).first() is None

    llm = _mock_llm(catalog=[])  # LLM returns nothing — we only care that it was called
    suggest_labels(db, f.id, llm=llm)

    llm.with_structured_output.assert_called()
    preset_labels = db.scalars(select(Label)).all()
    assert len(preset_labels) > 0

    db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
    db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
    for lbl in preset_labels:
        db.delete(lbl)
    db.delete(f)
    db.delete(path)
    db.commit()


def test_suggest_labels_returns_empty_when_no_chunks(db):
    path = RegisteredPath(path="/tmp/no_chunks_test")
    db.add(path)
    db.flush()
    f = File(
        path_id=path.id,
        filename="empty.pdf",
        full_path="/tmp/no_chunks_test/empty.pdf",
        file_type="pdf",
        file_size=0,
        file_hash="000abc",
        file_modified_at=datetime.now(timezone.utc),
    )
    label = Label(name="invoice")
    db.add(f)
    db.add(label)
    db.commit()

    mock_llm = MagicMock()
    result = suggest_labels(db, f.id, llm=mock_llm)

    assert result == []
    mock_llm.with_structured_output.assert_not_called()

    db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
    db.delete(label)
    db.delete(f)
    db.delete(path)
    db.commit()
