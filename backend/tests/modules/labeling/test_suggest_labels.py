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
from app.modules.labeling.service import (
    CONFIDENCE_THRESHOLD_DROP,
    LabelCandidate,
    LabelSuggestionOutput,
    suggest_labels,
)
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


def _mock_llm(suggestions: list[tuple[str, float]]) -> MagicMock:
    """Build a mock BaseChatModel whose structured-output chain returns `suggestions`."""
    output = LabelSuggestionOutput(
        labels=[LabelCandidate(name=n, confidence=c) for n, c in suggestions]
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = output

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


# --------------------------------------------------------------------------- #
# Happy-path tests
# --------------------------------------------------------------------------- #

def test_suggest_labels_stores_labels_above_threshold(db, seeded_db):
    file, _ = seeded_db
    # 0.9 → stored; 0.3 → below DROP threshold → discarded
    llm = _mock_llm([("invoice", 0.9), ("contract", 0.3)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1
    assert result[0].source == "llm"
    assert result[0].status == "suggested"
    assert result[0].confidence == 0.9

    rows = db.scalars(select(FileLabel).where(FileLabel.file_id == file.id)).all()
    assert len(rows) == 1


def test_suggest_labels_stores_medium_confidence_label(db, seeded_db):
    file, _ = seeded_db
    # 0.6 is between DROP (0.5) and HIGH (0.75) → stored as "suggested"
    llm = _mock_llm([("report", 0.6)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1
    assert result[0].confidence == 0.6


def test_suggest_labels_drops_all_below_threshold(db, seeded_db):
    file, _ = seeded_db
    llm = _mock_llm([("invoice", 0.2), ("contract", 0.4)])

    result = suggest_labels(db, file.id, llm=llm)

    assert result == []
    rows = db.scalars(select(FileLabel).where(FileLabel.file_id == file.id)).all()
    assert len(rows) == 0


def test_suggest_labels_ignores_unknown_label_names(db, seeded_db):
    file, _ = seeded_db
    # "receipt" is not in the label set
    llm = _mock_llm([("invoice", 0.9), ("receipt", 0.85)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_suggest_labels_deduplicates_repeated_label(db, seeded_db):
    file, _ = seeded_db
    # LLM returns the same label twice
    llm = _mock_llm([("invoice", 0.9), ("invoice", 0.75)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1


def test_suggest_labels_case_insensitive_name_match(db, seeded_db):
    file, _ = seeded_db
    # Labels are stored lowercase; LLM might return "Invoice"
    llm = _mock_llm([("Invoice", 0.85)])

    result = suggest_labels(db, file.id, llm=llm)

    assert len(result) == 1


# --------------------------------------------------------------------------- #
# Early-exit tests (LLM must NOT be called)
# --------------------------------------------------------------------------- #

def test_suggest_labels_returns_empty_when_no_labels_in_db(db, seeded_db):
    file, labels = seeded_db
    # Remove all labels from DB temporarily
    for lbl in labels:
        db.delete(lbl)
    db.commit()

    mock_llm = MagicMock()
    result = suggest_labels(db, file.id, llm=mock_llm)

    assert result == []
    mock_llm.with_structured_output.assert_not_called()

    # Restore for teardown
    db.add_all([Label(name=lbl.name) for lbl in labels])
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
