"""Tests for the MERGE semantics in _merge_candidates (called by suggest_labels).

All tests use a mock LLM so Ollama is never hit. The fixtures build a file with
existing file_label rows in various states, then call suggest_labels with a new
LLM output and assert the correct merge outcome.

Label names are test-scoped (e.g. "inv_t1") to avoid collisions with other test
modules that also insert labels into the shared test database.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from sqlalchemy import delete

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.service import (
    CatalogCandidate,
    FreetextCandidate,
    LabelSuggestionOutput,
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
# Catalog pick merge rules
# --------------------------------------------------------------------------- #

def test_confirmed_catalog_label_is_not_changed(db):
    """A confirmed catalog label must not be touched by re-labeling."""
    # Unique names scoped to this test to avoid DB collisions.
    label_name = "mrg_t1_inv"
    path_str = "/tmp/lfa_merge_t1"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t1.pdf", full_path=f"{path_str}/t1.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t1",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=label_name,
        source="llm", status="confirmed", confidence=0.9,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.5)])
        suggest_labels(db, f.id, llm=llm)

        db.refresh(existing_fl)
        assert existing_fl.status == "confirmed"
        assert existing_fl.confidence == 0.9
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_rejected_catalog_label_is_reset_to_suggested(db):
    """A rejected catalog label must become suggested again when re-suggested."""
    label_name = "mrg_t2_contract"
    path_str = "/tmp/lfa_merge_t2"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t2.pdf", full_path=f"{path_str}/t2.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t2",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=label_name,
        source="llm", status="rejected", confidence=0.4,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.85)])
        suggest_labels(db, f.id, llm=llm)

        db.refresh(existing_fl)
        assert existing_fl.status == "suggested"
        assert existing_fl.confidence == 0.85
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_suggested_catalog_label_confidence_is_updated(db):
    """A still-suggested catalog label gets its confidence updated."""
    label_name = "mrg_t3_report"
    path_str = "/tmp/lfa_merge_t3"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t3.pdf", full_path=f"{path_str}/t3.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t3",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=label_name,
        source="llm", status="suggested", confidence=0.5,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.95)])
        suggest_labels(db, f.id, llm=llm)

        db.refresh(existing_fl)
        assert existing_fl.status == "suggested"
        assert existing_fl.confidence == 0.95
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_new_catalog_candidate_is_inserted_as_suggested(db):
    """A catalog label not previously assigned is inserted as suggested."""
    label_name = "mrg_t4_tax_doc"
    path_str = "/tmp/lfa_merge_t4"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t4.pdf", full_path=f"{path_str}/t4.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t4",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=label_name)
    db.add(lbl)
    db.commit()

    try:
        llm = _mock_llm(catalog=[(label_name, 0.8)])
        result = suggest_labels(db, f.id, llm=llm)

        assert len(result) == 1
        assert result[0].label_id == lbl.id
        assert result[0].status == "suggested"
        assert result[0].confidence == 0.8
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_old_label_not_in_new_candidates_is_kept(db):
    """Labels absent from the new LLM output must not be deleted."""
    lbl_old_name = "mrg_t5_old"
    lbl_new_name = "mrg_t5_new"
    path_str = "/tmp/lfa_merge_t5"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t5.pdf", full_path=f"{path_str}/t5.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t5",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl_old = Label(name=lbl_old_name)
    lbl_new = Label(name=lbl_new_name)
    db.add_all([lbl_old, lbl_new])
    db.flush()

    old_fl = FileLabel(
        file_id=f.id, label_id=lbl_old.id, label_name=lbl_old_name,
        source="llm", status="confirmed", confidence=0.9,
    )
    db.add(old_fl)
    db.commit()

    try:
        # LLM suggests only lbl_new — lbl_old is not mentioned.
        llm = _mock_llm(catalog=[(lbl_new_name, 0.7)])
        suggest_labels(db, f.id, llm=llm)

        from sqlalchemy import select
        all_labels = list(db.scalars(select(FileLabel).where(FileLabel.file_id == f.id)))
        names = {fl.label_name for fl in all_labels}
        assert lbl_old_name in names
        assert lbl_new_name in names
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl_old)
        db.delete(lbl_new)
        db.delete(f)
        db.delete(path)
        db.commit()


# --------------------------------------------------------------------------- #
# Free-text merge rules
# --------------------------------------------------------------------------- #

def test_rejected_freetext_label_is_reset_to_suggested(db):
    """A rejected free-text label becomes suggested again when re-suggested."""
    catalog_label_name = "mrg_t6_catalog"
    freetext_name = "mrg_t6_car_rental"
    path_str = "/tmp/lfa_merge_t6"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t6.pdf", full_path=f"{path_str}/t6.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t6",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl_catalog = Label(name=catalog_label_name)
    db.add(lbl_catalog)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=None, label_name=freetext_name,
        source="llm", status="rejected", confidence=0.6,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_llm(freetext=[(freetext_name, 0.88)])
        suggest_labels(db, f.id, llm=llm)

        db.refresh(existing_fl)
        assert existing_fl.status == "suggested"
        assert existing_fl.confidence == 0.88
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl_catalog)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_confirmed_freetext_label_is_not_changed(db):
    """A confirmed free-text label must not be touched by re-labeling."""
    catalog_label_name = "mrg_t7_catalog"
    freetext_name = "mrg_t7_lease"
    path_str = "/tmp/lfa_merge_t7"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t7.pdf", full_path=f"{path_str}/t7.pdf",
        file_type="pdf", file_size=100, file_hash="mrg_hash_t7",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl_catalog = Label(name=catalog_label_name)
    db.add(lbl_catalog)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=None, label_name=freetext_name,
        source="llm", status="confirmed", confidence=0.7,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_llm(freetext=[(freetext_name, 0.3)])
        suggest_labels(db, f.id, llm=llm)

        db.refresh(existing_fl)
        assert existing_fl.status == "confirmed"
        assert existing_fl.confidence == 0.7
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl_catalog)
        db.delete(f)
        db.delete(path)
        db.commit()
