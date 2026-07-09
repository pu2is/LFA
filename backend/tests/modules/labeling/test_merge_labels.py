"""Tests for initial-mode write_initial_candidates and augment-mode append_augment_candidates
(app.modules.labeling.merge), exercised through suggest_labels / suggest_labels_augment.

Initial mode: pure INSERT (no existing rows expected).
Augment mode: append-only — confirmed/rejected/suggested NEVER touched;
only genuinely new names are inserted.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from sqlalchemy import delete, select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.prompts import (
    AugmentCandidate,
    AugmentSuggestionOutput,
    CatalogCandidate,
    FreetextCandidate,
    LabelSuggestionOutput,
)
from app.modules.labeling.service import normalize_label_name
from app.modules.labeling.suggestion import suggest_labels, suggest_labels_augment
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


def _mock_augment_llm(names: list[str]) -> MagicMock:
    output = AugmentSuggestionOutput(
        new_labels=[AugmentCandidate(name=n) for n in names],
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = output
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


# --------------------------------------------------------------------------- #
# Initial mode: write_initial_candidates
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


# --------------------------------------------------------------------------- #
# Augment mode: append_augment_candidates — append-only semantics
# --------------------------------------------------------------------------- #

def test_augment_appends_new_label(db):
    """Augment inserts a genuinely new label name."""
    catalog_name = "aug_t1_inv"
    path_str = "/tmp/lfa_aug_t1"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t1.pdf", full_path=f"{path_str}/t1.pdf",
        file_type="pdf", file_size=100, file_hash="aug_hash_t1",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=catalog_name)
    db.add(lbl)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=catalog_name,
        source="llm", status="confirmed", confidence=0.9,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_augment_llm(["aug_t1_new_angle"])
        result = suggest_labels_augment(db, f.id, llm=llm)

        assert len(result) == 1
        assert result[0].label_name == "aug_t1_new_angle"
        assert result[0].confidence is None
        assert result[0].source == "llm"
        assert result[0].status == "suggested"
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_augment_skips_existing_names(db):
    """Augment must NOT insert a label whose name already exists on the file."""
    catalog_name = "aug_t2_inv"
    path_str = "/tmp/lfa_aug_t2"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t2.pdf", full_path=f"{path_str}/t2.pdf",
        file_type="pdf", file_size=100, file_hash="aug_hash_t2",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=catalog_name)
    db.add(lbl)
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=catalog_name,
        source="llm", status="confirmed", confidence=0.9,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_augment_llm([catalog_name])
        result = suggest_labels_augment(db, f.id, llm=llm)
        assert len(result) == 0

        all_fl = list(db.scalars(select(FileLabel).where(FileLabel.file_id == f.id)))
        assert len(all_fl) == 1
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_augment_rejected_stays_rejected(db):
    """A rejected label must NEVER be flipped back to suggested by augment."""
    catalog_name = "aug_t3_catalog"
    rejected_name = "aug_t3_bad"
    path_str = "/tmp/lfa_aug_t3"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t3.pdf", full_path=f"{path_str}/t3.pdf",
        file_type="pdf", file_size=100, file_hash="aug_hash_t3",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=catalog_name)
    db.add(lbl)
    db.flush()

    rejected_fl = FileLabel(
        file_id=f.id, label_id=None, label_name=rejected_name,
        source="llm", status="rejected", confidence=0.6,
    )
    db.add(rejected_fl)
    db.commit()

    try:
        llm = _mock_augment_llm([rejected_name, "aug_t3_genuinely_new"])
        result = suggest_labels_augment(db, f.id, llm=llm)

        assert len(result) == 1
        assert result[0].label_name == "aug_t3_genuinely_new"

        db.refresh(rejected_fl)
        assert rejected_fl.status == "rejected"
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_augment_confirmed_stays_confirmed(db):
    """Confirmed labels are untouched by augment — only new names are appended."""
    catalog_name = "aug_t4_inv"
    path_str = "/tmp/lfa_aug_t4"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t4.pdf", full_path=f"{path_str}/t4.pdf",
        file_type="pdf", file_size=100, file_hash="aug_hash_t4",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl = Label(name=catalog_name)
    db.add(lbl)
    db.flush()

    confirmed_fl = FileLabel(
        file_id=f.id, label_id=lbl.id, label_name=catalog_name,
        source="llm", status="confirmed", confidence=0.9,
    )
    db.add(confirmed_fl)
    db.commit()

    try:
        llm = _mock_augment_llm([catalog_name, "aug_t4_extra"])
        result = suggest_labels_augment(db, f.id, llm=llm)

        assert len(result) == 1
        assert result[0].label_name == "aug_t4_extra"

        db.refresh(confirmed_fl)
        assert confirmed_fl.status == "confirmed"
        assert confirmed_fl.confidence == 0.9
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl)
        db.delete(f)
        db.delete(path)
        db.commit()


def test_augment_catalog_match_sets_label_id(db):
    """When augment suggests a name that matches the catalog, label_id is set."""
    existing_name = "aug_t5_old"
    catalog_name = "aug_t5_contract"
    path_str = "/tmp/lfa_aug_t5"

    path = RegisteredPath(path=path_str)
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id, filename="t5.pdf", full_path=f"{path_str}/t5.pdf",
        file_type="pdf", file_size=100, file_hash="aug_hash_t5",
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(f)
    db.flush()
    db.add(FileChunk(file_id=f.id, chunk_index=0, content="content"))

    lbl_old = Label(name=existing_name)
    lbl_cat = Label(name=catalog_name)
    db.add_all([lbl_old, lbl_cat])
    db.flush()

    existing_fl = FileLabel(
        file_id=f.id, label_id=lbl_old.id, label_name=existing_name,
        source="llm", status="suggested", confidence=0.5,
    )
    db.add(existing_fl)
    db.commit()

    try:
        llm = _mock_augment_llm([catalog_name])
        result = suggest_labels_augment(db, f.id, llm=llm)

        assert len(result) == 1
        assert result[0].label_id == lbl_cat.id
        assert result[0].label_name == catalog_name
    finally:
        db.execute(delete(FileLabel).where(FileLabel.file_id == f.id))
        db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
        db.delete(lbl_old)
        db.delete(lbl_cat)
        db.delete(f)
        db.delete(path)
        db.commit()
