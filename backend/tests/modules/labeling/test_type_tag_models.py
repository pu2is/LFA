"""Tests for the ADR-0001 foundation models: TypeLabel, TypeLabelFile, TagKind, TagLabel.

Mirrors tests/modules/jobs/test_models.py -- source/status are app-layer
vocabularies guarded by @validates, not DB CHECK constraints (see
docs/03_er-diagram.md), while name/(file, *) uniqueness is a real DB
constraint exercised against the database.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import (
    VALID_LABEL_SOURCES,
    VALID_LABEL_STATUSES,
    TagKind,
    TagLabel,
    TypeLabel,
    TypeLabelFile,
)


@pytest.fixture
def file_id(db):
    path = RegisteredPath(path="/tmp/lfa_type_tag_model_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/tmp/lfa_type_tag_model_test/sample.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="type-tag-model-test",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file.id


# --------------------------------------------------------------------------- #
# App-layer source/status vocabulary (construction-time, no DB needed)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source", sorted(VALID_LABEL_SOURCES))
def test_type_labels_files_accepts_valid_sources(source):
    row = TypeLabelFile(file_id=uuid.uuid4(), type_label_id=uuid.uuid4(), source=source)
    assert row.source == source


@pytest.mark.parametrize("status", sorted(VALID_LABEL_STATUSES))
def test_type_labels_files_accepts_valid_statuses(status):
    row = TypeLabelFile(file_id=uuid.uuid4(), type_label_id=uuid.uuid4(), source="llm", status=status)
    assert row.status == status


def test_type_labels_files_rejects_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        TypeLabelFile(file_id=uuid.uuid4(), type_label_id=uuid.uuid4(), source="bogus")


def test_type_labels_files_rejects_invalid_status_on_reassignment():
    row = TypeLabelFile(file_id=uuid.uuid4(), type_label_id=uuid.uuid4(), source="llm", status="suggested")
    with pytest.raises(ValueError, match="Invalid status"):
        row.status = "not_a_real_status"


@pytest.mark.parametrize("source", sorted(VALID_LABEL_SOURCES))
def test_tag_labels_accepts_valid_sources(source):
    row = TagLabel(file_id=uuid.uuid4(), kind_id=uuid.uuid4(), value="Berlin", source=source)
    assert row.source == source


@pytest.mark.parametrize("status", sorted(VALID_LABEL_STATUSES))
def test_tag_labels_accepts_valid_statuses(status):
    row = TagLabel(file_id=uuid.uuid4(), kind_id=uuid.uuid4(), value="Berlin", source="llm", status=status)
    assert row.status == status


def test_tag_labels_rejects_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        TagLabel(file_id=uuid.uuid4(), kind_id=uuid.uuid4(), value="Berlin", source="bogus")


def test_tag_labels_rejects_invalid_status_on_reassignment():
    row = TagLabel(file_id=uuid.uuid4(), kind_id=uuid.uuid4(), value="Berlin", source="llm", status="suggested")
    with pytest.raises(ValueError, match="Invalid status"):
        row.status = "not_a_real_status"


# --------------------------------------------------------------------------- #
# DB-level UNIQUE constraints
# --------------------------------------------------------------------------- #

def test_type_labels_name_unique(db):
    db.add(TypeLabel(name="invoice"))
    db.commit()

    db.add(TypeLabel(name="invoice"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_tag_kinds_name_unique(db):
    db.add(TagKind(name="person"))
    db.commit()

    db.add(TagKind(name="person"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_type_labels_files_unique_file_and_type(db, file_id):
    type_label = TypeLabel(name="contract")
    db.add(type_label)
    db.commit()
    db.refresh(type_label)

    db.add(TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm"))
    db.commit()

    # Same (file_id, type_label_id) pair again -- even with a different source/status.
    db.add(TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="user", status="confirmed"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_tag_labels_unique_file_kind_and_value(db, file_id):
    kind = TagKind(name="place")
    db.add(kind)
    db.commit()
    db.refresh(kind)

    db.add(TagLabel(file_id=file_id, kind_id=kind.id, value="Berlin", source="llm"))
    db.commit()

    # Same (file_id, kind_id, value) again is rejected...
    db.add(TagLabel(file_id=file_id, kind_id=kind.id, value="Berlin", source="user", status="confirmed"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    # ...but a different value under the same (file_id, kind_id) is fine.
    db.add(TagLabel(file_id=file_id, kind_id=kind.id, value="Munich", source="llm"))
    db.commit()
