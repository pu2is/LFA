"""Tests for POST /files/{file_id}/label-review (#68, ADR-0001b D4): the
keep/drop decision on a file whose content changed under an existing set of
labels. Includes the issue's required regression test for the
drop -> re-label -> initial-mode-routing chain (must not regress into #61)."""
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.scans.models import FileEvent
from tests.conftest import mock_rq_job


@pytest.fixture
def registered_path(db: Session) -> RegisteredPath:
    path = RegisteredPath(path="D:/lfa_test_label_review")
    db.add(path)
    db.commit()
    db.refresh(path)
    return path


def _needs_review_file(db: Session, *, path_id: uuid.UUID, status: str = "ready") -> File:
    file = File(
        path_id=path_id, filename="f.pdf", full_path="D:/lfa_test_label_review/f.pdf",
        file_type="pdf", file_size=1, file_hash="new-hash",
        file_modified_at=datetime.now(timezone.utc), status=status, labels_need_review=True,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def _make_rescan_job(db: Session) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def test_label_review_unknown_file_returns_404(client):
    resp = client.post(f"/files/{uuid.uuid4()}/label-review", json={"action": "keep"})
    assert resp.status_code == 404


def test_label_review_returns_409_when_nothing_pending(client, db, registered_path):
    file = _needs_review_file(db, path_id=registered_path.id)
    file.labels_need_review = False
    db.commit()

    resp = client.post(f"/files/{file.id}/label-review", json={"action": "keep"})

    assert resp.status_code == 409


def test_label_review_keep_only_clears_the_flag(client, db, registered_path):
    file = _needs_review_file(db, path_id=registered_path.id)
    type_label = TypeLabel(name="invoice")
    db.add(type_label)
    db.flush()
    db.add(TypeLabelFile(file_id=file.id, type_label_id=type_label.id, source="user", status="confirmed"))
    db.commit()

    resp = client.post(f"/files/{file.id}/label-review", json={"action": "keep"})

    assert resp.status_code == 200
    assert resp.json()["file"]["id"] == str(file.id)

    db.refresh(file)
    assert file.labels_need_review is False
    assert list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file.id)))


def test_label_review_surfaces_from_hash_to_hash_of_the_triggering_event(client, db, registered_path):
    file = _needs_review_file(db, path_id=registered_path.id)
    scan_job = _make_rescan_job(db)
    db.add(FileEvent(
        scan_id=scan_job.id, file_id=file.id, event_type="modified",
        from_path=file.full_path, to_path=file.full_path, from_hash="old-hash", to_hash="new-hash",
    ))
    db.commit()

    resp = client.post(f"/files/{file.id}/label-review", json={"action": "keep"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["from_hash"] == "old-hash"
    assert body["to_hash"] == "new-hash"


def test_label_review_drop_clears_all_associations_including_rejected(client, db, registered_path):
    file = _needs_review_file(db, path_id=registered_path.id)
    type_label = TypeLabel(name="invoice")
    tag_kind = TagKind(name="person")
    db.add_all([type_label, tag_kind])
    db.flush()
    db.add(TypeLabelFile(file_id=file.id, type_label_id=type_label.id, source="user", status="rejected"))
    db.add(TagLabel(file_id=file.id, kind_id=tag_kind.id, value="Alice", source="llm", status="confirmed"))
    db.commit()

    resp = client.post(f"/files/{file.id}/label-review", json={"action": "drop"})

    assert resp.status_code == 200
    db.refresh(file)
    assert file.labels_need_review is False
    assert not list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file.id)))
    assert not list(db.scalars(select(TagLabel).where(TagLabel.file_id == file.id)))


@patch("app.modules.labeling.routes.label_queue")
def test_drop_then_relabel_routes_back_to_initial_mode(mock_q, client, db, registered_path):
    """Regression test (issue #68 AC): a file dropped via label-review must
    have *zero* type_labels_files/tag_labels rows left -- including the
    'rejected' one seeded here -- so the next POST /label/files routes it to
    mode=initial, not augment (the #61 bug this must not reintroduce)."""
    mock_q.enqueue.return_value = mock_rq_job("rq-label-123")
    file = _needs_review_file(db, path_id=registered_path.id, status="ready")
    type_label = TypeLabel(name="invoice")
    db.add(type_label)
    db.flush()
    db.add(TypeLabelFile(file_id=file.id, type_label_id=type_label.id, source="llm", status="rejected"))
    db.commit()

    drop_resp = client.post(f"/files/{file.id}/label-review", json={"action": "drop"})
    assert drop_resp.status_code == 200

    label_resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert label_resp.status_code == 202
    data = label_resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["mode"] == "initial"
