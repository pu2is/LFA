"""Tests for POST /rescan-candidates/{id}/resolve (#68, ADR-0001b D5):
keep_labels/drop_labels/reject, and the mandatory re-verification against
drift since the Rescan that proposed the candidate."""
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.scans import discovery
from app.modules.scans.models import FileEvent, FileMatchCandidate
from tests.conftest import mock_rq_job


def _register(db: Session, path: Path) -> RegisteredPath:
    row = RegisteredPath(path=str(path.resolve()), last_scanned_at=datetime.now(timezone.utc))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _missing_file(db: Session, *, path_id: uuid.UUID, full_path: Path, file_hash: str) -> File:
    file = File(
        path_id=path_id, filename=full_path.name, full_path=str(full_path),
        file_type=full_path.suffix.lstrip("."), file_size=1, file_hash=file_hash,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="missing",
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def _rescan_job(db: Session) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual", status="succeeded")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _candidate(
    db: Session, *, scan_id: uuid.UUID, missing_file_id: uuid.UUID, candidate_path_id: uuid.UUID,
    candidate_path: Path, status: str = "pending",
) -> FileMatchCandidate:
    stat = candidate_path.stat()
    row = FileMatchCandidate(
        scan_id=scan_id, missing_file_id=missing_file_id, candidate_path_id=candidate_path_id,
        candidate_full_path=str(candidate_path), candidate_hash=discovery.compute_sha256(candidate_path),
        candidate_size=stat.st_size,
        candidate_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        similarity_score=0.95, status=status,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture
def scenario(db: Session, tmp_path):
    """A missing file (old_name.pdf, never actually on disk under the new
    path) and a pending candidate pointing at new_name.pdf, which *is* on
    disk with content matching the candidate's recorded hash/size/mtime."""
    registered_path = _register(db, tmp_path)
    new_path = tmp_path / "new_name.pdf"
    new_path.write_bytes(b"same-content")
    missing = _missing_file(db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf", file_hash="old-hash")
    scan_job = _rescan_job(db)
    candidate = _candidate(
        db, scan_id=scan_job.id, missing_file_id=missing.id,
        candidate_path_id=registered_path.id, candidate_path=new_path,
    )
    return registered_path, missing, scan_job, candidate, new_path


@pytest.fixture(autouse=True)
def _mock_ingest_queue():
    from unittest.mock import patch
    with patch("app.modules.scans.routes.ingest_queue") as mock_q:
        mock_q.enqueue.return_value = mock_rq_job("rq-ingest-123")
        yield mock_q


def test_resolve_unknown_candidate_returns_404(client):
    resp = client.post(f"/rescan-candidates/{uuid.uuid4()}/resolve", json={"action": "reject"})
    assert resp.status_code == 404


def test_resolve_already_resolved_candidate_returns_409(client, db, scenario):
    _, _, _, candidate, _ = scenario
    candidate.status = "rejected"
    db.commit()

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "reject"})

    assert resp.status_code == 409


def test_resolve_metadata_drift_returns_409(client, scenario):
    _, _, _, candidate, new_path = scenario
    new_path.write_bytes(b"changed-content-different-size")  # drifts size+mtime after the candidate snapshot

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "keep_labels"})

    assert resp.status_code == 409


def _drift_content_keep_metadata(path: Path) -> None:
    """Overwrite `path` with different content but the exact same size/mtime
    as before -- the drift only a re-hash (not a re-stat) can catch."""
    import os
    original_stat = path.stat()
    path.write_bytes(b"same-length!")  # same length as the fixture's b"same-content"
    os.utime(path, (original_stat.st_atime, original_stat.st_mtime))


def test_resolve_hash_drift_returns_409_for_keep_labels(client, scenario):
    _, _, _, candidate, new_path = scenario
    _drift_content_keep_metadata(new_path)

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "keep_labels"})
    assert resp.status_code == 409


def test_resolve_reject_does_not_verify_hash_and_succeeds_despite_content_drift(client, scenario):
    """Design decision: reject only creates an independent new file, so it
    trusts the metadata-confirmed candidate_hash rather than paying for a
    re-hash -- same least-cost trust diff_inventory's cheap diff already
    relies on elsewhere."""
    _, _, _, candidate, new_path = scenario
    _drift_content_keep_metadata(new_path)

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "reject"})
    assert resp.status_code == 200


def test_resolve_keep_labels_transplants_identity_and_keeps_labels(client, db, scenario):
    registered_path, missing, scan_job, candidate, new_path = scenario
    type_label = TypeLabel(name="invoice")
    db.add(type_label)
    db.flush()
    db.add(TypeLabelFile(file_id=missing.id, type_label_id=type_label.id, source="user", status="confirmed"))
    db.commit()

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "keep_labels"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["file"]["id"] == str(missing.id)
    assert body["file"]["full_path"] == str(new_path)
    assert body["file"]["status"] == "discovered"

    db.refresh(missing)
    assert missing.full_path == str(new_path)
    assert missing.status == "discovered"
    assert list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == missing.id)))

    db.refresh(candidate)
    assert candidate.status == "accepted_keep_labels"
    assert candidate.resolved_at is not None

    [event] = list(db.scalars(select(FileEvent).where(FileEvent.scan_id == scan_job.id, FileEvent.file_id == missing.id)))
    assert event.event_type == "recovered"

    [ingest_job] = list(db.scalars(select(Job).where(Job.file_id == missing.id, Job.type == "ingest")))
    assert ingest_job.rq_job_id == "rq-ingest-123"


def test_resolve_drop_labels_clears_all_associations_including_rejected(client, db, scenario):
    registered_path, missing, scan_job, candidate, new_path = scenario
    type_label = TypeLabel(name="invoice")
    tag_kind = TagKind(name="person")
    db.add_all([type_label, tag_kind])
    db.flush()
    db.add(TypeLabelFile(file_id=missing.id, type_label_id=type_label.id, source="user", status="rejected"))
    db.add(TagLabel(file_id=missing.id, kind_id=tag_kind.id, value="Alice", source="llm", status="confirmed"))
    db.commit()

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "drop_labels"})

    assert resp.status_code == 200
    assert not list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == missing.id)))
    assert not list(db.scalars(select(TagLabel).where(TagLabel.file_id == missing.id)))

    db.refresh(candidate)
    assert candidate.status == "accepted_drop_labels"


def test_resolve_reject_creates_new_file_and_leaves_missing_file_alone(client, db, scenario):
    registered_path, missing, scan_job, candidate, new_path = scenario

    resp = client.post(f"/rescan-candidates/{candidate.id}/resolve", json={"action": "reject"})

    assert resp.status_code == 200
    body = resp.json()
    new_file_id = uuid.UUID(body["file"]["id"])
    assert new_file_id != missing.id
    assert body["file"]["full_path"] == str(new_path)

    db.refresh(missing)
    assert missing.status == "missing"
    assert missing.full_path != str(new_path)

    new_file = db.get(File, new_file_id)
    assert new_file.status == "discovered"

    [event] = list(db.scalars(select(FileEvent).where(FileEvent.scan_id == scan_job.id, FileEvent.file_id == new_file_id)))
    assert event.event_type == "added"

    db.refresh(candidate)
    assert candidate.status == "rejected"
