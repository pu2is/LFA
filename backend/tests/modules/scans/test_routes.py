"""Tests for POST /scans enqueue bookkeeping (#33), POST /rescans's
precondition gate and GET /rescans/{id}'s scan report (#68, ADR-0001b D1),
and POST /rescans/{id}/resume's precondition gating (#67, ADR-0001b D6)."""
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.scans.models import FileEvent, FileMatchCandidate
from tests.conftest import mock_rq_job

_FAKE_PATH = "D:/lfa_test_scan_routes"


def _clear_paths(db: Session) -> None:
    """POST /rescans' precondition check looks at *every* registered path,
    so it's sensitive to whatever the shared dev DB already has registered
    (this repo's tests run against a real Postgres instance, isolated per
    test only via the db fixture's outer-transaction rollback -- see
    conftest.py). Clearing first, inside that same rolled-back transaction,
    keeps these tests deterministic without touching real data."""
    db.execute(delete(RegisteredPath))


@pytest.fixture
def registered_path(db: Session):
    _clear_paths(db)
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.commit()
    db.refresh(path)
    yield path


@pytest.fixture
def scanned_path(db: Session):
    """A registered path that has already completed its initial scan --
    POST /rescans' precondition 1 requires this for every registered path."""
    _clear_paths(db)
    path = RegisteredPath(path=_FAKE_PATH, last_scanned_at=datetime.now(timezone.utc))
    db.add(path)
    db.commit()
    db.refresh(path)
    yield path


def _make_file(db: Session, *, path_id: uuid.UUID, full_path: str = "D:/lfa_test_scan_routes/f.pdf") -> File:
    file = File(
        path_id=path_id, filename="f.pdf", full_path=full_path, file_type="pdf",
        file_size=1, file_hash="hash", file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


@patch("app.modules.scans.routes.scan_queue")
def test_create_scan_stores_rq_job_id(mock_q, client, db, registered_path):
    mock_q.enqueue.return_value = mock_rq_job("rq-scan-123")

    resp = client.post("/scans", json={"path_id": str(registered_path.id)})

    assert resp.status_code == 202
    assert resp.json()["rq_job_id"] == "rq-scan-123"

    job = db.get(Job, uuid.UUID(resp.json()["id"]))
    assert job.rq_job_id == "rq-scan-123"


@patch("app.modules.scans.routes.scan_queue")
def test_create_scan_unknown_path_returns_404(mock_q, client):
    resp = client.post("/scans", json={"path_id": str(uuid.uuid4())})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


def _make_rescan_job(db: Session, *, status: str, stage: str | None) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual", status=status, stage=stage)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@patch("app.modules.scans.routes.scan_queue")
def test_create_rescan_no_registered_paths_returns_409(mock_q, client, db):
    _clear_paths(db)
    db.commit()

    resp = client.post("/rescans")
    assert resp.status_code == 409
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_create_rescan_unscanned_path_returns_409(mock_q, client, registered_path):
    resp = client.post("/rescans")
    assert resp.status_code == 409
    assert _FAKE_PATH in resp.json()["detail"]
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_create_rescan_pending_candidate_returns_409(mock_q, client, db, scanned_path):
    prior_scan = _make_rescan_job(db, status="succeeded", stage="fan_out")
    missing_file = _make_file(db, path_id=scanned_path.id)
    db.add(FileMatchCandidate(
        scan_id=prior_scan.id, missing_file_id=missing_file.id, candidate_path_id=scanned_path.id,
        candidate_full_path="D:/lfa_test_scan_routes/new.pdf", candidate_hash="hash",
        candidate_size=1, candidate_modified_at=datetime.now(timezone.utc), similarity_score=0.95,
    ))
    db.commit()

    resp = client.post("/rescans")

    assert resp.status_code == 409
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_create_rescan_active_job_returns_409(mock_q, client, db, scanned_path):
    file = _make_file(db, path_id=scanned_path.id)
    db.add(Job(type="ingest", file_id=file.id, trigger="scan", status="queued"))
    db.commit()

    resp = client.post("/rescans")

    assert resp.status_code == 409
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_create_rescan_enqueues_and_stores_rq_job_id(mock_q, client, db, scanned_path):
    mock_q.enqueue.return_value = mock_rq_job("rq-rescan-123")

    resp = client.post("/rescans")

    assert resp.status_code == 202
    body = resp.json()
    assert body["rq_job_id"] == "rq-rescan-123"
    assert body["type"] == "scan"
    assert body["mode"] == "rescan"

    job = db.get(Job, uuid.UUID(body["id"]))
    assert job.rq_job_id == "rq-rescan-123"


def test_get_rescan_unknown_id_returns_404(client):
    resp = client.get(f"/rescans/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_rescan_rejects_an_initial_scan_job_id(client, db, registered_path):
    scan = Job(type="scan", path_id=registered_path.id, mode="initial", trigger="scan")
    db.add(scan)
    db.commit()
    db.refresh(scan)

    resp = client.get(f"/rescans/{scan.id}")

    assert resp.status_code == 404


def test_get_rescan_returns_event_counts_and_pending_candidates(client, db, scanned_path):
    job = _make_rescan_job(db, status="succeeded", stage="fan_out")
    file_a = _make_file(db, path_id=scanned_path.id, full_path="D:/lfa_test_scan_routes/a.pdf")
    file_b = _make_file(db, path_id=scanned_path.id, full_path="D:/lfa_test_scan_routes/b.pdf")
    db.add(FileEvent(
        scan_id=job.id, file_id=file_a.id, event_type="added", to_path=file_a.full_path, to_hash="h",
    ))
    db.add(FileEvent(
        scan_id=job.id, file_id=file_b.id, event_type="modified",
        from_path=file_b.full_path, to_path=file_b.full_path, from_hash="h1", to_hash="h2",
    ))
    db.add(FileMatchCandidate(
        scan_id=job.id, missing_file_id=file_a.id, candidate_path_id=scanned_path.id,
        candidate_full_path="D:/lfa_test_scan_routes/c.pdf", candidate_hash="hash",
        candidate_size=1, candidate_modified_at=datetime.now(timezone.utc), similarity_score=0.95,
    ))
    db.commit()

    resp = client.get(f"/rescans/{job.id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["event_counts"] == {"added": 1, "modified": 1}
    assert body["pending_candidate_count"] == 1


@patch("app.modules.scans.routes.scan_queue")
def test_resume_rescan_unknown_id_returns_404(mock_q, client):
    resp = client.post(f"/rescans/{uuid.uuid4()}/resume")
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_resume_rescan_rejects_an_initial_scan_job_id(mock_q, client, db, registered_path):
    scan = Job(type="scan", path_id=registered_path.id, mode="initial", trigger="scan", status="failed")
    db.add(scan)
    db.commit()
    db.refresh(scan)

    resp = client.post(f"/rescans/{scan.id}/resume")

    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


@pytest.mark.parametrize(("status_value", "stage"), [
    ("running", "fan_out"),
    ("succeeded", "fan_out"),
    ("failed", "apply"),
    ("failed", "diff"),
    ("failed", None),
])
@patch("app.modules.scans.routes.scan_queue")
def test_resume_rescan_rejects_jobs_not_in_failed_fan_out_state(mock_q, client, db, status_value, stage):
    job = _make_rescan_job(db, status=status_value, stage=stage)

    resp = client.post(f"/rescans/{job.id}/resume")

    assert resp.status_code == 409
    mock_q.enqueue.assert_not_called()


@patch("app.modules.scans.routes.scan_queue")
def test_resume_rescan_enqueues_and_stores_rq_job_id(mock_q, client, db):
    mock_q.enqueue.return_value = mock_rq_job("rq-resume-123")
    job = _make_rescan_job(db, status="failed", stage="fan_out")

    resp = client.post(f"/rescans/{job.id}/resume")

    assert resp.status_code == 202
    body = resp.json()
    assert body["rq_job_id"] == "rq-resume-123"
    assert body["status"] == "queued"

    db.refresh(job)
    assert job.rq_job_id == "rq-resume-123"
    assert job.status == "queued"
