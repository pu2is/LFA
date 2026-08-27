"""Tests for POST /scans enqueue bookkeeping (#33) and POST /rescans/{id}/
resume's precondition gating (#67, ADR-0001b D6)."""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.modules.files.models import RegisteredPath
from app.modules.jobs.models import Job
from tests.conftest import mock_rq_job

_FAKE_PATH = "D:/lfa_test_scan_routes"


@pytest.fixture
def registered_path(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.commit()
    db.refresh(path)
    yield path


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
