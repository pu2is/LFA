"""Tests for POST /scans enqueue bookkeeping (#33)."""
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
