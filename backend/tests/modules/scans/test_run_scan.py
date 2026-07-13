"""run_scan's job-lifecycle bookkeeping (real directory discovery itself is
covered in test_discovery.py / test_service.py, gated behind a real fixture
directory). This test mocks discovery entirely so it runs everywhere."""
from unittest.mock import patch

import pytest

from app.modules.files.models import RegisteredPath
from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed
from app.modules.scans.service import run_scan


@pytest.fixture
def scan_job(db):
    path = RegisteredPath(path="/tmp/lfa_run_scan_retry_fixture")
    db.add(path)
    db.flush()

    job = Job(type="scan", path_id=path.id, trigger="scan", mode="initial")
    db.add(job)
    db.commit()

    # Simulates a prior failed attempt (e.g. the path was unreadable last
    # time) being re-run -- realistic setup via the real transition helper,
    # not by hand-setting fields the code itself would never produce.
    mark_failed(db, job, RuntimeError("path was unreadable last time"))
    return job


@patch("app.modules.scans.service.discovery.iter_documents", return_value=[])
def test_run_scan_clears_stale_error_message_on_success(mock_iter_documents, db, scan_job):
    result_job, _ingest_jobs = run_scan(db, scan_job.id)

    assert result_job.status == "succeeded"
    assert result_job.error_message is None
