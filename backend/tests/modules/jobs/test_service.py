"""Unit tests for the shared job-lifecycle transition helpers."""
from datetime import datetime, timezone

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed, mark_progress, mark_running, mark_succeeded


@pytest.fixture
def file_id(db):
    path = RegisteredPath(path="/tmp/lfa_jobs_service_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/tmp/lfa_jobs_service_test/sample.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="jobs-service-test",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file.id


def test_mark_running_clears_stale_error_and_sets_timestamp(db, file_id):
    job = Job(type="embed", file_id=file_id, trigger="scan", status="failed", error_message="prior failure")
    db.add(job)
    db.commit()

    mark_running(db, job)

    assert job.status == "running"
    assert job.error_message is None
    assert job.started_at is not None


def test_mark_failed_records_error_and_completed_at(db, file_id):
    job = Job(type="embed", file_id=file_id, trigger="scan", status="running")
    db.add(job)
    db.commit()

    mark_failed(db, job, RuntimeError("boom"))

    assert job.status == "failed"
    assert job.error_message == "boom"
    assert job.completed_at is not None


def test_mark_succeeded_sets_completed_at(db, file_id):
    job = Job(type="embed", file_id=file_id, trigger="scan", status="running")
    db.add(job)
    db.commit()

    mark_succeeded(db, job)

    assert job.status == "succeeded"
    assert job.completed_at is not None


def test_mark_progress_does_not_change_status(db, file_id):
    job = Job(type="ingest", file_id=file_id, trigger="scan", status="running", stage="extract")
    db.add(job)
    db.commit()

    job.stage = "clean"
    mark_progress(db, job)

    assert job.status == "running"
    assert job.stage == "clean"
