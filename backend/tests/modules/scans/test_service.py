import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.scans.service import create_scan, get_scan, run_scan

TEST_DIR = Path("E:/lfa_test")

pytestmark = pytest.mark.skipif(
    not TEST_DIR.exists(), reason="E:\\lfa_test fixture directory not present on this machine"
)


@pytest.fixture
def registered_path(db):
    path = RegisteredPath(path=str(TEST_DIR.resolve()))
    db.add(path)
    db.commit()
    yield path


def _queue_scan(db, path_id: uuid.UUID) -> Job:
    job = Job(type="scan", path_id=path_id, trigger="scan", mode="initial")
    db.add(job)
    db.commit()
    return job


def test_run_scan_discovers_documents(db, registered_path):
    scan_job = _queue_scan(db, registered_path.id)

    result_job, ingest_jobs = run_scan(db, scan_job.id)

    assert result_job.status == "succeeded"
    assert result_job.started_at is not None
    assert result_job.completed_at is not None

    files = db.scalars(select(File).where(File.path_id == registered_path.id)).all()
    assert len(files) >= 1
    for file in files:
        assert file.status == "discovered"
        assert file.file_size > 0
        assert len(file.file_hash) == 64


def test_run_scan_creates_ingest_jobs_for_discovered_files(db, registered_path):
    scan_job = _queue_scan(db, registered_path.id)

    result_job, ingest_jobs = run_scan(db, scan_job.id)

    files = db.scalars(select(File).where(File.path_id == registered_path.id)).all()
    assert result_job.status == "succeeded"
    assert len(ingest_jobs) == len(files)
    for ij in ingest_jobs:
        assert ij.type == "ingest"
        assert ij.file_id is not None
        assert ij.parent_job_id == scan_job.id
        assert ij.trigger == "scan"
        assert ij.status == "queued"


def test_run_scan_does_not_create_ingest_for_already_processed(db, registered_path):
    """If files are already ready (from a previous scan+ingest), rescan should not
    fan-out new ingest jobs for them."""
    first_job = _queue_scan(db, registered_path.id)
    run_scan(db, first_job.id)

    # Mark all files as ready (simulate completed ingest)
    files = db.scalars(select(File).where(File.path_id == registered_path.id)).all()
    for f in files:
        f.status = "ready"
    db.commit()

    second_job = _queue_scan(db, registered_path.id)
    _, ingest_jobs = run_scan(db, second_job.id)

    assert len(ingest_jobs) == 0


def test_run_scan_updates_path_last_scanned_at(db, registered_path):
    assert registered_path.last_scanned_at is None

    scan_job = _queue_scan(db, registered_path.id)
    run_scan(db, scan_job.id)

    db.refresh(registered_path)
    assert registered_path.last_scanned_at is not None


def test_create_scan_starts_queued(db, registered_path):
    scan = create_scan(db, registered_path.id)

    assert scan.status == "queued"
    assert scan.type == "scan"
    assert scan.trigger == "scan"
    assert scan.mode == "initial"
    assert get_scan(db, scan.id).id == scan.id


def test_get_scan_returns_none_for_unknown_id(db):
    assert get_scan(db, uuid.uuid4()) is None


def test_rescan_is_idempotent(db, registered_path):
    first_job = _queue_scan(db, registered_path.id)
    run_scan(db, first_job.id)

    first_ids = {file.id for file in db.scalars(select(File).where(File.path_id == registered_path.id)).all()}
    assert len(first_ids) >= 1

    second_job = _queue_scan(db, registered_path.id)
    second_result, _ = run_scan(db, second_job.id)

    second_ids = {file.id for file in db.scalars(select(File).where(File.path_id == registered_path.id)).all()}
    assert second_result.status == "succeeded"
    assert second_ids == first_ids
