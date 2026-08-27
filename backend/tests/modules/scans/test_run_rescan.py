"""Tests for service.run_rescan (#67, ADR-0001b D6): the inventory -> diff ->
apply orchestration for a global Rescan job, up to (but not including)
fan-out -- see test_tasks.py for the RQ entrypoint that adds fan-out on top.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.scans import rescan
from app.modules.scans.service import get_pending_fan_out_jobs, get_rescan, run_rescan


def _rescan_job(db) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@pytest.fixture(autouse=True)
def _isolate_registered_paths(db):
    """run_rescan is deliberately global (ADR-0001b D1): it walks every
    RegisteredPath row, not just the one a test creates. Without this, these
    tests would also walk whatever paths are registered in the real dev DB
    this test's transaction is layered on top of (real directories on this
    machine) -- rolled back and harmless, but slow, non-deterministic, and
    reads real files unrelated to the test. Deleting them here only affects
    this test's own (rolled-back) transaction.
    """
    for job in db.scalars(select(Job)):
        db.delete(job)
    for path in db.scalars(select(RegisteredPath)):
        db.delete(path)
    db.commit()


@pytest.fixture
def registered_path(db, tmp_path: Path):
    (tmp_path / "report.pdf").write_bytes(b"pdf-content")
    path = RegisteredPath(path=str(tmp_path.resolve()))
    db.add(path)
    db.commit()
    db.refresh(path)
    return path


def test_run_rescan_succeeds_and_reaches_fan_out_stage(db, registered_path):
    scan_job = _rescan_job(db)

    result = run_rescan(db, scan_job.id)

    assert result.status == "running"  # apply succeeded; fan_out itself is tasks.py's job
    assert result.stage == "fan_out"
    assert result.started_at is not None

    [file] = list(db.scalars(select(File)))
    assert file.filename == "report.pdf"
    assert file.status == "discovered"

    db.refresh(registered_path)
    assert registered_path.last_scanned_at is not None


def test_run_rescan_creates_one_ingest_job_per_added_file(db, registered_path):
    scan_job = _rescan_job(db)

    result = run_rescan(db, scan_job.id)

    pending = get_pending_fan_out_jobs(db, result)
    assert len(pending) == 1
    assert pending[0].type == "ingest"


def test_run_rescan_fails_with_zero_manifest_changes_when_a_root_is_unreadable(db, registered_path, tmp_path):
    unreadable = RegisteredPath(path=str(tmp_path / "does_not_exist_on_disk"))
    db.add(unreadable)
    db.commit()
    scan_job = _rescan_job(db)

    result = run_rescan(db, scan_job.id)

    assert result.status == "failed"
    assert result.stage == "inventory"
    assert result.error_message is not None
    assert not list(db.scalars(select(File)))  # D2: zero side effects on an inventory failure

    db.refresh(registered_path)
    assert registered_path.last_scanned_at is None


def test_run_rescan_rolls_back_every_apply_mutation_on_a_db_error(db, registered_path):
    scan_job = _rescan_job(db)
    real_apply_diff = rescan.apply_diff

    def _apply_then_fail(db_, job, diff, registered_paths):
        real_apply_diff(db_, job, diff, registered_paths)  # let the real mutations happen first
        raise RuntimeError("simulated apply-transaction failure")

    with patch("app.modules.scans.service.rescan.apply_diff", side_effect=_apply_then_fail):
        result = run_rescan(db, scan_job.id)

    assert result.status == "failed"
    assert result.stage == "apply"  # not fan_out -- apply never durably completed
    assert not list(db.scalars(select(File)))  # D6: whole transaction rolled back

    db.refresh(registered_path)
    assert registered_path.last_scanned_at is None


def test_run_rescan_raises_for_unknown_job_id(db):
    with pytest.raises(ValueError):
        run_rescan(db, uuid.uuid4())


def test_get_rescan_returns_none_for_an_initial_scan_job(db, registered_path):
    initial = Job(type="scan", path_id=registered_path.id, trigger="scan", mode="initial")
    db.add(initial)
    db.commit()

    assert get_rescan(db, initial.id) is None


def test_get_rescan_returns_none_for_unknown_id(db):
    assert get_rescan(db, uuid.uuid4()) is None


def _dummy_file(db, path_id, name: str) -> File:
    file = File(
        path_id=path_id, filename=name, full_path=f"/nonexistent/{uuid.uuid4()}/{name}",
        file_type="pdf", file_size=1, file_hash=uuid.uuid4().hex,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    db.add(file)
    db.flush()
    return file


def test_get_pending_fan_out_jobs_excludes_children_already_enqueued(db, registered_path):
    scan_job = _rescan_job(db)
    already_enqueued_file = _dummy_file(db, registered_path.id, "a.pdf")
    still_pending_file = _dummy_file(db, registered_path.id, "b.pdf")
    enqueued = Job(
        type="ingest", file_id=already_enqueued_file.id, parent_job_id=scan_job.id,
        trigger="scan", rq_job_id="rq-1",
    )
    pending = Job(type="ingest", file_id=still_pending_file.id, parent_job_id=scan_job.id, trigger="scan")
    db.add_all([enqueued, pending])
    db.commit()

    result = get_pending_fan_out_jobs(db, scan_job)

    assert [job.id for job in result] == [pending.id]
