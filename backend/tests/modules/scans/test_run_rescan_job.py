"""Tests for run_rescan_job (#67, ADR-0001b D6): the RQ entrypoint that
serves both a Rescan's initial run and its resume, and the enqueue loop's
per-child commit that makes fan-out safely resumable.

Like test_tasks.py's run_scan_job tests, `service` and `ingest_queue` are
mocked entirely so these stay focused on run_rescan_job's own branching and
commit sequencing -- service.run_rescan and apply_diff have their own real-DB
tests in test_run_rescan.py / test_apply.py. SessionLocal is mocked too,
since run_rescan_job (unlike run_scan_job) reads job.stage via db.get()
before deciding what to do.
"""
import uuid
from unittest.mock import MagicMock, call, patch

from app.modules.jobs.models import Job
from app.modules.scans.tasks import run_rescan_job
from tests.conftest import mock_rq_job


def _job(**overrides) -> Job:
    defaults = {"id": uuid.uuid4(), "type": "scan", "mode": "rescan", "trigger": "manual"}
    defaults.update(overrides)
    return Job(**defaults)


@patch("app.modules.scans.tasks.mark_succeeded")
@patch("app.modules.scans.tasks.SessionLocal")
@patch("app.modules.scans.tasks.ingest_queue")
@patch("app.modules.scans.tasks.service")
def test_run_rescan_job_runs_pipeline_then_fans_out_when_not_yet_at_fan_out(
    mock_service, mock_queue, mock_session_local, mock_mark_succeeded,
):
    scan_id = uuid.uuid4()
    db = MagicMock()
    mock_session_local.return_value = db
    db.get.return_value = _job(id=scan_id, stage="apply")  # pre-apply: not resumed

    ran_job = _job(id=scan_id, stage="fan_out", status="running")
    mock_service.run_rescan.return_value = ran_job
    children = [_job(type="ingest", file_id=uuid.uuid4()), _job(type="ingest", file_id=uuid.uuid4())]
    mock_service.get_pending_fan_out_jobs.return_value = children
    mock_queue.enqueue.side_effect = [mock_rq_job("rq-1"), mock_rq_job("rq-2")]

    run_rescan_job(scan_id)

    mock_service.run_rescan.assert_called_once_with(db, scan_id)
    assert children[0].rq_job_id == "rq-1"
    assert children[1].rq_job_id == "rq-2"
    mock_mark_succeeded.assert_called_once_with(db, ran_job)


@patch("app.modules.scans.tasks.mark_succeeded")
@patch("app.modules.scans.tasks.SessionLocal")
@patch("app.modules.scans.tasks.ingest_queue")
@patch("app.modules.scans.tasks.service")
def test_run_rescan_job_stops_without_fan_out_when_apply_phase_fails(
    mock_service, mock_queue, mock_session_local, mock_mark_succeeded,
):
    """service.run_rescan already marks the job failed internally (an
    inventory/diff/apply failure) -- run_rescan_job must not then treat that
    job's (nonexistent) children as a normal empty fan-out and mark it
    succeeded, which would silently overwrite the real failure."""
    scan_id = uuid.uuid4()
    db = MagicMock()
    mock_session_local.return_value = db
    db.get.return_value = _job(id=scan_id, stage="diff")

    mock_service.run_rescan.return_value = _job(id=scan_id, stage="diff", status="failed")

    run_rescan_job(scan_id)

    mock_service.get_pending_fan_out_jobs.assert_not_called()
    mock_queue.enqueue.assert_not_called()
    mock_mark_succeeded.assert_not_called()


@patch("app.modules.scans.tasks.mark_succeeded")
@patch("app.modules.scans.tasks.mark_running")
@patch("app.modules.scans.tasks.SessionLocal")
@patch("app.modules.scans.tasks.ingest_queue")
@patch("app.modules.scans.tasks.service")
def test_run_rescan_job_resume_skips_run_rescan_and_only_enqueues_pending_children(
    mock_service, mock_queue, mock_session_local, mock_mark_running, mock_mark_succeeded,
):
    """Regression test (issue #67 AC4): a resumed job (already at
    status=failed, stage=fan_out) must not re-run inventory/diff/apply, and
    fan-out must only ever be handed the children service.
    get_pending_fan_out_jobs reports as pending (rq_job_id IS NULL) --
    already-enqueued children are never re-submitted.
    """
    scan_id = uuid.uuid4()
    db = MagicMock()
    mock_session_local.return_value = db
    resumed_job = _job(id=scan_id, stage="fan_out", status="failed")
    db.get.return_value = resumed_job

    still_pending = _job(type="ingest", file_id=uuid.uuid4())
    mock_service.get_pending_fan_out_jobs.return_value = [still_pending]
    mock_queue.enqueue.return_value = mock_rq_job("rq-resumed")

    run_rescan_job(scan_id)

    mock_service.run_rescan.assert_not_called()
    mock_mark_running.assert_called_once_with(db, resumed_job)
    mock_service.get_pending_fan_out_jobs.assert_called_once_with(db, resumed_job)
    assert mock_queue.enqueue.call_count == 1
    assert still_pending.rq_job_id == "rq-resumed"
    mock_mark_succeeded.assert_called_once_with(db, resumed_job)


@patch("app.modules.scans.tasks.mark_failed")
@patch("app.modules.scans.tasks.mark_succeeded")
@patch("app.modules.scans.tasks.SessionLocal")
@patch("app.modules.scans.tasks.ingest_queue")
@patch("app.modules.scans.tasks.service")
def test_run_rescan_job_commits_each_rq_job_id_immediately_so_a_later_failure_stays_resumable(
    mock_service, mock_queue, mock_session_local, mock_mark_succeeded, mock_mark_failed,
):
    """If the Nth child's enqueue call fails, the N-1 children enqueued
    before it must already have their rq_job_id committed -- otherwise a
    later resume would see rq_job_id IS NULL for them too and submit
    duplicate ingest jobs for files already queued (ADR-0001b D6)."""
    scan_id = uuid.uuid4()
    db = MagicMock()
    mock_session_local.return_value = db
    scan_job = _job(id=scan_id, stage="fan_out", status="running")
    db.get.return_value = scan_job

    first, second, third = (_job(type="ingest", file_id=uuid.uuid4()) for _ in range(3))
    mock_service.get_pending_fan_out_jobs.return_value = [first, second, third]
    mock_queue.enqueue.side_effect = [mock_rq_job("rq-1"), RuntimeError("redis unavailable")]

    run_rescan_job(scan_id)

    assert first.rq_job_id == "rq-1"
    assert second.rq_job_id is None
    assert third.rq_job_id is None
    # the successful child's assignment was committed on its own, before the
    # failing enqueue call was even attempted for the second child
    assert call() in db.commit.mock_calls
    mock_mark_failed.assert_called_once()
    assert mock_mark_failed.call_args[0][:2] == (db, scan_job)
    mock_mark_succeeded.assert_not_called()
