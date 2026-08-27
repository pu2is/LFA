"""Job.mode and Job.stage are per-type variants guarded at the app layer, not
by a DB CHECK constraint (see docs/03_er-diagram.md) -- these tests cover
that guard. The DB-level constraints below (ck_jobs_target's rescan shape,
ix_jobs_active_rescan) are covered separately since they can only be
observed on commit, not at construction time.
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.modules.jobs.models import Job, VALID_JOB_MODES, VALID_JOB_STAGES


@pytest.mark.parametrize("mode", sorted(VALID_JOB_MODES))
def test_job_accepts_valid_modes(mode):
    job = Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode=mode)
    assert job.mode == mode


def test_job_rejects_invalid_mode_at_construction():
    with pytest.raises(ValueError, match="Invalid job mode"):
        Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode="bogus")


def test_job_rejects_invalid_mode_on_reassignment():
    job = Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode="initial")
    with pytest.raises(ValueError, match="Invalid job mode"):
        job.mode = "not_a_real_mode"


@pytest.mark.parametrize("stage", sorted(VALID_JOB_STAGES))
def test_job_accepts_valid_stages(stage):
    job = Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode="initial", stage=stage)
    assert job.stage == stage


def test_job_accepts_none_stage():
    job = Job(type="scan", path_id=uuid.uuid4(), trigger="scan", mode="initial", stage=None)
    assert job.stage is None


def test_job_rejects_invalid_stage_at_construction():
    with pytest.raises(ValueError, match="Invalid job stage"):
        Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode="initial", stage="kind")


def test_job_rejects_invalid_stage_on_reassignment():
    job = Job(type="label", file_id=uuid.uuid4(), trigger="manual", mode="initial", stage="type")
    with pytest.raises(ValueError, match="Invalid job stage"):
        job.stage = "not_a_real_stage"


# --------------------------------------------------------------------------- #
# DB-level constraints (#64: ck_jobs_target's rescan shape, ix_jobs_active_rescan)
# --------------------------------------------------------------------------- #

def _rescan_job(*, status: str = "queued") -> Job:
    return Job(type="scan", path_id=None, file_id=None, trigger="manual", mode="rescan", status=status)


def test_only_one_active_rescan_allowed(db):
    """ADR-0001b D1: at most one active (queued/running) global Rescan, even
    if two requests both pass an app-level precondition check -- the
    partial unique index is the actual guarantee, not just app logic."""
    db.add(_rescan_job(status="queued"))
    db.commit()

    db.add(_rescan_job(status="running"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_completed_rescan_does_not_block_a_new_active_rescan(db):
    """The index is partial (WHERE status IN ('queued', 'running')) --
    finished Rescans must not permanently block future ones."""
    db.add(_rescan_job(status="succeeded"))
    db.commit()

    db.add(_rescan_job(status="queued"))
    db.commit()  # must not raise


def test_ck_jobs_target_rejects_rescan_with_path_id(db):
    db.add(Job(type="scan", path_id=uuid.uuid4(), file_id=None, trigger="manual", mode="rescan"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_ck_jobs_target_rejects_initial_scan_without_path_id(db):
    db.add(Job(type="scan", path_id=None, file_id=None, trigger="scan", mode="initial"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
