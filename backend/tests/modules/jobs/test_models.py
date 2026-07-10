"""Job.mode is a per-type variant guarded at the app layer, not by a DB
CHECK constraint (see docs/03_er-diagram.md) -- these tests cover that guard.
"""
import uuid

import pytest

from app.modules.jobs.models import Job, VALID_JOB_MODES


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
