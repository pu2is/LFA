"""Shared job-lifecycle transition helpers.

Each run_* function across scan/ingest/embed/label owns its own per-type
fields (job.stage, file.status, etc.) and sets them before calling one of
these -- the helpers only own the running/failed/succeeded status,
timestamp, commit, and publish mechanics that were otherwise duplicated
near-identically across every job type.

One exception: scan's succeeded transition (scans/service.py::run_scan)
stays inline rather than calling mark_succeeded. The file_count kwarg
alone would be an easy fix -- mark_succeeded could take **extra and
forward it, same as publish_job_status already does. The actual blocker
is registered_path.last_scanned_at: a SECOND model updated in the same
commit, using the exact timestamp mark_succeeded computes internally.
That timestamp doesn't exist before calling mark_succeeded, and setting
it after would mean a second db.commit(), breaking atomicity between
"job succeeded" and "path's last-scanned time updated". Solvable with a
pre-commit callback/hook parameter, but not worth adding one for a single
caller today -- revisit if a second caller ever needs the same thing.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.modules.jobs.models import Job
from app.shared.events import publish_job_status


def mark_running(db: Session, job: Job) -> None:
    """Transition a job to running, clearing any stale error from a prior
    failed attempt (RQ retries reuse the same job row, see #33)."""
    job.status = "running"
    job.error_message = None
    job.started_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(job)


def mark_failed(db: Session, job: Job, exc: Exception) -> None:
    job.status = "failed"
    job.error_message = str(exc)
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(job)


def mark_succeeded(db: Session, job: Job) -> None:
    job.status = "succeeded"
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(job)


def mark_progress(db: Session, job: Job) -> None:
    """Persist and broadcast an in-progress update (e.g. a stage change)
    without altering job.status."""
    db.commit()
    publish_job_status(job)
