"""Shared job-lifecycle transition helpers.

Each run_* function across scan/ingest/embed/label owns its own per-type
fields (job.stage, file.status, etc.) and sets them before calling one of
these -- the helpers only own the running/failed/succeeded status,
timestamp, commit, and publish mechanics that were otherwise duplicated
near-identically across every job type.

One exception: scan's succeeded transition (scans/service.py::run_scan)
stays inline rather than calling mark_succeeded. Not because of the extra
publish_job_status kwarg (publish_job_status already accepts **extra) --
the real blocker is that it also updates a SECOND model in the same
commit, registered_path.last_scanned_at, using the exact timestamp
mark_succeeded computes internally. That timestamp can't be obtained
before calling mark_succeeded (it doesn't exist yet) or supplied after
(that would mean a second db.commit(), breaking atomicity between "job
succeeded" and "path's last-scanned time updated"). Not worth a
callback/extra-commit-object parameter for one caller.
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
