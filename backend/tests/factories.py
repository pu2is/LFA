"""Shared test-data factories for building a File + Job pair quickly."""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job


def make_failed_job(
    db: Session,
    *,
    job_type: str,
    trigger: str = "scan",
    mode: str = "default",
    file_status: str = "discovered",
    error_message: str = "prior failure",
) -> Job:
    """Build a File with a Job already marked failed -- simulates a prior
    failed attempt that RQ is now retrying (#33)."""
    path = RegisteredPath(path=f"/tmp/lfa_{job_type}_retry_fixture")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path=f"{path.path}/sample.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash=f"hash-{job_type}",
        file_modified_at=datetime.now(timezone.utc),
        status=file_status,
    )
    db.add(file)
    db.flush()

    job = Job(
        type=job_type,
        file_id=file.id,
        trigger=trigger,
        mode=mode,
        status="failed",
        error_message=error_message,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
