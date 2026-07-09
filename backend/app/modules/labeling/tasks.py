import uuid
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.labeling.suggestion import suggest_labels, suggest_labels_augment
from app.shared.database import SessionLocal
from app.shared.events import publish_job_status


def run_label(
    db: Session,
    job_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
) -> Job:
    """Execute a label job: dispatch to initial or augment based on job.mode."""
    job = db.get(Job, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    file = db.get(File, job.file_id)
    if file is None:
        raise ValueError(f"File {job.file_id} not found")

    job.status = "running"
    job.stage = "labeling"
    job.started_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(job)

    try:
        if job.mode == "augment":
            suggest_labels_augment(db, file.id, llm=llm)
        else:
            suggest_labels(db, file.id, llm=llm)
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        publish_job_status(job)
        raise

    job.status = "succeeded"
    job.stage = None
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(job)

    return job


def run_label_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for a label job (initial or augment)."""
    db = SessionLocal()
    try:
        run_label(db, job_id)
    finally:
        db.close()
