import uuid

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed, mark_running, mark_succeeded
from app.modules.labeling.suggestion import suggest_labels, suggest_labels_augment
from app.shared.database import SessionLocal


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

    job.stage = "labeling"
    mark_running(db, job)

    try:
        if job.mode == "augment":
            suggest_labels_augment(db, file.id, llm=llm)
        else:
            suggest_labels(db, file.id, llm=llm)
    except Exception as exc:
        mark_failed(db, job, exc)
        raise

    job.stage = None
    mark_succeeded(db, job)

    return job


def run_label_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for a label job (initial or augment)."""
    db = SessionLocal()
    try:
        run_label(db, job_id)
    finally:
        db.close()
