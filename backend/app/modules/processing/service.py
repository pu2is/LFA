import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.processing import cleaning, extraction
from app.modules.processing.models import ProcessingJob
from app.modules.rag import service as rag_service


def create_processing_job(db: Session, file_id: uuid.UUID, triggered_by: str) -> ProcessingJob:
    job = ProcessingJob(file_id=file_id, triggered_by=triggered_by)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_processing_job(db: Session, job_id: uuid.UUID) -> ProcessingJob | None:
    return db.get(ProcessingJob, job_id)


def process_file(db: Session, job_id: uuid.UUID) -> ProcessingJob:
    """Run extract → clean → chunk synchronously for the file attached to `job_id`.

    Leaves the job at status "chunking" -- the labeling step (→ "succeeded")
    is added in issue #8. OCR fallback for scanned PDFs is deferred to #7.
    """
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise ValueError(f"ProcessingJob {job_id} not found")

    file = db.get(File, job.file_id)
    if file is None:
        raise ValueError(f"File {job.file_id} not found")

    job.status = "extracting"
    job.started_at = datetime.now(timezone.utc)
    file.status = "processing"
    db.commit()

    try:
        raw_text = extraction.extract_text(Path(file.full_path), file.file_type)
    except Exception as exc:
        # Any extraction failure (unsupported type, corrupted file, etc.) is
        # surfaced here so the job and file reflect the real outcome rather
        # than silently producing empty or garbage labels downstream.
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        file.status = "failed"
        db.commit()
        return job

    cleaned_text = cleaning.clean(raw_text)

    job.status = "chunking"
    db.commit()

    rag_service.chunk_and_store(db, file.id, cleaned_text)
    db.commit()

    return job
