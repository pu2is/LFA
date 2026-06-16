import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.labeling import service as labeling_service
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


def process_file(
    db: Session,
    job_id: uuid.UUID,
    llm: BaseChatModel | None = None,
) -> ProcessingJob:
    """Run extract → clean → chunk → label synchronously for the file attached to `job_id`.

    The llm parameter is injectable so tests can pass a mock without hitting Ollama.
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
        result = extraction.extract_text(Path(file.full_path), file.file_type)
    except Exception as exc:
        # Any extraction failure (unsupported type, corrupted file, no usable
        # text after OCR, etc.) is surfaced here so the job and file reflect
        # the real outcome rather than silently producing garbage labels.
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        file.status = "failed"
        db.commit()
        return job

    cleaned_text = cleaning.clean(result.text)

    job.status = "chunking"
    if result.ocr_applied:
        file.ocr_applied = True
    db.commit()

    rag_service.chunk_and_store(db, file.id, cleaned_text)
    db.commit()

    job.status = "labeling"
    db.commit()

    try:
        labeling_service.suggest_labels(db, file.id, llm=llm)
    except Exception as exc:
        # Only resource errors (Ollama connectivity) bubble up from suggest_labels;
        # non-resource errors are absorbed there and return [].
        # Mark the job failed and re-raise so RQ can retry.
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        file.status = "failed"
        db.commit()
        raise

    job.status = "succeeded"
    job.completed_at = datetime.now(timezone.utc)
    file.status = "ready"
    db.commit()

    return job
