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
from app.shared.events import publish_event


def _publish_job_status(job: ProcessingJob) -> None:
    publish_event("job_status", {
        "job_id": str(job.id),
        "file_id": str(job.file_id),
        "status": job.status,
        "error_message": job.error_message,
    })


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
    _publish_job_status(job)

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
        _publish_job_status(job)
        return job

    cleaned_text = cleaning.clean(result.text)

    job.status = "chunking"
    if result.ocr_applied:
        file.ocr_applied = True
    db.commit()
    _publish_job_status(job)

    rag_service.chunk_and_store(db, file.id, cleaned_text)
    db.commit()

    job.status = "labeling"
    db.commit()
    _publish_job_status(job)

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
        _publish_job_status(job)
        raise

    job.status = "succeeded"
    job.completed_at = datetime.now(timezone.utc)
    file.status = "ready"
    db.commit()
    _publish_job_status(job)

    return job


def relabel_file(
    db: Session,
    job_id: uuid.UUID,
    llm: BaseChatModel | None = None,
) -> ProcessingJob:
    """Re-run only the labeling stage on an already-processed file (WF1c).

    Skips extract/chunk entirely; existing chunks are reused.
    suggest_labels uses _merge_candidates internally so confirmed labels are
    preserved and rejected labels are reset to suggested.
    """
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise ValueError(f"ProcessingJob {job_id} not found")

    file = db.get(File, job.file_id)
    if file is None:
        raise ValueError(f"File {job.file_id} not found")

    job.status = "labeling"
    job.started_at = datetime.now(timezone.utc)
    db.commit()
    _publish_job_status(job)

    try:
        labeling_service.suggest_labels(db, file.id, llm=llm)
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        _publish_job_status(job)
        raise

    job.status = "succeeded"
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    _publish_job_status(job)

    return job
