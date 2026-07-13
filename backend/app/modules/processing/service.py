import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed, mark_progress, mark_running, mark_succeeded
from app.modules.processing import cleaning, extraction
from app.modules.rag import service as rag_service


def run_ingest(db: Session, job_id: uuid.UUID) -> tuple[Job, Job | None]:
    """Run extract -> clean -> chunk for the file attached to job_id.

    Uses the unified jobs table (type=ingest). No labeling step -- that is
    user-triggered via a separate label endpoint (#23).

    Returns (ingest_job, embed_job). embed_job is None when ingest failed.
    """
    job = db.get(Job, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    file = db.get(File, job.file_id)
    if file is None:
        raise ValueError(f"File {job.file_id} not found")

    job.stage = "extract"
    file.status = "processing"
    mark_running(db, job)

    try:
        result = extraction.extract_text(Path(file.full_path), file.file_type)
    except Exception as exc:
        file.status = "failed"
        mark_failed(db, job, exc)
        return job, None

    job.stage = "clean"
    if result.ocr_applied:
        file.ocr_applied = True
    mark_progress(db, job)

    cleaned_text = cleaning.clean(result.text)

    job.stage = "chunk"
    mark_progress(db, job)

    rag_service.chunk_and_store(db, file.id, cleaned_text)
    db.commit()

    file.status = "ready"
    mark_succeeded(db, job)

    embed_job = Job(
        type="embed",
        file_id=file.id,
        parent_job_id=job.id,
        trigger=job.trigger,
    )
    db.add(embed_job)
    db.commit()
    db.refresh(embed_job)

    return job, embed_job


