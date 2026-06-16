import uuid

from app.modules.processing import service
from app.modules.rag.tasks import run_embedding_job
from app.shared.database import SessionLocal
from app.shared.queue import embedding_queue


def run_processing_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for a processing job (Job1).

    Runs in the worker process, so it opens its own DB session rather than
    reusing a FastAPI request-scoped one -- same pattern as scans/tasks.py.
    After Job1 succeeds, enqueues Job2 (embedding) to the low-priority queue.
    """
    db = SessionLocal()
    try:
        job = service.process_file(db, job_id)
        if job.status == "succeeded":
            embedding_queue.enqueue(run_embedding_job, job.file_id)
    finally:
        db.close()
