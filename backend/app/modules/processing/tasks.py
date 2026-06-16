import uuid

from app.modules.processing import service
from app.shared.database import SessionLocal


def run_processing_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for a processing job.

    Runs in the worker process, so it opens its own DB session rather than
    reusing a FastAPI request-scoped one -- same pattern as scans/tasks.py.
    """
    db = SessionLocal()
    try:
        service.process_file(db, job_id)
    finally:
        db.close()
