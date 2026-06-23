import uuid

from app.modules.processing import service
from app.modules.rag.tasks import run_embedding_job
from app.shared.database import SessionLocal
from app.shared.queue import embedding_queue


def run_ingest_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for an ingest job (extract -> clean -> chunk).

    Uses the unified jobs table. No labeling -- that is a separate manual step.
    """
    db = SessionLocal()
    try:
        job = service.run_ingest(db, job_id)
        if job.status == "succeeded":
            embedding_queue.enqueue(run_embedding_job, job.file_id)
    finally:
        db.close()
