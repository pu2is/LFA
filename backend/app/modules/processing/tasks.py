import uuid

from app.modules.processing import service
from app.modules.rag.tasks import run_embedding_job
from app.shared.database import SessionLocal
from app.shared.queue import embedding_queue


def run_ingest_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for an ingest job (extract -> clean -> chunk).

    Uses the unified jobs table. No labeling -- that is a separate manual step.
    On success, fans out one embed job (same pattern as scan -> ingest).
    """
    db = SessionLocal()
    try:
        _ingest_job, embed_job = service.run_ingest(db, job_id)
        if embed_job is not None:
            embedding_queue.enqueue(run_embedding_job, embed_job.id)
    finally:
        db.close()
