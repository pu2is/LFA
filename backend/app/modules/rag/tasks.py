import uuid

from app.modules.rag import service
from app.shared.database import SessionLocal


def run_embedding_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for an embed job.

    Delegates to service.run_embed() which manages job lifecycle
    (status, timestamps, events) and calls embed_file().
    """
    db = SessionLocal()
    try:
        service.run_embed(db, job_id)
    finally:
        db.close()
