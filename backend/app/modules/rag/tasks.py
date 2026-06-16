import uuid

from app.modules.rag import service
from app.shared.database import SessionLocal


def run_embedding_job(file_id: uuid.UUID) -> None:
    """RQ entrypoint for Job2 (embedding).

    Runs in the worker process — opens its own DB session, same pattern as
    processing/tasks.py and scans/tasks.py.
    """
    db = SessionLocal()
    try:
        service.embed_file(db, file_id)
    finally:
        db.close()
