import uuid

from app.modules.labeling import service
from app.shared.database import SessionLocal


def run_label_job(job_id: uuid.UUID) -> None:
    """RQ entrypoint for a label job (initial or augment)."""
    db = SessionLocal()
    try:
        service.run_label(db, job_id)
    finally:
        db.close()
