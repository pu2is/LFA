import uuid

from app.modules.scans import service
from app.shared.database import SessionLocal


def run_scan_job(scan_id: uuid.UUID) -> None:
    """RQ entrypoint for a scan.

    Runs in the worker process, so it cannot reuse a FastAPI request-scoped
    session -- it opens and closes its own.
    """
    db = SessionLocal()
    try:
        service.run_scan(db, scan_id)
    finally:
        db.close()
