import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.scans import discovery
from app.modules.scans.models import Scan


def create_scan(db: Session, path_id: uuid.UUID) -> Scan:
    scan = Scan(path_id=path_id, status="queued")
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def get_scan(db: Session, scan_id: uuid.UUID) -> Scan | None:
    return db.get(Scan, scan_id)


def run_scan(db: Session, scan_id: uuid.UUID) -> Scan:
    """Walk the scan's registered path and upsert a File row per supported document.

    Synchronous on purpose (issue #4 "sync core"): no RQ enqueue, OCR, text
    extraction, embeddings, or move-detection diffing here. Those build on
    top of the File rows this function writes.
    """
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise ValueError(f"Scan {scan_id} not found")

    registered_path = files_service.get_path(db, scan.path_id)
    if registered_path is None:
        raise ValueError(f"Registered path {scan.path_id} not found")

    scan.status = "running"
    scan.started_at = datetime.now(timezone.utc)
    db.commit()

    try:
        for doc in discovery.iter_documents(Path(registered_path.path)):
            files_service.upsert_file(
                db,
                path_id=registered_path.id,
                full_path=str(doc.path),
                filename=doc.path.name,
                file_type=doc.file_type,
                file_size=doc.file_size,
                file_hash=doc.file_hash,
                file_created_at=doc.file_created_at,
                file_modified_at=doc.file_modified_at,
            )
    except OSError as exc:
        # Keep whatever files were already upserted before the error -- the
        # scan can simply be re-run later (upsert is idempotent) to pick up
        # the rest, instead of throwing away partial progress.
        scan.status = "failed"
        scan.error_message = str(exc)
        scan.completed_at = datetime.now(timezone.utc)
        db.commit()
        return scan

    scan.status = "completed"
    scan.completed_at = datetime.now(timezone.utc)
    registered_path.last_scanned_at = scan.completed_at
    db.commit()
    return scan
