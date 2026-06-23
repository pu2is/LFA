import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.jobs.models import Job
from app.modules.jobs.schemas import JobRead
from app.modules.scans import service
from app.modules.scans.schemas import ScanCreate
from app.modules.scans.tasks import run_scan_job
from app.shared.database import get_db
from app.shared.queue import scan_queue

router = APIRouter(tags=["scans"])


@router.post("/scans", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
def create_scan(payload: ScanCreate, db: Session = Depends(get_db)) -> Job:
    if files_service.get_path(db, payload.path_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path not found")

    scan = service.create_scan(db, payload.path_id)
    scan_queue.enqueue(run_scan_job, scan.id)
    return scan


@router.get("/scans/{scan_id}", response_model=JobRead)
def get_scan(scan_id: uuid.UUID, db: Session = Depends(get_db)) -> Job:
    scan = service.get_scan(db, scan_id)
    if scan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan
