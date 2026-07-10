import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.files import service
from app.modules.files.models import RegisteredPath
from app.modules.files.schemas import FileRead, PathCreate, PathRead
from app.shared.database import get_db

router = APIRouter(tags=["paths"])


@router.post("/paths", response_model=PathRead, status_code=status.HTTP_201_CREATED)
def register_path(payload: PathCreate, db: Session = Depends(get_db)) -> RegisteredPath:
    if service.get_path_by_value(db, payload.path) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Path is already registered")
    return service.create_path(db, payload.path)


@router.get("/paths", response_model=list[PathRead])
def list_registered_paths(db: Session = Depends(get_db)) -> list[RegisteredPath]:
    return service.list_paths(db)


@router.delete("/paths/{path_id}", response_model=PathRead)
def remove_path(path_id: uuid.UUID, db: Session = Depends(get_db)) -> PathRead:
    registered_path = service.get_path(db, path_id)
    if registered_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path not found")

    deleted = PathRead.model_validate(registered_path)
    service.delete_path(db, registered_path)
    return deleted


@router.get("/files", response_model=list[FileRead])
def list_files(path_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> list[FileRead]:
    if path_id is not None and service.get_path(db, path_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path not found")
    rows = service.list_files(db, path_id)
    return [
        FileRead.model_validate(file).model_copy(update={
            "processing_job_status": job_status,
            "processing_error_message": error_msg,
            "processing_job_type": job_type,
        })
        for file, job_status, error_msg, job_type in rows
    ]
