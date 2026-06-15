import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath


def get_path_by_value(db: Session, path: str) -> RegisteredPath | None:
    return db.scalar(select(RegisteredPath).where(RegisteredPath.path == path))


def get_path(db: Session, path_id: uuid.UUID) -> RegisteredPath | None:
    return db.get(RegisteredPath, path_id)


def list_paths(db: Session) -> list[RegisteredPath]:
    return list(db.scalars(select(RegisteredPath)))


def create_path(db: Session, path: str) -> RegisteredPath:
    registered_path = RegisteredPath(path=path)
    db.add(registered_path)
    db.commit()
    db.refresh(registered_path)
    return registered_path


def delete_path(db: Session, registered_path: RegisteredPath) -> None:
    db.delete(registered_path)
    db.commit()


def get_file_by_full_path(db: Session, full_path: str) -> File | None:
    return db.scalar(select(File).where(File.full_path == full_path))


def upsert_file(
    db: Session,
    *,
    path_id: uuid.UUID,
    full_path: str,
    filename: str,
    file_type: str,
    file_size: int,
    file_hash: str,
    file_created_at: datetime | None,
    file_modified_at: datetime,
) -> File:
    """Insert a new file row, or refresh filesystem metadata on an existing one.

    Matched by `full_path` (unique). Deliberately leaves `status` and
    `ocr_applied` untouched on existing rows: those reflect later pipeline
    stages (OCR, labeling) that a scan must not silently reset, even if the
    file's content changed (see er-diagram.md "modified" handling).
    """
    file = get_file_by_full_path(db, full_path)
    if file is not None:
        file.path_id = path_id
        file.filename = filename
        file.file_type = file_type
        file.file_size = file_size
        file.file_hash = file_hash
        file.file_created_at = file_created_at
        file.file_modified_at = file_modified_at
        return file

    file = File(
        path_id=path_id,
        filename=filename,
        full_path=full_path,
        file_type=file_type,
        file_size=file_size,
        file_hash=file_hash,
        file_created_at=file_created_at,
        file_modified_at=file_modified_at,
    )
    db.add(file)
    return file
