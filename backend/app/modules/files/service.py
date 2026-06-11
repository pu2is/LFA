import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files.models import RegisteredPath


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
