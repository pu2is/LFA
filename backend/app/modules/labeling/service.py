import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import FileLabel, Label


def normalize_label_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Label CRUD (pre-existing)
# --------------------------------------------------------------------------- #

def list_labels(db: Session) -> list[Label]:
    return list(db.scalars(select(Label)))


def get_label(db: Session, label_id: uuid.UUID) -> Label | None:
    return db.get(Label, label_id)


def bulk_create_labels(db: Session, names: list[str]) -> tuple[list[Label], list[str]]:
    # In-request dedupe, preserving first-seen order.
    unique_names = list(dict.fromkeys(names))

    existing_names = set(db.scalars(select(Label.name).where(Label.name.in_(unique_names))))
    skipped = [name for name in unique_names if name in existing_names]
    labels = [Label(name=name) for name in unique_names if name not in existing_names]

    # One transaction for the whole batch: all rows land or none do.
    db.add_all(labels)
    db.commit()
    for label in labels:
        db.refresh(label)
    return labels, skipped


def delete_label(db: Session, label: Label) -> None:
    db.delete(label)
    db.commit()


def file_has_labels(db: Session, file_id: uuid.UUID) -> bool:
    return db.scalar(
        select(FileLabel.id).where(FileLabel.file_id == file_id).limit(1)
    ) is not None


# --------------------------------------------------------------------------- #
# File-label review operations
# --------------------------------------------------------------------------- #

def list_file_labels(db: Session, file_id: uuid.UUID) -> list[FileLabel]:
    return list(db.scalars(select(FileLabel).where(FileLabel.file_id == file_id)))


def get_file_label_by_id(db: Session, file_label_id: uuid.UUID) -> FileLabel | None:
    """Fetch a file_label row by its own PK (works for both catalog and free-text rows)."""
    return db.get(FileLabel, file_label_id)


def get_file_label_by_catalog(db: Session, file_id: uuid.UUID, label_id: uuid.UUID) -> FileLabel | None:
    """Fetch a catalog file_label row by (file_id, label_id). Used for duplicate checks."""
    return db.scalar(
        select(FileLabel).where(
            FileLabel.file_id == file_id,
            FileLabel.label_id == label_id,
        )
    )


def batch_patch_file_labels(
    db: Session,
    file_id: uuid.UUID,
    operations: list[tuple[uuid.UUID, str]],
) -> list[FileLabel]:
    """Confirm or reject file_label rows in bulk, addressed by file_labels.id.

    All-or-nothing: raises ValueError listing any IDs not found or not belonging
    to this file so the caller can return a 404 before touching the DB.
    """
    to_update: list[tuple[FileLabel, str]] = []
    missing: list[str] = []
    for file_label_id, action in operations:
        fl = get_file_label_by_id(db, file_label_id)
        if fl is None or fl.file_id != file_id:
            missing.append(str(file_label_id))
        else:
            to_update.append((fl, action))
    if missing:
        raise ValueError(f"file_label not found for label_id(s): {', '.join(missing)}")

    for fl, action in to_update:
        fl.status = "confirmed" if action == "confirm" else "rejected"

    db.commit()
    for fl, _ in to_update:
        db.refresh(fl)
    return [fl for fl, _ in to_update]


def add_user_label(db: Session, file_id: uuid.UUID, label: Label) -> FileLabel:
    fl = FileLabel(
        file_id=file_id,
        label_id=label.id,
        label_name=label.name,
        source="user",
        status="confirmed",
        confidence=None,
    )
    db.add(fl)
    db.commit()
    db.refresh(fl)
    return fl


def remove_file_label(db: Session, fl: FileLabel) -> None:
    db.delete(fl)
    db.commit()
