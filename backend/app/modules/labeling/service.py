import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.labeling.presets import TAG_KIND_PRESETS, TYPE_LABEL_PRESETS


def normalize_label_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# ADR-0001 foundation: catalog seed helpers (type_labels / tag_kinds)
# --------------------------------------------------------------------------- #

def ensure_type_catalog(db: Session) -> list[TypeLabel]:
    """Return all type_labels; auto-populate from presets if empty."""
    types = list(db.scalars(select(TypeLabel)))
    if types:
        return types

    new_types = [TypeLabel(name=name) for name in TYPE_LABEL_PRESETS]
    db.add_all(new_types)
    db.commit()
    for t in new_types:
        db.refresh(t)
    return new_types


def ensure_tag_kind_catalog(db: Session) -> list[TagKind]:
    """Return all tag_kinds; auto-populate from TAG_KIND_PRESETS if empty."""
    kinds = list(db.scalars(select(TagKind)))
    if kinds:
        return kinds

    new_kinds = [TagKind(name=name) for name in TAG_KIND_PRESETS]
    db.add_all(new_kinds)
    db.commit()
    for k in new_kinds:
        db.refresh(k)
    return new_kinds


def get_files_with_type_or_tag_labels(db: Session, file_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    """Which of these file_ids have any type_labels_files or tag_labels rows yet.

    Drives /label's initial-vs-augment routing (docs/workflow/01c-file-label-
    augment.md): none yet -> mode=initial; any -> mode=augment. Batched (2
    queries for the whole list) rather than per-file, since callers route a
    whole batch of files at once (POST /label/files|paths).
    """
    if not file_ids:
        return set()
    typed = set(db.scalars(select(TypeLabelFile.file_id).where(TypeLabelFile.file_id.in_(file_ids))))
    tagged = set(db.scalars(select(TagLabel.file_id).where(TagLabel.file_id.in_(file_ids))))
    return typed | tagged


# --------------------------------------------------------------------------- #
# Type-label catalog CRUD (ADR-0001; mirrors Label CRUD above)
# --------------------------------------------------------------------------- #

def list_type_labels(db: Session) -> list[TypeLabel]:
    return list(db.scalars(select(TypeLabel)))


def get_type_label(db: Session, type_label_id: uuid.UUID) -> TypeLabel | None:
    return db.get(TypeLabel, type_label_id)


def bulk_create_type_labels(db: Session, names: list[str]) -> tuple[list[TypeLabel], list[str]]:
    unique_names = list(dict.fromkeys(names))

    existing_names = set(db.scalars(select(TypeLabel.name).where(TypeLabel.name.in_(unique_names))))
    skipped = [name for name in unique_names if name in existing_names]
    types = [TypeLabel(name=name) for name in unique_names if name not in existing_names]

    db.add_all(types)
    db.commit()
    for t in types:
        db.refresh(t)
    return types, skipped


def delete_type_label(db: Session, type_label: TypeLabel) -> None:
    db.delete(type_label)
    db.commit()


# --------------------------------------------------------------------------- #
# Tag-kind catalog CRUD (ADR-0001; mirrors Label CRUD above)
# --------------------------------------------------------------------------- #

def list_tag_kinds(db: Session) -> list[TagKind]:
    return list(db.scalars(select(TagKind)))


def get_tag_kind(db: Session, tag_kind_id: uuid.UUID) -> TagKind | None:
    return db.get(TagKind, tag_kind_id)


def bulk_create_tag_kinds(db: Session, names: list[str]) -> tuple[list[TagKind], list[str]]:
    unique_names = list(dict.fromkeys(names))

    existing_names = set(db.scalars(select(TagKind.name).where(TagKind.name.in_(unique_names))))
    skipped = [name for name in unique_names if name in existing_names]
    kinds = [TagKind(name=name) for name in unique_names if name not in existing_names]

    db.add_all(kinds)
    db.commit()
    for k in kinds:
        db.refresh(k)
    return kinds, skipped


def delete_tag_kind(db: Session, tag_kind: TagKind) -> None:
    db.delete(tag_kind)
    db.commit()


# --------------------------------------------------------------------------- #
# Type-label file review operations (ADR-0001 / 01x; mirrors file-label review above)
# --------------------------------------------------------------------------- #

def list_type_labels_files(db: Session, file_id: uuid.UUID) -> list[TypeLabelFile]:
    return list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file_id)))


def get_type_labels_file_by_id(db: Session, type_label_file_id: uuid.UUID) -> TypeLabelFile | None:
    return db.get(TypeLabelFile, type_label_file_id)


def get_type_labels_file_by_catalog(
    db: Session, file_id: uuid.UUID, type_label_id: uuid.UUID
) -> TypeLabelFile | None:
    """Fetch a (file_id, type_label_id) row. Used for duplicate checks on manual add."""
    return db.scalar(
        select(TypeLabelFile).where(
            TypeLabelFile.file_id == file_id,
            TypeLabelFile.type_label_id == type_label_id,
        )
    )


def batch_patch_type_labels_files(
    db: Session,
    file_id: uuid.UUID,
    operations: list[tuple[uuid.UUID, str]],
) -> list[TypeLabelFile]:
    """Confirm or reject type_labels_files rows in bulk, addressed by their own id.

    All-or-nothing: raises ValueError listing any IDs not found or not belonging
    to this file so the caller can return a 404 before touching the DB.
    """
    ids = [row_id for row_id, _ in operations]
    rows_by_id = {row.id: row for row in db.scalars(select(TypeLabelFile).where(TypeLabelFile.id.in_(ids)))}

    to_update: list[tuple[TypeLabelFile, str]] = []
    missing: list[str] = []
    for row_id, action in operations:
        row = rows_by_id.get(row_id)
        if row is None or row.file_id != file_id:
            missing.append(str(row_id))
        else:
            to_update.append((row, action))
    if missing:
        raise ValueError(f"type_labels_file not found for id(s): {', '.join(missing)}")

    for row, action in to_update:
        row.status = "confirmed" if action == "confirm" else "rejected"

    db.commit()
    for row, _ in to_update:
        db.refresh(row)
    return [row for row, _ in to_update]


def add_user_type_label(db: Session, file_id: uuid.UUID, type_label: TypeLabel) -> TypeLabelFile:
    row = TypeLabelFile(
        file_id=file_id,
        type_label_id=type_label.id,
        source="user",
        status="confirmed",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def remove_type_labels_file(db: Session, row: TypeLabelFile) -> None:
    db.delete(row)
    db.commit()


# --------------------------------------------------------------------------- #
# Tag-label file review operations (ADR-0001 / 01x; mirrors file-label review above)
# --------------------------------------------------------------------------- #

def list_tag_labels(db: Session, file_id: uuid.UUID) -> list[TagLabel]:
    return list(db.scalars(select(TagLabel).where(TagLabel.file_id == file_id)))


def get_tag_label_by_id(db: Session, tag_label_id: uuid.UUID) -> TagLabel | None:
    return db.get(TagLabel, tag_label_id)


def get_tag_label_by_kind_and_value(
    db: Session, file_id: uuid.UUID, kind_id: uuid.UUID, value: str
) -> TagLabel | None:
    """Fetch a (file_id, kind_id, value) row. Used for duplicate checks on manual add."""
    return db.scalar(
        select(TagLabel).where(
            TagLabel.file_id == file_id,
            TagLabel.kind_id == kind_id,
            TagLabel.value == value,
        )
    )


def batch_patch_tag_labels(
    db: Session,
    file_id: uuid.UUID,
    operations: list[tuple[uuid.UUID, str]],
) -> list[TagLabel]:
    """Confirm or reject tag_labels rows in bulk, addressed by their own id.

    All-or-nothing: raises ValueError listing any IDs not found or not belonging
    to this file so the caller can return a 404 before touching the DB.
    """
    ids = [row_id for row_id, _ in operations]
    rows_by_id = {row.id: row for row in db.scalars(select(TagLabel).where(TagLabel.id.in_(ids)))}

    to_update: list[tuple[TagLabel, str]] = []
    missing: list[str] = []
    for row_id, action in operations:
        row = rows_by_id.get(row_id)
        if row is None or row.file_id != file_id:
            missing.append(str(row_id))
        else:
            to_update.append((row, action))
    if missing:
        raise ValueError(f"tag_label not found for id(s): {', '.join(missing)}")

    for row, action in to_update:
        row.status = "confirmed" if action == "confirm" else "rejected"

    db.commit()
    for row, _ in to_update:
        db.refresh(row)
    return [row for row, _ in to_update]


def add_user_tag_label(db: Session, file_id: uuid.UUID, kind: TagKind, value: str) -> TagLabel:
    row = TagLabel(
        file_id=file_id,
        kind_id=kind.id,
        value=value,
        source="user",
        status="confirmed",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def remove_tag_label(db: Session, row: TagLabel) -> None:
    db.delete(row)
    db.commit()
