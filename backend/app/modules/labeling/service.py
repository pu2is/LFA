import uuid

from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.labeling.presets import TAG_KIND_PRESETS, TYPE_LABEL_PRESETS


def normalize_label_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# ADR-0001 foundation: catalog seed helpers (type_labels / tag_kinds)
# --------------------------------------------------------------------------- #

def ensure_type_catalog(db: Session) -> list[TypeLabel]:
    """Return all type_labels; auto-populate from presets if empty.

    #51: two RQ workers can both see an empty catalog on the first-ever label
    job and both try to seed it. Seeding is INSERT ... ON CONFLICT (lower(name))
    DO NOTHING (conflict target matches #49's case-insensitive expression
    index) instead of add_all + commit, so the loser's redundant inserts are
    silently skipped instead of raising IntegrityError and failing the job.
    Re-reads afterward -- the concurrent seeder may have inserted rows this
    call didn't.
    """
    types = list(db.scalars(select(TypeLabel)))
    if types:
        return types

    stmt = (
        pg_insert(TypeLabel)
        .values([{"name": name} for name in TYPE_LABEL_PRESETS])
        .on_conflict_do_nothing(index_elements=[func.lower(TypeLabel.name)])
    )
    db.execute(stmt)
    db.commit()
    return list(db.scalars(select(TypeLabel)))


def ensure_tag_kind_catalog(db: Session) -> list[TagKind]:
    """Return all tag_kinds; auto-populate from TAG_KIND_PRESETS if empty.
    See ensure_type_catalog for the #51 concurrency-safe seeding rationale.
    """
    kinds = list(db.scalars(select(TagKind)))
    if kinds:
        return kinds

    stmt = (
        pg_insert(TagKind)
        .values([{"name": name} for name in TAG_KIND_PRESETS])
        .on_conflict_do_nothing(index_elements=[func.lower(TagKind.name)])
    )
    db.execute(stmt)
    db.commit()
    return list(db.scalars(select(TagKind)))


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
    """#51: same check-then-insert race as ensure_type_catalog, user-triggered
    instead of first-job-triggered -- two concurrent bulk-creates for an
    overlapping name used to be able to crash one of them on the UNIQUE
    index. INSERT ... ON CONFLICT (lower(name)) DO NOTHING + RETURNING makes
    it race-safe: only actually-inserted rows come back, so `skipped` (names
    that already existed, whether before this call or via a losing race
    against a concurrent caller) falls out the same way either way.
    """
    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        return [], []

    stmt = (
        pg_insert(TypeLabel)
        .values([{"name": name} for name in unique_names])
        .on_conflict_do_nothing(index_elements=[func.lower(TypeLabel.name)])
        .returning(TypeLabel)
    )
    types = list(db.scalars(stmt))
    db.commit()
    for t in types:
        db.refresh(t)

    inserted_names = {t.name for t in types}
    skipped = [name for name in unique_names if name not in inserted_names]
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
    """#51: see bulk_create_type_labels for the concurrency-safe seeding rationale."""
    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        return [], []

    stmt = (
        pg_insert(TagKind)
        .values([{"name": name} for name in unique_names])
        .on_conflict_do_nothing(index_elements=[func.lower(TagKind.name)])
        .returning(TagKind)
    )
    kinds = list(db.scalars(stmt))
    db.commit()
    for k in kinds:
        db.refresh(k)

    inserted_names = {k.name for k in kinds}
    skipped = [name for name in unique_names if name not in inserted_names]
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


def upsert_user_type_label(
    db: Session, file_id: uuid.UUID, type_label_id: uuid.UUID
) -> tuple[TypeLabelFile, bool]:
    """Idempotent manual add (#50): INSERT ... ON CONFLICT (file_id, type_label_id)
    DO UPDATE SET status='confirmed'. Structurally removes the old check-then-
    insert race (two concurrent identical adds both land here safely: one
    insert, one no-op-ish update) and the dead end where a REJECTED row blocked
    re-adding via 409 -- a conflict now just flips it to confirmed.

    DO UPDATE touches status (and updated_at) only, never source: source is
    provenance (who originally suggested this), status is the user's verdict
    -- mirrors the existing PATCH confirm path, which also never rewrites
    source. updated_at is set explicitly here because this bypasses the ORM's
    onupdate=func.now() (that only fires for ORM-tracked attribute writes,
    not a Core on_conflict_do_update's SET clause).

    Returns (row, inserted) so the route can pick 201 (fresh insert) vs
    200 (existing row, now confirmed). inserted comes from the classic
    Postgres `xmax = 0` trick: a freshly inserted row's xmax is 0, while a
    row touched by the DO UPDATE branch has it set to the current transaction.
    """
    stmt = (
        pg_insert(TypeLabelFile)
        .values(file_id=file_id, type_label_id=type_label_id, source="user", status="confirmed")
        .on_conflict_do_update(
            index_elements=[TypeLabelFile.file_id, TypeLabelFile.type_label_id],
            set_={"status": "confirmed", "updated_at": func.now()},
        )
        .returning(TypeLabelFile, literal_column("(xmax = 0)").label("inserted"))
        .execution_options(populate_existing=True)
    )
    result = db.execute(stmt).one()
    row, inserted = result[0], result.inserted
    db.commit()
    db.refresh(row)
    return row, inserted


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


def upsert_user_tag_label(
    db: Session, file_id: uuid.UUID, kind_id: uuid.UUID, value: str
) -> tuple[TagLabel, bool]:
    """Idempotent manual add (#50), tag facet -- see upsert_user_type_label for
    the source/status split and race-elimination rationale.

    Conflict target is the case-insensitive expression index from #49
    (ix_tag_labels_file_kind_value_lower): "Berlin" and "berlin" collide here
    even though value itself is stored as-is, un-normalized.
    """
    stmt = (
        pg_insert(TagLabel)
        .values(file_id=file_id, kind_id=kind_id, value=value, source="user", status="confirmed")
        .on_conflict_do_update(
            index_elements=[TagLabel.file_id, TagLabel.kind_id, func.lower(TagLabel.value)],
            set_={"status": "confirmed", "updated_at": func.now()},
        )
        .returning(TagLabel, literal_column("(xmax = 0)").label("inserted"))
        .execution_options(populate_existing=True)
    )
    result = db.execute(stmt).one()
    row, inserted = result[0], result.inserted
    db.commit()
    db.refresh(row)
    return row, inserted


def remove_tag_label(db: Session, row: TagLabel) -> None:
    db.delete(row)
    db.commit()


# --------------------------------------------------------------------------- #
# Smart Search facet query (WF2a, ADR-0002a D3 / #56)
# --------------------------------------------------------------------------- #

def get_tag_facets(
    db: Session, type_label_ids: list[uuid.UUID] | None = None
) -> list[tuple[TagKind, list[str]]]:
    """Confirmed tag values that occur in the candidate file set, grouped by
    kind. Candidate set = all files if type_label_ids is empty, else files
    with a confirmed type_labels_files row for one of the given type ids.

    Case-insensitive dedup (#56 AC): "Berlin" and "berlin" collapse into one
    value. DISTINCT ON (kind_id, lower(value)) picks a deterministic
    representative -- the byte-order-smallest raw value for that (kind,
    lower(value)) pair -- instead of an arbitrary DB visitation order. The
    tie-break explicitly COLLATEs "C" so the choice doesn't depend on the
    Postgres cluster's default locale (e.g. some locales would otherwise
    sort "berlin" before "Berlin"). The same ORDER BY also gives
    alphabetical values per kind for free.
    """
    conditions = [TagLabel.status == "confirmed"]
    if type_label_ids:
        candidate_files = select(TypeLabelFile.file_id).where(
            TypeLabelFile.status == "confirmed",
            TypeLabelFile.type_label_id.in_(type_label_ids),
        )
        conditions.append(TagLabel.file_id.in_(candidate_files))

    stmt = (
        select(TagLabel.kind_id, TagLabel.value)
        .distinct(TagLabel.kind_id, func.lower(TagLabel.value))
        .where(*conditions)
        .order_by(TagLabel.kind_id, func.lower(TagLabel.value), TagLabel.value.collate("C"))
    )
    values_by_kind: dict[uuid.UUID, list[str]] = {}
    for kind_id, value in db.execute(stmt):
        values_by_kind.setdefault(kind_id, []).append(value)

    if not values_by_kind:
        return []

    kinds = {k.id: k for k in db.scalars(select(TagKind).where(TagKind.id.in_(values_by_kind.keys())))}
    return [(kinds[kind_id], values) for kind_id, values in values_by_kind.items()]
