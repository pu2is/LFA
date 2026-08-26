import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.labeling import service
from app.modules.labeling.schemas import (
    LabelByFileIdsRequest,
    LabelByPathIdsRequest,
    LabelJobEnqueued,
    LabelJobResult,
    LabelJobSkipped,
    TagKindBulkCreate,
    TagKindBulkCreateResult,
    TagKindRead,
    TagLabelAdd,
    TagLabelBatchPatch,
    TagLabelRead,
    TypeLabelBulkCreate,
    TypeLabelBulkCreateResult,
    TypeLabelFileAdd,
    TypeLabelFileBatchPatch,
    TypeLabelFileRead,
    TypeLabelRead,
)
from app.modules.labeling.tasks import run_label_job
from app.shared.database import get_db
from app.shared.queue import JOB_RETRY, label_queue


def _require_file(db: Session, file_id: uuid.UUID) -> None:
    if db.get(File, file_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")


# #55: constraint names for the one known inbound FK per catalog table
# (models.py); confirmed against the DB, not just the migration, since
# neither the ForeignKey() column defs nor the migration's
# ForeignKeyConstraint() calls pass an explicit name -- Postgres assigns the
# default "<table>_<column>_fkey" itself.
TYPE_LABEL_IN_USE_FK = "type_labels_files_type_label_id_fkey"
TAG_KIND_IN_USE_FK = "tag_labels_kind_id_fkey"


def _delete_catalog_entry_or_409(db: Session, delete_fn, entry, label: str, fk_constraint: str) -> None:
    """Shared body for the type-labels/tag-kinds catalog DELETE routes: run
    delete_fn, translating the FK-violation IntegrityError (entry still
    attached to files) into a 409 instead of an unhandled 500. Checks the
    violated constraint's name so an unrelated IntegrityError on the same
    delete (e.g. a second inbound FK added later) surfaces as-is instead of
    being misreported as "still referenced".
    """
    try:
        delete_fn(db, entry)
    except IntegrityError as exc:
        db.rollback()
        if exc.orig.diag.constraint_name != fk_constraint:
            raise
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{label} is still referenced by one or more files",
        ) from None


router = APIRouter(tags=["labels"])


def _enqueue_label_jobs(
    db: Session,
    files: list[File],
) -> LabelJobResult:
    """Create label jobs for eligible files and enqueue them.

    Batch semantics: all-or-nothing. Every Job row is created (flushed for an
    id) and handed to RQ before the single commit at the end, so a failure
    partway through (e.g. Redis unreachable on the 3rd file) rolls back every
    row from this call -- the caller never sees a batch that's half recorded.
    The one gap this can't close without a distributed transaction: if RQ has
    already accepted a job when the failure happens, that job is queued in
    Redis even though its DB row just got rolled back. Out of scope here (see
    #33 scope-out); acceptable because it only affects the failing request,
    not steady-state operation.
    """
    enqueued: list[LabelJobEnqueued] = []
    skipped: list[LabelJobSkipped] = []

    # One batched pair of queries for the whole request instead of one pair
    # per file -- see #47 code review.
    labeled_file_ids = service.get_files_with_type_or_tag_labels(db, [f.id for f in files])

    # #61: a labeled file whose most recent initial-mode job never finished
    # (Call 1 committed a type row, Call 2/3 crashed) must still route back
    # to initial, not augment -- only worth checking the subset that row
    # presence alone would otherwise send to augment.
    incomplete_initial_ids = service.get_files_with_incomplete_initial_job(db, list(labeled_file_ids))

    for file in files:
        if file.status != "ready":
            skipped.append(LabelJobSkipped(file_id=file.id, reason=f"status is '{file.status}', not 'ready'"))
            continue

        mode = "augment" if file.id in labeled_file_ids and file.id not in incomplete_initial_ids else "initial"

        job = Job(type="label", file_id=file.id, trigger="manual", mode=mode)
        db.add(job)
        db.flush()

        # No at_front: cross-queue priority (label always drained before
        # ingest/scan/embed) is enforced by worker queue listen order, see
        # docs/99_dev-setup.md. at_front only reordered *within* this queue,
        # which broke submission order for no priority benefit -- see #33.
        rq_job = label_queue.enqueue(run_label_job, job.id, retry=JOB_RETRY)
        job.rq_job_id = str(rq_job.id)

        enqueued.append(LabelJobEnqueued(job_id=job.id, file_id=file.id, mode=mode))

    db.commit()
    return LabelJobResult(enqueued=enqueued, skipped=skipped)


@router.post("/label/files", response_model=LabelJobResult, status_code=status.HTTP_202_ACCEPTED)
def label_files(payload: LabelByFileIdsRequest, db: Session = Depends(get_db)):
    """Enqueue label jobs for the given file IDs.

    Files in 'ready' status without existing labels get mode=initial;
    files with existing labels get mode=augment.
    Non-ready files are skipped. Unknown IDs return 404.
    """
    files: list[File] = []
    for file_id in payload.file_ids:
        file = db.get(File, file_id)
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found",
            )
        files.append(file)

    return _enqueue_label_jobs(db, files)


@router.post("/label/paths", response_model=LabelJobResult, status_code=status.HTTP_202_ACCEPTED)
def label_paths(payload: LabelByPathIdsRequest, db: Session = Depends(get_db)):
    """Enqueue label jobs for all ready files under the given path IDs."""
    for path_id in payload.path_ids:
        if files_service.get_path(db, path_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path {path_id} not found",
            )

    files = list(
        db.scalars(
            select(File).where(File.path_id.in_(payload.path_ids))
        )
    )
    return _enqueue_label_jobs(db, files)


# --- Type-label catalog CRUD (ADR-0001) ---

@router.get("/type-labels", response_model=list[TypeLabelRead])
def list_type_labels(db: Session = Depends(get_db)):
    return service.list_type_labels(db)


@router.post("/type-labels", response_model=TypeLabelBulkCreateResult, status_code=status.HTTP_201_CREATED)
def create_type_labels(payload: TypeLabelBulkCreate, db: Session = Depends(get_db)):
    created, skipped = service.bulk_create_type_labels(db, payload.names)
    return {"created": created, "skipped": skipped}


@router.delete("/type-labels/{type_label_id}", response_model=TypeLabelRead)
def remove_type_label(type_label_id: uuid.UUID, db: Session = Depends(get_db)) -> TypeLabelRead:
    type_label = service.get_type_label(db, type_label_id)
    if type_label is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Type label not found")

    deleted = TypeLabelRead.model_validate(type_label)
    _delete_catalog_entry_or_409(db, service.delete_type_label, type_label, "Type label", TYPE_LABEL_IN_USE_FK)
    return deleted


# --- Tag-kind catalog CRUD (ADR-0001) ---

@router.get("/tag-kinds", response_model=list[TagKindRead])
def list_tag_kinds(db: Session = Depends(get_db)):
    return service.list_tag_kinds(db)


@router.post("/tag-kinds", response_model=TagKindBulkCreateResult, status_code=status.HTTP_201_CREATED)
def create_tag_kinds(payload: TagKindBulkCreate, db: Session = Depends(get_db)):
    created, skipped = service.bulk_create_tag_kinds(db, payload.names)
    return {"created": created, "skipped": skipped}


@router.delete("/tag-kinds/{tag_kind_id}", response_model=TagKindRead)
def remove_tag_kind(tag_kind_id: uuid.UUID, db: Session = Depends(get_db)) -> TagKindRead:
    tag_kind = service.get_tag_kind(db, tag_kind_id)
    if tag_kind is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag kind not found")

    deleted = TagKindRead.model_validate(tag_kind)
    _delete_catalog_entry_or_409(db, service.delete_tag_kind, tag_kind, "Tag kind", TAG_KIND_IN_USE_FK)
    return deleted


# --- Type-label file review endpoints (ADR-0001 / 01x) ---

@router.get("/files/{file_id}/type-labels", response_model=list[TypeLabelFileRead])
def list_file_type_labels(file_id: uuid.UUID, db: Session = Depends(get_db)):
    _require_file(db, file_id)
    return service.list_type_labels_files(db, file_id)


@router.patch("/files/{file_id}/type-labels", response_model=list[TypeLabelFileRead])
def batch_patch_file_type_labels(
    file_id: uuid.UUID,
    payload: TypeLabelFileBatchPatch,
    db: Session = Depends(get_db),
):
    _require_file(db, file_id)
    try:
        return service.batch_patch_type_labels_files(
            db, file_id, [(op.type_label_file_id, op.action) for op in payload.operations]
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/files/{file_id}/type-labels", response_model=TypeLabelFileRead, status_code=status.HTTP_201_CREATED)
def add_user_type_label(
    file_id: uuid.UUID,
    payload: TypeLabelFileAdd,
    response: Response,
    db: Session = Depends(get_db),
):
    """Idempotent upsert (#50): a type_label_id already on this file (any
    status, including rejected) is confirmed in place -- 200, not 409/404."""
    _require_file(db, file_id)
    type_label = service.get_type_label(db, payload.type_label_id)
    if type_label is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Type label not found")
    row, inserted = service.upsert_user_type_label(db, file_id, payload.type_label_id)
    if not inserted:
        response.status_code = status.HTTP_200_OK
    return row


@router.delete("/files/{file_id}/type-labels/{type_label_file_id}", response_model=TypeLabelFileRead)
def remove_file_type_label(
    file_id: uuid.UUID,
    type_label_file_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    row = service.get_type_labels_file_by_id(db, type_label_file_id)
    if row is None or row.file_id != file_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Type label file not found")
    deleted = TypeLabelFileRead.model_validate(row)
    service.remove_type_labels_file(db, row)
    return deleted


# --- Tag-label file review endpoints (ADR-0001 / 01x) ---

@router.get("/files/{file_id}/tags", response_model=list[TagLabelRead])
def list_file_tags(file_id: uuid.UUID, db: Session = Depends(get_db)):
    _require_file(db, file_id)
    return service.list_tag_labels(db, file_id)


@router.patch("/files/{file_id}/tags", response_model=list[TagLabelRead])
def batch_patch_file_tags(
    file_id: uuid.UUID,
    payload: TagLabelBatchPatch,
    db: Session = Depends(get_db),
):
    _require_file(db, file_id)
    try:
        return service.batch_patch_tag_labels(
            db, file_id, [(op.tag_label_id, op.action) for op in payload.operations]
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/files/{file_id}/tags", response_model=TagLabelRead, status_code=status.HTTP_201_CREATED)
def add_user_tag(
    file_id: uuid.UUID,
    payload: TagLabelAdd,
    response: Response,
    db: Session = Depends(get_db),
):
    """Idempotent upsert (#50): a (kind_id, value) already on this file (any
    status, including rejected, matched case-insensitively per #49) is
    confirmed in place -- 200, not 409/404."""
    _require_file(db, file_id)
    kind = service.get_tag_kind(db, payload.kind_id)
    if kind is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag kind not found")
    row, inserted = service.upsert_user_tag_label(db, file_id, payload.kind_id, payload.value)
    if not inserted:
        response.status_code = status.HTTP_200_OK
    return row


@router.delete("/files/{file_id}/tags/{tag_label_id}", response_model=TagLabelRead)
def remove_file_tag(
    file_id: uuid.UUID,
    tag_label_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    row = service.get_tag_label_by_id(db, tag_label_id)
    if row is None or row.file_id != file_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found")
    deleted = TagLabelRead.model_validate(row)
    service.remove_tag_label(db, row)
    return deleted
