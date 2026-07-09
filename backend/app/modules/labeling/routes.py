import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.labeling import service
from app.modules.labeling.presets import get_preset_catalog
from app.modules.labeling.schemas import (
    FileLabelAdd,
    FileLabelBatchPatch,
    FileLabelRead,
    LabelBulkCreate,
    LabelBulkCreateResult,
    LabelByFileIdsRequest,
    LabelByPathIdsRequest,
    LabelJobEnqueued,
    LabelJobResult,
    LabelJobSkipped,
    LabelRead,
    PresetRead,
)
from app.modules.labeling.tasks import run_label_job
from app.shared.database import get_db
from app.shared.queue import JOB_RETRY, label_queue


def _require_file(db: Session, file_id: uuid.UUID) -> None:
    if db.get(File, file_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

router = APIRouter(tags=["labels"])


# Static path declared before any dynamic /labels/{label_id} route so
# "presets" can never be captured as a label_id.
@router.get("/labels/presets", response_model=list[PresetRead])
def list_presets() -> list[dict[str, str | bool]]:
    return get_preset_catalog()


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

    for file in files:
        if file.status != "ready":
            skipped.append(LabelJobSkipped(file_id=file.id, reason=f"status is '{file.status}', not 'ready'"))
            continue

        has_labels = service.file_has_labels(db, file.id)
        mode = "augment" if has_labels else "initial"

        job = Job(type="label", file_id=file.id, trigger="manual", mode=mode)
        db.add(job)
        db.flush()

        # No at_front: cross-queue priority (label always drained before
        # ingest/scan/embed) is enforced by worker queue listen order, see
        # docs/dev-setup.md. at_front only reordered *within* this queue,
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


@router.get("/labels", response_model=list[LabelRead])
def list_labels(db: Session = Depends(get_db)):
    return service.list_labels(db)


@router.post("/labels", response_model=LabelBulkCreateResult, status_code=status.HTTP_201_CREATED)
def create_labels(payload: LabelBulkCreate, db: Session = Depends(get_db)):
    created, skipped = service.bulk_create_labels(db, payload.names)
    return {"created": created, "skipped": skipped}


@router.delete("/labels/{label_id}", response_model=LabelRead)
def remove_label(label_id: uuid.UUID, db: Session = Depends(get_db)) -> LabelRead:
    label = service.get_label(db, label_id)
    if label is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Label not found")

    deleted = LabelRead.model_validate(label)
    service.delete_label(db, label)
    return deleted


# --- File-label review endpoints ---

@router.get("/files/{file_id}/labels", response_model=list[FileLabelRead])
def list_file_labels(file_id: uuid.UUID, db: Session = Depends(get_db)):
    _require_file(db, file_id)
    return service.list_file_labels(db, file_id)


@router.patch("/files/{file_id}/labels", response_model=list[FileLabelRead])
def batch_patch_file_labels(
    file_id: uuid.UUID,
    payload: FileLabelBatchPatch,
    db: Session = Depends(get_db),
):
    _require_file(db, file_id)
    try:
        return service.batch_patch_file_labels(
            db, file_id, [(op.file_label_id, op.action) for op in payload.operations]
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/files/{file_id}/labels", response_model=FileLabelRead, status_code=status.HTTP_201_CREATED)
def add_user_label(
    file_id: uuid.UUID,
    payload: FileLabelAdd,
    db: Session = Depends(get_db),
):
    _require_file(db, file_id)
    label = service.get_label(db, payload.label_id)
    if label is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Label not found")
    if service.get_file_label_by_catalog(db, file_id, payload.label_id) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Label already attached to this file")
    return service.add_user_label(db, file_id, label)


@router.delete("/files/{file_id}/labels/{file_label_id}", response_model=FileLabelRead)
def remove_file_label(
    file_id: uuid.UUID,
    file_label_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    fl = service.get_file_label_by_id(db, file_label_id)
    if fl is None or fl.file_id != file_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File label not found")
    deleted = FileLabelRead.model_validate(fl)
    service.remove_file_label(db, fl)
    return deleted
