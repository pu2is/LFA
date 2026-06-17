import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.labeling import service
from app.modules.labeling.presets import get_preset_catalog
from app.modules.labeling.schemas import (
    FileLabelAdd,
    FileLabelBatchPatch,
    FileLabelRead,
    LabelBulkCreate,
    LabelBulkCreateResult,
    LabelRead,
    PresetRead,
)
from app.shared.database import get_db


def _require_file(db: Session, file_id: uuid.UUID) -> None:
    if db.get(File, file_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

router = APIRouter(tags=["labels"])


# Static path declared before any dynamic /labels/{label_id} route so
# "presets" can never be captured as a label_id.
@router.get("/labels/presets", response_model=list[PresetRead])
def list_presets() -> list[dict[str, str | bool]]:
    return get_preset_catalog()


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
