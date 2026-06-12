import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.labeling import service
from app.modules.labeling.presets import get_preset_catalog
from app.modules.labeling.schemas import LabelBulkCreate, LabelBulkCreateResult, LabelRead, PresetRead
from app.shared.database import get_db

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
