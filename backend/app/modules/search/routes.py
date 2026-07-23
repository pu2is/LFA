import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.modules.labeling import service as labeling_service
from app.modules.search.schemas import TagFacetRead
from app.shared.database import get_db

router = APIRouter(tags=["search"])


@router.get("/search/tag-facets", response_model=list[TagFacetRead])
def get_tag_facets(
    type_label_ids: list[uuid.UUID] = Query(default=[]),
    db: Session = Depends(get_db),
):
    """WF2a (ADR-0002a D3): candidate-set tag facets for the manual filter's
    type->tag narrowing step. Empty type_label_ids = full candidate set."""
    for type_label_id in type_label_ids:
        if labeling_service.get_type_label(db, type_label_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Type label {type_label_id} not found")

    facets = labeling_service.get_tag_facets(db, type_label_ids)
    return [TagFacetRead(kind_id=kind.id, kind_name=kind.name, values=values) for kind, values in facets]
