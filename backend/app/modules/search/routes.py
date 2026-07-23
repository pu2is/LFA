import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.schemas import FileRead
from app.modules.labeling import service as labeling_service
from app.modules.search.schemas import SearchFilesRequest, TagFacetRead
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


@router.post("/search/files", response_model=list[FileRead])
def search_files(payload: SearchFilesRequest, db: Session = Depends(get_db)):
    """WF2a (ADR-0002a D3 / #57): execute the final type/tag filter and
    return matching files, sorted. An empty type_label_ids/tags list means
    that group is unrestricted (D2); both empty returns the full file list."""
    for type_label_id in payload.type_label_ids:
        if labeling_service.get_type_label(db, type_label_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Type label {type_label_id} not found")
    for tag in payload.tags:
        if labeling_service.get_tag_kind(db, tag.kind_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tag kind {tag.kind_id} not found")

    type_ids = (
        labeling_service.get_file_ids_by_confirmed_types(db, payload.type_label_ids)
        if payload.type_label_ids
        else None
    )
    tag_ids = (
        labeling_service.get_file_ids_by_confirmed_tags(db, [(tag.kind_id, tag.value) for tag in payload.tags])
        if payload.tags
        else None
    )

    if type_ids is None:
        file_ids = tag_ids
    elif tag_ids is None:
        file_ids = type_ids
    else:
        file_ids = type_ids & tag_ids

    return files_service.list_files_by_ids(db, file_ids, sort_by=payload.sort_by, sort_order=payload.sort_order)
