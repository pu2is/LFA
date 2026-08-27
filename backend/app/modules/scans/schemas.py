import uuid
from typing import Literal

from pydantic import BaseModel

from app.modules.files.schemas import FileRead
from app.modules.jobs.schemas import JobRead


class ScanCreate(BaseModel):
    path_id: uuid.UUID


class RescanRead(JobRead):
    """JobRead plus this Rescan's scan report (ADR-0001b): per-event-type
    counts and how many fuzzy recovery candidates are still pending."""

    event_counts: dict[str, int]
    pending_candidate_count: int


class RescanCandidateResolve(BaseModel):
    action: Literal["keep_labels", "drop_labels", "reject"]


class RescanCandidateResolveResult(BaseModel):
    candidate_id: uuid.UUID
    action: str
    file: FileRead


class LabelReviewRequest(BaseModel):
    action: Literal["keep", "drop"]


class LabelReviewResult(BaseModel):
    """file_events.from_hash/to_hash of the change that triggered review, so
    the frontend can show what changed without LFA judging drift itself
    (ADR-0001b D4 MVP1 decision)."""

    file: FileRead
    from_hash: str | None
    to_hash: str | None
