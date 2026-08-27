"""Rescan audit/recovery models (WF1b, ADR-0001b). Owned by scans -- not
files -- per docs/02_architecture.md: these record what a Rescan job did,
not the current manifest itself (that's still files.File)."""
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.shared.database import Base

# Same app-layer-guard rationale as jobs.VALID_JOB_MODES: closed sets that
# are cheap to extend without a migration.
VALID_FILE_EVENT_TYPES = frozenset({
    "added", "modified", "moved", "moved_modified", "missing", "recovered",
})
VALID_FILE_MATCH_CANDIDATE_STATUSES = frozenset({
    "pending", "accepted_keep_labels", "accepted_drop_labels", "rejected",
})


class FileEvent(Base):
    """Immutable record of one semantic manifest change a Rescan applied
    (ADR-0001b D4/D6). from_path/to_path/from_hash/to_hash/match_method let
    a scan report explain what evidence produced each event."""

    __tablename__ = "file_events"
    __table_args__ = (
        # Apply/fan-out retry must not double-write the same semantic event
        # for the same scan+file (ADR-0001b D4).
        UniqueConstraint("scan_id", "file_id", "event_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    event_type: Mapped[str] = mapped_column(String)
    from_path: Mapped[str | None] = mapped_column(String, nullable=True)
    to_path: Mapped[str | None] = mapped_column(String, nullable=True)
    from_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    to_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    # path | filesystem_id | hash | text_similarity_user -- descriptive
    # attribution, not a closed set worth app-layer validation (see jobs.trigger).
    match_method: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @validates("event_type")
    def _validate_event_type(self, _key: str, value: str) -> str:
        if value not in VALID_FILE_EVENT_TYPES:
            raise ValueError(f"Invalid event_type {value!r}; expected one of {sorted(VALID_FILE_EVENT_TYPES)}")
        return value


class FileMatchCandidate(Base):
    """Fuzzy recovery proposal awaiting a user decision (ADR-0001b D5) --
    never auto-applied. Snapshots the candidate's path/hash/size/mtime at
    proposal time so resolve-time re-stat can detect drift and 409."""

    __tablename__ = "file_match_candidates"
    __table_args__ = (
        # Same recovery proposal must not be raised twice by one scan.
        UniqueConstraint("scan_id", "missing_file_id", "candidate_full_path"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    missing_file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    candidate_path_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("paths.id"))
    candidate_full_path: Mapped[str] = mapped_column(String)
    candidate_hash: Mapped[str] = mapped_column(String)
    candidate_size: Mapped[int] = mapped_column(BigInteger)
    candidate_modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    similarity_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, default="pending", server_default=text("'pending'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @validates("status")
    def _validate_status(self, _key: str, value: str) -> str:
        if value not in VALID_FILE_MATCH_CANDIDATE_STATUSES:
            raise ValueError(
                f"Invalid status {value!r}; expected one of {sorted(VALID_FILE_MATCH_CANDIDATE_STATUSES)}"
            )
        return value
