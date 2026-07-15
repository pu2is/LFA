import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.shared.database import Base

# Per-type job variant (see docs/03_er-diagram.md). Not a DB CHECK constraint
# on purpose -- adding a new mode is a one-line change here, not a migration.
VALID_JOB_MODES = frozenset({"default", "initial", "augment", "check"})

# Per-type stage progression (see docs/03_er-diagram.md): ingest is
# extract|clean|chunk; label/mode=initial is type|kinds|tags; label/
# mode=augment only ever reaches tags; scan/embed leave stage NULL. Same
# app-layer-guard rationale as VALID_JOB_MODES.
VALID_JOB_STAGES = frozenset({"extract", "clean", "chunk", "type", "kinds", "tags"})


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "(type = 'scan' AND path_id IS NOT NULL AND file_id IS NULL) OR "
            "(type <> 'scan' AND file_id IS NOT NULL AND path_id IS NULL)",
            name="ck_jobs_target",
        ),
        Index("ix_jobs_file_type_created", "file_id", "type", text("created_at DESC")),
        Index("ix_jobs_path_created", "path_id", text("created_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    path_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paths.id"), nullable=True
    )
    file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id"), nullable=True
    )
    parent_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String, default="queued", server_default=text("'queued'")
    )
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    trigger: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(
        String, nullable=False, default="default", server_default=text("'default'")
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    rq_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @validates("mode")
    def _validate_mode(self, _key: str, value: str) -> str:
        if value not in VALID_JOB_MODES:
            raise ValueError(f"Invalid job mode {value!r}; expected one of {sorted(VALID_JOB_MODES)}")
        return value

    @validates("stage")
    def _validate_stage(self, _key: str, value: str | None) -> str | None:
        if value is not None and value not in VALID_JOB_STAGES:
            raise ValueError(f"Invalid job stage {value!r}; expected None or one of {sorted(VALID_JOB_STAGES)}")
        return value
