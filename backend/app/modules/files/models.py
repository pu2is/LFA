import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database import Base


class RegisteredPath(Base):
    __tablename__ = "paths"
    __table_args__ = (Index("ix_paths_parent_path_id", "parent_path_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    path: Mapped[str] = mapped_column(String, unique=True)
    # Nearest registered ancestor (self-ref FK, mirrors jobs.parent_job_id).
    # Set at registration time; see docs/workflow/00a-path-register.md.
    parent_path_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paths.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    path_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("paths.id"))
    filename: Mapped[str] = mapped_column(String)
    full_path: Mapped[str] = mapped_column(String, unique=True)
    file_type: Mapped[str] = mapped_column(String)
    file_size: Mapped[int] = mapped_column(BigInteger)
    # Non-unique: rescan move-detection (WF1b) pairs files by matching hash,
    # which requires identical-content files to be allowed to coexist.
    file_hash: Mapped[str] = mapped_column(String, index=True)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    # Nullable: not all filesystems expose a creation/birth time (see 03_er-diagram.md).
    file_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    file_modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="discovered", server_default="discovered")
    # Independent of files.status: Job2 (#10) fills in vectors without blocking labeling.
    embedding_status: Mapped[str] = mapped_column(String, default="pending", server_default=text("'pending'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
