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
    __table_args__ = (
        # Non-unique, both nullable: Rescan (WF1b) prefers this stable
        # filesystem identity for deterministic moved/moved_modified
        # matching when the platform exposes it, but still has to verify
        # one-to-one uniqueness within the unmatched set itself -- see
        # ADR-0001b D3 and docs/03_er-diagram.md.
        Index("ix_files_fs_device_file_id", "fs_device_id", "fs_file_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    path_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("paths.id"))
    filename: Mapped[str] = mapped_column(String)
    full_path: Mapped[str] = mapped_column(String, unique=True)
    file_type: Mapped[str] = mapped_column(String)
    file_size: Mapped[int] = mapped_column(BigInteger)
    # Non-unique: rescan move-detection (WF1b) pairs files by matching hash,
    # which requires identical-content files to be allowed to coexist.
    file_hash: Mapped[str] = mapped_column(String, index=True)
    # Filesystem identity components (WF1b, ADR-0001b D3) -- nullable, since
    # not every filesystem exposes a stable device/file id.
    fs_device_id: Mapped[str | None] = mapped_column(String, nullable=True)
    fs_file_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # 64-bit character n-gram SimHash of normalized text, as hex (WF1b,
    # ADR-0001b D3 fuzzy recovery) -- nullable until ingest computes/backfills it.
    text_signature: Mapped[str | None] = mapped_column(String, nullable=True)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    # Nullable: not all filesystems expose a creation/birth time (see 03_er-diagram.md).
    file_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    file_modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="discovered", server_default="discovered")
    # Independent of files.status: Job2 (#10) fills in vectors without blocking labeling.
    embedding_status: Mapped[str] = mapped_column(String, default="pending", server_default=text("'pending'"))
    # WF1b (ADR-0001b D4): set when Rescan finds modified/moved_modified
    # content on a file that already carries type/tag associations -- Smart
    # Search excludes it until the user chooses keep/drop (see files/{id}/label-review).
    labels_need_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
