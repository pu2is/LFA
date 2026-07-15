import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.shared.database import Base

# Shared source/status vocab for type_labels_files/tag_labels; app-layer guard, mirrors jobs.mode.
VALID_LABEL_SOURCES = frozenset({"llm", "user"})
VALID_LABEL_STATUSES = frozenset({"suggested", "confirmed", "rejected"})


# ADR-0001: faceted type/tag model. Supersedes the old labels/file_labels
# (nullable-FK "two identities in one table") design, dropped in #47.

class TypeLabel(Base):
    """Catalog table: type values are shared across files (true M:N)."""

    __tablename__ = "type_labels"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TypeLabelFile(Base):
    """Junction table: file <-> type_labels, with review state."""

    __tablename__ = "type_labels_files"
    __table_args__ = (
        # A file can't carry the same type twice.
        UniqueConstraint("file_id", "type_label_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    type_label_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("type_labels.id"))
    source: Mapped[str] = mapped_column(String)  # llm | user
    status: Mapped[str] = mapped_column(String, default="suggested", server_default=text("'suggested'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @validates("source")
    def _validate_source(self, _key: str, value: str) -> str:
        if value not in VALID_LABEL_SOURCES:
            raise ValueError(f"Invalid source {value!r}; expected one of {sorted(VALID_LABEL_SOURCES)}")
        return value

    @validates("status")
    def _validate_status(self, _key: str, value: str) -> str:
        if value not in VALID_LABEL_STATUSES:
            raise ValueError(f"Invalid status {value!r}; expected one of {sorted(VALID_LABEL_STATUSES)}")
        return value


class TagKind(Base):
    """Controlled vocabulary for tag *kinds* only -- tag values are not cataloged."""

    __tablename__ = "tag_kinds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TagLabel(Base):
    """File-local tag fact; value is free text, not drawn from a catalog."""

    __tablename__ = "tag_labels"
    __table_args__ = (
        # Augment (append-only) dedupes new suggestions against this.
        UniqueConstraint("file_id", "kind_id", "value"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    kind_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tag_kinds.id"))
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)  # llm | user
    status: Mapped[str] = mapped_column(String, default="suggested", server_default=text("'suggested'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @validates("source")
    def _validate_source(self, _key: str, value: str) -> str:
        if value not in VALID_LABEL_SOURCES:
            raise ValueError(f"Invalid source {value!r}; expected one of {sorted(VALID_LABEL_SOURCES)}")
        return value

    @validates("status")
    def _validate_status(self, _key: str, value: str) -> str:
        if value not in VALID_LABEL_STATUSES:
            raise ValueError(f"Invalid status {value!r}; expected one of {sorted(VALID_LABEL_STATUSES)}")
        return value
