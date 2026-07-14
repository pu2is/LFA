import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.shared.database import Base

# Shared source/status vocab for type_labels_files/tag_labels; app-layer guard, mirrors jobs.mode.
VALID_LABEL_SOURCES = frozenset({"llm", "user"})
VALID_LABEL_STATUSES = frozenset({"suggested", "confirmed", "rejected"})


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Unique at the DB level: the service layer skips duplicates politely,
    # but only this constraint prevents a race between concurrent requests.
    name: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FileLabel(Base):
    __tablename__ = "file_labels"
    __table_args__ = (
        # Prevents adding the same catalog label (by FK) twice to a file.
        Index("uq_file_labels_catalog", "file_id", "label_id", unique=True,
              postgresql_where=text("label_id IS NOT NULL")),
        # Prevents any two labels with the same name on the same file (catalog or free-text).
        Index("uq_file_labels_name", "file_id", "label_name", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    # Catalog path: label_id points to labels.id. Free-text path: NULL.
    label_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("labels.id"), nullable=True)
    # Always set: copied from labels.name for catalog picks; LLM-invented name for free-text.
    label_name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String)  # llm | user
    status: Mapped[str] = mapped_column(String, default="suggested", server_default=text("'suggested'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ADR-0001 foundation: faceted type/tag model (additive; labels/file_labels above are untouched).

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
