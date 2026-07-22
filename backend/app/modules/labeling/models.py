import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func, text
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
    __table_args__ = (
        # #49: case-insensitive uniqueness -- "Berlin" and "berlin" are the same
        # catalog entry. name is already lowercased app-side (normalize_label_name),
        # but the DB constraint enforces it independent of that.
        Index("ix_type_labels_name_lower", text("lower(name)"), unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String)
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
    __table_args__ = (
        # #49: case-insensitive uniqueness, same reasoning as TypeLabel above.
        Index("ix_tag_kinds_name_lower", text("lower(name)"), unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TagLabel(Base):
    """File-local tag fact; value is free text, not drawn from a catalog."""

    __tablename__ = "tag_labels"
    __table_args__ = (
        # Augment (append-only) dedupes new suggestions against this. #49:
        # case-insensitive -- "Berlin" and "berlin" under the same kind are
        # the same tag; value is free text so, unlike TypeLabel/TagKind.name,
        # nothing normalizes it app-side before it reaches this constraint.
        Index("ix_tag_labels_file_kind_value_lower", "file_id", "kind_id", text("lower(value)"), unique=True),
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
