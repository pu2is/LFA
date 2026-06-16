import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database import Base


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
        # A file cannot have the same label attached twice.
        UniqueConstraint("file_id", "label_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    label_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("labels.id"))
    source: Mapped[str] = mapped_column(String)  # llm | user
    status: Mapped[str] = mapped_column(String, default="suggested", server_default=text("'suggested'"))
    # NULL when source=user (no model confidence available for manual labels).
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
