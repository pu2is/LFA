import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database import Base

# Must match the bge-m3 embedding model's output size (see docs/03_er-diagram.md).
# Switching embedding models means re-embedding every chunk + an Alembic migration.
EMBEDDING_DIMENSIONS = 1024


class FileChunk(Base):
    __tablename__ = "file_chunks"
    __table_args__ = (
        # Re-running embedding (Job2) must not create duplicate chunks for a file.
        UniqueConstraint("file_id", "chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # NULL until Job2 (#10) backfills it via bge-m3.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
