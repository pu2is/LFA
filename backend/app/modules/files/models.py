import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared.database import Base


class RegisteredPath(Base):
    __tablename__ = "paths"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    path: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
