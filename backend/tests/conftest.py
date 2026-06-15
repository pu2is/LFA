from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from app.shared.database import SessionLocal


@pytest.fixture
def db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
