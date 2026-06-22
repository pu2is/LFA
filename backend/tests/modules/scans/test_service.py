import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.scans.models import Scan
from app.modules.scans.service import create_scan, get_scan, run_scan

TEST_DIR = Path("E:/lfa_test")

pytestmark = pytest.mark.skipif(
    not TEST_DIR.exists(), reason="E:\\lfa_test fixture directory not present on this machine"
)


@pytest.fixture
def registered_path(db):
    path = RegisteredPath(path=str(TEST_DIR.resolve()))
    db.add(path)
    db.commit()
    yield path


def _queue_scan(db, path_id: uuid.UUID) -> Scan:
    scan = Scan(path_id=path_id, status="queued")
    db.add(scan)
    db.commit()
    return scan


def test_run_scan_discovers_documents(db, registered_path):
    scan = _queue_scan(db, registered_path.id)

    result = run_scan(db, scan.id)

    assert result.status == "completed"
    assert result.started_at is not None
    assert result.completed_at is not None

    files = db.scalars(select(File).where(File.path_id == registered_path.id)).all()
    assert len(files) == 3
    for file in files:
        assert file.file_type == "pdf"
        assert file.status == "discovered"
        assert file.file_size > 0
        assert len(file.file_hash) == 64


def test_run_scan_updates_path_last_scanned_at(db, registered_path):
    assert registered_path.last_scanned_at is None

    scan = _queue_scan(db, registered_path.id)
    run_scan(db, scan.id)

    db.refresh(registered_path)
    assert registered_path.last_scanned_at is not None


def test_create_scan_starts_queued(db, registered_path):
    scan = create_scan(db, registered_path.id)

    assert scan.status == "queued"
    assert get_scan(db, scan.id).id == scan.id


def test_get_scan_returns_none_for_unknown_id(db):
    assert get_scan(db, uuid.uuid4()) is None


def test_rescan_is_idempotent(db, registered_path):
    first_scan = _queue_scan(db, registered_path.id)
    run_scan(db, first_scan.id)

    first_ids = {file.id for file in db.scalars(select(File).where(File.path_id == registered_path.id)).all()}
    assert len(first_ids) == 3

    second_scan = _queue_scan(db, registered_path.id)
    second_result = run_scan(db, second_scan.id)

    second_ids = {file.id for file in db.scalars(select(File).where(File.path_id == registered_path.id)).all()}
    assert second_result.status == "completed"
    assert second_ids == first_ids
