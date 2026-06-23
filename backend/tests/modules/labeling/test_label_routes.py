"""Tests for POST /label/files and POST /label/paths endpoints."""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import FileLabel, Label

_FAKE_PATH = "D:/lfa_test_label_routes"


@pytest.fixture
def path_and_ready_file(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="ready.pdf",
        full_path=f"{_FAKE_PATH}/ready.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="aabbccdd" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.commit()
    db.refresh(path)
    db.refresh(file)
    yield path, file


@pytest.fixture
def path_with_mixed_files(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    ready = File(
        path_id=path.id,
        filename="ready.pdf",
        full_path=f"{_FAKE_PATH}/ready.pdf",
        file_type="pdf",
        file_size=1024,
        file_hash="11223344" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    discovered = File(
        path_id=path.id,
        filename="new.pdf",
        full_path=f"{_FAKE_PATH}/new.pdf",
        file_type="pdf",
        file_size=512,
        file_hash="55667788" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="discovered",
    )
    db.add_all([ready, discovered])
    db.commit()
    db.refresh(ready)
    db.refresh(discovered)
    yield path, ready, discovered


@pytest.fixture
def ready_file_with_labels(db: Session):
    """A ready file that already has file_labels — triggers augment mode."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="labeled.pdf",
        full_path=f"{_FAKE_PATH}/labeled.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="cafebabe" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()

    lbl = Label(name="lr_test_invoice")
    db.add(lbl)
    db.flush()

    fl = FileLabel(
        file_id=file.id,
        label_id=lbl.id,
        label_name=lbl.name,
        source="llm",
        status="confirmed",
        confidence=0.9,
    )
    db.add(fl)
    db.commit()
    db.refresh(file)
    yield path, file


def _mock_rq_job() -> MagicMock:
    rq_job = MagicMock()
    rq_job.id = str(uuid.uuid4())
    return rq_job


# ---------------------------------------------------------------------------
# POST /label/files — initial mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.labeling_queue")
def test_label_files_initial_returns_202(mock_q, client, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["file_id"] == str(file.id)
    assert data["enqueued"][0]["mode"] == "initial"
    assert data["skipped"] == []


@patch("app.modules.labeling.routes.labeling_queue")
def test_label_files_skips_non_ready(mock_q, client, path_with_mixed_files):
    _, ready, discovered = path_with_mixed_files
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post(
        "/label/files",
        json={"file_ids": [str(ready.id), str(discovered.id)]},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["file_id"] == str(ready.id)
    assert len(data["skipped"]) == 1
    assert data["skipped"][0]["file_id"] == str(discovered.id)


@patch("app.modules.labeling.routes.labeling_queue")
def test_label_files_unknown_id_returns_404(mock_q, client):
    resp = client.post("/label/files", json={"file_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# POST /label/files — augment mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.labeling_queue")
def test_label_files_augment_for_labeled_file(mock_q, client, ready_file_with_labels):
    _, file = ready_file_with_labels
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["mode"] == "augment"


# ---------------------------------------------------------------------------
# POST /label/files — creates correct Job row
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.labeling_queue")
def test_label_files_creates_job_in_db(mock_q, client, db, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    data = resp.json()
    job_id = data["enqueued"][0]["job_id"]
    job = db.get(Job, uuid.UUID(job_id))
    assert job is not None
    assert job.type == "label"
    assert job.trigger == "manual"
    assert job.mode == "initial"
    assert job.file_id == file.id


# ---------------------------------------------------------------------------
# POST /label/paths
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.labeling_queue")
def test_label_paths_enqueues_ready_files(mock_q, client, path_with_mixed_files):
    path, ready, _discovered = path_with_mixed_files
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post("/label/paths", json={"path_ids": [str(path.id)]})

    assert resp.status_code == 202
    data = resp.json()
    enqueued_file_ids = {e["file_id"] for e in data["enqueued"]}
    assert str(ready.id) in enqueued_file_ids


@patch("app.modules.labeling.routes.labeling_queue")
def test_label_paths_unknown_path_returns_404(mock_q, client):
    resp = client.post("/label/paths", json={"path_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()
