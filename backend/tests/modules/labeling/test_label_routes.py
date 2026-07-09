"""Tests for POST /label/files and POST /label/paths endpoints."""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.main import app
from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import FileLabel, Label
from app.shared.database import get_db
from app.shared.queue import JOB_RETRY

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


@pytest.fixture
def three_ready_files(db: Session):
    """Three ready files under one path, for batch-order/atomicity tests."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    files = [
        File(
            path_id=path.id,
            filename=f"ready-{i}.pdf",
            full_path=f"{_FAKE_PATH}/ready-{i}.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash=f"{i}1223344" * 8,
            file_modified_at=datetime.now(timezone.utc),
            status="ready",
        )
        for i in range(3)
    ]
    db.add_all(files)
    db.commit()
    for file in files:
        db.refresh(file)
    yield files


def _mock_rq_job(job_id: str | None = None) -> MagicMock:
    rq_job = MagicMock()
    rq_job.id = job_id or str(uuid.uuid4())
    return rq_job


# ---------------------------------------------------------------------------
# POST /label/files — initial mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
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


@patch("app.modules.labeling.routes.label_queue")
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


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_unknown_id_returns_404(mock_q, client):
    resp = client.post("/label/files", json={"file_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# POST /label/files — augment mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
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

@patch("app.modules.labeling.routes.label_queue")
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

@patch("app.modules.labeling.routes.label_queue")
def test_label_paths_enqueues_ready_files(mock_q, client, path_with_mixed_files):
    path, ready, _discovered = path_with_mixed_files
    mock_q.enqueue.return_value = _mock_rq_job()

    resp = client.post("/label/paths", json={"path_ids": [str(path.id)]})

    assert resp.status_code == 202
    data = resp.json()
    enqueued_file_ids = {e["file_id"] for e in data["enqueued"]}
    assert str(ready.id) in enqueued_file_ids


@patch("app.modules.labeling.routes.label_queue")
def test_label_paths_unknown_path_returns_404(mock_q, client):
    resp = client.post("/label/paths", json={"path_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# Enqueue bookkeeping and retry consistency (#33)
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_stores_rq_job_id_on_job(mock_q, client, db, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = _mock_rq_job("rq-abc-123")

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    job_id = resp.json()["enqueued"][0]["job_id"]
    job = db.get(Job, uuid.UUID(job_id))
    assert job.rq_job_id == "rq-abc-123"


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_enqueues_without_at_front_using_shared_retry(mock_q, client, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = _mock_rq_job()

    client.post("/label/files", json={"file_ids": [str(file.id)]})

    _args, kwargs = mock_q.enqueue.call_args
    assert kwargs.get("retry") is JOB_RETRY
    assert "at_front" not in kwargs


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_batch_enqueues_in_submission_order(mock_q, client, three_ready_files):
    mock_q.enqueue.side_effect = [_mock_rq_job() for _ in three_ready_files]
    file_ids = [str(f.id) for f in three_ready_files]

    resp = client.post("/label/files", json={"file_ids": file_ids})

    enqueued_file_ids = [e["file_id"] for e in resp.json()["enqueued"]]
    assert enqueued_file_ids == file_ids

    # job.id passed as the 2nd positional arg to enqueue(), in call order.
    submitted_job_ids = [call.args[1] for call in mock_q.enqueue.call_args_list]
    response_job_ids = [uuid.UUID(e["job_id"]) for e in resp.json()["enqueued"]]
    assert submitted_job_ids == response_job_ids


def test_label_files_batch_rolls_back_all_jobs_if_one_enqueue_fails(db: Session, three_ready_files):
    """All-or-nothing: a mid-batch enqueue failure must not leave partial Job rows."""

    def override_get_db():
        # Mirror the real get_db's try/finally: an exception mid-request must
        # roll back whatever this request flushed, same as production.
        try:
            yield db
        finally:
            db.rollback()

    app.dependency_overrides[get_db] = override_get_db
    local_client = TestClient(app, raise_server_exceptions=False)

    with patch("app.modules.labeling.routes.label_queue") as mock_q:
        mock_q.enqueue.side_effect = [_mock_rq_job(), RuntimeError("redis unreachable"), _mock_rq_job()]
        file_ids = [str(f.id) for f in three_ready_files]

        resp = local_client.post("/label/files", json={"file_ids": file_ids})

    app.dependency_overrides.clear()

    assert resp.status_code == 500
    remaining = list(db.scalars(select(Job).where(Job.file_id.in_([f.id for f in three_ready_files]))))
    assert remaining == []
