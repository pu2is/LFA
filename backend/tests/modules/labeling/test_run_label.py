"""run_label's job-lifecycle bookkeeping (suggestion logic itself is covered
in test_suggest_labels.py / test_merge_labels.py)."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.tasks import run_label


@pytest.fixture
def label_job(db):
    path = RegisteredPath(path="/labeling-run-label-fixture")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/labeling-run-label-fixture/sample.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="d" * 64,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()

    job = Job(
        type="label",
        file_id=file.id,
        trigger="manual",
        mode="initial",
        # Simulates a prior failed attempt that RQ is now retrying (#33).
        status="failed",
        error_message="LLM parse error",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    yield job


@patch("app.modules.labeling.tasks.suggest_labels")
def test_run_label_clears_stale_error_message_on_success(mock_suggest_labels, db, label_job):
    mock_suggest_labels.return_value = []

    result = run_label(db, label_job.id)

    assert result.status == "succeeded"
    assert result.error_message is None
