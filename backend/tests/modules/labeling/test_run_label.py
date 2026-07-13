"""run_label's job-lifecycle bookkeeping (suggestion logic itself is covered
in test_suggest_labels.py / test_merge_labels.py)."""
from unittest.mock import patch

import pytest

from app.modules.labeling.tasks import run_label
from tests.factories import make_failed_job


@pytest.fixture
def label_job(db):
    return make_failed_job(
        db, job_type="label", trigger="manual", mode="initial", file_status="ready", error_message="LLM parse error"
    )


@patch("app.modules.labeling.tasks.suggest_labels")
def test_run_label_clears_stale_error_message_on_success(mock_suggest_labels, db, label_job):
    mock_suggest_labels.return_value = []

    result = run_label(db, label_job.id)

    assert result.status == "succeeded"
    assert result.error_message is None
