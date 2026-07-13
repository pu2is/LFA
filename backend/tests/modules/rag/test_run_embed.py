"""run_embed's job-lifecycle bookkeeping (separate from embed_file's own logic,
covered in test_embedding.py)."""
from unittest.mock import MagicMock

import pytest

from app.modules.rag.service import run_embed
from tests.factories import make_failed_job


@pytest.fixture
def embed_job(db):
    return make_failed_job(db, job_type="embed", error_message="Ollama unreachable")


def test_run_embed_clears_stale_error_message_on_success(db, embed_job):
    result = run_embed(db, embed_job.id, embeddings=MagicMock())

    assert result.status == "succeeded"
    assert result.error_message is None
