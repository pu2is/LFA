"""_invoke_or_raise's log message uses the immediate caller's function name,
read from the stack rather than a manually-passed string (#28 review)."""
import logging
import uuid
from unittest.mock import MagicMock

import pytest

from app.modules.labeling.suggestion import _invoke_or_raise


def _caller_a(structured_llm, messages, file_id):
    return _invoke_or_raise(structured_llm, messages, file_id=file_id)


def _caller_b(structured_llm, messages, file_id):
    return _invoke_or_raise(structured_llm, messages, file_id=file_id)


def test_invoke_or_raise_logs_the_actual_caller_name(caplog):
    file_id = uuid.uuid4()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        _caller_a(structured_llm, MagicMock(), file_id)

    assert any("_caller_a" in record.message for record in caplog.records)


def test_invoke_or_raise_logs_a_different_caller_name_from_a_different_call_site(caplog):
    file_id = uuid.uuid4()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        _caller_b(structured_llm, MagicMock(), file_id)

    assert any("_caller_b" in record.message for record in caplog.records)


def test_invoke_or_raise_returns_the_llm_result_on_success():
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = "some output"

    result = _invoke_or_raise(structured_llm, MagicMock(), file_id=uuid.uuid4())

    assert result == "some output"
