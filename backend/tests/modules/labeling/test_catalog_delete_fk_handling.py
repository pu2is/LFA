"""Tests for _delete_catalog_entry_or_409's constraint-name check (#55).

The known-FK 409 path (entry still referenced by a file) is already covered
end-to-end against the real DB by test_delete_type_label_in_use_returns_409
and test_delete_tag_kind_in_use_returns_409. These tests cover the helper in
isolation: the known FK still converts to 409, and an IntegrityError on a
*different* constraint is re-raised instead of being misreported as "still
referenced".
"""
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.modules.labeling.routes import (
    TAG_KIND_IN_USE_FK,
    TYPE_LABEL_IN_USE_FK,
    _delete_catalog_entry_or_409,
)


def _integrity_error(constraint_name: str) -> IntegrityError:
    orig = MagicMock()
    orig.diag.constraint_name = constraint_name
    return IntegrityError("stmt", {}, orig)


def test_known_fk_violation_converts_to_409(db):
    def delete_fn(_db, _entry):
        raise _integrity_error(TYPE_LABEL_IN_USE_FK)

    with pytest.raises(HTTPException) as exc_info:
        _delete_catalog_entry_or_409(db, delete_fn, object(), "Type label", TYPE_LABEL_IN_USE_FK)

    assert exc_info.value.status_code == 409
    assert "still referenced" in exc_info.value.detail


def test_unrelated_fk_violation_is_reraised(db):
    def delete_fn(_db, _entry):
        raise _integrity_error("some_other_table_some_column_fkey")

    with pytest.raises(IntegrityError):
        _delete_catalog_entry_or_409(db, delete_fn, object(), "Type label", TYPE_LABEL_IN_USE_FK)


def test_tag_kind_known_fk_violation_converts_to_409(db):
    def delete_fn(_db, _entry):
        raise _integrity_error(TAG_KIND_IN_USE_FK)

    with pytest.raises(HTTPException) as exc_info:
        _delete_catalog_entry_or_409(db, delete_fn, object(), "Tag kind", TAG_KIND_IN_USE_FK)

    assert exc_info.value.status_code == 409
