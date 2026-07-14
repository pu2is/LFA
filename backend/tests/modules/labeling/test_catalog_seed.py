"""Tests for the ADR-0001 foundation catalog seed helpers.

Mirrors the existing _ensure_label_catalog behavior in labeling/suggestion.py:
seed only when the table is completely empty; a non-empty catalog is left
untouched (no merge/backfill of missing presets).
"""
from sqlalchemy import select

from app.modules.labeling.models import TagKind, TypeLabel
from app.modules.labeling.presets import OPTIONAL_LABELS, RECOMMENDED_LABELS, TAG_KIND_PRESETS
from app.modules.labeling.service import ensure_tag_kind_catalog, ensure_type_catalog


# --------------------------------------------------------------------------- #
# ensure_type_catalog
# --------------------------------------------------------------------------- #

def test_ensure_type_catalog_seeds_all_presets_when_empty(db):
    assert db.scalars(select(TypeLabel)).first() is None

    types = ensure_type_catalog(db)

    assert {t.name for t in types} == set(RECOMMENDED_LABELS) | set(OPTIONAL_LABELS)
    assert len(types) == len(RECOMMENDED_LABELS) + len(OPTIONAL_LABELS)


def test_ensure_type_catalog_is_idempotent(db):
    first = ensure_type_catalog(db)
    second = ensure_type_catalog(db)

    assert {t.id for t in first} == {t.id for t in second}
    assert len(list(db.scalars(select(TypeLabel)))) == len(first)


def test_ensure_type_catalog_leaves_existing_catalog_untouched(db):
    """A single pre-existing row is enough to skip seeding entirely (no backfill)."""
    db.add(TypeLabel(name="custom_type"))
    db.commit()

    types = ensure_type_catalog(db)

    assert [t.name for t in types] == ["custom_type"]


# --------------------------------------------------------------------------- #
# ensure_tag_kind_catalog
# --------------------------------------------------------------------------- #

def test_ensure_tag_kind_catalog_seeds_all_presets_when_empty(db):
    assert db.scalars(select(TagKind)).first() is None

    kinds = ensure_tag_kind_catalog(db)

    assert {k.name for k in kinds} == set(TAG_KIND_PRESETS)
    assert len(kinds) == len(TAG_KIND_PRESETS)


def test_ensure_tag_kind_catalog_is_idempotent(db):
    first = ensure_tag_kind_catalog(db)
    second = ensure_tag_kind_catalog(db)

    assert {k.id for k in first} == {k.id for k in second}
    assert len(list(db.scalars(select(TagKind)))) == len(first)


def test_ensure_tag_kind_catalog_leaves_existing_catalog_untouched(db):
    db.add(TagKind(name="custom_kind"))
    db.commit()

    kinds = ensure_tag_kind_catalog(db)

    assert [k.name for k in kinds] == ["custom_kind"]
