"""Tests for labeling.service.normalize_label_name.

merge.py's write functions are all covered elsewhere now: write_initial_
candidates has no remaining caller (kept per #45's scope-out, not tested
further); write_type_candidates/select_kinds/write_tag_candidates are
covered in test_suggest_labels.py and test_suggest_labels_augment.py, since
ADR-0001's flows make them meaningful only in sequence, not in isolation.
"""
from app.modules.labeling.service import normalize_label_name


def test_normalize_label_name_replaces_spaces_with_underscores():
    assert normalize_label_name("Bank Statement") == "bank_statement"
    assert normalize_label_name("  Tax  ") == "tax"
    assert normalize_label_name("INVOICE") == "invoice"
    assert normalize_label_name("car_rental_agreement") == "car_rental_agreement"
