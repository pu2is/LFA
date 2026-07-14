"""Preset catalog of common personal-document types, shown during onboarding.

This is static product configuration, not user state: the database only
stores what the user actually selected. Changing the catalog is a code
change, not a migration.
"""

RECOMMENDED_LABELS: tuple[str, ...] = (
    "invoice",
    "contract",
    "report",
    "certificate",
    "receipt",
    "tax",
    "insurance",
    "manual",
    "correspondence",
    "other",
)

OPTIONAL_LABELS: tuple[str, ...] = (
    "bank_statement",
    "payslip",
    "utility_bill",
    "medical_record",
    "warranty",
    "resume",
    "presentation",
    "meeting_notes",
    "legal_document",
    "academic_paper",
)

# Starter controlled vocabulary for tag_kinds (ADR-0001 D1); "topic" is the catch-all.
TAG_KIND_PRESETS: tuple[str, ...] = (
    "person",
    "organization",
    "time",
    "place",
    "topic",
)


def get_preset_catalog() -> list[dict[str, str | bool]]:
    return [{"name": name, "recommended": True} for name in RECOMMENDED_LABELS] + [
        {"name": name, "recommended": False} for name in OPTIONAL_LABELS
    ]
