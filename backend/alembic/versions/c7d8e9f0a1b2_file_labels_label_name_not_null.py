"""file_labels: make label_name NOT NULL, remove XOR check, replace partial unique index

Revision ID: c7d8e9f0a1b2
Revises: b3f1c2d4e5a6
Create Date: 2026-06-17

Changes:
- Backfill label_name from labels.name for existing catalog-pick rows
- Drop XOR CHECK constraint (label_name is now always set, making it redundant)
- Drop partial unique index on (file_id, label_name) WHERE label_name IS NOT NULL
- Add full unique index on (file_id, label_name) — same file cannot have two labels with the same name
- Make label_name NOT NULL
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, Sequence[str], None] = "b3f1c2d4e5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Constraints must be dropped BEFORE the backfill UPDATE, not after.
    #
    # ck_file_labels_label_xor requires exactly one of (label_id, label_name) to be
    # set, so UPDATE-ing label_name on a row that already has label_id set would
    # immediately violate it.  uq_file_labels_freetext (WHERE label_name IS NOT NULL)
    # must also be dropped first: once the UPDATE sets label_name on catalog rows, those
    # rows enter the index and may collide with existing label_name values on the same file.

    # 1. Drop the XOR CHECK constraint — it blocks the backfill.
    op.drop_constraint("ck_file_labels_label_xor", "file_labels", type_="check")

    # 2. Drop the partial unique index on (file_id, label_name WHERE label_name IS NOT NULL)
    #    — the backfill can otherwise collide with rows that already have label_name set.
    op.drop_index("uq_file_labels_freetext", table_name="file_labels")

    # 3. Backfill label_name for catalog-pick rows that still have label_name IS NULL.
    op.execute(
        """
        UPDATE file_labels fl
        SET label_name = l.name
        FROM labels l
        WHERE fl.label_id = l.id
          AND fl.label_name IS NULL
        """
    )

    # 4. Remove any rows that still have label_name IS NULL after backfill.
    #    These are orphaned rows whose label_id no longer exists in labels — safe to drop.
    op.execute("DELETE FROM file_labels WHERE label_name IS NULL")

    # 5. Add full unique index on (file_id, label_name) — covers both catalog and free-text.
    op.create_index(
        "uq_file_labels_name",
        "file_labels",
        ["file_id", "label_name"],
        unique=True,
    )

    # 6. Make label_name NOT NULL now that all rows have a value.
    op.alter_column("file_labels", "label_name", nullable=False)


def downgrade() -> None:
    op.alter_column("file_labels", "label_name", nullable=True)
    op.drop_index("uq_file_labels_name", table_name="file_labels")
    op.create_index(
        "uq_file_labels_freetext",
        "file_labels",
        ["file_id", "label_name"],
        unique=True,
        postgresql_where=sa.text("label_name IS NOT NULL"),
    )
    op.create_check_constraint(
        "ck_file_labels_label_xor",
        "file_labels",
        "(label_id IS NOT NULL AND label_name IS NULL) OR "
        "(label_id IS NULL AND label_name IS NOT NULL)",
    )
