"""file_labels: make label_id nullable, add label_name for LLM free-text labels

Revision ID: b3f1c2d4e5a6
Revises: a4ca60952229
Create Date: 2026-06-17

Changes:
- label_id becomes nullable (NULL when LLM invents a label not in the catalog)
- label_name TEXT added (holds the free-text label name when label_id IS NULL)
- Old UNIQUE(file_id, label_id) replaced by two partial unique indexes + CHECK constraint
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3f1c2d4e5a6"
down_revision: Union[str, Sequence[str], None] = "a4ca60952229"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop the old full unique constraint on (file_id, label_id).
    op.drop_constraint("file_labels_file_id_label_id_key", "file_labels", type_="unique")

    # 2. Make label_id nullable.
    op.alter_column("file_labels", "label_id", nullable=True)

    # 3. Add label_name column for LLM free-text labels.
    op.add_column("file_labels", sa.Column("label_name", sa.String(), nullable=True))

    # 4. Partial unique index: one catalog label per file.
    op.create_index(
        "uq_file_labels_catalog",
        "file_labels",
        ["file_id", "label_id"],
        unique=True,
        postgresql_where=sa.text("label_id IS NOT NULL"),
    )

    # 5. Partial unique index: one free-text label name per file.
    op.create_index(
        "uq_file_labels_freetext",
        "file_labels",
        ["file_id", "label_name"],
        unique=True,
        postgresql_where=sa.text("label_name IS NOT NULL"),
    )

    # 6. CHECK: exactly one of label_id / label_name must be set.
    op.create_check_constraint(
        "ck_file_labels_label_xor",
        "file_labels",
        "(label_id IS NOT NULL AND label_name IS NULL) OR "
        "(label_id IS NULL AND label_name IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_file_labels_label_xor", "file_labels", type_="check")
    op.drop_index("uq_file_labels_freetext", table_name="file_labels")
    op.drop_index("uq_file_labels_catalog", table_name="file_labels")
    op.drop_column("file_labels", "label_name")
    op.alter_column("file_labels", "label_id", nullable=False)
    op.create_unique_constraint(
        "file_labels_file_id_label_id_key", "file_labels", ["file_id", "label_id"]
    )
