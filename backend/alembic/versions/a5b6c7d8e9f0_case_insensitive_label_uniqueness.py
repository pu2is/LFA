"""case-insensitive uniqueness for type_labels/tag_kinds/tag_labels

Revision ID: a5b6c7d8e9f0
Revises: 9e26075bd04c
Create Date: 2026-07-22

#49: augment (temperature=0.7) frequently re-suggests case variants of a
value already present (e.g. "berlin" alongside a confirmed "Berlin"),
which the old raw-string UNIQUE constraints didn't catch. Replaces the
three UNIQUE constraints created in 43baca0fc3f1 with case-insensitive
expression (lower()) indexes; app-side dedup (merge.py, service.py) is
updated to match in the same ticket.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a5b6c7d8e9f0"
down_revision: Union[str, Sequence[str], None] = "9e26075bd04c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("type_labels_name_key", "type_labels", type_="unique")
    op.create_index(
        "ix_type_labels_name_lower", "type_labels", [sa.text("lower(name)")], unique=True
    )

    op.drop_constraint("tag_kinds_name_key", "tag_kinds", type_="unique")
    op.create_index(
        "ix_tag_kinds_name_lower", "tag_kinds", [sa.text("lower(name)")], unique=True
    )

    op.drop_constraint("tag_labels_file_id_kind_id_value_key", "tag_labels", type_="unique")
    op.create_index(
        "ix_tag_labels_file_kind_value_lower",
        "tag_labels",
        ["file_id", "kind_id", sa.text("lower(value)")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_tag_labels_file_kind_value_lower", table_name="tag_labels")
    op.create_unique_constraint(
        "tag_labels_file_id_kind_id_value_key", "tag_labels", ["file_id", "kind_id", "value"]
    )

    op.drop_index("ix_tag_kinds_name_lower", table_name="tag_kinds")
    op.create_unique_constraint("tag_kinds_name_key", "tag_kinds", ["name"])

    op.drop_index("ix_type_labels_name_lower", table_name="type_labels")
    op.create_unique_constraint("type_labels_name_key", "type_labels", ["name"])
