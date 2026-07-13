"""add paths.parent_path_id

Revision ID: a1b2c3d4e5f6
Revises: f3a4b5c6d7e8
Create Date: 2026-07-13

Self-referencing FK modeling nested-path registration lineage (mirrors
jobs.parent_job_id). See docs/workflow/00a-path-register.md and
docs/03_er-diagram.md.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("paths", sa.Column("parent_path_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "fk_paths_parent_path_id_paths", "paths", "paths", ["parent_path_id"], ["id"]
    )
    op.create_index("ix_paths_parent_path_id", "paths", ["parent_path_id"])


def downgrade() -> None:
    op.drop_index("ix_paths_parent_path_id", table_name="paths")
    op.drop_constraint("fk_paths_parent_path_id_paths", "paths", type_="foreignkey")
    op.drop_column("paths", "parent_path_id")
