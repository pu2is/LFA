"""drop labels and file_labels (ADR-0001 contract step)

Revision ID: 9e26075bd04c
Revises: 43baca0fc3f1
Create Date: 2026-07-14

Contract step of ADR-0001's expand->contract migration (mirrors #25's
scans/processing_jobs drop). type_labels / type_labels_files / tag_kinds /
tag_labels (#43-#46) fully replaced these two tables; no code reads or
writes labels/file_labels anymore. Per ADR-0001's decided Open Questions,
existing dev data is discarded, not migrated -- downgrade recreates the
tables empty.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9e26075bd04c"
down_revision: Union[str, Sequence[str], None] = "43baca0fc3f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("file_labels")
    op.drop_table("labels")


def downgrade() -> None:
    op.create_table(
        "labels",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "file_labels",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=False),
        sa.Column("label_id", sa.UUID(), nullable=True),
        sa.Column("label_name", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("status", sa.String(), server_default=sa.text("'suggested'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["label_id"], ["labels.id"]),
    )
    op.create_index(
        "uq_file_labels_catalog", "file_labels", ["file_id", "label_id"],
        unique=True, postgresql_where=sa.text("label_id IS NOT NULL"),
    )
    op.create_index("uq_file_labels_name", "file_labels", ["file_id", "label_name"], unique=True)
