"""create type/tag label foundation tables

Revision ID: 43baca0fc3f1
Revises: a1b2c3d4e5f6
Create Date: 2026-07-14

ADR-0001 D1: normalizes labels/file_labels into a faceted type/tag model.
Additive only -- labels/file_labels are untouched; the contract (drop) side
of the expand->contract migration is a separate, later ticket. See
docs/adr/0001-normalize-label-schema.md and docs/03_er-diagram.md.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "43baca0fc3f1"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "type_labels",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "type_labels_files",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=False),
        sa.Column("type_label_id", sa.UUID(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("status", sa.String(), server_default=sa.text("'suggested'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["type_label_id"], ["type_labels.id"]),
        sa.UniqueConstraint("file_id", "type_label_id"),
    )
    op.create_table(
        "tag_kinds",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "tag_labels",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=False),
        sa.Column("kind_id", sa.UUID(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("status", sa.String(), server_default=sa.text("'suggested'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["kind_id"], ["tag_kinds.id"]),
        sa.UniqueConstraint("file_id", "kind_id", "value"),
    )


def downgrade() -> None:
    op.drop_table("tag_labels")
    op.drop_table("tag_kinds")
    op.drop_table("type_labels_files")
    op.drop_table("type_labels")
