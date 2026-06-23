"""create jobs table

Revision ID: d1e2f3a4b5c6
Revises: c7d8e9f0a1b2
Create Date: 2026-06-23

Unified jobs table replacing scans + processing_jobs.
Differentiates job types via `type` column (scan | ingest | label | embed).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("path_id", sa.UUID(), nullable=True),
        sa.Column("file_id", sa.UUID(), nullable=True),
        sa.Column("parent_job_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(), server_default=sa.text("'queued'"), nullable=False),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("trigger", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), server_default=sa.text("'default'"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("rq_job_id", sa.String(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["path_id"], ["paths.id"]),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["parent_job_id"], ["jobs.id"]),
        sa.CheckConstraint(
            "(type = 'scan' AND path_id IS NOT NULL AND file_id IS NULL) OR "
            "(type <> 'scan' AND file_id IS NOT NULL AND path_id IS NULL)",
            name="ck_jobs_target",
        ),
    )

    op.create_index(
        "ix_jobs_file_type_created",
        "jobs",
        ["file_id", "type", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_jobs_path_created",
        "jobs",
        ["path_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_path_created", table_name="jobs")
    op.drop_index("ix_jobs_file_type_created", table_name="jobs")
    op.drop_table("jobs")
