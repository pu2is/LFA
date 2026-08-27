"""rescan foundation schema

Revision ID: 1e7ea42aa609
Revises: a5b6c7d8e9f0
Create Date: 2026-08-27

#64 (ADR-0001b / WF1b): schema-only foundation for Global Rescan --
inventory/diff/matching/apply logic and the /rescans API are separate,
later tickets. Adds file_events + file_match_candidates (Rescan audit and
fuzzy-recovery tables, owned by scans per docs/02_architecture.md), extends
files with filesystem-identity/text-signature/review columns, replaces
jobs.mode's unused 'check' value with 'rescan', and widens ck_jobs_target
so type=scan splits into path-level mode=initial vs. global mode=rescan
(path_id NULL). Also adds a partial unique index guaranteeing at most one
active (queued/running) global Rescan even under concurrent requests.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "1e7ea42aa609"
down_revision: Union[str, Sequence[str], None] = "a5b6c7d8e9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("files", sa.Column("fs_device_id", sa.String(), nullable=True))
    op.add_column("files", sa.Column("fs_file_id", sa.String(), nullable=True))
    op.add_column("files", sa.Column("text_signature", sa.String(), nullable=True))
    op.add_column(
        "files",
        sa.Column("labels_need_review", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_index("ix_files_fs_device_file_id", "files", ["fs_device_id", "fs_file_id"])

    op.create_table(
        "file_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scan_id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("from_path", sa.String(), nullable=True),
        sa.Column("to_path", sa.String(), nullable=True),
        sa.Column("from_hash", sa.String(), nullable=True),
        sa.Column("to_hash", sa.String(), nullable=True),
        sa.Column("match_method", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scan_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.UniqueConstraint("scan_id", "file_id", "event_type"),
    )

    op.create_table(
        "file_match_candidates",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scan_id", sa.UUID(), nullable=False),
        sa.Column("missing_file_id", sa.UUID(), nullable=False),
        sa.Column("candidate_path_id", sa.UUID(), nullable=False),
        sa.Column("candidate_full_path", sa.String(), nullable=False),
        sa.Column("candidate_hash", sa.String(), nullable=False),
        sa.Column("candidate_size", sa.BigInteger(), nullable=False),
        sa.Column("candidate_modified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scan_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["missing_file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["candidate_path_id"], ["paths.id"]),
        sa.UniqueConstraint("scan_id", "missing_file_id", "candidate_full_path"),
    )

    op.drop_constraint("ck_jobs_target", "jobs", type_="check")
    op.create_check_constraint(
        "ck_jobs_target",
        "jobs",
        "(type = 'scan' AND file_id IS NULL AND ("
        "(mode = 'initial' AND path_id IS NOT NULL) OR "
        "(mode = 'rescan' AND path_id IS NULL)"
        ")) OR "
        "(type <> 'scan' AND file_id IS NOT NULL AND path_id IS NULL)",
    )

    op.create_index(
        "ix_jobs_active_rescan",
        "jobs",
        ["type", "mode"],
        unique=True,
        postgresql_where=sa.text("type = 'scan' AND mode = 'rescan' AND status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_active_rescan", table_name="jobs")

    op.drop_constraint("ck_jobs_target", "jobs", type_="check")
    op.create_check_constraint(
        "ck_jobs_target",
        "jobs",
        "(type = 'scan' AND path_id IS NOT NULL AND file_id IS NULL) OR "
        "(type <> 'scan' AND file_id IS NOT NULL AND path_id IS NULL)",
    )

    op.drop_table("file_match_candidates")
    op.drop_table("file_events")

    op.drop_index("ix_files_fs_device_file_id", table_name="files")
    op.drop_column("files", "labels_need_review")
    op.drop_column("files", "text_signature")
    op.drop_column("files", "fs_file_id")
    op.drop_column("files", "fs_device_id")
