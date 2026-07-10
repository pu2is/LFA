"""drop file_labels.confidence

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-07-10

Confidence scores are LLM self-reported, uncalibrated, and unreliable --
quality control is entirely the user's confirm/reject action, not a model-
reported number. Per ADR-0001 Decision D2 (docs/adr/0001-normalize-label-schema.md),
implemented ahead of that ADR's broader schema normalization since D2 stands
on its own and doesn't depend on it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f3a4b5c6d7e8"
down_revision: Union[str, Sequence[str], None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("file_labels", "confidence")


def downgrade() -> None:
    op.add_column("file_labels", sa.Column("confidence", sa.Float(), nullable=True))
