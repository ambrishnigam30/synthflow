"""add_is_default_to_llm_configs

Revision ID: 7a4c1f9e2b83
Revises: 2e67eaee50d3
Create Date: 2026-04-25 10:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a4c1f9e2b83'
down_revision: Union[str, None] = '2e67eaee50d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'llm_configs',
        sa.Column(
            'is_default',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )


def downgrade() -> None:
    op.drop_column('llm_configs', 'is_default')
