"""create memory tables (users, sessions, user_profiles, session_memories)

Revision ID: 0002_create_memory_tables
Revises: 0001_create_items
Create Date: 2026-06-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002_create_memory_tables"
down_revision: Union[str, None] = "0001_create_items"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=40), primary_key=True),
        sa.Column("channel", sa.String(length=20), nullable=True),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("channel", "external_id", name="uq_users_channel_external_id"),
    )

    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=80), primary_key=True),
        sa.Column("user_id", sa.String(length=40), sa.ForeignKey("users.user_id"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("thread_id", sa.String(length=100), nullable=True),
        sa.Column("turn_no", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_index("idx_sessions_expires_at", "sessions", ["expires_at"])

    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(length=40), sa.ForeignKey("users.user_id"), primary_key=True),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("profile_json", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("profile_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "session_memories",
        sa.Column("session_id", sa.String(length=80), sa.ForeignKey("sessions.session_id"), primary_key=True),
        sa.Column("user_id", sa.String(length=40), sa.ForeignKey("users.user_id"), nullable=False),
        sa.Column("memory_json", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_session_memories_user_id", "session_memories", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_session_memories_user_id", table_name="session_memories")
    op.drop_table("session_memories")
    op.drop_table("user_profiles")
    op.drop_index("idx_sessions_expires_at", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("users")
