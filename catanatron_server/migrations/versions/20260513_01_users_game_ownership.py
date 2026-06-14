"""add users and game ownership tables

Revision ID: 20260513_01
Revises:
Create Date: 2026-05-13 14:20:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260513_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "game_ownerships",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("game_uuid", sa.String(length=64), nullable=False),
        sa.Column("num_players", sa.Integer(), nullable=False),
        sa.Column("players_config", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "game_uuid", name="uq_game_ownership_user_game"),
    )
    op.create_index(op.f("ix_game_ownerships_game_uuid"), "game_ownerships", ["game_uuid"], unique=False)
    op.create_index(op.f("ix_game_ownerships_user_id"), "game_ownerships", ["user_id"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_game_ownerships_user_id"), table_name="game_ownerships")
    op.drop_index(op.f("ix_game_ownerships_game_uuid"), table_name="game_ownerships")
    op.drop_table("game_ownerships")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
