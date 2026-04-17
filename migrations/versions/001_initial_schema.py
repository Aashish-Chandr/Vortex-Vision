"""Initial schema: streams, anomaly_events, query_history

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "streams",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("stream_id", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("fps_limit", sa.Integer, default=30),
        sa.Column("active", sa.Integer, default=1),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "anomaly_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("stream_id", sa.String(64), nullable=False, index=True),
        sa.Column("timestamp", sa.Float, nullable=False, index=True),
        sa.Column("frame_id", sa.Integer),
        sa.Column("anomaly_type", sa.String(64)),
        sa.Column("confidence", sa.Float),
        sa.Column("autoencoder_score", sa.Float),
        sa.Column("transformer_score", sa.Float),
        sa.Column("clip_path", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "query_history",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("clips_found", sa.Integer, default=0),
        sa.Column("processing_ms", sa.Float),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("query_history")
    op.drop_table("anomaly_events")
    op.drop_table("streams")
