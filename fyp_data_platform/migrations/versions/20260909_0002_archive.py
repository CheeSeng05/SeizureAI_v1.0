"""Archive and resumable deletion state."""
from alembic import op
import sqlalchemy as sa

revision = "20260909_0002"
down_revision = "20260815_0001"
branch_labels = None
depends_on = None

def upgrade():
    for table in ("recordings", "model_runs"):
        op.add_column(table, sa.Column("archived", sa.Boolean(), server_default=sa.false(), nullable=False))
        op.add_column(table, sa.Column("deleting", sa.Boolean(), server_default=sa.false(), nullable=False))

def downgrade():
    for table in ("model_runs", "recordings"):
        op.drop_column(table, "deleting")
        op.drop_column(table, "archived")
