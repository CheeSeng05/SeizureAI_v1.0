from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_engine_and_session(database_url: str, *, echo: bool = False) -> tuple[Engine, sessionmaker[Session]]:
    engine = create_engine(database_url, echo=echo, future=True, pool_pre_ping=True)
    return engine, sessionmaker(engine, expire_on_commit=False)
