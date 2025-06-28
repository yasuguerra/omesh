from __future__ import annotations

"""SQLAlchemy engine and session helpers."""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from ..settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWD, DB_DATABASE

_DB_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}" if DB_HOST else ""
)

engine: Engine | None = None

if _DB_URL:
    engine = create_engine(
        _DB_URL,
        pool_pre_ping=True,
        future=True,
    )

    _Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
else:
    _Session = sessionmaker()

@contextmanager
def get_session() -> Session:
    """Provide a transactional scope around a series of operations."""
    session: Session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

