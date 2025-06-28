from __future__ import annotations

"""
SQLAlchemy database connector for MariaDB using PyMySQL.

This module provides a SQLAlchemy engine and a session management context manager
for interacting with the database defined in the application settings.
It supports connection pooling with pre-ping and uses SQLAlchemy 2.0 features.
"""

from contextlib import contextmanager
import typing  # Import typing for Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession  # Alias to avoid confusion
from sqlalchemy.engine import Engine

from ..settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWD, DB_DATABASE

_DB_URL: str = ""
if DB_HOST and DB_PORT and DB_USER and DB_DATABASE:  # Ensure all essential components are present
    _DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

engine: Engine | None = None
_SessionMaker: sessionmaker[SQLAlchemySession]

if _DB_URL:
    engine = create_engine(
        _DB_URL,
        pool_pre_ping=True,  # Enable connection pool pre-ping
        future=True,  # Enable SQLAlchemy 2.0 features
        echo=False,  # Set to True for debugging SQL queries
    )
    _SessionMaker = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
else:
    # Fallback if DB configuration is not complete, allows app to run without DB for some features
    _SessionMaker = sessionmaker(future=True)

@contextmanager
def get_session() -> typing.Iterator[SQLAlchemySession]:
    """
    Provide a transactional scope around a series of database operations.

    This context manager handles session creation, commit, rollback, and closing.
    Example:
        with get_session() as session:
            session.add(MyModel(name="example"))
    """
    session: SQLAlchemySession = _SessionMaker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

