"""Async database engine and session factory — no globals, pure DI."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_engine(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine from a database URL.

    Args:
        database_url: PostgreSQL (or other) connection URL.

    Returns:
        A configured ``AsyncEngine`` instance.
    """
    return create_async_engine(database_url, echo=False)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the given engine.

    Args:
        engine: An ``AsyncEngine`` instance (typically created via ``create_engine``).

    Returns:
        A configured ``async_sessionmaker`` ready to produce sessions.
    """
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
