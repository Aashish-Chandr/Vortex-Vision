"""
SQLAlchemy async database setup.
Stores streams, anomaly events, and query history persistently.
Engine is created lazily so test env vars set before import take effect.
"""
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


def _make_async_url(url: str) -> str:
    """Convert a sync DB URL to its async driver equivalent."""
    if url.startswith("sqlite:///") and "aiosqlite" not in url:
        return url.replace("sqlite:///", "sqlite+aiosqlite:///")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


def _build_engine():
    """Build engine from current environment (called lazily)."""
    from config.settings import get_settings
    settings = get_settings()
    url = _make_async_url(settings.database_url)
    return create_async_engine(
        url,
        echo=settings.debug,
        pool_pre_ping=True,
        # SQLite doesn't support connection pooling
        **({} if "sqlite" in url else {"pool_size": 10, "max_overflow": 20}),
    )


# Lazy singletons — built on first access
_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


# Public aliases
def get_engine():
    return _get_engine()


def get_session_factory():
    return _get_session_factory()


class Base(DeclarativeBase):
    pass


class StreamRecord(Base):
    __tablename__ = "streams"
    id = Column(Integer, primary_key=True)
    stream_id = Column(String(64), unique=True, nullable=False, index=True)
    source = Column(Text, nullable=False)
    fps_limit = Column(Integer, default=30)
    active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyEventRecord(Base):
    __tablename__ = "anomaly_events"
    id = Column(Integer, primary_key=True)
    stream_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(Float, nullable=False, index=True)
    frame_id = Column(Integer)
    anomaly_type = Column(String(64))
    confidence = Column(Float)
    autoencoder_score = Column(Float)
    transformer_score = Column(Float)
    clip_path = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class QueryHistoryRecord(Base):
    __tablename__ = "query_history"
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    clips_found = Column(Integer, default=0)
    processing_ms = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


async def init_db():
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
