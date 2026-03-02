"""
Database layer – async SQLAlchemy with asyncpg.
Provides a shared engine, session factory, and raw-query helpers.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import text

from config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / context-manager that yields a DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# DDL – create tables if they don't exist
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS email_store (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    message_id      TEXT NOT NULL,
    thread_id       TEXT,
    user_email      TEXT NOT NULL,
    from_email      TEXT,
    from_name       TEXT,
    subject         TEXT,
    received_at     TIMESTAMPTZ,
    clean_body      TEXT,
    category        TEXT,
    priority        TEXT,
    intent          TEXT,
    confidence      FLOAT,
    needs_reply     BOOLEAN DEFAULT FALSE,
    needs_calendar  BOOLEAN DEFAULT FALSE,
    needs_reminder  BOOLEAN DEFAULT FALSE,
    event_details   JSONB,
    entities        JSONB,
    reply_tone      TEXT,
    suggested_reply TEXT,
    reminder_date   TIMESTAMPTZ,
    reminder_text   TEXT,
    embedding       vector(768),
    action_status   TEXT DEFAULT 'pending',
    UNIQUE (message_id, user_email)
);

CREATE TABLE IF NOT EXISTS reminders (
    id            SERIAL PRIMARY KEY,
    message_id    TEXT,
    from_email    TEXT,
    subject       TEXT,
    reminder_text TEXT,
    reminder_date TIMESTAMPTZ,
    status        TEXT DEFAULT 'pending',
    priority      TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    sent_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS eod_summaries (
    id               SERIAL PRIMARY KEY,
    summary_date     DATE NOT NULL,
    markdown_summary TEXT,
    email_count      INT DEFAULT 0,
    user_email       TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
"""


async def init_db() -> None:
    """Run DDL on startup."""
    async with engine.begin() as conn:
        for statement in CREATE_TABLES_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                await conn.execute(text(stmt))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

async def fetchone(session: AsyncSession, query: str, *params: Any) -> dict | None:
    result = await session.execute(text(query), dict(enumerate(params)))
    row = result.mappings().first()
    return dict(row) if row else None


async def fetchall(session: AsyncSession, query: str, *params: Any) -> list[dict]:
    result = await session.execute(text(query), dict(enumerate(params)))
    return [dict(row) for row in result.mappings().all()]


async def execute(session: AsyncSession, query: str, params: dict) -> None:
    await session.execute(text(query), params)
