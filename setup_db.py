#!/usr/bin/env python3
"""
Run this once to set up your local database.
  python setup_db.py
"""
import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# ── paste your DATABASE_URL here if .env isn't loading ──────────────────────
import os, pathlib
env_file = pathlib.Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

print(f"Using: {DATABASE_URL}")

# ── DDL ──────────────────────────────────────────────────────────────────────
STATEMENTS = [
    # Try to install pgvector — may fail if not available, that's OK
    "CREATE EXTENSION IF NOT EXISTS vector",

    """CREATE TABLE IF NOT EXISTS email_store (
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
    )""",

    """CREATE TABLE IF NOT EXISTS reminders (
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
    )""",

    """CREATE TABLE IF NOT EXISTS eod_summaries (
        id               SERIAL PRIMARY KEY,
        summary_date     DATE NOT NULL,
        markdown_summary TEXT,
        email_count      INT DEFAULT 0,
        user_email       TEXT NOT NULL,
        created_at       TIMESTAMPTZ DEFAULT NOW(),
        updated_at       TIMESTAMPTZ DEFAULT NOW()
    )""",
]

async def main():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            label = stmt.strip()[:60].replace("\n", " ")
            try:
                await conn.execute(text(stmt))
                print(f"  ✓  {label}")
            except Exception as e:
                err = str(e).split("\n")[0]
                if "already exists" in err or "vector" in err.lower():
                    print(f"  ~  {label}  ({err})")
                else:
                    print(f"  ✗  {label}\n     ERROR: {err}")
                    raise
    await engine.dispose()
    print("\n✅ Database ready!")

if __name__ == "__main__":
    asyncio.run(main())