"""
Email Manager — FastAPI Application
=====================================
A complete Python port of the 6-workflow n8n Email Manager system.

Workflows implemented:
  1. POST /email-manager         — Main ingestion + action routing
  2. POST /ingest_email_bulk     — Bulk email ingestion
  3. POST /get-emails            — Retrieve stored emails for frontend
  4a. POST /eod-summary-generate — Generate end-of-day summary
  4b. POST /show-eod             — Show / regenerate EOD summary
  4c. POST /send-eod-email       — Email the EOD summary
  5. POST /chat-email            — Semantic search + AI chat
  6. POST /action_status         — Update action status on records

Manual action endpoints:
  POST /manual-create-calendar
  POST /manual-send-reply
  POST /manual-set-reminder

Usage:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db.database import init_db
from routers import ingestion, get_emails, eod_summary, email_chat, actions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Email Manager API…")
    try:
        await init_db()
        log.info("Database tables initialised.")
    except Exception as exc:
        log.warning("DB init skipped (check DATABASE_URL): %s", exc)
    yield
    log.info("Email Manager API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Manager API",
    description=(
        "AI-powered email management system with Gmail integration, "
        "LLM analysis, vector search, calendar automation, and EOD summaries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "path": str(request.url.path)},
    )


# ---------------------------------------------------------------------------
# Include routers (webhook paths match original n8n paths exactly)
# ---------------------------------------------------------------------------

app.include_router(ingestion.router,   tags=["Ingestion"])
app.include_router(get_emails.router,  tags=["Emails"])
app.include_router(eod_summary.router, tags=["EOD Summary"])
app.include_router(email_chat.router,  tags=["Chat"])
app.include_router(actions.router,     tags=["Actions"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "service": "email-manager-api"}


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service":  "Email Manager API",
        "version":  "1.0.0",
        "endpoints": [
            "POST /email-manager",
            "POST /ingest_email_bulk",
            "POST /get-emails",
            "POST /eod-summary-generate",
            "POST /show-eod",
            "POST /send-eod-email",
            "POST /chat-email",
            "POST /action_status",
            "POST /manual-create-calendar",
            "POST /manual-send-reply",
            "POST /manual-set-reminder",
            "GET  /health",
            "GET  /docs",
        ],
    }
