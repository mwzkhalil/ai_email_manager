"""
Email Manager — FastAPI Application
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Email Manager API…")
    await init_db()          # ← let it raise so you see real errors on startup
    log.info("Database tables initialised.")
    yield
    log.info("Email Manager API shutting down.")


app = FastAPI(
    title="Email Manager API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "path": str(request.url.path)},
    )


app.include_router(ingestion.router,   tags=["Ingestion"])
app.include_router(get_emails.router,  tags=["Emails"])
app.include_router(eod_summary.router, tags=["EOD Summary"])
app.include_router(email_chat.router,  tags=["Chat"])
app.include_router(actions.router,     tags=["Actions"])


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "service": "email-manager-api"}


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service": "Email Manager API",
        "version": "1.0.0",
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