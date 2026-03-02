"""
Workflow 5: Email Chat — semantic search + AI chat over stored emails.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from config import get_settings
from db.database import get_db
from models.schemas import ChatRequest
from utils.ai import create_embedding, embedding_to_pg_vector, chat_with_emails

log = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


def _format_email_context(rows: list[dict]) -> str:
    """Convert DB rows into a readable context block for the LLM."""
    lines: list[str] = []
    for i, row in enumerate(rows, 1):
        lines.append(
            f"[Email {i}]\n"
            f"From: {row.get('from_name', '')} <{row.get('from_email', '')}>\n"
            f"Subject: {row.get('subject', '')}\n"
            f"Received: {row.get('received_at', '')}\n"
            f"Priority: {row.get('priority', '')}\n"
            f"Category: {row.get('category', '')}\n"
            f"Intent: {row.get('intent', '')}\n"
            f"Needs Reply: {row.get('needs_reply', False)}\n"
            f"Needs Calendar: {row.get('needs_calendar', False)}\n"
            f"Needs Reminder: {row.get('needs_reminder', False)}\n"
            f"Body:\n{row.get('clean_body', '')}\n"
            "---"
        )
    return "\n".join(lines)


@router.post("/chat-email")
async def chat_email(req: ChatRequest) -> dict:
    # 1. Embed the query
    try:
        embedding = await create_embedding(req.query)
        pg_vec = embedding_to_pg_vector(embedding)
    except Exception as exc:
        log.error("Query embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    # 2. Semantic similarity search
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT *, embedding <-> :vec::vector AS distance
                FROM email_store
                WHERE user_email = :ue
                  AND embedding IS NOT NULL
                ORDER BY distance
                LIMIT :limit
            """),
            {
                "vec":   pg_vec,
                "ue":    req.user_email,
                "limit": settings.semantic_search_limit,
            },
        )
        rows = [dict(r) for r in result.mappings().all()]

    if not rows:
        return {
            "success": True,
            "type": "chat",
            "answer": (
                "I couldn't find any relevant emails matching your query. "
                "Make sure your emails have been processed and embeddings generated."
            ),
            "llm": "none",
        }

    # 3. Format context
    context = _format_email_context(rows)

    # 4. LLM chat
    try:
        answer, provider = await chat_with_emails(context, req.query)
    except Exception as exc:
        log.error("Chat LLM failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")

    return {
        "success": True,
        "type":    "chat",
        "answer":  answer,
        "llm":     provider,
    }
