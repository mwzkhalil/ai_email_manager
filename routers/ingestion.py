"""
Workflow 1: Email Manager – main ingestion + action routing
Workflow 2: Bulk Email Ingestion
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from config import get_settings
from db.database import get_db
from models.schemas import EmailManagerRequest, BulkIngestRequest
from utils.ai import analyze_email, create_embedding, embedding_to_pg_vector
from utils.calendar import build_event_datetimes, check_free_busy, create_event
from utils.gmail import (
    list_messages,
    get_message,
    parse_message,
    send_reply,
    create_label,
    add_label_to_message,
)

log = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _count_user_emails(user_email: str) -> int:
    async with get_db() as db:
        result = await db.execute(
            text("SELECT COUNT(*) FROM email_store WHERE user_email = :ue"),
            {"ue": user_email},
        )
        return result.scalar() or 0


async def _is_duplicate(message_id: str, user_email: str) -> bool:
    async with get_db() as db:
        result = await db.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM email_store "
                "WHERE message_id = :mid AND user_email = :ue)"
            ),
            {"mid": message_id, "ue": user_email},
        )
        return result.scalar()


async def _save_email(parsed: dict, analysis: dict, user_email: str) -> None:
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO email_store (
                    message_id, thread_id, user_email, from_email, from_name,
                    subject, received_at, clean_body, category, priority,
                    intent, confidence, needs_reply, needs_calendar, needs_reminder,
                    event_details, entities, reply_tone, suggested_reply,
                    reminder_date, reminder_text
                ) VALUES (
                    :message_id, :thread_id, :user_email, :from_email, :from_name,
                    :subject, :received_at, :clean_body, :category, :priority,
                    :intent, :confidence, :needs_reply, :needs_calendar, :needs_reminder,
                    :event_details::jsonb, :entities::jsonb, :reply_tone, :suggested_reply,
                    :reminder_date, :reminder_text
                )
                ON CONFLICT (message_id, user_email) DO NOTHING
            """),
            {
                "message_id":    parsed["message_id"],
                "thread_id":     parsed.get("thread_id"),
                "user_email":    user_email,
                "from_email":    parsed.get("from_email"),
                "from_name":     parsed.get("from_name"),
                "subject":       parsed.get("subject"),
                "received_at":   parsed.get("received_date"),
                "clean_body":    parsed.get("clean_body"),
                "category":      analysis.get("category"),
                "priority":      analysis.get("priority"),
                "intent":        analysis.get("intent"),
                "confidence":    analysis.get("confidence"),
                "needs_reply":   analysis.get("needsReply", False),
                "needs_calendar": analysis.get("needsCalendarEvent", False),
                "needs_reminder": analysis.get("needsReminder", False),
                "event_details":  str(analysis.get("eventDetails") or "null").replace("'", '"'),
                "entities":      str(analysis.get("entities", {})).replace("'", '"'),
                "reply_tone":    analysis.get("replyTone"),
                "suggested_reply": analysis.get("suggestedReply"),
                "reminder_date": analysis.get("reminderDate"),
                "reminder_text": analysis.get("reminderText"),
            },
        )


async def _save_embedding(message_id: str, user_email: str, embedding_vec: str) -> None:
    async with get_db() as db:
        await db.execute(
            text(
                "UPDATE email_store SET embedding = :emb::vector "
                "WHERE message_id = :mid AND user_email = :ue"
            ),
            {"emb": embedding_vec, "mid": message_id, "ue": user_email},
        )


async def _store_reminder(analysis: dict, parsed: dict) -> None:
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO reminders
                    (message_id, from_email, subject, reminder_text, reminder_date, priority)
                VALUES
                    (:mid, :fe, :sub, :rt, :rd, :pr)
            """),
            {
                "mid": parsed["message_id"],
                "fe":  parsed.get("from_email"),
                "sub": parsed.get("subject"),
                "rt":  analysis.get("reminderText"),
                "rd":  analysis.get("reminderDate"),
                "pr":  analysis.get("priority", "medium"),
            },
        )


async def _update_action_status(message_id: str, user_email: str, status: str) -> None:
    async with get_db() as db:
        await db.execute(
            text(
                "UPDATE email_store SET action_status = :s "
                "WHERE message_id = :mid AND user_email = :ue"
            ),
            {"s": status, "mid": message_id, "ue": user_email},
        )


# ---------------------------------------------------------------------------
# Core processing pipeline (shared by Workflow 1 and 2)
# ---------------------------------------------------------------------------

async def process_single_email(
    message_id: str,
    access_token: str,
    user_email: str,
    auto_mode: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline for one email:
    fetch → parse → dedup check → AI analysis → save → embed → auto-actions
    """
    # 1. Fetch from Gmail
    try:
        raw = await get_message(access_token, message_id)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Gmail fetch failed: {exc}")

    # 2. Parse
    parsed = parse_message(raw)
    parsed["user_email"] = user_email

    # 3. Dedup check
    if await _is_duplicate(parsed["message_id"], user_email):
        return {"status": "duplicate", "message_id": parsed["message_id"]}

    # 4. AI Analysis
    try:
        analysis, provider = await analyze_email(
            from_name=parsed.get("from_name", ""),
            from_email=parsed.get("from_email", ""),
            subject=parsed.get("subject", ""),
            received_date=parsed.get("received_date", ""),
            clean_body=parsed.get("clean_body", ""),
        )
    except Exception as exc:
        log.error("AI analysis failed for %s: %s", message_id, exc)
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}")

    # 5. Save to DB
    await _save_email(parsed, analysis, user_email)

    # 6. Embeddings
    try:
        embedding_text = (
            f"{parsed.get('from_name', '')} {parsed.get('from_email', '')} "
            f"{parsed.get('subject', '')} {analysis.get('intent', '')} "
            f"{analysis.get('priority', '')} {parsed.get('clean_body', '')}"
        )
        embedding = await create_embedding(embedding_text)
        pg_vec = embedding_to_pg_vector(embedding)
        await _save_embedding(parsed["message_id"], user_email, pg_vec)
    except Exception as exc:
        log.warning("Embedding failed for %s: %s", message_id, exc)

    # 7. Auto-mode action routing
    action_results: dict[str, Any] = {}
    if auto_mode:
        action_results = await _route_actions(parsed, analysis, access_token, user_email)

    return {
        "status": "processed",
        "message_id": parsed["message_id"],
        "provider": provider,
        "analysis": analysis,
        "actions": action_results,
    }


async def _route_actions(
    parsed: dict,
    analysis: dict,
    access_token: str,
    user_email: str,
) -> dict[str, Any]:
    """Handle auto-mode action routing (calendar / reply / reminder)."""
    results: dict[str, Any] = {}

    # All matching outputs (run all that apply)
    tasks = []

    if analysis.get("needsCalendarEvent") and analysis.get("eventDetails"):
        tasks.append(("calendar", _handle_calendar(parsed, analysis, access_token, user_email)))

    if analysis.get("needsReply"):
        tasks.append(("reply", _handle_reply(parsed, analysis, access_token, user_email)))

    if analysis.get("needsReminder"):
        tasks.append(("reminder", _handle_reminder(parsed, analysis)))

    import asyncio
    for name, coro in tasks:
        try:
            results[name] = await coro
        except Exception as exc:
            log.warning("Action %s failed: %s", name, exc)
            results[name] = {"error": str(exc)}

    return results


async def _handle_calendar(
    parsed: dict,
    analysis: dict,
    access_token: str,
    user_email: str,
) -> dict:
    event = analysis["eventDetails"]
    try:
        duration = int(event.get("duration", 60))
    except (ValueError, TypeError):
        duration = 60

    start_iso, end_iso = build_event_datetimes(
        event.get("date", ""),
        event.get("time", ""),
        duration,
    )

    available = await check_free_busy(access_token, user_email, start_iso, end_iso)
    if not available:
        await _update_action_status(parsed["message_id"], user_email, "slot_unavailable")
        return {"status": "slot_unavailable", "start": start_iso}

    created = await create_event(
        access_token=access_token,
        user_email=user_email,
        title=event.get("title", parsed.get("subject", "Meeting")),
        start=start_iso,
        end=end_iso,
        description=event.get("description", ""),
        location=event.get("location", ""),
        attendees=event.get("attendees", []),
    )

    # Send confirmation email to sender
    try:
        await send_reply(
            access_token=access_token,
            to=parsed["from_email"],
            subject=parsed.get("subject", ""),
            body=(
                f"Hi,\n\nA calendar event '{event.get('title')}' has been created "
                f"for {event.get('date')} at {event.get('time')}.\n\nBest regards"
            ),
            thread_id=parsed["thread_id"],
            message_id=parsed["message_id"],
        )
    except Exception as exc:
        log.warning("Calendar confirmation email failed: %s", exc)

    await _update_action_status(parsed["message_id"], user_email, "completed")
    return {"status": "created", "event_id": created.get("id"), "start": start_iso}


async def _handle_reply(
    parsed: dict,
    analysis: dict,
    access_token: str,
    user_email: str,
) -> dict:
    reply_body = analysis.get("suggestedReply", "")
    if not reply_body:
        return {"status": "skipped", "reason": "no suggested reply"}

    await send_reply(
        access_token=access_token,
        to=parsed["from_email"],
        subject=parsed.get("subject", ""),
        body=reply_body,
        thread_id=parsed["thread_id"],
        message_id=parsed["message_id"],
    )

    # Label the original message
    try:
        await create_label(access_token, "Auto-Replied")
        await add_label_to_message(access_token, parsed["message_id"], ["Label_2"])
    except Exception as exc:
        log.warning("Labelling failed: %s", exc)

    await _update_action_status(parsed["message_id"], user_email, "completed")
    return {"status": "sent"}


async def _handle_reminder(parsed: dict, analysis: dict) -> dict:
    await _store_reminder(analysis, parsed)
    return {"status": "stored", "reminder_date": analysis.get("reminderDate")}


# ---------------------------------------------------------------------------
# Fetch all message IDs with optional pagination
# ---------------------------------------------------------------------------

async def _fetch_all_message_ids(
    access_token: str,
    max_results: int = 10,
    query: str = "",
) -> list[str]:
    ids: list[str] = []
    page_token: str | None = None

    while True:
        resp = await list_messages(access_token, max_results, query, page_token)
        messages = resp.get("messages", [])
        ids.extend(m["id"] for m in messages)
        page_token = resp.get("nextPageToken")
        if not page_token or len(ids) >= max_results:
            break

    return ids[:max_results]


# ---------------------------------------------------------------------------
# Route: Workflow 1 — /email-manager
# ---------------------------------------------------------------------------

@router.post("/email-manager")
async def email_manager(req: EmailManagerRequest) -> dict:
    email_count = await _count_user_emails(req.user_email)

    # Routing decision
    if email_count >= settings.bulk_threshold:
        fetch_limit = settings.email_fetch_limit
    else:
        # Trigger bulk ingestion by routing to the bulk logic directly
        bulk_req = BulkIngestRequest(
            access_token=req.access_token,
            refresh_token=req.refresh_token,
            user_email=req.user_email,
            autoMode=req.auto_mode,
        )
        return await bulk_ingest(bulk_req)

    # Fetch latest N messages
    try:
        message_ids = await _fetch_all_message_ids(req.access_token, fetch_limit)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Gmail list failed: {exc}")

    results = []
    for mid in message_ids:
        try:
            result = await process_single_email(mid, req.access_token, req.user_email, req.auto_mode)
            results.append(result)
        except Exception as exc:
            log.error("Failed to process message %s: %s", mid, exc)
            results.append({"message_id": mid, "status": "error", "error": str(exc)})

    processed = sum(1 for r in results if r.get("status") == "processed")
    duplicates = sum(1 for r in results if r.get("status") == "duplicate")
    errors     = sum(1 for r in results if r.get("status") == "error")

    return {
        "success": True,
        "mode": "normal",
        "total_fetched": len(message_ids),
        "processed": processed,
        "duplicates": duplicates,
        "errors": errors,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Route: Workflow 2 — /ingest_email_bulk
# ---------------------------------------------------------------------------

@router.post("/ingest_email_bulk")
async def bulk_ingest(req: BulkIngestRequest) -> dict:
    try:
        message_ids = await _fetch_all_message_ids(
            req.access_token,
            max_results=500,   # fetch up to 500 for bulk
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Gmail list failed: {exc}")

    results = []
    # Process in batches of 10 to avoid overwhelming downstream APIs
    batch_size = 10
    for i in range(0, len(message_ids), batch_size):
        batch = message_ids[i : i + batch_size]
        import asyncio
        batch_results = await asyncio.gather(
            *[
                process_single_email(mid, req.access_token, req.user_email, req.auto_mode)
                for mid in batch
            ],
            return_exceptions=True,
        )
        for mid, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results.append({"message_id": mid, "status": "error", "error": str(result)})
            else:
                results.append(result)

    processed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "processed")
    return {
        "success": True,
        "mode": "bulk",
        "total_fetched": len(message_ids),
        "processed": processed,
        "results": results,
    }
