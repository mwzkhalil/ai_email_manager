"""
Workflow 6: Action Status — update action_status on email records.
Manual Action Endpoints: calendar, reply, reminder.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from db.database import get_db
from models.schemas import (
    ActionStatusRequest,
    ManualCalendarRequest,
    ManualReplyRequest,
    ManualReminderRequest,
)
from utils.calendar import build_event_datetimes, check_free_busy, create_event
from utils.gmail import send_reply

log = logging.getLogger(__name__)
router = APIRouter()

VALID_STATUSES = {"pending", "completed", "dismissed"}


# ---------------------------------------------------------------------------
# Workflow 6: Action Status
# ---------------------------------------------------------------------------

@router.post("/action_status")
async def action_status(req: ActionStatusRequest) -> dict:
    if req.action_status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{req.action_status}'. Must be one of: {VALID_STATUSES}",
        )

    async with get_db() as db:
        result = await db.execute(
            text("""
                UPDATE email_store
                SET action_status = :s
                WHERE message_id = :mid AND user_email = :ue
                RETURNING *
            """),
            {"s": req.action_status, "mid": req.message_id, "ue": req.user_email},
        )
        row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Email record not found.")

    return {"success": True, "data": dict(row)}


# ---------------------------------------------------------------------------
# Manual: Create Calendar Event
# ---------------------------------------------------------------------------

@router.post("/manual-create-calendar")
async def manual_create_calendar(req: ManualCalendarRequest) -> dict:
    # Fetch email record for context
    async with get_db() as db:
        result = await db.execute(
            text("SELECT * FROM email_store WHERE message_id = :mid LIMIT 1"),
            {"mid": req.message_id},
        )
        row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Email not found.")

    event = req.event_details
    try:
        duration = int(event.duration or 60)
    except (ValueError, TypeError):
        duration = 60

    start_iso, end_iso = build_event_datetimes(event.date, event.time, duration)

    available = await check_free_busy(req.access_token, req.user_email, start_iso, end_iso)

    if not available:
        async with get_db() as db:
            await db.execute(
                text("UPDATE email_store SET action_status = 'slot_unavailable' WHERE message_id = :mid"),
                {"mid": req.message_id},
            )
        return {
            "success": False,
            "status": "slot_unavailable",
            "message": f"Calendar slot at {start_iso} is not available.",
        }

    created = await create_event(
        access_token=req.access_token,
        user_email=req.user_email,
        title=event.title or dict(row).get("subject", "Meeting"),
        start=start_iso,
        end=end_iso,
        description=event.description,
        location=event.location,
        attendees=event.attendees,
    )

    async with get_db() as db:
        await db.execute(
            text("UPDATE email_store SET action_status = 'completed' WHERE message_id = :mid"),
            {"mid": req.message_id},
        )

    return {
        "success":  True,
        "status":   "created",
        "event_id": created.get("id"),
        "start":    start_iso,
        "end":      end_iso,
    }


# ---------------------------------------------------------------------------
# Manual: Send Reply
# ---------------------------------------------------------------------------

@router.post("/manual-send-reply")
async def manual_send_reply(req: ManualReplyRequest) -> dict:
    async with get_db() as db:
        result = await db.execute(
            text("SELECT * FROM email_store WHERE message_id = :mid LIMIT 1"),
            {"mid": req.message_id},
        )
        row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Email not found.")

    row_dict = dict(row)

    try:
        await send_reply(
            access_token=req.access_token,
            to=row_dict.get("from_email", ""),
            subject=row_dict.get("subject", ""),
            body=req.reply_text,
            thread_id=row_dict.get("thread_id", ""),
            message_id=req.message_id,
            cc=req.cc or [],
            bcc=req.bcc or [],
            attachments=req.attachments or [],
        )
    except Exception as exc:
        log.error("Manual reply failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to send reply: {exc}")

    async with get_db() as db:
        await db.execute(
            text("UPDATE email_store SET action_status = 'completed' WHERE message_id = :mid"),
            {"mid": req.message_id},
        )

    return {"success": True, "status": "sent"}


# ---------------------------------------------------------------------------
# Manual: Set Reminder
# ---------------------------------------------------------------------------

@router.post("/manual-set-reminder")
async def manual_set_reminder(req: ManualReminderRequest) -> dict:
    rd = req.reminder_details

    async with get_db() as db:
        result = await db.execute(
            text("""
                INSERT INTO reminders
                    (message_id, from_email, subject, reminder_text, reminder_date, priority, status)
                VALUES
                    (:mid, :fe, :sub, :rt, :rd, :pr, 'pending')
                RETURNING id
            """),
            {
                "mid": req.message_id,
                "fe":  rd.from_email,
                "sub": rd.subject,
                "rt":  rd.text,
                "rd":  rd.date,
                "pr":  rd.priority,
            },
        )
        row = result.mappings().first()

    return {
        "success":     True,
        "status":      "stored",
        "reminder_id": row["id"] if row else None,
        "reminder_date": rd.date,
    }
