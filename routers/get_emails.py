"""
Workflow 3: Get Emails — retrieve stored emails from DB for frontend.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from db.database import get_db
from models.schemas import GetEmailsRequest

log = logging.getLogger(__name__)
router = APIRouter()


def _compute_action_items(row: dict) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if row.get("needs_reply"):
        status = row.get("action_status", "pending")
        items.append({
            "type": "reply",
            "title": "Reply to Email",
            "status": "completed" if status == "completed" else "pending",
        })

    if row.get("needs_calendar"):
        event = row.get("event_details")
        if event:
            action_status = row.get("action_status", "pending")
            cal_status = (
                "completed"        if action_status == "completed"
                else "slot_unavailable" if action_status == "slot_unavailable"
                else "pending"
            )
            items.append({
                "type": "calendar",
                "title": "Create Calendar Event",
                "status": cal_status,
            })

    confidence = float(row.get("confidence") or 1.0)
    if confidence < 0.7:
        items.append({
            "type": "review",
            "title": "Manual Review Required",
            "status": "pending",
        })

    if not items:
        items.append({
            "type": "info",
            "title": "No Action Required",
            "status": "done",
        })

    return items


def _row_to_email_record(row: dict) -> dict[str, Any]:
    def _safe_json(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return val
        try:
            return json.loads(val)
        except Exception:
            return val

    action_items = _compute_action_items(row)
    event_details = _safe_json(row.get("event_details"))
    entities = _safe_json(row.get("entities"))
    confidence = float(row.get("confidence") or 0)

    return {
        "success": True,
        "email": {
            "id":       row.get("message_id"),
            "threadId": row.get("thread_id"),
            "from": {
                "name":  row.get("from_name"),
                "email": row.get("from_email"),
            },
            "subject":     row.get("subject"),
            "body": {
                "text":  None,
                "html":  None,
                "clean": row.get("clean_body"),
            },
            "receivedDate": str(row.get("received_at", "")),
            "labels":      [],
            "attachments": [],
        },
        "actionItems": action_items,
        "analysis": {
            "category":          row.get("category"),
            "intent":            row.get("intent"),
            "priority":          row.get("priority"),
            "confidence":        confidence,
            "needsReply":        bool(row.get("needs_reply")),
            "needsCalendarEvent": bool(row.get("needs_calendar")),
            "needsReminder":     bool(row.get("needs_reminder")),
            "suggestedReply":    row.get("suggested_reply"),
            "replyTone":         row.get("reply_tone"),
            "eventDetails":      event_details,
            "entities":          entities,
            "reminderDate":      str(row.get("reminder_date", "")) or None,
            "reminderText":      row.get("reminder_text"),
        },
        "flags": {
            "needsManualReview": confidence < 0.7,
            "autoMode":          True,
        },
        "metadata": {
            "processedAt": str(row.get("received_at", "")),
        },
    }


@router.post("/get-emails")
async def get_emails(req: GetEmailsRequest) -> dict:
    async with get_db() as db:
        result = await db.execute(
            text(
                "SELECT * FROM email_store "
                "WHERE user_email = :ue "
                "ORDER BY received_at DESC"
            ),
            {"ue": req.user_email},
        )
        rows = [dict(r) for r in result.mappings().all()]

    emails = [_row_to_email_record(row) for row in rows]

    return {
        "success": True,
        "count": len(emails),
        "emails": emails,
    }
