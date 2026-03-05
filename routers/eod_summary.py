from __future__ import annotations

import base64
import logging
from datetime import date, datetime

import pytz
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from config import get_settings
from db.database import get_db
from models.schemas import EODGenerateRequest, EODShowRequest, EODEmailRequest
from utils.ai import generate_eod_summary
from utils.gmail import send_message
from utils.markdown import format_eod_html_email

log = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    tz = pytz.timezone(settings.default_timezone)
    return datetime.now(tz).strftime("%Y-%m-%d")


async def _get_todays_emails(user_email: str, today: str) -> list[dict]:
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT message_id, from_name, from_email, subject,
                       received_at, category, priority, intent, confidence,
                       needs_reply, needs_calendar, needs_reminder
                FROM email_store
                WHERE user_email = :ue
                  AND received_at::date = :today::date
                ORDER BY received_at DESC
            """),
            {"ue": user_email, "today": today},
        )
        return [dict(r) for r in result.mappings().all()]


async def _get_existing_eod(user_email: str, today: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT * FROM eod_summaries
                WHERE user_email = :ue
                  AND summary_date::date = CAST(:today AS DATE)
                LIMIT 1
            """),
            {"ue": user_email, "today": today},
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def _upsert_eod(
    user_email: str,
    today: str,
    markdown_summary: str,
    email_count: int,
    eod_id: int | None,
) -> dict:
    async with get_db() as db:
        if eod_id is not None:
            await db.execute(
                text("""
                    UPDATE eod_summaries
                    SET markdown_summary = :ms,
                        email_count      = :ec,
                        updated_at       = NOW()
                    WHERE id = :id
                """),
                {"ms": markdown_summary, "ec": email_count, "id": eod_id},
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO eod_summaries
                        (summary_date, markdown_summary, email_count, user_email)
                    VALUES (CAST(:sd AS DATE), :ms, :ec, :ue)
                """),
                {
                    "sd": today,
                    "ms": markdown_summary,
                    "ec": email_count,
                    "ue": user_email,
                },
            )
    return await _get_existing_eod(user_email, today) or {}


# ---------------------------------------------------------------------------
# 4a: Generate EOD Summary
# ---------------------------------------------------------------------------

@router.post("/eod-summary-generate")
async def eod_summary_generate(req: EODGenerateRequest) -> dict:
    today = _today_str()
    emails = await _get_todays_emails(req.user_email, today)

    if not emails:
        return {
            "success": True,
            "email_count": 0,
            "summary_date": today,
            "markdown_summary": f"# EOD Summary — {today}\n\nNo emails received today.",
        }

    stats = {
        "total":          len(emails),
        "high":           sum(1 for e in emails if e.get("priority") == "high"),
        "medium":         sum(1 for e in emails if e.get("priority") == "medium"),
        "low":            sum(1 for e in emails if e.get("priority") == "low"),
        "needs_reply":    sum(1 for e in emails if e.get("needs_reply")),
        "needs_calendar": sum(1 for e in emails if e.get("needs_calendar")),
        "needs_reminder": sum(1 for e in emails if e.get("needs_reminder")),
    }
    high_priority = [e for e in emails if e.get("priority") == "high"]

    try:
        markdown_summary, provider = await generate_eod_summary(today, stats, high_priority, emails)
    except Exception as exc:
        log.error("EOD LLM failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"EOD generation failed: {exc}")

    saved = await _upsert_eod(req.user_email, today, markdown_summary, len(emails), req.eod_id)

    return {
        "success":          True,
        "email_count":      len(emails),
        "summary_date":     today,
        "markdown_summary": markdown_summary,
        "eod_id":           saved.get("id"),
        "provider":         provider,
    }


# ---------------------------------------------------------------------------
# 4b: Show EOD Summary
# ---------------------------------------------------------------------------

@router.post("/show-eod")
async def show_eod(req: EODShowRequest) -> dict:
    today = _today_str()
    existing = await _get_existing_eod(req.user_email, today)

    if req.regenerate and existing:
        gen_req = EODGenerateRequest(
            access_token="",
            user_email=req.user_email,
            eod_id=existing["id"],
        )
        return await eod_summary_generate(gen_req)

    if existing and existing.get("id"):
        return {
            "success":          True,
            "email_count":      existing.get("email_count", 0),
            "summary_date":     str(existing.get("summary_date", today)),
            "markdown_summary": existing.get("markdown_summary", ""),
            "eod_id":           existing.get("id"),
        }

    gen_req = EODGenerateRequest(
        access_token="",
        user_email=req.user_email,
        eod_id=None,
    )
    return await eod_summary_generate(gen_req)


# ---------------------------------------------------------------------------
# 4c: Send EOD Email
# ---------------------------------------------------------------------------

@router.post("/send-eod-email")
async def send_eod_email(req: EODEmailRequest) -> dict:
    import base64 as _b64
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    today = _today_str()
    existing = await _get_existing_eod(req.user_email, today)

    if not existing:
        raise HTTPException(
            status_code=404,
            detail="No EOD summary found for today. Generate one first.",
        )

    markdown_summary = existing.get("markdown_summary", "")
    email_count      = existing.get("email_count", 0)
    html_body        = format_eod_html_email(markdown_summary, today)

    subject     = f"📊 EOD Summary - {today}"
    subject_b64 = base64.b64encode(subject.encode("utf-8")).decode("utf-8")

    msg = MIMEMultipart("alternative")
    msg["To"]      = req.user_email
    msg["Subject"] = f"=?UTF-8?B?{subject_b64}?="
    msg.attach(MIMEText(markdown_summary, "plain", "utf-8"))
    msg.attach(MIMEText(html_body,        "html",  "utf-8"))

    raw = _b64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    try:
        await send_message(req.access_token, raw)
    except Exception as exc:
        log.error("EOD email send failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Email send failed: {exc}")

    return {
        "success":      True,
        "email_count":  email_count,
        "summary_date": today,
    }