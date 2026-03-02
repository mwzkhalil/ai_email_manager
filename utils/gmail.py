"""
Gmail API helpers.
All functions are async and accept OAuth tokens directly (no stored credentials).
"""
from __future__ import annotations

import base64
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Any, Optional

import httpx

GMAIL_BASE = "https://www.googleapis.com/gmail/v1/users/me"


def _auth_headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

async def list_messages(
    access_token: str,
    max_results: int = 10,
    query: str = "",
    page_token: str | None = None,
) -> dict:
    """Return raw Gmail messages.list response."""
    params: dict[str, Any] = {"maxResults": max_results}
    if query:
        params["q"] = query
    if page_token:
        params["pageToken"] = page_token

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GMAIL_BASE}/messages",
            headers=_auth_headers(access_token),
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


async def get_message(access_token: str, message_id: str) -> dict:
    """Fetch full message payload."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GMAIL_BASE}/messages/{message_id}",
            headers=_auth_headers(access_token),
            params={"format": "full"},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def _decode_base64(data: str) -> str:
    """Decode URL-safe base64 string."""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _walk_parts(payload: dict) -> tuple[str, str]:
    """Recursively collect text/plain and text/html parts."""
    body_text = ""
    body_html = ""
    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        body_text = _decode_base64(payload.get("body", {}).get("data", ""))
    elif mime_type == "text/html":
        body_html = _decode_base64(payload.get("body", {}).get("data", ""))
    elif parts:
        for part in parts:
            t, h = _walk_parts(part)
            body_text = body_text or t
            body_html = body_html or h

    return body_text, body_html


def parse_message(raw: dict) -> dict:
    """
    Convert a raw Gmail API message into a normalised dict matching
    the Email Preprocessor output.
    """
    payload = raw.get("payload", {})
    headers = payload.get("headers", [])

    from_raw = _extract_header(headers, "From")
    subject = _extract_header(headers, "Subject")
    date_str = _extract_header(headers, "Date")

    # Parse from_email / from_name from "Name <email>" or "email"
    name_match = re.match(r"^(.*?)\s*<(.+?)>$", from_raw)
    if name_match:
        from_name = name_match.group(1).strip().strip('"')
        from_email = name_match.group(2).strip()
    else:
        from_email = from_raw.strip()
        from_name = from_raw.strip()

    body_text, body_html = _walk_parts(payload)
    clean_body = (body_text or re.sub(r"<[^>]+>", " ", body_html)).strip()

    from config import get_settings
    max_chars = get_settings().max_body_chars

    return {
        "message_id": raw.get("id", ""),
        "thread_id": raw.get("threadId", ""),
        "from_email": from_email,
        "from_name": from_name,
        "subject": subject,
        "received_date": date_str,
        "body_text": body_text,
        "body_html": body_html,
        "clean_body": clean_body[:max_chars],
    }


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def _build_reply_raw(
    to: str,
    subject: str,
    body: str,
    thread_id: str,
    in_reply_to: str,
    references: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    attachments: list[dict] | None = None,
) -> str:
    """Build a base64url-encoded RFC 2822 email."""
    msg = MIMEMultipart("mixed")
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        msg["Bcc"] = ", ".join(bcc)
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    msg.attach(MIMEText(body, "plain"))

    for att in (attachments or []):
        part = MIMEBase("application", "octet-stream")
        part.set_payload(base64.b64decode(att.get("data", "")))
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{att.get("filename", "attachment")}"',
        )
        msg.attach(part)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return raw


async def send_message(
    access_token: str,
    raw_message: str,
    thread_id: str | None = None,
) -> dict:
    body: dict[str, Any] = {"raw": raw_message}
    if thread_id:
        body["threadId"] = thread_id

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GMAIL_BASE}/messages/send",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def send_reply(
    access_token: str,
    to: str,
    subject: str,
    body: str,
    thread_id: str,
    message_id: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    raw = _build_reply_raw(
        to=to,
        subject=f"Re: {subject}" if not subject.startswith("Re:") else subject,
        body=body,
        thread_id=thread_id,
        in_reply_to=message_id,
        references=message_id,
        cc=cc,
        bcc=bcc,
        attachments=attachments,
    )
    return await send_message(access_token, raw, thread_id)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

async def create_label(access_token: str, name: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{GMAIL_BASE}/labels",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"name": name},
        )
        # 409 means label already exists – that's fine
        if resp.status_code not in (200, 201, 409):
            resp.raise_for_status()
        return resp.json() if resp.status_code != 409 else {"name": name}


async def add_label_to_message(
    access_token: str, message_id: str, label_ids: list[str]
) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{GMAIL_BASE}/messages/{message_id}/modify",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"addLabelIds": label_ids},
        )
        resp.raise_for_status()
        return resp.json()
