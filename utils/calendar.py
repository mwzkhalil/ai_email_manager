"""
Google Calendar API helpers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import httpx
import pytz

from config import get_settings

settings = get_settings()
CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"


def _auth_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def build_event_datetimes(
    date_str: str,       # YYYY-MM-DD
    time_str: str,       # HH:MM
    duration_minutes: int = 60,
    timezone: str | None = None,
) -> tuple[str, str]:
    """Return ISO 8601 start/end strings."""
    tz_name = timezone or settings.default_timezone
    tz = pytz.timezone(tz_name)

    # Parse date + time
    if time_str:
        dt_str = f"{date_str}T{time_str}:00"
        fmt = "%Y-%m-%dT%H:%M:%S"
    else:
        dt_str = f"{date_str}T09:00:00"
        fmt = "%Y-%m-%dT%H:%M:%S"

    naive = datetime.strptime(dt_str, fmt)
    local_dt = tz.localize(naive)
    end_dt = local_dt + timedelta(minutes=duration_minutes)

    return local_dt.isoformat(), end_dt.isoformat()


async def check_free_busy(
    access_token: str,
    user_email: str,
    start: str,
    end: str,
) -> bool:
    """Returns True if the slot is available (no busy blocks)."""
    payload = {
        "timeMin": start,
        "timeMax": end,
        "items": [{"id": "primary"}],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{CALENDAR_BASE}/freeBusy",
            headers=_auth_headers(access_token),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    busy_slots = data.get("calendars", {}).get("primary", {}).get("busy", [])
    return len(busy_slots) == 0


async def create_event(
    access_token: str,
    user_email: str,
    title: str,
    start: str,
    end: str,
    description: str = "",
    location: str = "",
    attendees: list[str] | None = None,
    timezone: str | None = None,
) -> dict:
    """Create a Google Calendar event and return the API response."""
    tz_name = timezone or settings.default_timezone
    event_body: dict[str, Any] = {
        "summary": title,
        "description": description,
        "location": location,
        "start": {"dateTime": start, "timeZone": tz_name},
        "end":   {"dateTime": end,   "timeZone": tz_name},
    }
    if attendees:
        event_body["attendees"] = [{"email": a} for a in attendees]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{CALENDAR_BASE}/calendars/{user_email}/events",
            headers=_auth_headers(access_token),
            json=event_body,
        )
        resp.raise_for_status()
        return resp.json()
