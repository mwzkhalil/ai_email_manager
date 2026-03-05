"""
AI provider helpers.
Implements the LLM fallback chain:
  OpenRouter → Groq → Gemini → OpenAI
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are an intelligent email analysis assistant.
Analyze the provided email and return ONLY valid JSON with this exact structure:
{
  "category": "CALENDAR_EVENT,REQUIRES_REPLY",
  "intent": "string",
  "entities": {
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "keyTopics": []
  },
  "needsCalendarEvent": true,
  "eventDetails": {
    "title": "string",
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "duration": "60",
    "location": "string",
    "description": "string"
  },
  "needsReply": true,
  "replyTone": "professional|casual|formal",
  "suggestedReply": "string",
  "needsReminder": true,
  "reminderDate": "YYYY-MM-DD",
  "reminderText": "string",
  "confidence": 0.95,
  "priority": "high|medium|low"
}

Categories (multi-value, comma-separated): CALENDAR_EVENT, REQUIRES_REPLY, REMINDER, INFORMATIONAL
Return ONLY valid JSON. No markdown, no explanation."""

EOD_SYSTEM_PROMPT = """You are a professional executive assistant.
Generate a concise, well-structured end-of-day email summary in markdown format.
Include: executive overview, high-priority items, action items, calendar events, reminders.
Return your response in the JSON format: {"response": "markdown content here"}"""

CHAT_SYSTEM_PROMPT = """You are an intelligent email assistant with access to the user's emails.
Answer questions based ONLY on the provided email context.
Do not use any outside information. If the answer isn't in the emails, say so clearly."""


def build_analysis_prompt(
    from_name: str,
    from_email: str,
    subject: str,
    received_date: str,
    clean_body: str,
) -> str:
    return (
        f"From: {from_name} <{from_email}>\n"
        f"Subject: {subject}\n"
        f"Date: {received_date}\n\n"
        f"{clean_body}"
    )


def build_eod_prompt(
    date: str,
    stats: dict,
    high_priority: list[dict],
    all_emails: list[dict],
) -> str:
    email_lines = "\n".join(
        f"- [{e.get('priority','?').upper()}] {e.get('subject','(no subject)')} "
        f"from {e.get('from_name','')} | {e.get('category','')} | intent: {e.get('intent','')}"
        for e in all_emails
    )
    hp_lines = "\n".join(
        f"- {e.get('subject','(no subject)')} from {e.get('from_name','')} – {e.get('intent','')}"
        for e in high_priority
    )
    return (
        f"Date: {date}\n"
        f"Total emails: {stats.get('total', 0)}\n"
        f"High priority: {stats.get('high', 0)} | Medium: {stats.get('medium', 0)} | Low: {stats.get('low', 0)}\n"
        f"Need reply: {stats.get('needs_reply', 0)} | Calendar: {stats.get('needs_calendar', 0)} | Reminder: {stats.get('needs_reminder', 0)}\n\n"
        f"HIGH PRIORITY EMAILS:\n{hp_lines}\n\n"
        f"ALL EMAILS:\n{email_lines}"
    )


def build_chat_prompt(context: str, query: str) -> str:
    return f"Email context:\n{context}\n\nUser question: {query}"


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _openrouter_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def _openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def _call_gemini(system: str, user: str, max_tokens: int = 1500) -> str:
    """Call Gemini via REST since there's no async SDK parity needed."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={settings.gemini_api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

_ANALYSIS_PROVIDERS = [
    ("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free", lambda: _openrouter_client()),
    ("groq",       "llama-3.3-70b-versatile", lambda: _groq_client()),
    ("openai",     "gpt-4.1-mini",            lambda: _openai_client()),
]

_EOD_PROVIDERS = [
    ("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free", lambda: _openrouter_client()),
    ("groq",       "llama-3.3-70b-versatile", lambda: _groq_client()),
    ("openai",     "gpt-4o-mini",             lambda: _openai_client()),
]

_CHAT_PROVIDERS = [
    ("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free", lambda: _openrouter_client()),
    ("groq",       "llama-3.3-70b-versatile",            lambda: _groq_client()),
    ("openai",     "gpt-4.1-mini",                       lambda: _openai_client()),
]


async def _call_provider(
    name: str, model: str, client_factory: Any, system: str, user: str, max_tokens: int
) -> str:
    if name == "gemini":
        return await _call_gemini(system, user, max_tokens)
    client = client_factory()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def _run_with_fallback(
    providers: list[tuple], system: str, user: str, max_tokens: int = 1500
) -> tuple[str, str]:
    """Try each provider in order. Returns (raw_text, provider_name)."""
    last_error: Exception | None = None
    for name, model, factory in providers:
        try:
            log.info("Trying LLM provider: %s / %s", name, model)
            text = await _call_provider(name, model, factory, system, user, max_tokens)
            return text, name
        except Exception as exc:
            log.warning("Provider %s failed: %s", name, exc)
            last_error = exc
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> str:
    """Pull the first {...} block from text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _parse_json_robust(raw: str) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to escape unescaped newlines inside string values
        fixed = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
        return json.loads(fixed)


async def analyze_email(
    from_name: str,
    from_email: str,
    subject: str,
    received_date: str,
    clean_body: str,
) -> tuple[dict, str]:
    """
    Run the email analysis fallback chain.
    Returns (parsed_analysis_dict, provider_name).
    """
    user_prompt = build_analysis_prompt(from_name, from_email, subject, received_date, clean_body)
    raw, provider = await _run_with_fallback(_ANALYSIS_PROVIDERS, ANALYSIS_SYSTEM_PROMPT, user_prompt)
    raw_json = _extract_json(raw)
    data = _parse_json_robust(raw_json)

    # Normalise & validate
    data.setdefault("entities", {})
    data["entities"].setdefault("people", [])
    data["entities"].setdefault("organizations", [])
    data["entities"].setdefault("locations", [])
    data["entities"].setdefault("dates", [])
    data["entities"].setdefault("keyTopics", [])
    data.setdefault("suggestedReply", "")
    data.setdefault("reminderText", "")

    confidence = float(data.get("confidence", 0.95))
    from config import get_settings as _gs
    if confidence < _gs().ai_confidence_threshold:
        data["needsManualReview"] = True
    else:
        data.setdefault("needsManualReview", False)

    # Auto-add sender to attendees when calendar event needed
    if data.get("needsCalendarEvent") and data.get("eventDetails"):
        attendees = data["eventDetails"].get("attendees", [])
        if not attendees:
            attendees = [from_email]
        data["eventDetails"]["attendees"] = attendees

    return data, provider


async def generate_eod_summary(
    date: str,
    stats: dict,
    high_priority: list[dict],
    all_emails: list[dict],
) -> tuple[str, str]:
    """Returns (markdown_summary, provider_name)."""
    user_prompt = build_eod_prompt(date, stats, high_priority, all_emails)
    raw, provider = await _run_with_fallback(_EOD_PROVIDERS, EOD_SYSTEM_PROMPT, user_prompt, max_tokens=2000)

    # Try to unwrap {"response": "..."} wrapper
    try:
        cleaned = _strip_fences(raw)
        wrapped = json.loads(cleaned)
        if isinstance(wrapped, dict) and "response" in wrapped:
            return wrapped["response"], provider
    except Exception:
        pass

    return raw.strip(), provider


async def chat_with_emails(context: str, query: str) -> tuple[str, str]:
    """Returns (answer, provider_name)."""
    user_prompt = build_chat_prompt(context, query)
    raw, provider = await _run_with_fallback(_CHAT_PROVIDERS, CHAT_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    return raw.strip(), provider


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

async def create_embedding(text: str) -> list[float]:
    """Call Ollama nomic-embed-text to generate a vector embedding."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            settings.ollama_embedding_url,
            json={"model": settings.embedding_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])


def embedding_to_pg_vector(embedding: list[float]) -> str:
    """Format a float list as a PostgreSQL vector string '[x,y,z,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"
