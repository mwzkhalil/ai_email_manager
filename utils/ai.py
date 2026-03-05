"""
AI provider helpers.
Implements the LLM fallback chain:
  OpenRouter → Groq → Gemini → OpenAI

Changes:
- Exponential backoff with jitter for 429 / 5xx errors per provider
- LLM concurrency semaphore to prevent thundering-herd on OpenRouter (8 req/min limit)
- Concurrent batch embeddings via asyncio semaphore (nomic-embed-text)
- Provider retry config is centralised and easy to tune
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx
from openai import AsyncOpenAI, RateLimitError

from config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Retry / backoff config
# ---------------------------------------------------------------------------

#: Maximum attempts per provider before moving to the next one
MAX_RETRIES_PER_PROVIDER = 3
#: Base delay (seconds) for exponential backoff
BACKOFF_BASE = 1.5
#: Cap on any single wait (seconds)
BACKOFF_MAX = 30.0

#: Max concurrent LLM requests — OpenRouter free tier caps at 8 req/min,
#: so limit to 4 to avoid thundering-herd when many emails arrive at once.
LLM_CONCURRENCY = 4
_llm_semaphore: asyncio.Semaphore | None = None

#: Max concurrent Ollama embedding requests — tune to your hardware
EMBEDDING_CONCURRENCY = 8
_embedding_semaphore: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Lazily create the LLM semaphore inside a running event loop."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    return _llm_semaphore


def _get_embedding_semaphore() -> asyncio.Semaphore:
    """Lazily create the embedding semaphore inside a running event loop."""
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
    return _embedding_semaphore


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with ±25 % jitter."""
    delay = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
    jitter = delay * 0.25 * random.random()
    return delay + jitter


def _is_retryable(exc: Exception) -> bool:
    """Return True for rate-limit or transient server errors."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


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
# Fallback chain — free OpenRouter models first, paid as last resort
#
# Free models ordered by token throughput (highest first).
# The chain auto-advances on 429 / rate-limit so you rarely hit paid APIs.
# ---------------------------------------------------------------------------

_FREE_OR_MODELS: list[tuple] = [
    # model-id                                                   throughput  ctx
    ("openrouter", "openai/gpt-oss-120b:free",                  _openrouter_client),  # 4.4B tok  131K
    ("openrouter", "arcee-ai/arcee-trinity-large-preview:free", _openrouter_client),  # 585B tok  128K
    ("openrouter", "stepfun/step-3-5-flash:free",               _openrouter_client),  # 548B tok  256K
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free",    _openrouter_client),  # 2.64B tok 128K
    ("openrouter", "openai/gpt-oss-20b:free",                   _openrouter_client),  # 1.35B tok 131K
    ("openrouter", "z-ai/glm-4-5-air:free",                     _openrouter_client),  #  62B tok  131K
    ("openrouter", "nvidia/llama-3.3-nemotron-super-49b-v1:free", _openrouter_client),
    ("openrouter", "arcee-ai/arcee-trinity-mini:free",          _openrouter_client),
]

_PAID_FALLBACKS: list[tuple] = [
    ("groq",   "llama-3.3-70b-versatile", _groq_client),
    ("openai", "gpt-4.1-mini",            _openai_client),
]

_ANALYSIS_PROVIDERS = [*_FREE_OR_MODELS, *_PAID_FALLBACKS]
_EOD_PROVIDERS      = [*_FREE_OR_MODELS, ("groq", "llama-3.3-70b-versatile", _groq_client), ("openai", "gpt-4o-mini", _openai_client)]
_CHAT_PROVIDERS     = [*_FREE_OR_MODELS, *_PAID_FALLBACKS]


async def _call_provider(
    name: str,
    model: str,
    client_factory: Any,
    system: str,
    user: str,
    max_tokens: int,
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


async def _try_provider_with_retry(
    name: str,
    model: str,
    factory: Any,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    """
    Attempt one provider up to MAX_RETRIES_PER_PROVIDER times.
    Retries only on rate-limit / transient errors; raises immediately on
    anything else so the fallback chain can move to the next provider.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_PROVIDER):
        try:
            log.info("Trying LLM provider: %s / %s (attempt %d)", name, model, attempt + 1)
            return await _call_provider(name, model, factory, system, user, max_tokens)
        except Exception as exc:
            if _is_retryable(exc):
                wait = _backoff_delay(attempt)
                log.warning(
                    "Provider %s hit rate-limit / transient error (attempt %d/%d). "
                    "Retrying in %.1fs. Error: %s",
                    name, attempt + 1, MAX_RETRIES_PER_PROVIDER, wait, exc,
                )
                last_exc = exc
                await asyncio.sleep(wait)
            else:
                # Non-retryable (auth error, bad request, etc.) — skip provider immediately
                log.warning("Provider %s failed with non-retryable error: %s", name, exc)
                raise

    raise RuntimeError(
        f"Provider {name} exhausted {MAX_RETRIES_PER_PROVIDER} retries. Last error: {last_exc}"
    )


async def _run_with_fallback(
    providers: list[tuple],
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> tuple[str, str]:
    """
    Try each provider in order with per-provider retry. Returns (raw_text, provider_name).
    Acquires the global LLM semaphore to cap concurrent requests and avoid
    thundering-herd 429s when many emails are processed simultaneously.
    """
    sem = _get_llm_semaphore()
    async with sem:
        last_error: Exception | None = None
        for name, model, factory in providers:
            try:
                result = await _try_provider_with_retry(name, model, factory, system, user, max_tokens)
                log.info("LLM provider succeeded: %s / %s", name, model)
                return result, name
            except Exception as exc:
                log.warning("Provider %s fully failed, trying next. Error: %s", name, exc)
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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _parse_json_robust(raw: str) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
        return json.loads(fixed)


async def analyze_email(
    from_name: str,
    from_email: str,
    subject: str,
    received_date: str,
    clean_body: str,
) -> tuple[dict, str]:
    """Run the email analysis fallback chain. Returns (parsed_dict, provider_name)."""
    user_prompt = build_analysis_prompt(from_name, from_email, subject, received_date, clean_body)
    raw, provider = await _run_with_fallback(_ANALYSIS_PROVIDERS, ANALYSIS_SYSTEM_PROMPT, user_prompt)
    raw_json = _extract_json(raw)
    data = _parse_json_robust(raw_json)

    # Normalise & validate
    data.setdefault("entities", {})
    for key in ("people", "organizations", "locations", "dates", "keyTopics"):
        data["entities"].setdefault(key, [])
    data.setdefault("suggestedReply", "")
    data.setdefault("reminderText", "")

    confidence = float(data.get("confidence", 0.95))
    if confidence < settings.ai_confidence_threshold:
        data["needsManualReview"] = True
    else:
        data.setdefault("needsManualReview", False)

    if data.get("needsCalendarEvent") and data.get("eventDetails"):
        attendees = data["eventDetails"].get("attendees") or [from_email]
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
# Embeddings — nomic-embed-text via Ollama, concurrent via semaphore
# ---------------------------------------------------------------------------

async def create_embedding(text: str) -> list[float]:
    """
    Generate a single vector embedding via Ollama nomic-embed-text.
    Respects the global EMBEDDING_CONCURRENCY semaphore so parallel calls
    don't overwhelm the local Ollama server.
    """
    sem = _get_embedding_semaphore()
    async with sem:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                settings.ollama_embedding_url,
                json={"model": settings.embedding_model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])


async def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts concurrently.
    All tasks run in parallel, throttled by EMBEDDING_CONCURRENCY.

    Usage:
        vectors = await create_embeddings_batch(["text one", "text two", ...])
    """
    if not texts:
        return []

    log.info("Creating embeddings for %d texts (concurrency=%d)", len(texts), EMBEDDING_CONCURRENCY)
    tasks = [create_embedding(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    embeddings: list[list[float]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.error("Embedding failed for text[%d]: %s", i, result)
            embeddings.append([])          # empty vector as sentinel; caller can filter
        else:
            embeddings.append(result)      # type: ignore[arg-type]

    return embeddings


def embedding_to_pg_vector(embedding: list[float]) -> str:
    """Format a float list as a PostgreSQL vector string '[x,y,z,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"