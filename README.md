# 📧 AI Email Manager

An email management backend built with FastAPI, Gmail API, and LLM providers. Automatically ingests, analyses, and acts on emails with calendar creation, smart replies, reminders, EOD summaries, and semantic chat.

---

## Features

- **Smart Ingestion** — fetches emails from Gmail, deduplicates, and runs AI analysis on each
- **LLM Fallback Chain** — OpenRouter → Groq → OpenAI with exponential backoff and per-provider retry on 429s
- **Auto Actions** — creates Google Calendar events, sends replies, stores reminders
- **EOD Summaries** — generates end-of-day markdown summaries via LLM
- **Semantic Chat** — query your emails in natural language
- **Vector Embeddings** — nomic-embed-text via Ollama, with concurrent batch processing

---

## Tech Stack

- **FastAPI** + **asyncpg** / SQLAlchemy async
- **PostgreSQL** (with pgvector for embeddings)
- **Gmail API** + **Google Calendar API**
- **OpenRouter / Groq / OpenAI** (LLM fallback chain)
- **Ollama** (local embeddings via nomic-embed-text)

---

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Ollama running locally with `nomic-embed-text` pulled

```bash
ollama pull nomic-embed-text
```

### 2. Install pgvector

```bash
sudo apt install postgresql-15-pgvector -y
```

> If pgvector is unavailable, the schema falls back to `FLOAT[]` — embeddings still work, you just lose vector similarity search.

### 3. Clone & install dependencies

```bash
cd ai_email_manager
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DATABASE_URL=postgresql+asyncpg://mailuser:mailpass@localhost:5432/ai_email_manager

OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

GROQ_API_KEY=your_key
GROQ_BASE_URL=https://api.groq.com/openai/v1

OPENAI_API_KEY=your_key

GEMINI_API_KEY=your_key

OLLAMA_EMBEDDING_URL=http://localhost:11434/api/embeddings
EMBEDDING_MODEL=nomic-embed-text

DEFAULT_TIMEZONE=UTC
AI_CONFIDENCE_THRESHOLD=0.7
BULK_THRESHOLD=50
EMAIL_FETCH_LIMIT=20
```

### 5. Create the database

```bash
sudo -u postgres psql -c "CREATE USER mailuser WITH PASSWORD 'mailpass';"
sudo -u postgres psql -c "CREATE DATABASE ai_email_manager OWNER mailuser;"
python setup_db.py
```

### 6. Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/email-manager` | Main ingestion — fetch, analyse, and act on emails |
| POST | `/ingest_email_bulk` | Bulk ingest up to 500 emails |
| POST | `/get-emails` | Retrieve stored emails for a user |
| POST | `/eod-summary-generate` | Generate end-of-day summary |
| POST | `/show-eod` | Show or regenerate today's EOD summary |
| POST | `/send-eod-email` | Email the EOD summary to the user |
| POST | `/chat-email` | Chat with your emails using natural language |
| POST | `/action_status` | Update action status on an email record |
| POST | `/manual-create-calendar` | Manually create a calendar event |
| POST | `/manual-send-reply` | Manually send a reply |
| POST | `/manual-set-reminder` | Manually set a reminder |
| GET  | `/health` | Health check |
| GET  | `/docs` | Swagger UI |

---

## Request Examples

### Ingest emails
```bash
curl -X POST http://localhost:8000/email-manager \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "YOUR_GMAIL_TOKEN",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "user_email": "you@gmail.com",
    "autoMode": false
  }'
```

### Show EOD summary
```bash
curl -X POST http://localhost:8000/show-eod \
  -H "Content-Type: application/json" \
  -d '{"user_email": "you@gmail.com"}'
```

### Chat with emails
```bash
curl -X POST http://localhost:8000/chat-email \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "you@gmail.com",
    "query": "Any urgent emails I need to reply to?"
  }'
```

---

## Architecture

```
Gmail API
   │
   ▼
Ingestion Router
   ├── Dedup check (PostgreSQL)
   ├── AI Analysis (OpenRouter → Groq → OpenAI fallback)
   ├── Save to DB
   ├── Generate embedding (Ollama nomic-embed-text)
   └── Auto Actions (Calendar / Reply / Reminder)

EOD Summary Router
   ├── Aggregate today's emails
   ├── LLM summary generation
   └── Optional email delivery

Chat Router
   └── Semantic search + LLM answer
```

---

## LLM Fallback & Rate Limiting

The system uses a **3-provider fallback chain** with exponential backoff:

- Each provider retries up to **3 times** on 429 / 5xx errors
- Backoff starts at 1.5s and caps at 30s with ±25% jitter
- A **concurrency semaphore** (default: 4) prevents thundering-herd when processing many emails simultaneously
- If all providers fail, the request raises a clear error

To increase throughput on a paid plan, raise `LLM_CONCURRENCY` in `utils/ai.py`.

---
