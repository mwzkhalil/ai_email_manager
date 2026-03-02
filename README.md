# Email Manager API

An implementation of the Email Manager system.

## Architecture

```
email_manager/
├── main.py                    # FastAPI app + router registration
├── config.py                  # Settings via pydantic-settings + .env
├── requirements.txt
├── .env.example
├── models/
│   └── schemas.py             # All Pydantic request/response models
├── db/
│   └── database.py            # Async SQLAlchemy engine, session, DDL
├── utils/
│   ├── ai.py                  # LLM fallback chain + embeddings
│   ├── gmail.py               # Gmail API (fetch, parse, send, label)
│   ├── calendar.py            # Google Calendar API
│   └── markdown.py            # Markdown → HTML for EOD emails
└── routers/
    ├── ingestion.py           # Workflows 1 & 2 (main + bulk ingestion)
    ├── get_emails.py          # Workflow 3 (get emails for frontend)
    ├── eod_summary.py         # Workflow 4 (EOD generate/show/send)
    ├── email_chat.py          # Workflow 5 (semantic chat)
    └── actions.py             # Workflow 6 + manual endpoints
```

## Endpoints

| Method | Path | Workflow | Description |
|--------|------|----------|-------------|
| POST | `/email-manager` | 1 | Fetch + analyse + store Gmail emails |
| POST | `/ingest_email_bulk` | 2 | Bulk ingest all inbox emails |
| POST | `/get-emails` | 3 | Retrieve stored emails for frontend |
| POST | `/eod-summary-generate` | 4a | Generate AI end-of-day summary |
| POST | `/show-eod` | 4b | Show or regenerate today's EOD |
| POST | `/send-eod-email` | 4c | Email the EOD summary to user |
| POST | `/chat-email` | 5 | Semantic search + AI chat over emails |
| POST | `/action_status` | 6 | Update action status on email record |
| POST | `/manual-create-calendar` | — | Manually create calendar event |
| POST | `/manual-send-reply` | — | Manually send email reply |
| POST | `/manual-set-reminder` | — | Manually set reminder |
| GET | `/health` | — | Health check |
| GET | `/docs` | — | Swagger UI |

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 14+ with [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.ai) running `nomic-embed-text` for embeddings

### 2. Install dependencies

```bash
cd email_manager
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required values in `.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/email_manager
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

### 4. Create the database

```sql
CREATE DATABASE email_manager;
\c email_manager
CREATE EXTENSION IF NOT EXISTS vector;
```

Tables are created automatically on startup.

### 5. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Usage Examples

### Trigger email ingestion

```bash
curl -X POST http://localhost:8000/email-manager \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "ya29...",
    "refresh_token": "1//...",
    "user_email": "user@gmail.com",
    "autoMode": true
  }'
```

### Chat with your emails

```bash
curl -X POST http://localhost:8000/chat-email \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "user@gmail.com",
    "query": "What did John say about the Q4 budget?"
  }'
```

### Get today's EOD summary

```bash
curl -X POST http://localhost:8000/show-eod \
  -H "Content-Type: application/json" \
  -d '{"user_email": "user@gmail.com", "regenerate": false}'
```

---

## AI / LLM Fallback Chain

Each workflow uses a cascading fallback across providers:

| Order | Provider | Model |
|-------|----------|-------|
| 1 | OpenRouter | `openai/gpt-4o-free` |
| 2 | Groq | `llama-3.3-70b-versatile` |
| 3 | OpenAI | `gpt-4.1-mini` / `gpt-4o-mini` |

If all providers fail, a `500` error is returned.

---

## Database Tables

- **`email_store`** — Processed emails + AI analysis + vector embeddings
- **`reminders`** — Reminder records with dates and status
- **`eod_summaries`** — Daily AI-generated summaries per user

---

## Auto-Mode Actions

When `autoMode: true` is passed, the system automatically:

- **Creates Google Calendar events** for emails with meeting requests (checks freeBusy first)
- **Sends AI-drafted replies** for emails requiring a response
- **Stores reminders** for time-sensitive emails
- **Flags for manual review** when AI confidence < 0.7
