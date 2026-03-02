from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / common
# ---------------------------------------------------------------------------

class TokensBase(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    user_email: str


# ---------------------------------------------------------------------------
# Workflow 1 – Email Manager (main ingestion)
# ---------------------------------------------------------------------------

class EmailManagerRequest(TokensBase):
    auto_mode: bool = Field(False, alias="autoMode")
    ingest_mode: Optional[str] = Field(None, alias="ingest_mode")

    class Config:
        populate_by_name = True


class EventDetails(BaseModel):
    title: str = ""
    date: str = ""           # YYYY-MM-DD
    time: str = ""           # HH:MM
    duration: str = "60"
    location: str = ""
    description: str = ""
    attendees: list[str] = []


class Entities(BaseModel):
    people: list[str] = []
    organizations: list[str] = []
    locations: list[str] = []
    dates: list[str] = []
    key_topics: list[str] = Field([], alias="keyTopics")

    class Config:
        populate_by_name = True


class AIAnalysis(BaseModel):
    category: str = "INFORMATIONAL"
    intent: str = ""
    entities: Entities = Field(default_factory=Entities)
    needs_calendar_event: bool = Field(False, alias="needsCalendarEvent")
    event_details: Optional[EventDetails] = Field(None, alias="eventDetails")
    needs_reply: bool = Field(False, alias="needsReply")
    reply_tone: str = Field("professional", alias="replyTone")
    suggested_reply: str = Field("", alias="suggestedReply")
    needs_reminder: bool = Field(False, alias="needsReminder")
    reminder_date: Optional[str] = Field(None, alias="reminderDate")
    reminder_text: str = Field("", alias="reminderText")
    confidence: float = 0.95
    priority: str = "medium"
    needs_manual_review: bool = Field(False, alias="needsManualReview")

    class Config:
        populate_by_name = True


class ParsedEmail(BaseModel):
    message_id: str
    thread_id: str
    user_email: str
    from_email: str
    from_name: str
    subject: str
    received_date: str
    body_text: str = ""
    body_html: str = ""
    clean_body: str = ""
    tokens: dict[str, Any] = {}
    auto_mode: bool = False


# ---------------------------------------------------------------------------
# Workflow 2 – Bulk Ingestion (same as main but bulk flag)
# ---------------------------------------------------------------------------

class BulkIngestRequest(TokensBase):
    auto_mode: bool = Field(False, alias="autoMode")

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Workflow 3 – Get Emails
# ---------------------------------------------------------------------------

class GetEmailsRequest(BaseModel):
    user_email: str


class ActionItem(BaseModel):
    type: str
    title: str
    status: Optional[str] = None


class EmailAnalysisResponse(BaseModel):
    category: str
    intent: str
    priority: str
    confidence: float
    needs_reply: bool
    needs_calendar_event: bool
    needs_reminder: bool
    suggested_reply: Optional[str]
    reply_tone: Optional[str]
    event_details: Optional[dict]
    reminder_date: Optional[str]
    reminder_text: Optional[str]


class EmailRecord(BaseModel):
    success: bool = True
    email: dict[str, Any]
    action_items: list[ActionItem]
    analysis: EmailAnalysisResponse
    flags: dict[str, Any]
    metadata: dict[str, Any]


class GetEmailsResponse(BaseModel):
    success: bool
    count: int
    emails: list[EmailRecord]


# ---------------------------------------------------------------------------
# Workflow 4 – EOD Summary
# ---------------------------------------------------------------------------

class EODGenerateRequest(TokensBase):
    eod_id: Optional[int] = None


class EODShowRequest(BaseModel):
    user_email: str
    regenerate: bool = False


class EODEmailRequest(TokensBase):
    pass


class EODResponse(BaseModel):
    success: bool
    email_count: int
    summary_date: str
    markdown_summary: str


# ---------------------------------------------------------------------------
# Workflow 5 – Email Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_email: str
    query: str


class ChatResponse(BaseModel):
    success: bool
    type: str = "chat"
    answer: str
    llm: str


# ---------------------------------------------------------------------------
# Workflow 6 – Action Status
# ---------------------------------------------------------------------------

class ActionStatusRequest(BaseModel):
    action_status: str   # pending | completed | dismissed
    message_id: str
    user_email: str


# ---------------------------------------------------------------------------
# Manual Action Endpoints
# ---------------------------------------------------------------------------

class ManualCalendarRequest(BaseModel):
    message_id: str = Field(alias="messageId")
    access_token: str
    refresh_token: Optional[str] = None
    user_email: str
    event_details: EventDetails = Field(alias="eventDetails")

    class Config:
        populate_by_name = True


class ManualReplyRequest(BaseModel):
    message_id: str = Field(alias="messageId")
    access_token: str
    reply_text: str = Field(alias="replyText")
    cc: list[str] = []
    bcc: list[str] = []
    attachments: list[Any] = []

    class Config:
        populate_by_name = True


class ReminderDetails(BaseModel):
    date: str
    text: str
    priority: str = "medium"
    from_email: str = Field("", alias="fromEmail")
    subject: str = ""

    class Config:
        populate_by_name = True


class ManualReminderRequest(BaseModel):
    message_id: str = Field(alias="messageId")
    reminder_details: ReminderDetails = Field(alias="reminderDetails")

    class Config:
        populate_by_name = True
