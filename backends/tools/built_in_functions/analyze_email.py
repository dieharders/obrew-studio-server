"""Backend tool for structured email analysis with schema-constrained LLM output."""

from pydantic import BaseModel, Field


class Params(BaseModel):
    """Analyze an email and extract a structured insight including a concise summary, suggested reply drafts, and any calendar events implied by the email content."""

    summary: str = Field(
        ...,
        description="A concise summary of the email in 3 sentences or fewer. Capture the key intent, any action items, and relevant context.",
    )

    # ── Suggested Responses (3 fixed slots) ──────────────────────

    response_1_label: str = Field(
        default="",
        description="Short action label for the first suggested reply (e.g. 'Confirm attendance', 'Request details'). Leave empty if no reply is appropriate.",
    )
    response_1_body: str = Field(
        default="",
        description="Full draft reply text for the first suggested response. Professional and appropriate tone.",
    )
    response_1_tone: str = Field(
        default="formal",
        description="The tone of the first response.",
        options=["formal", "casual", "brief"],
    )

    response_2_label: str = Field(
        default="",
        description="Short action label for the second suggested reply. Leave empty if not needed.",
    )
    response_2_body: str = Field(
        default="",
        description="Full draft reply text for the second suggested response.",
    )
    response_2_tone: str = Field(
        default="formal",
        description="The tone of the second response.",
        options=["formal", "casual", "brief"],
    )

    response_3_label: str = Field(
        default="",
        description="Short action label for the third suggested reply. Leave empty if not needed.",
    )
    response_3_body: str = Field(
        default="",
        description="Full draft reply text for the third suggested response.",
    )
    response_3_tone: str = Field(
        default="formal",
        description="The tone of the third response.",
        options=["formal", "casual", "brief"],
    )

    # ── Suggested Events (2 fixed slots) ─────────────────────────

    event_1_title: str = Field(
        default="",
        description="Title for the first suggested calendar event. Leave empty if no events are implied by the email.",
    )
    event_1_description: str = Field(
        default="",
        description="Brief description of the first suggested event.",
    )
    event_1_suggested_start_date: str = Field(
        default="",
        description="ISO 8601 datetime string for the EARLIEST date the event could occur. For a range like 'next week', use the coming Monday. For 'sometime this week', use today. Leave empty if the email implies 'anytime' or 'as soon as possible'. IMPORTANT: this must differ from the target date when the email implies a range of days.",
    )
    event_1_suggested_target_date: str = Field(
        default="",
        description="ISO 8601 datetime string for the LATEST date the event should occur by. For a range like 'next week', use the Friday of that week. For a specific day like 'next Tuesday', use that Tuesday (same as start). Leave empty if unknown. IMPORTANT: for date ranges this MUST be later than the start date.",
    )
    event_1_duration_minutes: int = Field(
        default=30,
        description="Duration in minutes for the first event.",
    )
    event_1_attendees: str = Field(
        default="",
        description="Comma-separated email addresses of attendees for the first event.",
    )
    event_1_type: str = Field(
        default="meeting",
        description="The type of the first event.",
        options=["meeting", "follow-up", "recurring", "deadline"],
    )

    event_2_title: str = Field(
        default="",
        description="Title for the second suggested calendar event. Leave empty if not applicable.",
    )
    event_2_description: str = Field(
        default="",
        description="Brief description of the second suggested event.",
    )
    event_2_suggested_start_date: str = Field(
        default="",
        description="ISO 8601 datetime for the EARLIEST date the second event could occur. For date ranges, use the first possible day. Leave empty if the email implies 'anytime'.",
    )
    event_2_suggested_target_date: str = Field(
        default="",
        description="ISO 8601 datetime for the LATEST date the second event should occur by. For date ranges this MUST be later than the start date. Leave empty if unknown.",
    )
    event_2_duration_minutes: int = Field(
        default=30,
        description="Duration in minutes for the second event.",
    )
    event_2_attendees: str = Field(
        default="",
        description="Comma-separated attendee emails for the second event.",
    )
    event_2_type: str = Field(
        default="meeting",
        description="The type of the second event.",
        options=["meeting", "follow-up", "recurring", "deadline"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "John is requesting a meeting next Tuesday to discuss Q3 budget. He needs the latest projections spreadsheet beforehand. Action required: confirm availability and share the document.",
                    "response_1_label": "Confirm and share",
                    "response_1_body": "Hi John,\n\nTuesday works for me. I'll share the Q3 projections spreadsheet by end of day today.\n\nBest regards",
                    "response_1_tone": "formal",
                    "response_2_label": "Request reschedule",
                    "response_2_body": "Hi John,\n\nI'm unable to make Tuesday. Would Wednesday or Thursday work instead?\n\nBest",
                    "response_2_tone": "formal",
                    "response_3_label": "Acknowledge and defer",
                    "response_3_body": "Hi John,\n\nThanks for the heads up. Let me check my schedule and get back to you by tomorrow.\n\nBest",
                    "response_3_tone": "casual",
                    "event_1_title": "Q3 Budget Discussion",
                    "event_1_description": "Review Q3 budget projections with John",
                    "event_1_suggested_start_date": "2025-01-14T00:00:00Z",
                    "event_1_suggested_target_date": "2025-01-14T23:59:59Z",
                    "event_1_duration_minutes": 60,
                    "event_1_attendees": "john@example.com",
                    "event_1_type": "meeting",
                    "event_2_title": "Share Q3 Projections Spreadsheet",
                    "event_2_description": "Send the latest Q3 projections spreadsheet to John before the budget meeting",
                    "event_2_suggested_start_date": "2025-01-13T00:00:00Z",
                    "event_2_suggested_target_date": "2025-01-13T23:59:59Z",
                    "event_2_duration_minutes": 15,
                    "event_2_attendees": "john@example.com",
                    "event_2_type": "deadline",
                },
            ]
        }
    }


VALID_TONES = {"formal", "casual", "brief"}
VALID_EVENT_TYPES = {"meeting", "follow-up", "recurring", "deadline"}


async def main(**kwargs) -> dict:
    """
    Return the structured email analysis as JSON.

    The LLM fills in all Params fields via schema-constrained output.
    This function assembles them into the EmailInsightData-compatible shape
    expected by the frontend (summary, suggestedResponses, suggestedEvents).

    IMPORTANT: This tool should be called with tool_response_type="result"
    to return the raw JSON directly without LLM interpretation.
    """
    summary = kwargs.get("summary", "")

    # Assemble suggested responses (filter out empty slots)
    suggested_responses = []
    for i in range(1, 4):
        label = (kwargs.get(f"response_{i}_label") or "").strip()
        body = (kwargs.get(f"response_{i}_body") or "").strip()
        tone = (kwargs.get(f"response_{i}_tone") or "formal").strip()
        if label and body:
            if tone not in VALID_TONES:
                tone = "formal"
            suggested_responses.append(
                {
                    "label": label,
                    "body": body,
                    "tone": tone,
                }
            )

    # Assemble suggested events (filter out empty slots)
    suggested_events = []
    for i in range(1, 3):
        title = (kwargs.get(f"event_{i}_title") or "").strip()
        if title:
            attendees_str = (kwargs.get(f"event_{i}_attendees") or "").strip()
            attendees = (
                [a.strip() for a in attendees_str.split(",") if a.strip()]
                if attendees_str
                else []
            )
            event_type = (kwargs.get(f"event_{i}_type") or "meeting").strip()
            if event_type not in VALID_EVENT_TYPES:
                event_type = "meeting"
            suggested_events.append(
                {
                    "title": title,
                    "description": (kwargs.get(f"event_{i}_description") or "").strip(),
                    "suggestedStartDate": (
                        kwargs.get(f"event_{i}_suggested_start_date") or ""
                    ).strip(),
                    "suggestedDate": (
                        kwargs.get(f"event_{i}_suggested_target_date") or ""
                    ).strip(),
                    "durationMinutes": kwargs.get(f"event_{i}_duration_minutes", 30),
                    "attendees": attendees,
                    "type": event_type,
                }
            )

    result = {
        "summary": summary,
        "suggestedResponses": suggested_responses,
        "suggestedEvents": suggested_events,
    }

    return result
