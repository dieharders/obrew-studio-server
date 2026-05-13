import re
from typing import Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Schedule or manage an online meeting by extracting the subject, description, attendees, timing, and action from user intent. Use this tool when the user wants to schedule, create, or look up a meeting.

    Fields by action:
    - 'create': subject, body, start_date_time, end_date_time or duration_minutes, attendees
    - 'get': meeting_id
    - 'find_availability': attendees, duration_minutes
    """

    subject: Optional[str] = Field(
        default="",
        description="Meeting title/subject. Generate an appropriate title based on the user's intent if not explicitly provided.",
    )
    body: Optional[str] = Field(
        default="",
        description="Meeting description or agenda in plain text. Generate a professional agenda based on the user's request and any provided context.",
    )
    start_date_time: Optional[str] = Field(
        default="",
        description="ISO 8601 datetime for the meeting start (e.g. '2025-01-15T14:00:00'). Leave empty if the user wants to find availability first.",
    )
    end_date_time: Optional[str] = Field(
        default="",
        description="ISO 8601 datetime for the meeting end (e.g. '2025-01-15T15:00:00'). Can be omitted if duration_minutes is provided.",
    )
    duration_minutes: Optional[int] = Field(
        default=30,
        description="Meeting duration in minutes. Common values: 15, 30, 45, 60, 90, 120. Default: 30.",
    )
    attendees: Optional[str] = Field(
        default="",
        description="Comma-separated email addresses of meeting attendees/guests. Extract from context or user's request.",
    )
    action: Optional[str] = Field(
        default="create",
        description="The meeting action. One of: 'create' (schedule a new meeting), 'get' (look up an existing meeting), 'find_availability' (find free time for all attendees). Default: 'create'.",
    )
    meeting_id: Optional[str] = Field(
        default="",
        description="Existing meeting ID for 'get' action. Leave empty for 'create' or 'find_availability'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "subject": "Q1 Planning Review",
                    "body": "Agenda:\n1. Review Q1 milestones\n2. Discuss blockers\n3. Plan next steps",
                    "start_date_time": "2025-01-15T14:00:00",
                    "end_date_time": "2025-01-15T15:00:00",
                    "duration_minutes": 60,
                    "attendees": "alice@example.com, bob@example.com",
                    "action": "create",
                    "meeting_id": "",
                },
                {
                    "subject": "Quick Sync",
                    "body": "Brief sync on project status.",
                    "start_date_time": "",
                    "end_date_time": "",
                    "duration_minutes": 30,
                    "attendees": "charlie@example.com",
                    "action": "find_availability",
                    "meeting_id": "",
                },
            ]
        }
    }


async def main(**kwargs) -> dict:
    """Return structured meeting data for widget rendering.

    All fields are optional - the widget UI will handle displaying
    placeholders for any missing data and guide the user through
    the multi-step meeting creation flow.
    """
    subject = kwargs.get("subject", "") or ""
    body = kwargs.get("body", "") or ""
    start_date_time = kwargs.get("start_date_time", "") or ""
    end_date_time = kwargs.get("end_date_time", "") or ""
    duration_minutes = kwargs.get("duration_minutes", 30)
    attendees_str = kwargs.get("attendees", "") or ""
    action = kwargs.get("action", "create") or "create"
    meeting_id = kwargs.get("meeting_id", "") or ""

    # Validate ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SS with optional timezone)
    warnings = []
    iso_pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?$"
    )
    if start_date_time and not iso_pattern.match(start_date_time):
        warnings.append(
            f"Invalid start_date_time '{start_date_time}' — expected ISO 8601 format (e.g. 2025-01-15T14:00:00). Value was cleared."
        )
        start_date_time = ""
    if end_date_time and not iso_pattern.match(end_date_time):
        warnings.append(
            f"Invalid end_date_time '{end_date_time}' — expected ISO 8601 format (e.g. 2025-01-15T15:00:00). Value was cleared."
        )
        end_date_time = ""

    # Parse comma-separated attendees into guest objects
    guests = []
    if attendees_str:
        for email in attendees_str.split(","):
            email = email.strip()
            if email:
                guests.append({"email": email, "name": ""})

    # Ensure duration is a valid integer
    try:
        duration_minutes = int(duration_minutes)
    except (ValueError, TypeError):
        duration_minutes = 30

    # Determine the initial wizard step based on available data
    if action == "find_availability":
        step = "availability"
    elif action == "get":
        step = "complete"
    elif guests:
        step = "guests" if not start_date_time else "confirm"
    else:
        step = "details"

    return {
        "step": step,
        "subject": subject,
        "body": body,
        "guests": guests,
        "availableSlots": [],
        "meetingDurationMinutes": duration_minutes,
        "startDateTime": start_date_time if start_date_time else None,
        "endDateTime": end_date_time if end_date_time else None,
        "meetingId": meeting_id if meeting_id else None,
        **({"warnings": warnings} if warnings else {}),
    }
