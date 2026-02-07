import json
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Display calendar events, check availability, or propose new meetings. Use this tool when the user asks about their schedule, meetings, availability, or wants to create a calendar event."""

    date: Optional[str] = Field(
        default="",
        description="ISO date string for the day being viewed (e.g. '2025-01-15'). Defaults to today if not specified.",
    )
    events: Optional[str] = Field(
        default="[]",
        description="JSON array of event objects. Each event should have: title (string), start (ISO datetime), end (ISO datetime), and optionally: location, isAllDay (bool), organizer, attendees (array of emails), description, showAs ('free'|'tentative'|'busy'|'oof'|'unknown').",
    )
    action_mode: Optional[str] = Field(
        default="view",
        description="The calendar display mode. One of: 'view' (show events), 'availability' (show free/busy slots), 'propose' (suggest a new meeting). Default: 'view'.",
    )
    events_today: Optional[int] = Field(
        default=0,
        description="Number of events today. Used for the summary display.",
    )
    events_this_week: Optional[int] = Field(
        default=0,
        description="Number of events this week. Used for the summary display.",
    )
    next_available_slot: Optional[str] = Field(
        default="",
        description="ISO datetime of the next available free slot. Leave empty if unknown.",
    )
    available_slots: Optional[str] = Field(
        default="[]",
        description="JSON array of availability slot objects, each with 'start' and 'end' ISO datetime strings. Used when action_mode is 'availability'.",
    )
    proposed_title: Optional[str] = Field(
        default="",
        description="Title/subject of the proposed calendar event.",
    )
    proposed_start: Optional[str] = Field(
        default="",
        description="ISO datetime for the start of the proposed event.",
    )
    proposed_end: Optional[str] = Field(
        default="",
        description="ISO datetime for the end of the proposed event.",
    )
    proposed_location: Optional[str] = Field(
        default="",
        description="Location for the proposed event.",
    )
    proposed_description: Optional[str] = Field(
        default="",
        description="Description/body text for the proposed event.",
    )
    proposed_attendees: Optional[str] = Field(
        default="",
        description="Comma-separated email addresses of attendees for the proposed event.",
    )
    proposed_is_all_day: Optional[bool] = Field(
        default=False,
        description="Whether the proposed event is an all-day event.",
    )
    context: Optional[str] = Field(
        default="",
        description="Optional contextual text such as meeting notes, project background, or scheduling constraints.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": "2025-01-15",
                    "events": '[{"title": "Team Standup", "start": "2025-01-15T09:00:00", "end": "2025-01-15T09:30:00", "location": "Conference Room A", "showAs": "busy"}, {"title": "Lunch with Client", "start": "2025-01-15T12:00:00", "end": "2025-01-15T13:00:00", "location": "Downtown Cafe", "showAs": "busy"}]',
                    "action_mode": "view",
                    "events_today": 2,
                    "events_this_week": 8,
                    "next_available_slot": "2025-01-15T14:00:00",
                    "context": "",
                },
                {
                    "date": "2025-01-16",
                    "events": "[]",
                    "action_mode": "propose",
                    "proposed_title": "Project Review Meeting",
                    "proposed_start": "2025-01-16T14:00:00",
                    "proposed_end": "2025-01-16T15:00:00",
                    "proposed_location": "Virtual - Teams",
                    "proposed_description": "Review Q1 project milestones and discuss next steps.",
                    "proposed_attendees": "alice@example.com, bob@example.com",
                    "proposed_is_all_day": False,
                    "context": "Follow-up from last week's planning session.",
                },
            ]
        }
    }


async def main(**kwargs) -> dict:
    """Return structured calendar data for widget rendering.

    All fields are optional - the widget UI will handle displaying
    placeholders for any missing data.
    """
    date_str = kwargs.get("date", "") or ""
    events_json = kwargs.get("events", "[]") or "[]"
    action_mode = kwargs.get("action_mode", "view") or "view"
    events_today = kwargs.get("events_today", 0) or 0
    events_this_week = kwargs.get("events_this_week", 0) or 0
    next_available_slot = kwargs.get("next_available_slot", "") or ""
    available_slots_json = kwargs.get("available_slots", "[]") or "[]"
    proposed_title = kwargs.get("proposed_title", "") or ""
    proposed_start = kwargs.get("proposed_start", "") or ""
    proposed_end = kwargs.get("proposed_end", "") or ""
    proposed_location = kwargs.get("proposed_location", "") or ""
    proposed_description = kwargs.get("proposed_description", "") or ""
    proposed_attendees = kwargs.get("proposed_attendees", "") or ""
    proposed_is_all_day = kwargs.get("proposed_is_all_day", False) or False
    context_str = kwargs.get("context", "") or ""

    # Default date to today if not provided
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Parse events JSON string into list
    try:
        events_list = json.loads(events_json) if isinstance(events_json, str) else events_json
    except (json.JSONDecodeError, TypeError):
        events_list = []

    # Ensure each event has an id
    for idx, evt in enumerate(events_list):
        if isinstance(evt, dict) and "id" not in evt:
            evt["id"] = f"event-{idx}"

    # Parse available slots JSON string into list
    try:
        available_slots_list = json.loads(available_slots_json) if isinstance(available_slots_json, str) else available_slots_json
    except (json.JSONDecodeError, TypeError):
        available_slots_list = []

    # Build proposed event dict (only if we have at least a title or start)
    proposed_event = None
    if proposed_title or proposed_start:
        attendees_list = (
            [email.strip() for email in proposed_attendees.split(",") if email.strip()]
            if proposed_attendees
            else []
        )
        proposed_event = {
            "title": proposed_title,
            "start": proposed_start,
            "end": proposed_end,
            "location": proposed_location,
            "description": proposed_description,
            "attendees": attendees_list,
            "isAllDay": proposed_is_all_day,
        }

    return {
        "date": date_str,
        "events": events_list,
        "summary": {
            "eventsToday": int(events_today),
            "eventsThisWeek": int(events_this_week),
            "nextAvailableSlot": next_available_slot if next_available_slot else None,
        },
        "actionMode": action_mode,
        "availableSlots": available_slots_list if available_slots_list else [],
        "proposedEvent": proposed_event,
        "context": context_str if context_str else None,
    }
