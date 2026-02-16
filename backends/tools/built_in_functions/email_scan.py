"""Email scan tool - lists available emails from context items with metadata.

Requires request.state.context_items to be populated with email objects
by the frontend/middleware before tool invocation. If not set, returns
empty results.
"""

from typing import Optional
from pydantic import BaseModel, Field
from .email_utils import extract_sender_short


class Params(BaseModel):
    """List available emails showing metadata (subject, sender, date, importance). Use this to discover what emails are available before previewing or reading them."""

    max_results: int = Field(
        default=50,
        description="Maximum number of emails to return.",
    )
    sort_by: Optional[str] = Field(
        default="date",
        description="Field to sort by. Options: date, sender, subject. Defaults to date (most recent first).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "max_results": 20,
                    "sort_by": "date",
                }
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    List available emails from context items with metadata.
    Returns dict with: emails (list of metadata), total_available
    """
    max_results = kwargs.get("max_results", 50)
    sort_by = kwargs.get("sort_by", "date")

    # Read email items from request.state.context_items (injected by frontend)
    items = []
    request = kwargs.get("request")
    if (
        request
        and hasattr(request, "state")
        and hasattr(request.state, "context_items")
    ):
        items = request.state.context_items or []
    else:
        print(
            "[email_scan] Warning: no request.state.context_items available.",
            flush=True,
        )

    if not items:
        return {
            "emails": [],
            "total_available": 0,
        }

    # Extract metadata from each email
    email_list = []
    for idx, email in enumerate(items):
        email_id = email.get("id", f"email_{idx}")
        subject = email.get("subject", "(No Subject)")
        sender = extract_sender_short(email)
        date = email.get("receivedDateTime", "")
        importance = email.get("importance", "normal")
        has_attachments = email.get("hasAttachments", False)
        is_read = email.get("isRead", False)

        email_list.append(
            {
                "id": email_id,
                "subject": subject,
                "sender": sender,
                "date": date,
                "importance": importance,
                "has_attachments": has_attachments,
                "is_read": is_read,
            }
        )

    # Sort
    if sort_by == "date":
        email_list.sort(key=lambda e: e.get("date", ""), reverse=True)
    elif sort_by == "sender":
        email_list.sort(key=lambda e: e.get("sender", "").lower())
    elif sort_by == "subject":
        email_list.sort(key=lambda e: e.get("subject", "").lower())

    return {
        "emails": email_list[:max_results],
        "total_available": len(items),
    }
