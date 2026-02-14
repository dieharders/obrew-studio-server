"""Email preview tool - returns bodyPreview and metadata for selected emails."""

from typing import List
from pydantic import BaseModel, Field
from .email_utils import extract_sender, extract_recipients


class Params(BaseModel):
    """Get preview content (bodyPreview, sender, subject, date) for specific emails by ID. Use this after email_scan to see more detail before deciding which emails to read fully."""

    email_ids: List[str] = Field(
        ...,
        description="List of email IDs to preview.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email_ids": ["AAMkAGI2...", "AAMkAGI3..."],
                }
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Get preview content for specific emails by ID.
    Returns dict with: previews (list), found, not_found
    """
    email_ids = kwargs.get("email_ids", [])
    if not email_ids:
        raise ValueError("email_ids is required")

    # Read email items from request.state.context_items
    items = []
    request = kwargs.get("request")
    if (
        request
        and hasattr(request, "state")
        and hasattr(request.state, "context_items")
    ):
        items = request.state.context_items or []

    if not items:
        return {
            "previews": [],
            "found": 0,
            "not_found": len(email_ids),
        }

    # Build ID → email index
    id_to_email = {}
    for idx, email in enumerate(items):
        eid = email.get("id", f"email_{idx}")
        id_to_email[eid] = email

    # Look up requested emails
    previews = []
    not_found_ids = []

    for eid in email_ids:
        email = id_to_email.get(eid)
        if not email:
            not_found_ids.append(eid)
            continue

        previews.append(
            {
                "id": eid,
                "subject": email.get("subject", "(No Subject)"),
                "sender": extract_sender(email),
                "to": extract_recipients(email.get("toRecipients", [])),
                "cc": extract_recipients(email.get("ccRecipients", [])),
                "date": email.get("receivedDateTime", ""),
                "importance": email.get("importance", "normal"),
                "has_attachments": email.get("hasAttachments", False),
                "is_read": email.get("isRead", False),
                "body_preview": email.get("bodyPreview", ""),
                "conversation_id": email.get("conversationId"),
            }
        )

    return {
        "previews": previews,
        "found": len(previews),
        "not_found": len(not_found_ids),
    }
