"""Email preview tool - returns bodyPreview and metadata for selected emails.

Requires request.state.context_items to be populated with email objects
by the frontend/middleware before tool invocation. If not set, returns
empty results.
"""

from typing import List
from pydantic import BaseModel, Field
from .._email_utils import extract_sender, extract_recipients, get_context_emails, build_email_id_index


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

    items = get_context_emails(kwargs)
    if not items:
        return {
            "previews": [],
            "found": 0,
            "not_found": len(email_ids),
        }

    id_to_email = build_email_id_index(items)

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
