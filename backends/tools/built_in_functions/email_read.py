"""Email read tool - returns full body content for selected emails.

Requires request.state.context_items to be populated with email objects
by the frontend/middleware before tool invocation. If not set, returns
empty results.
"""

from typing import List
from pydantic import BaseModel, Field
from .email_utils import extract_sender, extract_recipients, extract_body_text


class Params(BaseModel):
    """Read the full body content of specific emails by ID. Use this after email_preview to get the complete email text for emails you've identified as relevant."""

    email_ids: List[str] = Field(
        ...,
        description="List of email IDs to read fully.",
    )
    max_length: int = Field(
        default=5000,
        description="Maximum characters of body content per email.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email_ids": ["AAMkAGI2..."],
                    "max_length": 5000,
                }
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Read full body content for specific emails by ID.
    Returns dict with: emails (list with full content), found, not_found
    """
    email_ids = kwargs.get("email_ids", [])
    max_length = kwargs.get("max_length", 5000)

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
            "emails": [],
            "found": 0,
            "not_found": len(email_ids),
        }

    # Build ID → email index
    id_to_email = {}
    for idx, email in enumerate(items):
        eid = email.get("id", f"email_{idx}")
        id_to_email[eid] = email

    # Look up and extract full content
    emails = []
    not_found_ids = []

    for eid in email_ids:
        email = id_to_email.get(eid)
        if not email:
            not_found_ids.append(eid)
            continue

        body_text = extract_body_text(email, max_length)

        emails.append(
            {
                "id": eid,
                "subject": email.get("subject", "(No Subject)"),
                "sender": extract_sender(email),
                "to": extract_recipients(email.get("toRecipients", [])),
                "cc": extract_recipients(email.get("ccRecipients", [])),
                "date": email.get("receivedDateTime", ""),
                "importance": email.get("importance", "normal"),
                "has_attachments": email.get("hasAttachments", False),
                "conversation_id": email.get("conversationId"),
                "body": body_text,
                "body_length": len(body_text),
                "truncated": len(body_text) >= max_length,
            }
        )

    return {
        "emails": emails,
        "found": len(emails),
        "not_found": len(not_found_ids),
    }
