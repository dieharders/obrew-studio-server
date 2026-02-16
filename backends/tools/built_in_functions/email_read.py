"""Email read tool - returns full body content for selected emails.

Requires request.state.context_items to be populated with email objects
by the frontend/middleware before tool invocation. If not set, returns
empty results.
"""

from typing import List
from pydantic import BaseModel, Field
from .._email_utils import extract_sender, extract_recipients, extract_body_text, get_context_emails, build_email_id_index


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

    items = get_context_emails(kwargs)
    if not items:
        return {
            "emails": [],
            "found": 0,
            "not_found": len(email_ids),
        }

    id_to_email = build_email_id_index(items)

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
