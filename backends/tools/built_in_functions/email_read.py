"""Email read tool - returns full body content for selected emails."""

import re
from typing import List
from pydantic import BaseModel, Field


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


def _extract_sender(email: dict) -> str:
    """Extract sender display string from a Graph API email object."""
    from_data = email.get("from", {})
    if isinstance(from_data, dict):
        addr = from_data.get("emailAddress", {})
        name = addr.get("name", "")
        address = addr.get("address", "")
        if name:
            return f"{name} <{address}>" if address else name
        return address or "Unknown"
    return str(from_data) if from_data else "Unknown"


def _extract_recipients(recipients: list) -> str:
    """Extract recipient display string."""
    if not recipients or not isinstance(recipients, list):
        return ""
    parts = []
    for r in recipients[:5]:
        if isinstance(r, dict):
            addr = r.get("emailAddress", {})
            name = addr.get("name", "")
            address = addr.get("address", "")
            parts.append(f"{name} <{address}>" if name else address)
    result = ", ".join(parts)
    if len(recipients) > 5:
        result += f" (+{len(recipients) - 5} more)"
    return result


def _html_to_text(html: str) -> str:
    """Simple HTML to plain text conversion."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_body_text(email: dict, max_length: int) -> str:
    """Extract full body as plain text from a Graph API email object."""
    body = email.get("body", {})
    if isinstance(body, dict):
        content_type = body.get("contentType", "").lower()
        content = body.get("content", "")
        if content_type == "html" and content:
            text = _html_to_text(content)
        else:
            text = content or ""
    else:
        text = str(body) if body else ""

    # Fall back to bodyPreview if body is empty
    if not text.strip():
        text = email.get("bodyPreview", "")

    return text[:max_length] if text else ""


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

        body_text = _extract_body_text(email, max_length)

        emails.append({
            "id": eid,
            "subject": email.get("subject", "(No Subject)"),
            "sender": _extract_sender(email),
            "to": _extract_recipients(email.get("toRecipients", [])),
            "cc": _extract_recipients(email.get("ccRecipients", [])),
            "date": email.get("receivedDateTime", ""),
            "importance": email.get("importance", "normal"),
            "has_attachments": email.get("hasAttachments", False),
            "conversation_id": email.get("conversationId"),
            "body": body_text,
            "body_length": len(body_text),
            "truncated": len(body_text) >= max_length,
        })

    return {
        "emails": emails,
        "found": len(emails),
        "not_found": len(not_found_ids),
    }
