"""Email grep tool for pattern-based search across email items."""
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Search for a pattern across email items by sender, subject, body content, or date. Use this to find relevant emails matching a keyword or pattern."""

    pattern: str = Field(
        ...,
        description="The search pattern (plain text or regular expression).",
    )
    search_fields: Optional[List[str]] = Field(
        default=None,
        description="Which email fields to search. Options: subject, from, to, bodyPreview, body. Defaults to subject, from, bodyPreview, body.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search should be case-sensitive.",
    )
    use_regex: bool = Field(
        default=False,
        description="Whether to interpret the pattern as a regular expression.",
    )
    max_results: int = Field(
        default=25,
        description="Maximum number of matching emails to return.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pattern": "quarterly report",
                    "search_fields": ["subject", "bodyPreview"],
                }
            ]
        }
    }


def _extract_field_value(email: Dict[str, Any], field: str) -> str:
    """Extract a searchable string value from an email field."""
    if field == "from":
        from_data = email.get("from", "")
        if isinstance(from_data, dict):
            addr = from_data.get("emailAddress", {})
            name = addr.get("name", "")
            address = addr.get("address", "")
            return f"{name} {address}".strip()
        return str(from_data)

    if field == "to":
        to_data = email.get("to", email.get("toRecipients", []))
        if isinstance(to_data, list):
            parts = []
            for recipient in to_data:
                if isinstance(recipient, dict):
                    addr = recipient.get("emailAddress", {})
                    name = addr.get("name", "")
                    address = addr.get("address", "")
                    parts.append(f"{name} {address}".strip())
                else:
                    parts.append(str(recipient))
            return " ".join(parts)
        return str(to_data)

    value = email.get(field, "")
    if isinstance(value, dict):
        # Handle body object with contentType/content
        return value.get("content", str(value))
    return str(value) if value else ""


async def main(**kwargs: Params) -> dict:
    """
    Search for a pattern across email items.
    Returns dict with: pattern, matches, emails_matched, items_searched
    """
    pattern_str = kwargs.get("pattern")

    # Read email items from request.state.context_items (injected by frontend)
    items = []
    request = kwargs.get("request")
    if request and hasattr(request, "state") and hasattr(request.state, "context_items"):
        items = request.state.context_items or []

    search_fields = kwargs.get("search_fields") or [
        "subject",
        "from",
        "bodyPreview",
        "body",
    ]
    case_sensitive = kwargs.get("case_sensitive", False)
    use_regex = kwargs.get("use_regex", False)
    max_results = kwargs.get("max_results", 25)

    if not pattern_str:
        raise ValueError("pattern is required")

    if not items:
        return {
            "pattern": pattern_str,
            "matches": [],
            "emails_matched": 0,
            "items_searched": 0,
        }

    # Compile the pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if use_regex:
        try:
            pattern = re.compile(pattern_str, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    else:
        pattern = re.compile(re.escape(pattern_str), flags)

    all_matches = []
    items_searched = 0

    for idx, email in enumerate(items):
        items_searched += 1
        if len(all_matches) >= max_results:
            break

        email_id = email.get("id") or f"email_{idx}"
        subject = email.get("subject") or "(No Subject)"
        field_matches = []

        for field in search_fields:
            field_value = _extract_field_value(email, field)
            if not field_value:
                continue

            match = pattern.search(field_value)
            if match:
                # Extract a snippet around the match
                start = max(0, match.start() - 40)
                end = min(len(field_value), match.end() + 40)
                snippet = field_value[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(field_value):
                    snippet = snippet + "..."

                field_matches.append(
                    {
                        "field": field,
                        "snippet": snippet,
                    }
                )

        if field_matches:
            all_matches.append(
                {
                    "id": email_id,
                    "subject": subject,
                    "from": _extract_field_value(email, "from"),
                    "date": email.get("receivedDateTime", ""),
                    "conversationId": email.get("conversationId"),
                    "matched_fields": field_matches,
                }
            )

    return {
        "pattern": pattern_str,
        "matches": all_matches,
        "emails_matched": len(all_matches),
        "items_searched": items_searched,
        "truncated": len(all_matches) >= max_results,
    }
