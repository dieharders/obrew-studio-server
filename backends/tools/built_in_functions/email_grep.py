"""Email grep tool for pattern-based search across email items."""

import re
from typing import List, Optional
from pydantic import BaseModel, Field
from .._email_utils import extract_field_value


class Params(BaseModel):
    """Search for a pattern across email items by sender, subject, body content, or date. Use this to find relevant emails matching a keyword or pattern."""

    pattern: str = Field(
        ...,
        description="The search pattern (plain text keyword or phrase).",
    )
    search_fields: Optional[List[str]] = Field(
        default=None,
        description="Which email fields to search the pattern in (not what to return). Options: subject, from, to, bodyPreview, body, conversationId. Defaults to subject, from, bodyPreview, body. Use 'subject' to search email titles/subjects.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search should be case-sensitive.",
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


# Note: kwargs type hint omitted because all built-in tools receive unpacked
# fields (not a Params instance). The Params schema is used for LLM
# JSON-schema generation only. See also: other tools follow this convention.
async def main(**kwargs) -> dict:
    """
    Search for a pattern across email items.
    Returns dict with: pattern, matches, emails_matched, items_searched
    """
    pattern_str = kwargs.get("pattern")

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
            "[email_grep] Warning: no request.state.context_items available, returning empty results.",
            flush=True,
        )

    # Normalize field aliases (e.g. "title" -> "subject")
    field_aliases = {"title": "subject", "sender": "from", "recipient": "to"}
    raw_fields = kwargs.get("search_fields") or [
        "subject",
        "from",
        "bodyPreview",
        "body",
    ]
    search_fields = [field_aliases.get(f, f) for f in raw_fields]
    case_sensitive = kwargs.get("case_sensitive", False)
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

    # @TODO We could include a `regex_pattern` for advanced search that an agent could pass from another search tool.
    # Compile the pattern (always escaped plain text)
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(pattern_str), flags)

    all_matches = []
    items_searched = 0
    truncated = False

    for idx, email in enumerate(items):
        if len(all_matches) >= max_results:
            truncated = True
            break
        items_searched += 1

        email_id = email.get("id") or f"email_{idx}"
        subject = email.get("subject") or "(No Subject)"
        field_matches = []

        for field in search_fields:
            field_value = extract_field_value(email, field)
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
                    "from": extract_field_value(email, "from"),
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
        "truncated": truncated,
    }
