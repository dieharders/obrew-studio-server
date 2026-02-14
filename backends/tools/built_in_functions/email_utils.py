"""Shared utility functions for working with Microsoft Graph API email objects.

Used by email tool functions (email_scan, email_preview, email_read, email_grep)
and the EmailProvider search provider.
"""

import html as html_lib
import re
from typing import Dict, Any


def extract_sender(email: Dict[str, Any]) -> str:
    """Extract sender display string from a Graph API email object.

    Returns format: "Name <email@example.com>" or just the address/name.
    """
    from_data = email.get("from", {})
    if isinstance(from_data, dict):
        addr = from_data.get("emailAddress", {})
        name = addr.get("name", "")
        address = addr.get("address", "")
        if name:
            return f"{name} <{address}>" if address else name
        return address or "Unknown"
    return str(from_data) if from_data else "Unknown"


def extract_sender_short(email: Dict[str, Any]) -> str:
    """Extract short sender name from a Graph API email object.

    Returns just the display name or email address.
    """
    from_data = email.get("from", {})
    if isinstance(from_data, dict):
        addr = from_data.get("emailAddress", {})
        return addr.get("name", "") or addr.get("address", "") or "Unknown"
    return str(from_data) if from_data else "Unknown"


def extract_recipients(recipients: list) -> str:
    """Extract recipient display string from a Graph API recipients list.

    Handles toRecipients/ccRecipients format. Limits output to 5 recipients.
    """
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


def html_to_text(html_content: str) -> str:
    """Convert HTML to plain text.

    Handles:
    - Style/script tag removal
    - Link text extraction (<a href="url">text</a> -> text (url))
    - Table cell separation and row breaks
    - List item conversion
    - Block element conversion (br, p, div)
    - HTML entity decoding (named, decimal, and hex via html.unescape)
    - Whitespace normalization
    """
    if not html_content:
        return ""

    text = html_content

    # Remove style and script tags and their content
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Extract link text with URL: <a href="url">text</a> -> text (url)
    text = re.sub(
        r'<a\s[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r"\2 (\1)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Table cells: add tab separation between adjacent cells
    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", "\t", text, flags=re.IGNORECASE)
    # Table rows and list items: add newlines
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)

    # Replace br/p/div closing tags with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode all HTML entities (named, decimal, and hex)
    text = html_lib.unescape(text)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def extract_body_text(email: Dict[str, Any], max_length: int = 5000) -> str:
    """Extract full body as plain text from a Graph API email object.

    Converts HTML bodies to plain text. Falls back to bodyPreview if body is empty.
    """
    body = email.get("body", {})
    if isinstance(body, dict):
        content_type = body.get("contentType", "").lower()
        content = body.get("content", "")
        if content_type == "html" and content:
            text = html_to_text(content)
        else:
            text = content or ""
    else:
        text = str(body) if body else ""

    # Fall back to bodyPreview if body is empty
    if not text.strip():
        text = email.get("bodyPreview", "")

    return text[:max_length] if text else ""


def extract_field_value(email: Dict[str, Any], field: str) -> str:
    """Extract a searchable string value from a Graph API email field.

    Handles nested objects for 'from' and 'to' fields, and body objects
    with contentType/content structure.
    """
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
