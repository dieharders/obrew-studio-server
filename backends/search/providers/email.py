"""
EmailProvider - Search provider for email data from Microsoft Graph API.

This provider implements the SearchProvider protocol for searching
over email data sent by frontend applications. The emails are fetched
by the frontend from the MS Graph API and passed in the request body.

The provider is email-aware: discover shows metadata (sender, subject, date),
preview shows the bodyPreview (~255 chars), and extract returns the full
body content (HTML converted to plain text).
"""

import re
from typing import List, Dict, Optional, Any

from ..harness import (
    SearchProvider,
    SearchItem,
    DEFAULT_CONTENT_EXTRACT_LENGTH,
)


def _extract_sender(email: Dict[str, Any]) -> str:
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


def _extract_sender_short(email: Dict[str, Any]) -> str:
    """Extract short sender name from a Graph API email object."""
    from_data = email.get("from", {})
    if isinstance(from_data, dict):
        addr = from_data.get("emailAddress", {})
        return addr.get("name", "") or addr.get("address", "") or "Unknown"
    return str(from_data) if from_data else "Unknown"


def _html_to_text(html: str) -> str:
    """Simple HTML to plain text conversion."""
    # Remove style and script tags and their content
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace br/p tags with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_body_text(email: Dict[str, Any], max_length: int = DEFAULT_CONTENT_EXTRACT_LENGTH) -> str:
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


class EmailProvider(SearchProvider):
    """
    Search provider for email data from Microsoft Graph API.

    Operates on email objects sent in the request body by the frontend.
    The data exists only for the duration of the request.

    Search phases:
    - discover: Shows email metadata (sender, subject, date) for LLM triage
    - preview: Shows bodyPreview content for LLM to assess relevance
    - extract: Returns full body content (HTML→text) for selected emails
    """

    def __init__(
        self,
        app: Any,
        emails: List[Dict[str, Any]],
    ):
        """
        Initialize the EmailProvider.

        Args:
            app: FastAPI application instance
            emails: List of raw email objects (Microsoft Graph API format)
        """
        self.app = app
        self.emails = emails
        self._search_items: List[SearchItem] = []

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover emails from the provided data.

        Returns SearchItems with email metadata only (sender, subject, date).
        The LLM uses this metadata to decide which emails to preview.

        Args:
            scope: Unused (all emails provided upfront)
            **kwargs: Must include 'query'

        Returns:
            List of SearchItem objects representing emails
        """
        from core import common

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for email search")

        # Return cached items if already discovered
        if self._search_items:
            return self._search_items

        search_items = []
        for idx, email in enumerate(self.emails):
            email_id = email.get("id", f"email_{idx}")
            subject = email.get("subject", "(No Subject)")
            sender = _extract_sender_short(email)
            date = email.get("receivedDateTime", "")
            importance = email.get("importance", "normal")
            has_attachments = email.get("hasAttachments", False)

            # Name: what the LLM sees in the selection list
            name = f"{sender} — {subject}"

            # Preview: brief metadata for initial triage
            meta_parts = []
            if date:
                # Show just the date portion
                date_short = date[:10] if len(date) >= 10 else date
                meta_parts.append(date_short)
            if importance and importance.lower() != "normal":
                meta_parts.append(f"importance: {importance}")
            if has_attachments:
                meta_parts.append("has attachments")
            preview = " | ".join(meta_parts) if meta_parts else ""

            search_item = SearchItem(
                id=email_id,
                name=name,
                type="email",
                preview=preview,
                metadata={
                    "index": idx,
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "importance": importance,
                    "has_attachments": has_attachments,
                },
                requires_extraction=True,
            )
            search_items.append(search_item)

        self._search_items = search_items

        print(
            f"{common.PRNT_API} [EmailProvider] Loaded {len(search_items)} email(s)",
            flush=True,
        )

        return search_items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content (bodyPreview) for the given emails.

        The LLM uses this preview to decide which emails to read fully.

        Args:
            items: List of email items to preview

        Returns:
            Items with preview field populated with bodyPreview content
        """
        for item in items:
            idx = item.metadata.get("index") if item.metadata else None
            if idx is not None and 0 <= idx < len(self.emails):
                email = self.emails[idx]
                body_preview = email.get("bodyPreview", "")
                sender = _extract_sender(email)
                subject = email.get("subject", "(No Subject)")
                date = email.get("receivedDateTime", "")

                # Rich preview with metadata + body preview
                preview_parts = [
                    f"From: {sender}",
                    f"Subject: {subject}",
                ]
                if date:
                    preview_parts.append(f"Date: {date}")
                if body_preview:
                    preview_parts.append(f"Preview: {body_preview}")

                item.preview = "\n".join(preview_parts)

        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full body content from the given emails.

        Converts HTML body to plain text and truncates to prevent
        oversized payloads.

        Args:
            items: List of email items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        context = []

        for item in items:
            idx = item.metadata.get("index") if item.metadata else None

            if idx is not None and 0 <= idx < len(self.emails):
                email = self.emails[idx]
                sender = _extract_sender(email)
                subject = email.get("subject", "(No Subject)")
                date = email.get("receivedDateTime", "")
                body_text = _extract_body_text(email)

                # Format as a structured email document
                content_parts = [
                    f"From: {sender}",
                    f"Subject: {subject}",
                    f"Date: {date}",
                    "",
                    body_text,
                ]

                context.append(
                    {
                        "source": f"[email] {sender} — {subject}",
                        "content": "\n".join(content_parts),
                    }
                )
            else:
                # Fallback to preview if index not found
                context.append(
                    {
                        "source": f"[email] {item.name}",
                        "content": item.preview or "",
                    }
                )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return additional scopes for expansion.

        Email data is provided upfront by the frontend, so there are
        no additional scopes to expand into.

        Returns:
            Empty list (no expansion)
        """
        return []
