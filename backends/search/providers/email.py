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
    DEFAULT_CONTENT_PREVIEW_LENGTH,
)
from backends.tools.built_in_functions.email_utils import (
    extract_sender,
    extract_sender_short,
    extract_body_text,
    extract_field_value,
)


_FIELD_ALIASES = {"title": "subject", "sender": "from", "recipient": "to"}

_DEFAULT_GREP_FIELDS = ["subject", "from", "to", "bodyPreview", "body"]


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
            sender = extract_sender_short(email)
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
                sender = extract_sender(email)
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
                sender = extract_sender(email)
                subject = email.get("subject", "(No Subject)")
                date = email.get("receivedDateTime", "")
                body_text = extract_body_text(email, DEFAULT_CONTENT_EXTRACT_LENGTH)

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

    @property
    def supports_grep(self) -> bool:
        return True

    @property
    def grep_fields(self) -> List[str]:
        return _DEFAULT_GREP_FIELDS

    async def grep(
        self, items: List[SearchItem], pattern: str, **kwargs
    ) -> Optional[List[SearchItem]]:
        """
        Filter discovered email items by text pattern matching.

        Searches across email fields (subject, from, to, bodyPreview, body)
        and returns only items whose underlying email data matches the pattern.
        Matched items get their preview enriched with match snippets.

        Args:
            items: Discovered SearchItem list from discover()
            pattern: Text pattern to search for (plain text, not regex)
            **kwargs: Optional 'search_fields' list and 'case_sensitive' bool

        Returns:
            Filtered list of SearchItems with enriched previews, or None on error
        """
        from core import common

        if not pattern or not items:
            return None

        # Resolve field aliases and defaults
        raw_fields = kwargs.get("search_fields") or _DEFAULT_GREP_FIELDS
        search_fields = [_FIELD_ALIASES.get(f, f) for f in raw_fields]
        case_sensitive = kwargs.get("case_sensitive", False)

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(re.escape(pattern), flags)

        matched_items: List[SearchItem] = []

        for item in items:
            idx = item.metadata.get("index") if item.metadata else None
            if idx is None or idx < 0 or idx >= len(self.emails):
                continue

            email = self.emails[idx]
            snippets: List[str] = []

            for field in search_fields:
                field_value = extract_field_value(email, field)
                if not field_value:
                    continue

                match = compiled.search(field_value)
                if match:
                    # Extract a snippet around the match
                    start = max(0, match.start() - 40)
                    end = min(len(field_value), match.end() + 40)
                    snippet = field_value[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(field_value):
                        snippet = snippet + "..."
                    snippets.append(f"{field}: {snippet}")

            if snippets:
                # Enrich preview with match context
                enriched_preview = " | ".join(snippets)
                if item.preview:
                    enriched_preview = f"{item.preview} | matches: {enriched_preview}"

                matched_item = item.model_copy(
                    update={
                        "preview": enriched_preview[
                            : DEFAULT_CONTENT_PREVIEW_LENGTH * 5
                        ]
                    }
                )
                matched_items.append(matched_item)

        print(
            f"{common.PRNT_API} [EmailProvider] Grep '{pattern}' matched {len(matched_items)}/{len(items)} emails",
            flush=True,
        )

        return matched_items if matched_items else None
