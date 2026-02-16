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
from tools._email_utils import (
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

    Expansion:
    When auto_expand is enabled, emails are grouped by conversationId.
    The initial discover returns the largest conversation group, and
    expansion searches the remaining groups.
    """

    def __init__(
        self,
        app: Any,
        emails: List[Dict[str, Any]],
        auto_expand: bool = False,
    ):
        """
        Initialize the EmailProvider.

        Args:
            app: FastAPI application instance
            emails: List of raw email objects (Microsoft Graph API format)
            auto_expand: Whether to group by conversationId for expansion
        """
        self.app = app
        self.emails = emails
        self.auto_expand = auto_expand
        self._search_items: List[SearchItem] = []

        # Build conversation groups for auto_expand
        self._groups: Dict[str, List[int]] = {}  # conversationId → email indices
        self._searched_groups: List[str] = []
        if auto_expand:
            self._build_conversation_groups()

    def _build_conversation_groups(self):
        """Group emails by conversationId for scope expansion."""
        for idx, email in enumerate(self.emails):
            conv_id = email.get("conversationId", "_ungrouped")
            if not conv_id:
                conv_id = "_ungrouped"
            if conv_id not in self._groups:
                self._groups[conv_id] = []
            self._groups[conv_id].append(idx)

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover emails from the provided data.

        Returns SearchItems with email metadata and bodyPreview snippet.
        The LLM uses this to decide which emails to preview in detail.

        When auto_expand is enabled, discovers emails from the specified
        conversation group (scope). On the first call (scope=None), returns
        emails from the largest conversation group.

        Note: The query is validated but not used for filtering here because
        all emails are provided upfront by the frontend. Query-based filtering
        happens in Phase 1.5 (grep pre-filter) and Phase 2 (LLM selection).

        Args:
            scope: conversationId group to discover (when auto_expand enabled)
            **kwargs: Must include 'query'

        Returns:
            List of SearchItem objects representing emails
        """
        from core import common

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for email search")

        # Determine which email indices to process
        if self.auto_expand and self._groups:
            if scope is not None:
                # Search specific conversation group
                indices = self._groups.get(scope, [])
                emails_to_process = [(idx, self.emails[idx]) for idx in indices]
                if scope not in self._searched_groups:
                    self._searched_groups.append(scope)
            elif not self._searched_groups:
                # First call — pick the largest conversation group
                largest_group = max(self._groups, key=lambda k: len(self._groups[k]))
                indices = self._groups[largest_group]
                emails_to_process = [(idx, self.emails[idx]) for idx in indices]
                self._searched_groups.append(largest_group)
            else:
                # Already discovered, return cached
                return self._search_items
        else:
            # No grouping — process all emails (only once)
            if self._search_items:
                return self._search_items
            emails_to_process = list(enumerate(self.emails))

        search_items = []
        for idx, email in emails_to_process:
            email_id = email.get("id", f"email_{idx}")
            subject = email.get("subject", "(No Subject)")
            sender = extract_sender_short(email)
            date = email.get("receivedDateTime", "")
            importance = email.get("importance", "normal")
            has_attachments = email.get("hasAttachments", False)
            body_preview = email.get("bodyPreview", "")

            # Name: what the LLM sees in the selection list
            name = f"{sender} — {subject}"

            # Preview: metadata + bodyPreview snippet for initial triage
            meta_parts = []
            if date:
                # Show just the date portion
                date_short = date[:10] if len(date) >= 10 else date
                meta_parts.append(date_short)
            if importance and importance.lower() != "normal":
                meta_parts.append(f"importance: {importance}")
            if has_attachments:
                meta_parts.append("has attachments")
            if body_preview:
                # Include bodyPreview snippet to give LLM better triage context
                snippet = body_preview[:DEFAULT_CONTENT_PREVIEW_LENGTH]
                if len(body_preview) > DEFAULT_CONTENT_PREVIEW_LENGTH:
                    snippet += "..."
                meta_parts.append(snippet)
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

        # Cache items (append for grouped, replace for ungrouped)
        if self.auto_expand:
            self._search_items.extend(search_items)
        else:
            self._search_items = search_items

        scope_info = f" (conversation: {scope})" if scope else ""
        print(
            f"{common.PRNT_API} [EmailProvider] Loaded {len(search_items)} email(s){scope_info}",
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
        Return additional conversation groups that can be searched.

        When auto_expand is enabled, returns conversationId groups
        that haven't been searched yet.

        Returns:
            List of unsearched conversationId group keys
        """
        if not self.auto_expand:
            return []

        return [
            group
            for group in self._groups.keys()
            if group not in self._searched_groups
        ]

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
