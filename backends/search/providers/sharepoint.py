"""
SharePointProvider - Search provider for SharePoint file data.

This provider implements the SearchProvider protocol for searching
over SharePoint file data sent by frontend applications. The files are
fetched by the frontend from the MS Graph API and passed in the request body.

The provider is file-aware: discover shows file metadata (name, author, date),
preview shows content snippets, and extract returns the full text content.
"""

import re
from core import common
from typing import List, Dict, Optional, Any
from ..harness import (
    SearchProvider,
    SearchItem,
    DEFAULT_CONTENT_EXTRACT_LENGTH,
)

# Intermediate preview length between discover snippet and full extract.
# Discover shows ~100 chars (metadata triage), preview shows ~800 chars
# (relevance assessment), extract shows up to 5000 chars (full content).
SHAREPOINT_PREVIEW_LENGTH = 800


class SharePointProvider(SearchProvider):
    """
    Search provider for SharePoint file data from Microsoft Graph API.

    Operates on file objects sent in the request body by the frontend.
    The data exists only for the duration of the request.

    Search phases:
    - discover: Shows file metadata (name, author, date, type) for LLM triage
    - preview: Shows content snippet for LLM to assess relevance
    - extract: Returns full text content for selected files
    """

    def __init__(
        self,
        app: Any,
        items: List[Dict[str, Any]],
    ):
        """
        Initialize the SharePointProvider.

        Args:
            app: FastAPI application instance
            items: List of SharePoint file objects with content
        """
        self.app = app
        self.items = items
        self._search_items: List[SearchItem] = []

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover files from the provided SharePoint data.

        Returns SearchItems with file metadata and content snippet.
        The LLM uses this to decide which files to preview in detail.

        Args:
            scope: Not used (no expansion for SharePoint)
            **kwargs: Must include 'query'

        Returns:
            List of SearchItem objects representing SharePoint files
        """
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for SharePoint search")

        # Only discover once
        if self._search_items:
            return self._search_items

        search_items = []
        for idx, item in enumerate(self.items):
            item_id = item.get("id", f"sp_file_{idx}")
            name = item.get("name", "(Unnamed File)")
            content = item.get("content", "")
            web_url = item.get("web_url", "")
            mime_type = item.get("mime_type", "")
            drive_id = item.get("drive_id", "")
            last_modified = item.get("last_modified", "")
            last_modified_by = item.get("last_modified_by", "")

            # Name: what the LLM sees in the selection list
            display_name = name
            if last_modified_by:
                display_name = f"{name} (by {last_modified_by})"

            # Preview: metadata-only for initial triage (no content yet).
            # The LLM decides which files to preview based on name, type,
            # date, and author — content is revealed in the preview phase.
            meta_parts = []
            if mime_type:
                meta_parts.append(mime_type)
            if last_modified:
                date_short = (
                    last_modified[:10] if len(last_modified) >= 10 else last_modified
                )
                meta_parts.append(date_short)
            size_chars = len(content)
            meta_parts.append(f"{size_chars} chars")
            preview = " | ".join(meta_parts) if meta_parts else ""

            search_item = SearchItem(
                id=item_id,
                name=display_name,
                type="sharepoint_file",
                preview=preview,
                metadata={
                    "index": idx,
                    "name": name,
                    "web_url": web_url,
                    "mime_type": mime_type,
                    "drive_id": drive_id,
                    "last_modified": last_modified,
                    "last_modified_by": last_modified_by,
                },
                requires_extraction=True,
            )
            search_items.append(search_item)

        self._search_items = search_items

        print(
            f"{common.PRNT_API} [SharePointProvider] Loaded {len(search_items)} file(s)",
            flush=True,
        )

        return search_items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given SharePoint files.

        The LLM uses this preview to decide which files to read fully.

        Args:
            items: List of file items to preview

        Returns:
            Items with preview field populated with content snippet
        """
        for item in items:
            idx = item.metadata.get("index") if item.metadata else None
            if idx is not None and 0 <= idx < len(self.items):
                sp_item = self.items[idx]
                content = sp_item.get("content", "")
                name = sp_item.get("name", "(Unnamed)")
                web_url = sp_item.get("web_url", "")
                last_modified_by = sp_item.get("last_modified_by", "")

                preview_parts = [f"File: {name}"]
                if last_modified_by:
                    preview_parts.append(f"Last modified by: {last_modified_by}")
                if web_url:
                    preview_parts.append(f"URL: {web_url}")
                snippet = content[:SHAREPOINT_PREVIEW_LENGTH]
                if len(content) > SHAREPOINT_PREVIEW_LENGTH:
                    snippet += "..."
                preview_parts.append(f"Content: {snippet}")

                item.preview = "\n".join(preview_parts)

        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given SharePoint files.

        Args:
            items: List of file items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        context = []

        for item in items:
            idx = item.metadata.get("index") if item.metadata else None

            if idx is not None and 0 <= idx < len(self.items):
                sp_item = self.items[idx]
                name = sp_item.get("name", "(Unnamed)")
                content = sp_item.get("content", "")
                web_url = sp_item.get("web_url", "")
                drive_id = sp_item.get("drive_id", "")
                last_modified_by = sp_item.get("last_modified_by", "")

                # Truncate content if too long
                if len(content) > DEFAULT_CONTENT_EXTRACT_LENGTH:
                    content = (
                        content[:DEFAULT_CONTENT_EXTRACT_LENGTH] + "\n... [truncated]"
                    )

                # Format as a structured file document
                content_parts = [f"File: {name}"]
                if last_modified_by:
                    content_parts.append(f"Last modified by: {last_modified_by}")
                if web_url:
                    content_parts.append(f"SharePoint URL: {web_url}")
                if drive_id:
                    content_parts.append(f"Drive ID: {drive_id}")
                content_parts.append("")
                content_parts.append(content)

                context.append(
                    {
                        "source": f"[sharepoint] {name}",
                        "content": "\n".join(content_parts),
                    }
                )
            else:
                # Fallback to preview if index not found
                context.append(
                    {
                        "source": f"[sharepoint] {item.name}",
                        "content": item.preview or "(no content available)",
                    }
                )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """No expansion for SharePoint (frontend controls scope)."""
        return []

    @property
    def supports_grep(self) -> bool:
        """SharePoint provider supports grep pre-filtering."""
        return True

    @property
    def grep_fields(self) -> List[str]:
        """Available fields for grep searching."""
        return ["name", "content", "web_url", "last_modified_by"]

    async def grep(
        self, items: List[SearchItem], pattern: str, **kwargs
    ) -> List[SearchItem]:
        """
        Filter SharePoint files by text pattern matching.

        Searches across file name and content.

        Args:
            items: SearchItems to filter
            pattern: Text pattern to search for

        Returns:
            Filtered list of matching SearchItems
        """
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

        matching = []
        for item in items:
            idx = item.metadata.get("index") if item.metadata else None
            if idx is not None and 0 <= idx < len(self.items):
                sp_item = self.items[idx]
                fields_to_search = [
                    str(sp_item.get("name", "")),
                    str(sp_item.get("content", "")),
                    str(sp_item.get("web_url", "")),
                    str(sp_item.get("last_modified_by", "")),
                ]
                if any(regex.search(field) for field in fields_to_search):
                    matching.append(item)

        return matching
