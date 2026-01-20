"""
StructuredProvider - Search provider for client-provided structured data.

This provider implements the SearchProvider protocol for searching
over ephemeral data sent by frontend applications. The data exists only
for the duration of the request - nothing is stored on the server.

Use cases:
- Searching conversation history
- Finding relevant project metadata
- Searching workflow data
- Any data the frontend has that the server cannot access
"""

from typing import List, Dict, Optional

from ..base import SearchProvider, SearchItem
from ..classes import StructuredItem


class StructuredProvider(SearchProvider):
    """
    Search provider for structured client-provided data.

    Unlike other providers that access server-side resources (files, vector
    databases, web), this provider operates on client-provided data that
    exists only for the duration of the request.
    """

    def __init__(
        self,
        app,
        items: List[StructuredItem],
        item_type: str = "item",
    ):
        """
        Initialize the StructuredProvider.

        Args:
            app: FastAPI application instance
            items: List of structured items to search over
            item_type: Type label for items (e.g., "conversation", "memo")
        """
        self.app = app
        self.items = items
        self.item_type = item_type
        self._search_items: List[SearchItem] = []

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover items from the provided structured data.

        Unlike other providers that scan external resources, this provider
        already has all items - just convert them to SearchItem format.

        Args:
            scope: Ignored - items are provided at initialization
            **kwargs: Additional arguments (must include 'query')

        Returns:
            List of SearchItem objects
        """
        from core import common

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for structured search")

        # Return cached items if already processed
        if self._search_items:
            return self._search_items

        for idx, item in enumerate(self.items):
            # Generate id/name if not provided
            item_id = item.id or f"{self.item_type}_{idx}"
            item_name = item.name or f"Item {idx}"

            # Truncate content for preview (first ~300 chars)
            preview = item.content[:300] if item.content else ""
            if len(item.content) > 300:
                preview += "..."

            search_item = SearchItem(
                id=item_id,
                name=item_name,
                type=self.item_type,
                preview=preview,
                metadata={
                    "index": idx,
                    "content_length": len(item.content),
                    **(item.metadata or {}),
                },
                requires_extraction=len(item.content) > 300,
            )
            self._search_items.append(search_item)

        print(
            f"{common.PRNT_API} [StructuredProvider] Loaded {len(self._search_items)} {self.item_type}(s)",
            flush=True,
        )

        return self._search_items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given items.

        Preview is already populated from discover phase.

        Args:
            items: List of items to preview

        Returns:
            Same items (preview already populated)
        """
        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given items.

        Retrieves the original full content from the StructuredItem list.

        Args:
            items: List of items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        context = []

        for item in items:
            # Get index from metadata to retrieve original content
            idx = item.metadata.get("index") if item.metadata else None

            if idx is not None and 0 <= idx < len(self.items):
                original = self.items[idx]
                context.append(
                    {
                        "source": f"[{self.item_type}] {item.name}",
                        "content": original.content[:5000],  # Limit context size
                    }
                )
            else:
                # Fallback to preview if index not found
                context.append(
                    {
                        "source": f"[{self.item_type}] {item.name}",
                        "content": item.preview or "",
                    }
                )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return additional scopes that can be searched.

        Structured data does not support scope expansion - all data
        is provided upfront by the client.

        Args:
            current_scope: Ignored

        Returns:
            Empty list (no expansion available)
        """
        return []
