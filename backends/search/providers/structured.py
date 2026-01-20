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

from typing import List, Dict, Optional, Any

from ..harness import SearchProvider, SearchItem
from ..classes import StructuredItem


def _get_content_preview(content: Any, max_length: int = 300) -> str:
    """Generate a preview string from content based on its type."""
    if isinstance(content, str):
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content
    elif isinstance(content, dict):
        keys = list(content.keys())
        if len(keys) <= 5:
            return f"{{keys: {keys}}}"
        return f"{{keys: {keys[:5]}... (+{len(keys) - 5} more)}}"
    elif isinstance(content, list):
        return f"[array with {len(content)} items]"
    else:
        str_val = str(content)
        if len(str_val) > max_length:
            return str_val[:max_length] + "..."
        return str_val


def _get_content_length(content: Any) -> int:
    """Get a size metric for content based on its type."""
    if isinstance(content, str):
        return len(content)
    elif isinstance(content, dict):
        # Approximate size by converting to string
        return len(str(content))
    elif isinstance(content, list):
        return len(content)
    else:
        return len(str(content))


def _content_to_string(content: Any, max_length: int = 5000) -> str:
    """Convert content to a string representation for extraction."""
    if isinstance(content, str):
        return content[:max_length]
    elif isinstance(content, (dict, list)):
        import json
        try:
            result = json.dumps(content, indent=2, default=str)
            return result[:max_length]
        except (TypeError, ValueError):
            return str(content)[:max_length]
    else:
        return str(content)[:max_length]


class StructuredProvider(SearchProvider):
    """
    Search provider for structured client-provided data.

    Unlike other providers that access server-side resources (files, vector
    databases, web), this provider operates on client-provided data that
    exists only for the duration of the request.

    Supports grouping items by a metadata field for auto_expand functionality.
    """

    def __init__(
        self,
        app,
        items: List[StructuredItem],
        group_by: Optional[str] = None,
    ):
        """
        Initialize the StructuredProvider.

        Args:
            app: FastAPI application instance
            items: List of structured items to search over
            group_by: Optional metadata field to group items by for expansion
        """
        self.app = app
        self.items = items
        self.group_by = group_by
        self._search_items: List[SearchItem] = []
        self._groups: Dict[str, List[int]] = {}  # group_value -> item indices
        self._searched_groups: List[str] = []

        # Build group index if group_by is specified
        if group_by:
            self._build_group_index()

    def _build_group_index(self):
        """Build index of items by group field."""
        for idx, item in enumerate(self.items):
            if item.metadata:
                group_value = str(item.metadata.get(self.group_by, "_ungrouped"))
            else:
                group_value = "_ungrouped"

            if group_value not in self._groups:
                self._groups[group_value] = []
            self._groups[group_value].append(idx)

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover items from the provided structured data.

        Unlike other providers that scan external resources, this provider
        already has all items - just convert them to SearchItem format.

        Args:
            scope: When using group_by, specifies which group to search.
                   If None and grouping enabled, searches first group.
            **kwargs: Additional arguments (must include 'query')

        Returns:
            List of SearchItem objects
        """
        from core import common

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for structured search")

        # Determine which items to process based on scope/grouping
        if self.group_by and self._groups:
            if scope is not None:
                # Search specific group
                indices = self._groups.get(scope, [])
                items_to_process = [(idx, self.items[idx]) for idx in indices]
                if scope not in self._searched_groups:
                    self._searched_groups.append(scope)
            elif not self._searched_groups:
                # No scope specified, first call - use first group
                first_group = list(self._groups.keys())[0]
                indices = self._groups[first_group]
                items_to_process = [(idx, self.items[idx]) for idx in indices]
                self._searched_groups.append(first_group)
            else:
                # Already searched, return cached items for current scope
                return self._search_items
        else:
            # No grouping - process all items (only once)
            if self._search_items:
                return self._search_items
            items_to_process = list(enumerate(self.items))

        # Convert to SearchItems
        search_items = []
        for idx, item in items_to_process:
            # Generate id/name if not provided
            item_id = item.id or f"item_{idx}"
            item_name = item.name or f"Item {idx}"

            # Generate preview based on content type
            preview = _get_content_preview(item.content)
            content_length = _get_content_length(item.content)

            search_item = SearchItem(
                id=item_id,
                name=item_name,
                type="item",
                preview=preview,
                metadata={
                    "index": idx,
                    "content_length": content_length,
                    "content_type": type(item.content).__name__,
                    **(item.metadata or {}),
                },
                requires_extraction=content_length > 300,
            )
            search_items.append(search_item)

        # Cache items (append for grouped, replace for ungrouped)
        if self.group_by:
            self._search_items.extend(search_items)
        else:
            self._search_items = search_items

        scope_info = f" (group: {scope})" if scope else ""
        print(
            f"{common.PRNT_API} [StructuredProvider] Loaded {len(search_items)} item(s){scope_info}",
            flush=True,
        )

        return search_items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given items.

        For structured data, generates smarter previews showing
        metadata and content structure.

        Args:
            items: List of items to preview

        Returns:
            Items with enhanced preview information
        """
        for item in items:
            idx = item.metadata.get("index") if item.metadata else None
            if idx is not None and 0 <= idx < len(self.items):
                original = self.items[idx]
                # Enhance preview with metadata info
                metadata_preview = ""
                if original.metadata:
                    meta_keys = list(original.metadata.keys())[:3]
                    meta_vals = [f"{k}={original.metadata[k]}" for k in meta_keys]
                    metadata_preview = f" [{', '.join(meta_vals)}]"

                content_preview = _get_content_preview(original.content, 400)
                item.preview = f"{content_preview}{metadata_preview}"

        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given items.

        Retrieves the original full content from the StructuredItem list.
        Handles both string and object content.

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
                content_str = _content_to_string(original.content)
                context.append(
                    {
                        "source": f"[item] {item.name}",
                        "content": content_str,
                    }
                )
            else:
                # Fallback to preview if index not found
                context.append(
                    {
                        "source": f"[item] {item.name}",
                        "content": item.preview or "",
                    }
                )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return additional scopes (groups) that can be searched.

        When group_by is configured, returns other group values
        that haven't been searched yet.

        Args:
            current_scope: The group that was just searched

        Returns:
            List of other group values not yet searched
        """
        if not self.group_by:
            return []  # No expansion without grouping

        return [
            group for group in self._groups.keys()
            if group not in self._searched_groups
        ]
