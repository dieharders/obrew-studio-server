"""Shared utility functions for item tools."""

from typing import List, Dict, Any, Optional


def get_context_items(kwargs: Dict[str, Any]) -> list:
    """Extract context items from request.state.context_items.

    Args:
        kwargs: Tool kwargs dict, expected to contain a 'request' key

    Returns:
        List of context item dicts, or empty list if unavailable
    """
    request = kwargs.get("request")
    if (
        request
        and hasattr(request, "state")
        and hasattr(request.state, "context_items")
    ):
        return request.state.context_items or []
    return []


def find_item_by_id(
    item_id: str, items: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find an item by ID or index.

    Supports:
    - Exact ID match (e.g., "config-001")
    - Index-based lookup with any prefix (e.g., "item_0", "sp_file_3", "email_1")
    - Plain numeric index (e.g., "0", "3")

    Args:
        item_id: The ID or index-based identifier of the item
        items: List of items to search in

    Returns:
        The matching item or None if not found
    """
    # Try exact ID match first
    for item in items:
        if item.get("id") == item_id:
            return item

    # Try index-based lookup: extract trailing integer after last "_"
    # Handles "item_0", "sp_file_3", "email_1", or plain "0"
    try:
        parts = item_id.rsplit("_", 1)
        if len(parts) == 2:
            idx = int(parts[1])
        else:
            idx = int(item_id)
        if 0 <= idx < len(items):
            return items[idx]
    except (ValueError, IndexError):
        pass

    return None
