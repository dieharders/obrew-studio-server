"""Shared utility functions for item tools."""

from typing import List, Dict, Any, Optional


def find_item_by_id(
    item_id: str, items: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find an item by ID or index.

    Supports:
    - Exact ID match (e.g., "config-001")
    - Index-based lookup (e.g., "item_0" or "0")

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

    # Try index-based lookup (e.g., "item_0", "0")
    try:
        if item_id.startswith("item_"):
            idx = int(item_id.split("_")[1])
        else:
            idx = int(item_id)
        if 0 <= idx < len(items):
            return items[idx]
    except (ValueError, IndexError):
        pass

    return None
