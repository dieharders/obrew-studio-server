"""Item preview tool for showing item structure and metadata."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Preview a specific structured item, showing its metadata and content structure without full extraction."""

    item_id: str = Field(
        ...,
        description="The ID of the item to preview.",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of structured items to search in.",
    )
    max_depth: int = Field(
        default=2,
        description="Maximum depth to show for nested object structures.",
    )
    max_string_preview: int = Field(
        default=500,
        description="Maximum length of string content preview.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "item_id": "config-001",
                    "items": [
                        {
                            "id": "config-001",
                            "name": "app_config",
                            "content": {"auth": {"provider": "jwt"}},
                            "metadata": {"category": "settings"},
                        }
                    ],
                    "max_depth": 2,
                }
            ]
        }
    }


def _find_item_by_id(item_id: str, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find an item by ID or index."""
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


def _summarize_object(obj: Any, max_depth: int, current_depth: int = 0) -> Any:
    """Create a summary of an object structure up to max_depth."""
    if current_depth >= max_depth:
        if isinstance(obj, dict):
            return f"{{...{len(obj)} keys}}"
        elif isinstance(obj, list):
            return f"[...{len(obj)} items]"
        elif isinstance(obj, str):
            return obj[:100] + ("..." if len(obj) > 100 else "")
        return obj

    if isinstance(obj, dict):
        return {
            key: _summarize_object(value, max_depth, current_depth + 1)
            for key, value in list(obj.items())[:10]  # Limit keys shown
        }
    elif isinstance(obj, list):
        if len(obj) <= 3:
            return [_summarize_object(item, max_depth, current_depth + 1) for item in obj]
        else:
            # Show first 2 and indicate more
            preview = [_summarize_object(item, max_depth, current_depth + 1) for item in obj[:2]]
            preview.append(f"...+{len(obj) - 2} more items")
            return preview
    elif isinstance(obj, str):
        if len(obj) > 200:
            return obj[:200] + "..."
        return obj
    else:
        return obj


def _get_structure_info(content: Any) -> Dict[str, Any]:
    """Get detailed structure information about content."""
    if isinstance(content, str):
        return {
            "type": "string",
            "length": len(content),
            "line_count": content.count("\n") + 1,
        }
    elif isinstance(content, dict):
        return {
            "type": "object",
            "keys": list(content.keys()),
            "key_count": len(content),
            "nested_types": {
                key: type(value).__name__
                for key, value in list(content.items())[:10]
            },
        }
    elif isinstance(content, list):
        item_types = set(type(item).__name__ for item in content[:10])
        return {
            "type": "array",
            "length": len(content),
            "item_types": list(item_types),
        }
    elif isinstance(content, (int, float)):
        return {"type": "number", "value": content}
    elif isinstance(content, bool):
        return {"type": "boolean", "value": content}
    elif content is None:
        return {"type": "null"}
    else:
        return {"type": type(content).__name__}


async def main(**kwargs) -> Dict[str, Any]:
    """
    Preview a specific structured item.
    Returns dict with: id, name, metadata, content_preview, structure
    """
    item_id = kwargs.get("item_id")
    items = kwargs.get("items", [])
    max_depth = kwargs.get("max_depth", 2)
    max_string_preview = kwargs.get("max_string_preview", 500)

    if not item_id:
        raise ValueError("item_id is required")

    if not items:
        raise ValueError("items list is empty")

    item = _find_item_by_id(item_id, items)
    if not item:
        raise ValueError(f"Item not found: {item_id}")

    content = item.get("content")
    metadata = item.get("metadata", {})

    # Generate content preview based on type
    if isinstance(content, str):
        content_preview = content[:max_string_preview]
        if len(content) > max_string_preview:
            content_preview += "..."
    else:
        content_preview = _summarize_object(content, max_depth)

    return {
        "id": item.get("id") or item_id,
        "name": item.get("name") or "Unnamed",
        "metadata": metadata,
        "content_preview": content_preview,
        "structure": _get_structure_info(content),
    }
