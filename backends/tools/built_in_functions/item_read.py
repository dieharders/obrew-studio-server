"""Item read tool for reading full content or navigating into nested structures."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .._item_utils import find_item_by_id


class Params(BaseModel):
    """Read the full content of a structured item, or navigate to a specific path within it."""

    item_id: str = Field(
        ...,
        description="The ID of the item to read.",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of structured items to search in.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Optional path to navigate into the content (e.g., 'auth.provider', 'users[0].name'). If not provided, returns full content.",
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
                            "content": {"auth": {"provider": "jwt", "expiry": 3600}},
                        }
                    ],
                    "path": "auth.provider",
                }
            ]
        }
    }


def _navigate_path(obj: Any, path: str) -> Any:
    """
    Navigate into an object using a dot/bracket path.
    Supports: 'key', 'key.nested', 'arr[0]', 'key.arr[0].nested'
    """
    if not path:
        return obj

    current = obj
    # Split on dots, but handle array indices
    import re

    parts = re.split(r"\.(?![^\[]*\])", path)

    for part in parts:
        if not part:
            continue

        # Check for array index: key[0] or just [0]
        array_match = re.match(r"^(\w*)?\[(\d+)\]$", part)
        if array_match:
            key = array_match.group(1)
            idx = int(array_match.group(2))

            if key:
                if not isinstance(current, dict) or key not in current:
                    raise KeyError(f"Key not found: {key}")
                current = current[key]

            if not isinstance(current, list):
                raise TypeError(f"Cannot index into non-array at: {part}")
            if idx >= len(current):
                raise IndexError(f"Index out of range: {idx} (length: {len(current)})")
            current = current[idx]
        else:
            # Regular key access
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key not found: {part}")
                current = current[part]
            else:
                raise TypeError(
                    f"Cannot access key '{part}' on {type(current).__name__}"
                )

    return current


async def main(**kwargs) -> Dict[str, Any]:
    """
    Read content from a structured item, optionally navigating to a specific path.
    Returns dict with: id, name, path (if provided), content/value, type info
    """
    item_id = kwargs.get("item_id")
    items = kwargs.get("items", [])
    path = kwargs.get("path")

    if not item_id:
        raise ValueError("item_id is required")

    if not items:
        raise ValueError("items list is empty")

    item = find_item_by_id(item_id, items)
    if not item:
        raise ValueError(f"Item not found: {item_id}")

    content = item.get("content")

    # If path provided, navigate into the content
    if path:
        try:
            value = _navigate_path(content, path)
            return {
                "id": item.get("id") or item_id,
                "name": item.get("name") or "Unnamed",
                "path": path,
                "value": value,
                "type": type(value).__name__,
            }
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Path navigation failed: {e}")

    # Return full content with type info
    result = {
        "id": item.get("id") or item_id,
        "name": item.get("name") or "Unnamed",
        "type": type(content).__name__,
    }

    if isinstance(content, str):
        result["content"] = content
        result["total_chars"] = len(content)
    elif isinstance(content, dict):
        result["content"] = content
        result["keys"] = list(content.keys())
    elif isinstance(content, list):
        result["content"] = content
        result["length"] = len(content)
    else:
        result["content"] = content

    return result
