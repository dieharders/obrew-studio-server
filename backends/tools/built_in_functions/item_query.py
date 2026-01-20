"""Object query tool for advanced navigation and querying into nested object structures."""
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Query into nested object structures within a structured item. Supports wildcards and advanced paths."""

    item_id: str = Field(
        ...,
        description="The ID of the item to query.",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of structured items to search in.",
    )
    path: str = Field(
        ...,
        description="Path to query (e.g., 'data.users[0].name', 'results[*].score' for all scores, 'config.*.enabled' for wildcard key matching).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "item_id": "data-001",
                    "items": [
                        {
                            "id": "data-001",
                            "name": "users",
                            "content": {
                                "users": [
                                    {"name": "Alice", "role": "admin"},
                                    {"name": "Bob", "role": "user"},
                                ]
                            },
                        }
                    ],
                    "path": "users[*].name",
                }
            ]
        }
    }


def _find_item_by_id(item_id: str, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find an item by ID or index."""
    for item in items:
        if item.get("id") == item_id:
            return item

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


def _query_path(obj: Any, path: str) -> Dict[str, Any]:
    """
    Query into an object using an advanced path with wildcard support.

    Supports:
    - 'key' - simple key access
    - 'key.nested' - nested access
    - 'arr[0]' - array index
    - 'arr[*]' - all array elements
    - 'obj.*' - all object values
    - 'arr[*].field' - field from all array elements
    """
    if not path:
        return {"result": obj, "is_multiple": False}

    # Parse path into segments
    segments = []
    remaining = path

    while remaining:
        # Match array access: [0] or [*]
        array_match = re.match(r'^\[(\d+|\*)\]', remaining)
        if array_match:
            idx = array_match.group(1)
            segments.append(("index", idx))
            remaining = remaining[array_match.end():]
            if remaining.startswith('.'):
                remaining = remaining[1:]
            continue

        # Match key followed by array access or dot
        key_match = re.match(r'^([^.\[\]]+)', remaining)
        if key_match:
            key = key_match.group(1)
            segments.append(("key", key))
            remaining = remaining[key_match.end():]
            if remaining.startswith('.'):
                remaining = remaining[1:]
            continue

        raise ValueError(f"Invalid path syntax at: {remaining}")

    # Execute query
    results = [obj]
    is_multiple = False

    for seg_type, seg_value in segments:
        new_results = []

        for current in results:
            if seg_type == "key":
                if seg_value == "*":
                    # Wildcard key - get all values
                    if isinstance(current, dict):
                        new_results.extend(current.values())
                        is_multiple = True
                    else:
                        raise TypeError(f"Cannot use wildcard key on {type(current).__name__}")
                else:
                    # Regular key access
                    if isinstance(current, dict):
                        if seg_value in current:
                            new_results.append(current[seg_value])
                        else:
                            raise KeyError(f"Key not found: {seg_value}")
                    else:
                        raise TypeError(f"Cannot access key '{seg_value}' on {type(current).__name__}")

            elif seg_type == "index":
                if seg_value == "*":
                    # Wildcard index - get all elements
                    if isinstance(current, list):
                        new_results.extend(current)
                        is_multiple = True
                    else:
                        raise TypeError(f"Cannot iterate over {type(current).__name__}")
                else:
                    # Regular index access
                    idx = int(seg_value)
                    if isinstance(current, list):
                        if 0 <= idx < len(current):
                            new_results.append(current[idx])
                        else:
                            raise IndexError(f"Index out of range: {idx}")
                    else:
                        raise TypeError(f"Cannot index into {type(current).__name__}")

        results = new_results

    if is_multiple:
        return {"result": results, "is_multiple": True, "count": len(results)}
    elif len(results) == 1:
        return {"result": results[0], "is_multiple": False}
    elif len(results) == 0:
        return {"result": None, "is_multiple": False, "error": "No results found"}
    else:
        return {"result": results, "is_multiple": True, "count": len(results)}


async def main(**kwargs) -> Dict[str, Any]:
    """
    Query into nested object structures.
    Returns dict with: id, name, path, result, type, is_multiple, count (if multiple)
    """
    item_id = kwargs.get("item_id")
    items = kwargs.get("items", [])
    path = kwargs.get("path")

    if not item_id:
        raise ValueError("item_id is required")

    if not path:
        raise ValueError("path is required")

    if not items:
        raise ValueError("items list is empty")

    item = _find_item_by_id(item_id, items)
    if not item:
        raise ValueError(f"Item not found: {item_id}")

    content = item.get("content")

    try:
        query_result = _query_path(content, path)
    except (KeyError, TypeError, IndexError, ValueError) as e:
        raise ValueError(f"Query failed: {e}")

    result = query_result["result"]
    is_multiple = query_result.get("is_multiple", False)

    response = {
        "id": item.get("id") or item_id,
        "name": item.get("name") or "Unnamed",
        "path": path,
        "result": result,
        "type": type(result).__name__ if not is_multiple else f"list[{type(result[0]).__name__}]" if result else "list",
        "is_multiple": is_multiple,
    }

    if is_multiple:
        response["count"] = query_result.get("count", len(result) if result else 0)

    return response
