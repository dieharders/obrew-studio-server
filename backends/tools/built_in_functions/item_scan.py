"""Item scan tool for listing structured items with metadata."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """List all structured items with their metadata and content structure preview."""

    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of structured items to scan. Each item should have id, name, content, and optional metadata.",
    )
    max_preview_length: int = Field(
        default=200,
        description="Maximum length of content preview for string content.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "id": "item-001",
                            "name": "config",
                            "content": {"key": "value"},
                            "metadata": {"category": "settings"},
                        }
                    ],
                    "max_preview_length": 200,
                }
            ]
        }
    }


def _get_content_preview(content: Any, max_length: int) -> str:
    """Generate a preview of the content based on its type."""
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
        return str(content)[:max_length]


def _get_content_structure(content: Any) -> Dict[str, Any]:
    """Get structure information about the content."""
    if isinstance(content, str):
        return {"type": "string", "length": len(content)}
    elif isinstance(content, dict):
        return {"type": "object", "keys": list(content.keys()), "key_count": len(content)}
    elif isinstance(content, list):
        return {"type": "array", "length": len(content)}
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
    Scan structured items and return summary information.
    Returns dict with: items (list of item summaries), total_count
    """
    items = kwargs.get("items", [])
    max_preview_length = kwargs.get("max_preview_length", 200)

    if not items:
        return {"items": [], "total_count": 0}

    scanned_items = []
    for idx, item in enumerate(items):
        item_id = item.get("id") or f"item_{idx}"
        item_name = item.get("name") or f"Item {idx}"
        content = item.get("content")
        metadata = item.get("metadata", {})

        scanned_items.append({
            "id": item_id,
            "name": item_name,
            "content_preview": _get_content_preview(content, max_preview_length),
            "content_structure": _get_content_structure(content),
            "metadata": metadata,
            "metadata_keys": list(metadata.keys()) if metadata else [],
        })

    return {
        "items": scanned_items,
        "total_count": len(scanned_items),
    }
