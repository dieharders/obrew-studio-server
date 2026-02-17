"""SharePoint file list tool for listing pre-fetched SharePoint file metadata.

Requires request.state.context_items to be populated with SharePoint file
objects by the frontend/middleware before tool invocation. If not set,
returns empty results.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """List available SharePoint files showing metadata (name, size, type, date, author). Use this to discover what files are available before reading them."""

    filter_name: Optional[str] = Field(
        default=None,
        description="Optional substring filter for file names (case-insensitive).",
    )
    filter_type: Optional[str] = Field(
        default=None,
        description="Optional filter for MIME type (case-insensitive substring match).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {},
                {
                    "filter_name": "report",
                    "filter_type": "spreadsheet",
                },
            ]
        }
    }


def _get_context_items(kwargs: Dict[str, Any]) -> list:
    """Extract SharePoint items from request.state.context_items."""
    request = kwargs.get("request")
    if (
        request
        and hasattr(request, "state")
        and hasattr(request.state, "context_items")
    ):
        return request.state.context_items or []
    return []


def _format_size(bytes_val: int) -> str:
    """Format byte count to human-readable string."""
    if bytes_val <= 0:
        return "0 B"
    sizes = ["B", "KB", "MB", "GB"]
    i = 0
    val = float(bytes_val)
    while val >= 1024 and i < len(sizes) - 1:
        val /= 1024
        i += 1
    return f"{val:.1f} {sizes[i]}"


async def main(**kwargs) -> Dict[str, Any]:
    """
    List SharePoint files from context items with metadata.
    Returns dict with: files (list), total_count, filtered_count
    """
    filter_name = kwargs.get("filter_name")
    filter_type = kwargs.get("filter_type")

    items = _get_context_items(kwargs)

    if not items:
        return {
            "files": [],
            "total_count": 0,
            "filtered_count": 0,
        }

    total_count = len(items)

    # Apply filters
    filtered = items
    if filter_name:
        filter_name_lower = filter_name.lower()
        filtered = [
            f for f in filtered
            if filter_name_lower in str(f.get("name", "")).lower()
        ]
    if filter_type:
        filter_type_lower = filter_type.lower()
        filtered = [
            f for f in filtered
            if filter_type_lower in str(f.get("mime_type", "")).lower()
        ]

    # Format output
    formatted_files = []
    for f in filtered:
        entry = {
            "id": f.get("id", ""),
            "name": f.get("name", "(unnamed)"),
            "size": _format_size(f.get("size", 0)),
            "web_url": f.get("web_url", ""),
            "mime_type": f.get("mime_type", ""),
            "last_modified": f.get("last_modified", ""),
            "last_modified_by": f.get("last_modified_by", ""),
        }
        formatted_files.append(entry)

    return {
        "files": formatted_files,
        "total_count": total_count,
        "filtered_count": len(filtered),
    }
