"""SharePoint file read tool for reading pre-fetched SharePoint file content.

Requires request.state.context_items to be populated with SharePoint file
objects by the frontend/middleware before tool invocation. If not set,
returns empty results.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from .._item_utils import find_item_by_id


class Params(BaseModel):
    """Read the contents of a SharePoint file by ID. The file data must be pre-fetched by the frontend from Microsoft Graph API and available in context items."""

    file_id: str = Field(
        ...,
        description="The ID of the SharePoint file to read.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number (1-indexed). If not provided, starts from beginning.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (inclusive). If not provided, reads to end.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_id": "01ABCDEF...",
                },
                {
                    "file_id": "sp_file_0",
                    "start_line": 2,
                    "end_line": 4,
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


async def main(**kwargs) -> Dict[str, Any]:
    """
    Read the contents of a SharePoint file by ID from context items.
    Returns dict with: content, file_name, total_lines, lines_read, start_line, end_line
    """
    file_id = kwargs.get("file_id")
    start_line = kwargs.get("start_line")
    end_line = kwargs.get("end_line")

    if not file_id:
        raise ValueError("file_id is required")

    items = _get_context_items(kwargs)
    if not items:
        return {
            "content": "",
            "file_name": "unknown",
            "total_lines": 0,
            "lines_read": 0,
            "start_line": 0,
            "end_line": 0,
        }

    item = find_item_by_id(file_id, items)
    if not item:
        raise ValueError(f"SharePoint file not found: {file_id}")

    file_name = item.get("name", "unknown")
    file_content = item.get("content", "")

    if not file_content:
        return {
            "content": "",
            "file_name": file_name,
            "total_lines": 0,
            "lines_read": 0,
            "start_line": 0,
            "end_line": 0,
        }

    all_lines = file_content.splitlines(keepends=True)
    total_lines = len(all_lines)

    # Validate line numbers
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    # Determine actual line range
    actual_start = (start_line or 1) - 1  # Convert to 0-indexed
    actual_end = end_line or total_lines

    # Clamp to valid range
    actual_start = max(0, min(actual_start, total_lines))
    actual_end = max(actual_start, min(actual_end, total_lines))

    selected_lines = all_lines[actual_start:actual_end]
    content = "".join(selected_lines)

    return {
        "content": content,
        "file_name": file_name,
        "total_lines": total_lines,
        "lines_read": len(selected_lines),
        "start_line": actual_start + 1,
        "end_line": actual_end,
    }
