"""SharePoint file read tool for reading pre-fetched SharePoint file content."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Read the contents of a SharePoint file. The content must be pre-fetched by the frontend from Microsoft Graph API."""

    file_content: str = Field(
        ...,
        description="The pre-fetched text content of the SharePoint file.",
    )
    file_name: str = Field(
        ...,
        description="The name of the SharePoint file.",
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
                    "file_content": "This is the content of a SharePoint document...",
                    "file_name": "report.txt",
                },
                {
                    "file_content": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
                    "file_name": "notes.md",
                    "start_line": 2,
                    "end_line": 4,
                },
            ]
        }
    }


async def main(**kwargs) -> Dict[str, Any]:
    """
    Read the contents of a pre-fetched SharePoint file.
    Returns dict with: content, file_name, total_lines, lines_read, start_line, end_line
    """
    file_content = kwargs.get("file_content")
    file_name = kwargs.get("file_name", "unknown")
    start_line = kwargs.get("start_line")
    end_line = kwargs.get("end_line")

    if not file_content:
        raise ValueError("file_content is required")

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
