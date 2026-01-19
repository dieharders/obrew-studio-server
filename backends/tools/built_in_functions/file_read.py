"""File read tool for reading raw file contents with optional line ranges."""
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Read the raw contents of a text file. Optionally specify line ranges for large files."""

    file_path: str = Field(
        ...,
        description="The path to the file to read.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number (1-indexed). If not provided, starts from beginning.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (inclusive). If not provided, reads to end of file.",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding (default: utf-8).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/documents/notes.txt",
                },
                {
                    "file_path": "/logs/app.log",
                    "start_line": 100,
                    "end_line": 150,
                },
            ]
        }
    }


async def main(**kwargs) -> Dict[str, Any]:
    """
    Read the contents of a text file.
    Returns dict with: content, total_lines, lines_read, start_line, end_line
    """
    file_path_str = kwargs.get("file_path")
    start_line = kwargs.get("start_line")
    end_line = kwargs.get("end_line")
    encoding = kwargs.get("encoding", "utf-8")

    if not file_path_str:
        raise ValueError("file_path is required")

    file_path = Path(file_path_str)

    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path_str}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path_str}")

    # Validate line numbers
    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            all_lines = f.readlines()
    except IOError as e:
        raise ValueError(f"Unable to read file: {e}")

    total_lines = len(all_lines)

    # Determine actual line range
    actual_start = (start_line or 1) - 1  # Convert to 0-indexed
    actual_end = end_line or total_lines

    # Clamp to valid range
    actual_start = max(0, min(actual_start, total_lines))
    actual_end = max(0, min(actual_end, total_lines))

    # Extract requested lines
    selected_lines = all_lines[actual_start:actual_end]
    content = "".join(selected_lines)

    return {
        "content": content,
        "total_lines": total_lines,
        "lines_read": len(selected_lines),
        "start_line": actual_start + 1,  # Convert back to 1-indexed
        "end_line": actual_end,
        "file_path": str(file_path.resolve()),
    }
