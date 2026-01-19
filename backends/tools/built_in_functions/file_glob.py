"""File glob tool for pattern-based file matching in directories."""
import glob
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Find files matching a glob pattern within a directory. Use wildcards like *.pdf, **/*.txt for recursive search."""

    pattern: str = Field(
        ...,
        description="The glob pattern to match files (e.g., '*.pdf', '**/*.txt', 'report_*.docx').",
    )
    directory_path: str = Field(
        ...,
        description="The directory path to search in.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pattern": "*.pdf",
                    "directory_path": "/documents/reports",
                },
                {
                    "pattern": "**/*.txt",
                    "directory_path": "/home/user/notes",
                },
            ]
        }
    }


async def main(**kwargs) -> List[str]:
    """
    Find all files matching the glob pattern in the specified directory.
    Returns a list of absolute file paths.
    """
    pattern = kwargs.get("pattern", "*")
    directory_path = kwargs.get("directory_path", ".")

    # Ensure directory exists
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Build the full pattern
    full_pattern = str(dir_path / pattern)

    # Find all matching files
    matches = glob.glob(full_pattern, recursive=True)

    # Filter to only files (not directories) and convert to absolute paths
    file_matches = [
        str(Path(m).resolve()) for m in matches if Path(m).is_file()
    ]

    return sorted(file_matches)
